"""Async SQLite persistence layer for Life Agent (v2: Two-Phase Gating, Tiered Scene, FollowUps)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from .models import (
    BufferItem,
    BufferStatus,
    BufferZone,
    ChatTurn,
    FeedbackLogEntry,
    FollowUpIntent,
    FollowUpStatus,
    GatingRule,
    GatingRuleStatus,
    GradeSnapshot,
    PendingPush,
    PushStatus,
    ScheduledPassiveTrigger,
    SlowMemoryItem,
    SlowMemoryKind,
    SlowMemoryStatus,
    SourceAffinity,
    StateTier,
    TieredSceneEntry,
    TimetableEntry,
    TimetableStatus,
    TrustTier,
    utc_now,
)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(val: str | None) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val)
    except ValueError:
        return None


class Storage:
    """Structured SQLite store with smooth idempotent schema migrations."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS timetable (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                location TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT 'other',
                is_mandatory INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'scheduled',
                notes TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'manual',
                reminded INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scheduled_triggers (
                id TEXT PRIMARY KEY,
                trigger_name TEXT NOT NULL,
                trigger_at TEXT NOT NULL,
                directive TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'schedule_reminder',
                priority REAL NOT NULL DEFAULT 9.0,
                related_timetable_id TEXT,
                fired INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS slow_memory (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'idea',
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                planned_start TEXT,
                planned_end TEXT,
                location TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT 'other',
                estimated_minutes INTEGER,
                energy_level TEXT NOT NULL DEFAULT '',
                is_mandatory INTEGER NOT NULL DEFAULT 0,
                reminder_lead_minutes INTEGER NOT NULL DEFAULT 20,
                source TEXT NOT NULL DEFAULT 'manual',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS buffer_items (
                id TEXT PRIMARY KEY,
                zone TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                weight REAL NOT NULL DEFAULT 4.0,
                intrinsic_value REAL NOT NULL DEFAULT 4.0,
                urgency REAL NOT NULL DEFAULT 4.0,
                starvation_boost REAL NOT NULL DEFAULT 0.0,
                trust_tier TEXT NOT NULL DEFAULT 'medium',
                needs_confirmation INTEGER NOT NULL DEFAULT 0,
                return_count INTEGER NOT NULL DEFAULT 0,
                has_explicit_time INTEGER NOT NULL DEFAULT 0,
                is_mandatory_task INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                next_review_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scene_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tiered_scene_state (
                key TEXT PRIMARY KEY,
                tier TEXT NOT NULL DEFAULT 't0_ephemeral',
                value_json TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'system',
                expires_at TEXT,
                renewal_prompted INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS gating_rules (
                id TEXT PRIMARY KEY,
                scope_type TEXT NOT NULL DEFAULT 'topic',
                scope_key TEXT NOT NULL DEFAULT '',
                effect TEXT NOT NULL DEFAULT 'boost',
                instruction TEXT NOT NULL,
                score_delta REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL DEFAULT 1.0,
                origin TEXT NOT NULL DEFAULT 'default',
                expires_at TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS grade_snapshots (
                version INTEGER PRIMARY KEY AUTOINCREMENT,
                grade_text TEXT NOT NULL,
                rule_ids_json TEXT NOT NULL DEFAULT '[]',
                summary_note TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback_log (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                source_key TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                category TEXT NOT NULL DEFAULT '',
                activity_mode TEXT NOT NULL DEFAULT '',
                hour_bucket INTEGER NOT NULL DEFAULT 0,
                sentiment TEXT NOT NULL DEFAULT 'neutral',
                signal_value REAL NOT NULL DEFAULT 0.0,
                raw_text TEXT NOT NULL DEFAULT '',
                is_explicit INTEGER NOT NULL DEFAULT 0,
                processed_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_affinity (
                key TEXT PRIMARY KEY,
                affinity REAL NOT NULL DEFAULT 0.0,
                sample_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS follow_ups (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                context_hint TEXT NOT NULL DEFAULT '',
                intent_type TEXT NOT NULL DEFAULT 'deadline_backtrack',
                deadline_at TEXT,
                anchor_event_id TEXT,
                safety_margin_minutes INTEGER NOT NULL DEFAULT 60,
                computed_check_at TEXT,
                priority REAL NOT NULL DEFAULT 8.0,
                status TEXT NOT NULL DEFAULT 'open',
                check_count INTEGER NOT NULL DEFAULT 0,
                max_checks INTEGER NOT NULL DEFAULT 2,
                source TEXT NOT NULL DEFAULT 'conversation',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                channel TEXT NOT NULL DEFAULT 'qq',
                parsed_by_rtir INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pending_pushes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                trigger_source TEXT NOT NULL DEFAULT '',
                source_key TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                category TEXT NOT NULL DEFAULT 'general',
                urgency REAL NOT NULL DEFAULT 8.0,
                qq_target TEXT NOT NULL DEFAULT '',
                weixin_target TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'waiting_qq_reply',
                sent_at TEXT NOT NULL,
                failover_at TEXT NOT NULL,
                resolved_at TEXT,
                error_message TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS seen_external_ids (
                namespace TEXT NOT NULL,
                external_id TEXT NOT NULL,
                seen_at TEXT NOT NULL,
                PRIMARY KEY (namespace, external_id)
            );

            CREATE TABLE IF NOT EXISTS trigger_runs (
                trigger_key TEXT PRIMARY KEY,
                fired_at TEXT NOT NULL
            );
            """
        )
        await self._migrate_existing_tables()
        await self._conn.commit()

    async def _migrate_existing_tables(self) -> None:
        """Idempotently add new columns when upgrading an older SQLite database."""
        migrations = {
            "buffer_items": [
                ("intrinsic_value", "REAL NOT NULL DEFAULT 4.0"),
                ("urgency", "REAL NOT NULL DEFAULT 4.0"),
                ("starvation_boost", "REAL NOT NULL DEFAULT 0.0"),
                ("trust_tier", "TEXT NOT NULL DEFAULT 'medium'"),
                ("needs_confirmation", "INTEGER NOT NULL DEFAULT 0"),
            ],
            "pending_pushes": [
                ("source_key", "TEXT NOT NULL DEFAULT ''"),
                ("tags_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("category", "TEXT NOT NULL DEFAULT 'general'"),
                ("urgency", "REAL NOT NULL DEFAULT 8.0"),
            ],
        }
        for table, cols in migrations.items():
            cur = await self.conn.execute(f"PRAGMA table_info({table})")
            existing = {r["name"] for r in await cur.fetchall()}
            for col_name, col_def in cols:
                if col_name not in existing:
                    await self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Storage not initialized. Call await storage.init() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Timetable CRUD (Global Single Source of Truth <= 7 days)
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_timetable(row: aiosqlite.Row) -> TimetableEntry:
        return TimetableEntry(
            id=row["id"],
            title=row["title"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            location=row["location"],
            category=row["category"],
            is_mandatory=bool(row["is_mandatory"]),
            status=TimetableStatus(row["status"]),
            notes=row["notes"],
            source=row["source"],
            reminded=bool(row["reminded"]),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
        )

    async def upsert_timetable(self, entry: TimetableEntry) -> TimetableEntry:
        now = utc_now()
        cursor = await self.conn.execute("SELECT id FROM timetable WHERE id = ?", (entry.id,))
        existing = await cursor.fetchone()
        if existing is None:
            cursor2 = await self.conn.execute(
                "SELECT id FROM timetable WHERE lower(trim(title)) = lower(trim(?)) "
                "AND start_time = ? AND status != 'cancelled'",
                (entry.title, entry.start_time),
            )
            dup = await cursor2.fetchone()
            if dup is not None:
                entry.id = dup["id"]
                existing = dup

        if existing is not None:
            entry.updated_at = now
            await self.conn.execute(
                """
                UPDATE timetable
                SET title = ?, start_time = ?, end_time = ?, location = ?,
                    category = ?, is_mandatory = ?, status = ?, notes = ?,
                    source = ?, reminded = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    entry.title,
                    entry.start_time,
                    entry.end_time,
                    entry.location,
                    entry.category,
                    int(entry.is_mandatory),
                    entry.status.value,
                    entry.notes,
                    entry.source,
                    int(entry.reminded),
                    _iso(entry.updated_at),
                    entry.id,
                ),
            )
        else:
            entry.created_at = now
            entry.updated_at = now
            await self.conn.execute(
                """
                INSERT INTO timetable (
                    id, title, start_time, end_time, location, category,
                    is_mandatory, status, notes, source, reminded, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.title,
                    entry.start_time,
                    entry.end_time,
                    entry.location,
                    entry.category,
                    int(entry.is_mandatory),
                    entry.status.value,
                    entry.notes,
                    entry.source,
                    int(entry.reminded),
                    _iso(entry.created_at),
                    _iso(entry.updated_at),
                ),
            )
        await self.conn.commit()
        return entry

    async def get_timetable_entry(self, entry_id: str) -> TimetableEntry | None:
        cursor = await self.conn.execute("SELECT * FROM timetable WHERE id = ?", (entry_id,))
        row = await cursor.fetchone()
        return self._row_to_timetable(row) if row else None

    async def list_timetable(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        status: str | None = None,
        keyword: str | None = None,
        limit: int = 100,
    ) -> list[TimetableEntry]:
        clauses: list[str] = []
        params: list[Any] = []
        if date_from:
            clauses.append("start_time >= ?")
            params.append(date_from)
        if date_to:
            bound = f"{date_to}T23:59:59" if len(date_to) == 10 else date_to
            clauses.append("start_time <= ?")
            params.append(bound)
        if status and status != "all":
            clauses.append("status = ?")
            params.append(status)
        if keyword:
            like = f"%{keyword}%"
            clauses.append("(title LIKE ? OR location LIKE ? OR notes LIKE ?)")
            params.extend([like, like, like])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM timetable {where} ORDER BY start_time ASC LIMIT ?"
        params.append(limit)
        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_timetable(r) for r in rows]

    async def delete_timetable(self, entry_id: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM timetable WHERE id = ?", (entry_id,))
        await self.conn.execute(
            "DELETE FROM scheduled_triggers WHERE related_timetable_id = ?",
            (entry_id,),
        )
        await self.conn.commit()
        return cursor.rowcount > 0

    async def mark_timetable_reminded(self, entry_id: str) -> None:
        await self.conn.execute(
            "UPDATE timetable SET reminded = 1, updated_at = ? WHERE id = ?",
            (_iso(utc_now()), entry_id),
        )
        await self.conn.commit()

    # ------------------------------------------------------------------
    # Scheduled Passive Triggers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_scheduled_trigger(row: aiosqlite.Row) -> ScheduledPassiveTrigger:
        return ScheduledPassiveTrigger(
            id=row["id"],
            trigger_name=row["trigger_name"],
            trigger_at=row["trigger_at"],
            directive=row["directive"],
            category=row["category"],
            priority=float(row["priority"]),
            related_timetable_id=row["related_timetable_id"],
            fired=bool(row["fired"]),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
        )

    async def upsert_scheduled_trigger(
        self, trigger: ScheduledPassiveTrigger
    ) -> ScheduledPassiveTrigger:
        existing = None
        if trigger.related_timetable_id:
            cursor = await self.conn.execute(
                "SELECT id FROM scheduled_triggers WHERE related_timetable_id = ? AND trigger_at = ?",
                (trigger.related_timetable_id, trigger.trigger_at),
            )
            existing = await cursor.fetchone()
            if existing is not None:
                trigger.id = existing["id"]
        else:
            cursor = await self.conn.execute(
                "SELECT id FROM scheduled_triggers WHERE trigger_name = ? AND trigger_at = ?",
                (trigger.trigger_name, trigger.trigger_at),
            )
            existing = await cursor.fetchone()
            if existing is not None:
                trigger.id = existing["id"]

        if existing is not None:
            await self.conn.execute(
                """
                UPDATE scheduled_triggers
                SET trigger_name = ?, directive = ?, category = ?, priority = ?, fired = ?
                WHERE id = ?
                """,
                (
                    trigger.trigger_name,
                    trigger.directive,
                    trigger.category,
                    trigger.priority,
                    int(trigger.fired),
                    trigger.id,
                ),
            )
        else:
            trigger.created_at = utc_now()
            await self.conn.execute(
                """
                INSERT INTO scheduled_triggers (
                    id, trigger_name, trigger_at, directive, category,
                    priority, related_timetable_id, fired, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trigger.id,
                    trigger.trigger_name,
                    trigger.trigger_at,
                    trigger.directive,
                    trigger.category,
                    trigger.priority,
                    trigger.related_timetable_id,
                    int(trigger.fired),
                    _iso(trigger.created_at),
                ),
            )
        await self.conn.commit()
        return trigger

    async def list_scheduled_triggers(
        self,
        *,
        include_fired: bool = False,
        limit: int = 100,
    ) -> list[ScheduledPassiveTrigger]:
        where = "" if include_fired else "WHERE fired = 0"
        cursor = await self.conn.execute(
            f"SELECT * FROM scheduled_triggers {where} ORDER BY trigger_at ASC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_scheduled_trigger(r) for r in rows]

    async def has_scheduled_trigger_for_timetable(self, timetable_id: str) -> bool:
        cursor = await self.conn.execute(
            "SELECT 1 FROM scheduled_triggers WHERE related_timetable_id = ? LIMIT 1",
            (timetable_id,),
        )
        return (await cursor.fetchone()) is not None

    async def mark_scheduled_trigger_fired(self, trigger_id: str) -> None:
        await self.conn.execute(
            "UPDATE scheduled_triggers SET fired = 1 WHERE id = ?",
            (trigger_id,),
        )
        await self.conn.commit()

    async def delete_scheduled_trigger(self, trigger_id: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM scheduled_triggers WHERE id = ?", (trigger_id,))
        await self.conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Slow Loop Memory CRUD (BM25-indexed)
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_slow_memory(row: aiosqlite.Row) -> SlowMemoryItem:
        return SlowMemoryItem(
            id=row["id"],
            kind=SlowMemoryKind(row["kind"]),
            title=row["title"],
            summary=row["summary"],
            details=row["details"],
            tags=json.loads(row["tags_json"] or "[]"),
            planned_start=row["planned_start"],
            planned_end=row["planned_end"],
            location=row["location"],
            category=row["category"],
            estimated_minutes=row["estimated_minutes"],
            energy_level=row["energy_level"],
            is_mandatory=bool(row["is_mandatory"]),
            reminder_lead_minutes=int(row["reminder_lead_minutes"]),
            source=row["source"],
            status=SlowMemoryStatus(row["status"]),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
        )

    async def upsert_slow_memory(self, item: SlowMemoryItem) -> SlowMemoryItem:
        now = utc_now()
        cursor = await self.conn.execute("SELECT id FROM slow_memory WHERE id = ?", (item.id,))
        existing = await cursor.fetchone()
        if existing is None:
            if item.planned_start:
                cursor2 = await self.conn.execute(
                    "SELECT id FROM slow_memory WHERE lower(trim(title)) = lower(trim(?)) "
                    "AND planned_start = ? AND status = 'active'",
                    (item.title, item.planned_start),
                )
            else:
                cursor2 = await self.conn.execute(
                    "SELECT id FROM slow_memory WHERE lower(trim(title)) = lower(trim(?)) "
                    "AND kind = ? AND planned_start IS NULL AND status = 'active'",
                    (item.title, item.kind.value),
                )
            dup = await cursor2.fetchone()
            if dup is not None:
                item.id = dup["id"]
                existing = dup

        if existing is not None:
            item.updated_at = now
            await self.conn.execute(
                """
                UPDATE slow_memory
                SET kind = ?, title = ?, summary = ?, details = ?, tags_json = ?,
                    planned_start = ?, planned_end = ?, location = ?, category = ?,
                    estimated_minutes = ?, energy_level = ?, is_mandatory = ?,
                    reminder_lead_minutes = ?, source = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    item.kind.value,
                    item.title,
                    item.summary,
                    item.details,
                    json.dumps(item.tags, ensure_ascii=False),
                    item.planned_start,
                    item.planned_end,
                    item.location,
                    item.category,
                    item.estimated_minutes,
                    item.energy_level,
                    int(item.is_mandatory),
                    item.reminder_lead_minutes,
                    item.source,
                    item.status.value,
                    _iso(item.updated_at),
                    item.id,
                ),
            )
        else:
            item.created_at = now
            item.updated_at = now
            await self.conn.execute(
                """
                INSERT INTO slow_memory (
                    id, kind, title, summary, details, tags_json, planned_start, planned_end,
                    location, category, estimated_minutes, energy_level, is_mandatory,
                    reminder_lead_minutes, source, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.kind.value,
                    item.title,
                    item.summary,
                    item.details,
                    json.dumps(item.tags, ensure_ascii=False),
                    item.planned_start,
                    item.planned_end,
                    item.location,
                    item.category,
                    item.estimated_minutes,
                    item.energy_level,
                    int(item.is_mandatory),
                    item.reminder_lead_minutes,
                    item.source,
                    item.status.value,
                    _iso(item.created_at),
                    _iso(item.updated_at),
                ),
            )
        await self.conn.commit()
        return item

    async def get_slow_memory(self, item_id: str) -> SlowMemoryItem | None:
        cursor = await self.conn.execute("SELECT * FROM slow_memory WHERE id = ?", (item_id,))
        row = await cursor.fetchone()
        return self._row_to_slow_memory(row) if row else None

    async def list_slow_memory(
        self,
        *,
        kind: str | None = None,
        category: str | None = None,
        status: str | None = "active",
        limit: int = 500,
    ) -> list[SlowMemoryItem]:
        clauses: list[str] = []
        params: list[Any] = []
        if kind and kind != "all":
            clauses.append("kind = ?")
            params.append(kind)
        if category and category != "all":
            clauses.append("category = ?")
            params.append(category)
        if status and status != "all":
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM slow_memory {where} ORDER BY COALESCE(planned_start, '9999') ASC, updated_at DESC LIMIT ?"
        params.append(limit)
        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_slow_memory(r) for r in rows]

    async def delete_slow_memory(self, item_id: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM slow_memory WHERE id = ?", (item_id,))
        await self.conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Buffer CRUD (Dual-axis scoring + Trust Tier)
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_buffer(row: aiosqlite.Row) -> BufferItem:
        return BufferItem(
            id=row["id"],
            zone=BufferZone(row["zone"]),
            source=row["source"],
            title=row["title"],
            summary=row["summary"],
            details=row["details"],
            tags=json.loads(row["tags_json"] or "[]"),
            weight=float(row["weight"]),
            intrinsic_value=float(row["intrinsic_value"]),
            urgency=float(row["urgency"]),
            starvation_boost=float(row["starvation_boost"]),
            trust_tier=TrustTier(row["trust_tier"] or "medium"),
            needs_confirmation=bool(row["needs_confirmation"]),
            return_count=int(row["return_count"]),
            has_explicit_time=bool(row["has_explicit_time"]),
            is_mandatory_task=bool(row["is_mandatory_task"]),
            status=BufferStatus(row["status"]),
            next_review_at=_parse_dt(row["next_review_at"]),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
        )

    async def add_buffer_item(self, item: BufferItem) -> BufferItem:
        now = utc_now()
        item.created_at = now
        item.updated_at = now
        await self.conn.execute(
            """
            INSERT INTO buffer_items (
                id, zone, source, title, summary, details, tags_json, weight,
                intrinsic_value, urgency, starvation_boost, trust_tier, needs_confirmation,
                return_count, has_explicit_time, is_mandatory_task, status,
                next_review_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.zone.value,
                item.source,
                item.title,
                item.summary,
                item.details,
                json.dumps(item.tags, ensure_ascii=False),
                item.weight,
                item.intrinsic_value,
                item.urgency,
                item.starvation_boost,
                item.trust_tier.value,
                int(item.needs_confirmation),
                item.return_count,
                int(item.has_explicit_time),
                int(item.is_mandatory_task),
                item.status.value,
                _iso(item.next_review_at),
                _iso(item.created_at),
                _iso(item.updated_at),
            ),
        )
        await self.conn.commit()
        return item

    async def get_buffer_item(self, item_id: str) -> BufferItem | None:
        cursor = await self.conn.execute("SELECT * FROM buffer_items WHERE id = ?", (item_id,))
        row = await cursor.fetchone()
        return self._row_to_buffer(row) if row else None

    async def update_buffer_item(self, item: BufferItem) -> BufferItem:
        item.updated_at = utc_now()
        await self.conn.execute(
            """
            UPDATE buffer_items
            SET zone = ?, source = ?, title = ?, summary = ?, details = ?,
                tags_json = ?, weight = ?, intrinsic_value = ?, urgency = ?,
                starvation_boost = ?, trust_tier = ?, needs_confirmation = ?,
                return_count = ?, has_explicit_time = ?, is_mandatory_task = ?,
                status = ?, next_review_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                item.zone.value,
                item.source,
                item.title,
                item.summary,
                item.details,
                json.dumps(item.tags, ensure_ascii=False),
                item.weight,
                item.intrinsic_value,
                item.urgency,
                item.starvation_boost,
                item.trust_tier.value,
                int(item.needs_confirmation),
                item.return_count,
                int(item.has_explicit_time),
                int(item.is_mandatory_task),
                item.status.value,
                _iso(item.next_review_at),
                _iso(item.updated_at),
                item.id,
            ),
        )
        await self.conn.commit()
        return item

    async def delete_buffer_item(self, item_id: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM buffer_items WHERE id = ?", (item_id,))
        await self.conn.commit()
        return cursor.rowcount > 0

    async def list_buffer_items(
        self,
        *,
        zone: str | None = None,
        status: str | None = "active",
        keyword: str | None = None,
        limit: int = 50,
    ) -> list[BufferItem]:
        clauses: list[str] = []
        params: list[Any] = []
        if zone and zone != "all":
            clauses.append("zone = ?")
            params.append(zone)
        if status and status != "all":
            clauses.append("status = ?")
            params.append(status)
        if keyword:
            like = f"%{keyword}%"
            clauses.append("(title LIKE ? OR summary LIKE ? OR details LIKE ?)")
            params.extend([like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT * FROM buffer_items {where} "
            "ORDER BY weight DESC, urgency DESC, updated_at DESC LIMIT ?"
        )
        params.append(limit)
        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_buffer(r) for r in rows]

    async def list_due_buffer_items(self, now: datetime, limit: int = 5) -> list[BufferItem]:
        cursor = await self.conn.execute(
            """
            SELECT * FROM buffer_items
            WHERE status = 'active'
              AND next_review_at IS NOT NULL
              AND next_review_at <= ?
            ORDER BY weight DESC, urgency DESC, created_at ASC
            LIMIT ?
            """,
            (_iso(now), limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_buffer(r) for r in rows]

    async def expire_stale_buffer_items(self, ttl_hours: float) -> int:
        cutoff = utc_now() - timedelta(hours=ttl_hours)
        cursor = await self.conn.execute(
            """
            UPDATE buffer_items
            SET status = 'dismissed', updated_at = ?
            WHERE status = 'active' AND created_at < ?
            """,
            (_iso(utc_now()), _iso(cutoff)),
        )
        await self.conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Tiered Scene State (T0–T3 TTL) + Backward-Compatible Scene Dict
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_tiered_scene(row: aiosqlite.Row) -> TieredSceneEntry:
        try:
            val = json.loads(row["value_json"])
        except json.JSONDecodeError:
            val = row["value_json"]
        return TieredSceneEntry(
            key=row["key"],
            tier=StateTier(row["tier"]),
            value=val,
            source=row["source"],
            expires_at=_parse_dt(row["expires_at"]),
            renewal_prompted=bool(row["renewal_prompted"]),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
        )

    async def upsert_tiered_scene(self, entry: TieredSceneEntry) -> TieredSceneEntry:
        entry.updated_at = utc_now()
        await self.conn.execute(
            """
            INSERT INTO tiered_scene_state (key, tier, value_json, source, expires_at, renewal_prompted, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                tier = excluded.tier,
                value_json = excluded.value_json,
                source = excluded.source,
                expires_at = excluded.expires_at,
                renewal_prompted = excluded.renewal_prompted,
                updated_at = excluded.updated_at
            """,
            (
                entry.key,
                entry.tier.value,
                json.dumps(entry.value, ensure_ascii=False),
                entry.source,
                _iso(entry.expires_at),
                int(entry.renewal_prompted),
                _iso(entry.updated_at),
            ),
        )
        await self.conn.commit()
        return entry

    async def get_tiered_scene(self, key: str) -> TieredSceneEntry | None:
        cursor = await self.conn.execute("SELECT * FROM tiered_scene_state WHERE key = ?", (key,))
        row = await cursor.fetchone()
        return self._row_to_tiered_scene(row) if row else None

    async def list_tiered_scene(self, tier: StateTier | None = None) -> list[TieredSceneEntry]:
        if tier is not None:
            cursor = await self.conn.execute(
                "SELECT * FROM tiered_scene_state WHERE tier = ? ORDER BY updated_at DESC",
                (tier.value,),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM tiered_scene_state ORDER BY tier ASC, updated_at DESC"
            )
        rows = await cursor.fetchall()
        return [self._row_to_tiered_scene(r) for r in rows]

    async def delete_tiered_scene(self, key: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM tiered_scene_state WHERE key = ?", (key,))
        await self.conn.commit()
        return cursor.rowcount > 0

    async def get_scene_state(self) -> dict[str, Any]:
        """Flat compatibility view combining legacy scene_state and active tiered_scene_state."""
        result: dict[str, Any] = {}
        cursor = await self.conn.execute("SELECT key, value_json FROM scene_state ORDER BY key")
        for row in await cursor.fetchall():
            try:
                result[row["key"]] = json.loads(row["value_json"])
            except json.JSONDecodeError:
                result[row["key"]] = row["value_json"]
        for entry in await self.list_tiered_scene():
            result[entry.key] = entry.value
        return result

    async def merge_scene_state(
        self,
        updates: dict[str, Any],
        *,
        tier: StateTier = StateTier.T0_EPHEMERAL,
        expires_at: datetime | None = None,
        source: str = "gateway",
    ) -> dict[str, Any]:
        for key, value in updates.items():
            if value is None:
                await self.delete_tiered_scene(key)
                await self.conn.execute("DELETE FROM scene_state WHERE key = ?", (key,))
                await self.conn.commit()
            else:
                await self.upsert_tiered_scene(
                    TieredSceneEntry(
                        key=key,
                        tier=tier,
                        value=value,
                        source=source,
                        expires_at=expires_at,
                        renewal_prompted=False,
                    )
                )
        return await self.get_scene_state()

    # ------------------------------------------------------------------
    # Dynamic Gating Rules, Versioned Grade Snapshots, Feedback Log, Affinity
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_gating_rule(row: aiosqlite.Row) -> GatingRule:
        return GatingRule(
            id=row["id"],
            scope_type=row["scope_type"],
            scope_key=row["scope_key"],
            effect=row["effect"],
            instruction=row["instruction"],
            score_delta=float(row["score_delta"]),
            confidence=float(row["confidence"]),
            origin=row["origin"],
            expires_at=_parse_dt(row["expires_at"]),
            status=GatingRuleStatus(row["status"]),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
        )

    async def upsert_gating_rule(self, rule: GatingRule) -> GatingRule:
        now = utc_now()
        cursor = await self.conn.execute("SELECT id FROM gating_rules WHERE id = ?", (rule.id,))
        existing = await cursor.fetchone()
        if existing is None and rule.scope_key:
            cursor2 = await self.conn.execute(
                "SELECT id FROM gating_rules WHERE scope_type = ? AND lower(trim(scope_key)) = lower(trim(?)) AND effect = ?",
                (rule.scope_type, rule.scope_key, rule.effect),
            )
            dup = await cursor2.fetchone()
            if dup is not None:
                rule.id = dup["id"]
                existing = dup

        if existing is not None:
            rule.updated_at = now
            await self.conn.execute(
                """
                UPDATE gating_rules
                SET scope_type = ?, scope_key = ?, effect = ?, instruction = ?,
                    score_delta = ?, confidence = ?, origin = ?, expires_at = ?,
                    status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    rule.scope_type,
                    rule.scope_key,
                    rule.effect,
                    rule.instruction,
                    rule.score_delta,
                    rule.confidence,
                    rule.origin,
                    _iso(rule.expires_at),
                    rule.status.value,
                    _iso(rule.updated_at),
                    rule.id,
                ),
            )
        else:
            rule.created_at = now
            rule.updated_at = now
            await self.conn.execute(
                """
                INSERT INTO gating_rules (
                    id, scope_type, scope_key, effect, instruction, score_delta,
                    confidence, origin, expires_at, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.id,
                    rule.scope_type,
                    rule.scope_key,
                    rule.effect,
                    rule.instruction,
                    rule.score_delta,
                    rule.confidence,
                    rule.origin,
                    _iso(rule.expires_at),
                    rule.status.value,
                    _iso(rule.created_at),
                    _iso(rule.updated_at),
                ),
            )
        await self.conn.commit()
        return rule

    async def get_gating_rule(self, rule_id: str) -> GatingRule | None:
        cursor = await self.conn.execute("SELECT * FROM gating_rules WHERE id = ?", (rule_id,))
        row = await cursor.fetchone()
        return self._row_to_gating_rule(row) if row else None

    async def list_gating_rules(self, status: str = "all") -> list[GatingRule]:
        if status and status != "all":
            cursor = await self.conn.execute(
                "SELECT * FROM gating_rules WHERE status = ? ORDER BY confidence DESC, updated_at DESC",
                (status,),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM gating_rules ORDER BY status ASC, confidence DESC, updated_at DESC"
            )
        return [self._row_to_gating_rule(r) for r in await cursor.fetchall()]

    async def delete_gating_rule(self, rule_id: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM gating_rules WHERE id = ?", (rule_id,))
        await self.conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_grade_snapshot(row: aiosqlite.Row) -> GradeSnapshot:
        return GradeSnapshot(
            version=int(row["version"]),
            grade_text=row["grade_text"],
            rule_ids=json.loads(row["rule_ids_json"] or "[]"),
            summary_note=row["summary_note"],
            is_active=bool(row["is_active"]),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
        )

    async def publish_grade_snapshot(
        self,
        *,
        grade_text: str,
        rule_ids: list[str],
        summary_note: str = "",
    ) -> GradeSnapshot:
        """Atomically deactivate previous snapshot and publish a new active `grade_vN`."""
        now = utc_now()
        await self.conn.execute("UPDATE grade_snapshots SET is_active = 0 WHERE is_active = 1")
        cursor = await self.conn.execute(
            """
            INSERT INTO grade_snapshots (grade_text, rule_ids_json, summary_note, is_active, created_at)
            VALUES (?, ?, ?, 1, ?)
            """,
            (grade_text, json.dumps(rule_ids, ensure_ascii=False), summary_note, _iso(now)),
        )
        await self.conn.commit()
        ver = int(cursor.lastrowid or 1)
        return GradeSnapshot(
            version=ver,
            grade_text=grade_text,
            rule_ids=rule_ids,
            summary_note=summary_note,
            is_active=True,
            created_at=now,
        )

    async def get_active_grade_snapshot(self) -> GradeSnapshot | None:
        cursor = await self.conn.execute(
            "SELECT * FROM grade_snapshots WHERE is_active = 1 ORDER BY version DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return self._row_to_grade_snapshot(row) if row else None

    async def list_grade_snapshots(self, limit: int = 20) -> list[GradeSnapshot]:
        cursor = await self.conn.execute(
            "SELECT * FROM grade_snapshots ORDER BY version DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_grade_snapshot(r) for r in await cursor.fetchall()]

    async def activate_grade_version(self, version: int) -> GradeSnapshot | None:
        cursor = await self.conn.execute("SELECT * FROM grade_snapshots WHERE version = ?", (version,))
        row = await cursor.fetchone()
        if row is None:
            return None
        await self.conn.execute("UPDATE grade_snapshots SET is_active = 0")
        await self.conn.execute("UPDATE grade_snapshots SET is_active = 1 WHERE version = ?", (version,))
        await self.conn.commit()
        snap = self._row_to_grade_snapshot(row)
        snap.is_active = True
        return snap

    @staticmethod
    def _row_to_feedback(row: aiosqlite.Row) -> FeedbackLogEntry:
        return FeedbackLogEntry(
            id=row["id"],
            event_type=row["event_type"],
            source_key=row["source_key"],
            tags=json.loads(row["tags_json"] or "[]"),
            category=row["category"],
            activity_mode=row["activity_mode"],
            hour_bucket=int(row["hour_bucket"]),
            sentiment=row["sentiment"],
            signal_value=float(row["signal_value"]),
            raw_text=row["raw_text"],
            is_explicit=bool(row["is_explicit"]),
            processed_at=_parse_dt(row["processed_at"]),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
        )

    async def append_feedback_log(self, entry: FeedbackLogEntry) -> FeedbackLogEntry:
        """Append-only daytime Forward Pass feedback ledger write."""
        await self.conn.execute(
            """
            INSERT INTO feedback_log (
                id, event_type, source_key, tags_json, category, activity_mode,
                hour_bucket, sentiment, signal_value, raw_text, is_explicit,
                processed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.event_type,
                entry.source_key,
                json.dumps(entry.tags, ensure_ascii=False),
                entry.category,
                entry.activity_mode,
                entry.hour_bucket,
                entry.sentiment,
                entry.signal_value,
                entry.raw_text,
                int(entry.is_explicit),
                _iso(entry.processed_at),
                _iso(entry.created_at),
            ),
        )
        await self.conn.commit()
        return entry

    async def list_unprocessed_feedback(self, limit: int = 500) -> list[FeedbackLogEntry]:
        cursor = await self.conn.execute(
            "SELECT * FROM feedback_log WHERE processed_at IS NULL ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        return [self._row_to_feedback(r) for r in await cursor.fetchall()]

    async def list_recent_feedback(self, limit: int = 50) -> list[FeedbackLogEntry]:
        cursor = await self.conn.execute(
            "SELECT * FROM feedback_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_feedback(r) for r in await cursor.fetchall()]

    async def mark_feedback_processed(self, feedback_ids: list[str], processed_at: datetime | None = None) -> None:
        if not feedback_ids:
            return
        ts = _iso(processed_at or utc_now())
        placeholders = ",".join("?" for _ in feedback_ids)
        await self.conn.execute(
            f"UPDATE feedback_log SET processed_at = ? WHERE id IN ({placeholders})",
            [ts, *feedback_ids],
        )
        await self.conn.commit()

    async def get_source_affinity(self, key: str) -> SourceAffinity | None:
        cursor = await self.conn.execute("SELECT * FROM source_affinity WHERE key = ?", (key,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return SourceAffinity(
            key=row["key"],
            affinity=float(row["affinity"]),
            sample_count=int(row["sample_count"]),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
        )

    async def upsert_source_affinity(self, aff: SourceAffinity) -> SourceAffinity:
        aff.updated_at = utc_now()
        await self.conn.execute(
            """
            INSERT INTO source_affinity (key, affinity, sample_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                affinity = excluded.affinity,
                sample_count = excluded.sample_count,
                updated_at = excluded.updated_at
            """,
            (aff.key, aff.affinity, aff.sample_count, _iso(aff.updated_at)),
        )
        await self.conn.commit()
        return aff

    async def list_source_affinities(self, limit: int = 100) -> list[SourceAffinity]:
        cursor = await self.conn.execute(
            "SELECT * FROM source_affinity ORDER BY abs(affinity) DESC, sample_count DESC LIMIT ?",
            (limit,),
        )
        return [
            SourceAffinity(
                key=r["key"],
                affinity=float(r["affinity"]),
                sample_count=int(r["sample_count"]),
                updated_at=_parse_dt(r["updated_at"]) or utc_now(),
            )
            for r in await cursor.fetchall()
        ]

    # ------------------------------------------------------------------
    # Dynamic Follow-Up Intents (100% arbitrated by Message Gateway)
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_followup(row: aiosqlite.Row) -> FollowUpIntent:
        return FollowUpIntent(
            id=row["id"],
            topic=row["topic"],
            context_hint=row["context_hint"],
            intent_type=row["intent_type"],
            deadline_at=row["deadline_at"],
            anchor_event_id=row["anchor_event_id"],
            safety_margin_minutes=int(row["safety_margin_minutes"]),
            computed_check_at=row["computed_check_at"],
            priority=float(row["priority"]),
            status=FollowUpStatus(row["status"]),
            check_count=int(row["check_count"]),
            max_checks=int(row["max_checks"]),
            source=row["source"],
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
        )

    async def upsert_followup(self, fu: FollowUpIntent) -> FollowUpIntent:
        now = utc_now()
        cursor = await self.conn.execute("SELECT id FROM follow_ups WHERE id = ?", (fu.id,))
        existing = await cursor.fetchone()
        if existing is None:
            cursor2 = await self.conn.execute(
                "SELECT id FROM follow_ups WHERE lower(trim(topic)) = lower(trim(?)) AND status = 'open'",
                (fu.topic,),
            )
            dup = await cursor2.fetchone()
            if dup is not None:
                fu.id = dup["id"]
                existing = dup

        if existing is not None:
            fu.updated_at = now
            await self.conn.execute(
                """
                UPDATE follow_ups
                SET topic = ?, context_hint = ?, intent_type = ?, deadline_at = ?,
                    anchor_event_id = ?, safety_margin_minutes = ?, computed_check_at = ?,
                    priority = ?, status = ?, check_count = ?, max_checks = ?,
                    source = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    fu.topic,
                    fu.context_hint,
                    fu.intent_type,
                    fu.deadline_at,
                    fu.anchor_event_id,
                    fu.safety_margin_minutes,
                    fu.computed_check_at,
                    fu.priority,
                    fu.status.value,
                    fu.check_count,
                    fu.max_checks,
                    fu.source,
                    _iso(fu.updated_at),
                    fu.id,
                ),
            )
        else:
            fu.created_at = now
            fu.updated_at = now
            await self.conn.execute(
                """
                INSERT INTO follow_ups (
                    id, topic, context_hint, intent_type, deadline_at, anchor_event_id,
                    safety_margin_minutes, computed_check_at, priority, status,
                    check_count, max_checks, source, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fu.id,
                    fu.topic,
                    fu.context_hint,
                    fu.intent_type,
                    fu.deadline_at,
                    fu.anchor_event_id,
                    fu.safety_margin_minutes,
                    fu.computed_check_at,
                    fu.priority,
                    fu.status.value,
                    fu.check_count,
                    fu.max_checks,
                    fu.source,
                    _iso(fu.created_at),
                    _iso(fu.updated_at),
                ),
            )
        await self.conn.commit()
        return fu

    async def get_followup(self, followup_id: str) -> FollowUpIntent | None:
        cursor = await self.conn.execute("SELECT * FROM follow_ups WHERE id = ?", (followup_id,))
        row = await cursor.fetchone()
        return self._row_to_followup(row) if row else None

    async def list_followups(self, status: str = "open", limit: int = 50) -> list[FollowUpIntent]:
        if status and status != "all":
            cursor = await self.conn.execute(
                "SELECT * FROM follow_ups WHERE status = ? ORDER BY COALESCE(computed_check_at, '9999') ASC LIMIT ?",
                (status, limit),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM follow_ups ORDER BY status ASC, COALESCE(computed_check_at, '9999') ASC LIMIT ?",
                (limit,),
            )
        return [self._row_to_followup(r) for r in await cursor.fetchall()]

    async def delete_followup(self, followup_id: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM follow_ups WHERE id = ?", (followup_id,))
        await self.conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Chat Context & RTIR cursor
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_chat(row: aiosqlite.Row) -> ChatTurn:
        return ChatTurn(
            id=int(row["id"]),
            role=row["role"],
            content=row["content"],
            channel=row["channel"],
            parsed_by_rtir=bool(row["parsed_by_rtir"]),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
        )

    async def append_chat_turn(self, turn: ChatTurn) -> ChatTurn:
        cursor = await self.conn.execute(
            """
            INSERT INTO chat_messages (role, content, channel, parsed_by_rtir, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                turn.role,
                turn.content,
                turn.channel,
                int(turn.parsed_by_rtir),
                _iso(turn.created_at),
            ),
        )
        await self.conn.commit()
        turn.id = cursor.lastrowid
        return turn

    async def get_recent_chat_turns(self, limit: int = 40) -> list[ChatTurn]:
        cursor = await self.conn.execute(
            """
            SELECT * FROM (
                SELECT * FROM chat_messages ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_chat(r) for r in rows]

    async def get_unparsed_chat_turns(self) -> list[ChatTurn]:
        cursor = await self.conn.execute(
            "SELECT * FROM chat_messages WHERE parsed_by_rtir = 0 ORDER BY id ASC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_chat(r) for r in rows]

    async def mark_chat_turns_parsed(self, up_to_id: int) -> None:
        await self.conn.execute(
            "UPDATE chat_messages SET parsed_by_rtir = 1 WHERE id <= ? AND parsed_by_rtir = 0",
            (up_to_id,),
        )
        await self.conn.commit()

    # ------------------------------------------------------------------
    # Pending QQ -> WeChat Failover Pushes
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_push(row: aiosqlite.Row) -> PendingPush:
        return PendingPush(
            id=row["id"],
            content=row["content"],
            trigger_source=row["trigger_source"],
            source_key=row["source_key"] or "",
            tags=json.loads(row["tags_json"] or "[]"),
            category=row["category"] or "general",
            urgency=float(row["urgency"]),
            qq_target=row["qq_target"],
            weixin_target=row["weixin_target"],
            status=PushStatus(row["status"]),
            sent_at=_parse_dt(row["sent_at"]) or utc_now(),
            failover_at=_parse_dt(row["failover_at"]) or utc_now(),
            resolved_at=_parse_dt(row["resolved_at"]),
            error_message=row["error_message"],
        )

    async def create_pending_push(self, push: PendingPush) -> PendingPush:
        await self.conn.execute(
            """
            INSERT INTO pending_pushes (
                id, content, trigger_source, source_key, tags_json, category, urgency,
                qq_target, weixin_target, status, sent_at, failover_at, resolved_at, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                push.id,
                push.content,
                push.trigger_source,
                push.source_key,
                json.dumps(push.tags, ensure_ascii=False),
                push.category,
                push.urgency,
                push.qq_target,
                push.weixin_target,
                push.status.value,
                _iso(push.sent_at),
                _iso(push.failover_at),
                _iso(push.resolved_at),
                push.error_message,
            ),
        )
        await self.conn.commit()
        return push

    async def get_pending_push(self, push_id: str) -> PendingPush | None:
        cursor = await self.conn.execute("SELECT * FROM pending_pushes WHERE id = ?", (push_id,))
        row = await cursor.fetchone()
        return self._row_to_push(row) if row else None

    async def list_pending_pushes(self, status: str | None = None, limit: int = 50) -> list[PendingPush]:
        if status:
            cursor = await self.conn.execute(
                "SELECT * FROM pending_pushes WHERE status = ? ORDER BY sent_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM pending_pushes ORDER BY sent_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_push(r) for r in rows]

    async def update_pending_push(
        self,
        push_id: str,
        *,
        status: PushStatus,
        resolved_at: datetime | None = None,
        error_message: str | None = None,
    ) -> None:
        await self.conn.execute(
            """
            UPDATE pending_pushes
            SET status = ?, resolved_at = COALESCE(?, resolved_at),
                error_message = COALESCE(?, error_message)
            WHERE id = ?
            """,
            (status.value, _iso(resolved_at), error_message, push_id),
        )
        await self.conn.commit()

    # ------------------------------------------------------------------
    # Deduplication & Trigger Execution History
    # ------------------------------------------------------------------

    async def is_seen(self, namespace: str, external_id: str) -> bool:
        cursor = await self.conn.execute(
            "SELECT 1 FROM seen_external_ids WHERE namespace = ? AND external_id = ?",
            (namespace, external_id),
        )
        return (await cursor.fetchone()) is not None

    async def mark_seen(self, namespace: str, external_id: str) -> None:
        await self.conn.execute(
            """
            INSERT OR IGNORE INTO seen_external_ids (namespace, external_id, seen_at)
            VALUES (?, ?, ?)
            """,
            (namespace, external_id, _iso(utc_now())),
        )
        await self.conn.commit()

    async def has_trigger_fired(self, trigger_key: str) -> bool:
        cursor = await self.conn.execute(
            "SELECT 1 FROM trigger_runs WHERE trigger_key = ?",
            (trigger_key,),
        )
        return (await cursor.fetchone()) is not None

    async def mark_trigger_fired(self, trigger_key: str) -> None:
        await self.conn.execute(
            "INSERT OR IGNORE INTO trigger_runs (trigger_key, fired_at) VALUES (?, ?)",
            (trigger_key, _iso(utc_now())),
        )
        await self.conn.commit()
