"""BM25-indexed Slow Loop storage for undated ideas and >7-day future plans/tasks.

Design principles:
  - Keeps Fast Loop context clean (only <= 7-day Timetable, active Buffer, Scene State, Context).
  - Stores undated ideas / free-time activity lists / > 7-day schedules & tasks structurally in SQLite.
  - Uses pure lexical BM25 (Lucene non-negative IDF BM25Okapi + jieba Chinese/English tokenization)
    — NO heavy vector DB or embedding pipeline.
  - Automatically promotes > 7-day scheduled items into the Fast Loop Timetable + Passive Triggers
    once they enter the 7-day rolling horizon.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from ..models import (
    ScheduledPassiveTrigger,
    SlowMemoryItem,
    SlowMemoryStatus,
    TimetableEntry,
    TimetableStatus,
    utc_now,
)
from ..storage import Storage

logger = logging.getLogger(__name__)

# Silence jieba startup banner to keep CLI output tidy
jieba.setLogLevel(logging.WARNING)

_TOKEN_CLEAN_RE = re.compile(r"[^\w\u4e00-\u9fff]+", flags=re.UNICODE)
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "那",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "and",
        "or",
        "is",
        "are",
    }
)


def parse_iso_datetime(raw: str | None, ref_now: datetime | None = None) -> datetime | None:
    """Parse ISO-8601 or common YYYY-MM-DD HH:MM timestamps, attaching local tz if naive."""
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    ref = ref_now or utc_now()
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None and ref.tzinfo is not None:
            dt = dt.replace(tzinfo=ref.tzinfo)
        return dt
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            if ref.tzinfo is not None:
                dt = dt.replace(tzinfo=ref.tzinfo)
            return dt
        except ValueError:
            continue
    return None


def is_beyond_horizon(
    start_time: str | None,
    *,
    now: datetime | None = None,
    horizon_days: int = 7,
) -> bool:
    """Return True if start_time is strictly more than horizon_days in the future."""
    ref = now or utc_now()
    dt = parse_iso_datetime(start_time, ref)
    if dt is None:
        return False
    return dt > (ref + timedelta(days=horizon_days))


def build_passive_trigger_for_event(
    entry: TimetableEntry,
    *,
    lead_minutes: int = 20,
    priority: float = 9.0,
    now: datetime | None = None,
) -> ScheduledPassiveTrigger | None:
    """Create a dated passive trigger firing lead_minutes before a fast-loop TimetableEntry."""
    ref = now or utc_now()
    start_dt = parse_iso_datetime(entry.start_time, ref)
    if start_dt is None:
        return None
    trigger_dt = start_dt - timedelta(minutes=max(0, lead_minutes))
    # If event starts in less than lead_minutes (but still in the future), fire immediately at now
    if trigger_dt < ref <= start_dt:
        trigger_dt = ref

    loc_part = f"，地点：{entry.location}" if entry.location else ""
    notes_part = f"，备注：{entry.notes}" if entry.notes else ""
    time_hm = start_dt.strftime("%H:%M")

    if entry.category == "class":
        label = "即将上课提醒"
        trigger_cat = "class_reminder"
        action_hint = "请主动提醒用户前往对应教学楼/教室上课，并提示相关注意事项。"
    elif entry.category == "shift":
        label = "排班上岗提醒"
        trigger_cat = "shift_reminder"
        action_hint = "请主动提醒用户按时到岗值班/排班，确认相关交接或准备事项。"
    else:
        label = "日程临近提醒"
        trigger_cat = "schedule_reminder"
        action_hint = "请主动提醒用户准时处理或出发。"

    directive = (
        f"{label}：{time_hm}「{entry.title}」（{entry.start_time}）即将开始"
        f"{loc_part}{notes_part}。{action_hint}"
    )
    return ScheduledPassiveTrigger(
        trigger_name=f"{trigger_cat}_{entry.id}",
        trigger_at=trigger_dt.isoformat(),
        directive=directive,
        category=trigger_cat,
        priority=priority,
        related_timetable_id=entry.id,
        fired=False,
    )


def tokenize_mixed(text: str) -> list[str]:
    """Mixed Chinese + English tokenizer for BM25 (jieba search mode + CJK bigrams)."""
    if not text:
        return []
    cleaned = _TOKEN_CLEAN_RE.sub(" ", text.lower())
    tokens: list[str] = []
    for piece in jieba.cut_for_search(cleaned):
        tok = piece.strip()
        if not tok or tok in _STOPWORDS:
            continue
        tokens.append(tok)

    # Add CJK character bigrams so slightly different phrasing still gets high recall
    for run in _CJK_RUN_RE.findall(cleaned):
        if len(run) >= 2:
            for i in range(len(run) - 1):
                bg = run[i : i + 2]
                if bg not in _STOPWORDS:
                    tokens.append(bg)
    return tokens


class _LuceneBM25(BM25Okapi):
    """BM25Okapi with Lucene non-negative IDF: ln(1 + (N - n + 0.5) / (n + 0.5)).

    Standard Robertson IDF becomes <= 0 when a term appears in >= 50% of a tiny corpus
    (e.g. N=1..3 personal memory notes). Lucene IDF stays strictly positive for matches
    while keeping score == 0.0 for documents that contain none of the query terms.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(
                1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
            )


class SlowLoopBM25Store:
    """BM25 retrieval and lifecycle manager for Slow Loop structured memories."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self._dirty = True
        self._cached_items: list[SlowMemoryItem] = []
        self._cached_tokens: list[list[str]] = []
        self._bm25: _LuceneBM25 | None = None

    def invalidate(self) -> None:
        self._dirty = True

    @staticmethod
    def _item_to_tokens(item: SlowMemoryItem) -> list[str]:
        """Build field-weighted token list (title 3x, tags 2x, structured fields 1x)."""
        title_tokens = tokenize_mixed(item.title)
        tag_tokens = tokenize_mixed(" ".join(item.tags))
        meta_parts = [
            item.summary,
            item.details,
            item.location,
            item.category,
            item.kind.value,
            item.planned_start or "",
            item.energy_level,
        ]
        # Add Chinese semantic aliases for kind/category so natural queries hit easily
        alias_map = {
            "idea": "想法 灵感 空闲 活动 待办 心愿",
            "future_event": "未来 日程 安排 课表 排班 活动",
            "future_task": "未来 任务 计划 截止 作业",
            "plan": "规划 计划 目标 长期",
            "class": "课程 上课 课表 教学楼",
            "shift": "排班 值班 班次 上岗",
        }
        if item.kind.value in alias_map:
            meta_parts.append(alias_map[item.kind.value])
        if item.category in alias_map:
            meta_parts.append(alias_map[item.category])

        body_tokens = tokenize_mixed(" ".join(meta_parts))
        return (title_tokens * 3) + (tag_tokens * 2) + body_tokens

    async def _ensure_index(self) -> None:
        if not self._dirty:
            return
        items = await self.storage.list_slow_memory(status="all", limit=5000)
        self._cached_items = items
        self._cached_tokens = [self._item_to_tokens(it) for it in items]
        self._bm25 = _LuceneBM25(self._cached_tokens) if self._cached_tokens else None
        self._dirty = False

    async def add_or_update(self, item: SlowMemoryItem) -> SlowMemoryItem:
        saved = await self.storage.upsert_slow_memory(item)
        self.invalidate()
        return saved

    async def delete(self, item_id: str) -> bool:
        deleted = await self.storage.delete_slow_memory(item_id)
        if deleted:
            self.invalidate()
        return deleted

    async def get(self, item_id: str) -> SlowMemoryItem | None:
        return await self.storage.get_slow_memory(item_id)

    async def list_all(
        self,
        *,
        kind: str | None = None,
        category: str | None = None,
        status: str | None = "active",
        limit: int = 200,
    ) -> list[SlowMemoryItem]:
        return await self.storage.list_slow_memory(
            kind=kind,
            category=category,
            status=status,
            limit=limit,
        )

    async def search(
        self,
        query: str = "",
        *,
        kind: str = "all",
        category: str | None = None,
        status: str = "active",
        max_minutes: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """BM25 search over Slow Loop items with optional structured pre-filtering."""
        await self._ensure_index()
        if not self._cached_items:
            return []

        # Collect eligible indices after structured filters
        eligible: list[int] = []
        date_to_bound = f"{date_to}T23:59:59" if date_to and len(date_to) == 10 else date_to
        for idx, item in enumerate(self._cached_items):
            if status != "all" and item.status.value != status:
                continue
            if kind != "all" and item.kind.value != kind:
                continue
            if category and category != "all" and item.category != category:
                continue
            if max_minutes is not None and item.estimated_minutes is not None:
                if item.estimated_minutes > max_minutes:
                    continue
            if date_from:
                if not item.planned_start or item.planned_start < date_from:
                    continue
            if date_to_bound:
                if not item.planned_start or item.planned_start > date_to_bound:
                    continue
            eligible.append(idx)

        if not eligible:
            return []

        query_tokens = tokenize_mixed(query)
        scored: list[tuple[float, SlowMemoryItem]] = []

        if query_tokens and self._bm25 is not None:
            # Build subset BM25 if heavily filtered, or score globally and filter
            all_scores = self._bm25.get_scores(query_tokens)
            for idx in eligible:
                score = float(all_scores[idx])
                if score > 0.0:
                    scored.append((score, self._cached_items[idx]))
            scored.sort(key=lambda pair: (-pair[0], pair[1].planned_start or "9999"))

        # Graceful browse fallback when query is empty or only broad filter criteria matched
        if not scored and (not query_tokens or kind != "all" or category or date_from or date_to):
            fallback_items = [self._cached_items[idx] for idx in eligible]
            fallback_items.sort(
                key=lambda it: (it.planned_start or "9999", -it.updated_at.timestamp())
            )
            scored = [(0.0, it) for it in fallback_items[:top_k]]
        else:
            scored = scored[:top_k]

        results: list[dict[str, Any]] = []
        for score, it in scored:
            results.append(
                {
                    "id": it.id,
                    "kind": it.kind.value,
                    "title": it.title,
                    "summary": it.summary,
                    "details": it.details,
                    "tags": it.tags,
                    "planned_start": it.planned_start,
                    "planned_end": it.planned_end,
                    "location": it.location,
                    "category": it.category,
                    "estimated_minutes": it.estimated_minutes,
                    "energy_level": it.energy_level,
                    "status": it.status.value,
                    "source": it.source,
                    "bm25_score": round(score, 4),
                }
            )
        return results

    async def promote_due_items(
        self,
        *,
        now: datetime | None = None,
        horizon_days: int = 7,
        default_lead_minutes: int = 20,
        default_priority: float = 9.0,
    ) -> list[TimetableEntry]:
        """Promote Slow Loop timed events that have entered the 7-day Fast Loop window.

        Moves qualifying active items into `timetable` AND writes their corresponding
        `ScheduledPassiveTrigger` so future imported classes/shifts automatically become
        active reminders when their week arrives!
        """
        ref = now or utc_now()
        cutoff = ref + timedelta(days=horizon_days)
        past_floor = ref - timedelta(hours=2)

        active_items = await self.storage.list_slow_memory(status=SlowMemoryStatus.ACTIVE.value, limit=2000)
        promoted: list[TimetableEntry] = []

        for item in active_items:
            if not item.planned_start:
                continue
            start_dt = parse_iso_datetime(item.planned_start, ref)
            if start_dt is None:
                continue
            if past_floor <= start_dt <= cutoff:
                entry = TimetableEntry(
                    title=item.title,
                    start_time=item.planned_start,
                    end_time=item.planned_end,
                    location=item.location,
                    category=item.category,
                    is_mandatory=item.is_mandatory or item.category in ("class", "shift", "deadline"),
                    status=TimetableStatus.SCHEDULED,
                    notes=item.summary if not item.details else f"{item.summary} | {item.details}",
                    source=f"promoted_from_slow:{item.id}",
                )
                saved_entry = await self.storage.upsert_timetable(entry)

                lead = item.reminder_lead_minutes or default_lead_minutes
                trig = build_passive_trigger_for_event(
                    saved_entry,
                    lead_minutes=lead,
                    priority=default_priority,
                    now=ref,
                )
                if trig is not None:
                    await self.storage.upsert_scheduled_trigger(trig)

                item.status = SlowMemoryStatus.PROMOTED
                await self.storage.upsert_slow_memory(item)
                promoted.append(saved_entry)
                logger.info(
                    "Promoted Slow Loop item [%s] (%s @ %s) into Fast Loop Timetable + Passive Trigger.",
                    item.id,
                    item.title,
                    item.planned_start,
                )

        if promoted:
            self.invalidate()
        return promoted
