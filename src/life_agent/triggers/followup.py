"""Dynamic Follow-Up Timing Calculator (no separate timetable; 100% arbitrated by Message Gateway).

Computes and recalculates `computed_check_at` for every open `FollowUpIntent` directly
against the global single-source-of-truth `timetable`:
  1. Event-anchored (e.g. "待会下课去交作业"): anchor_event.end_time + 5 min.
  2. Deadline-backtracked (e.g. "驿站 20:00 关门"): deadline_at - safety_margin_minutes,
     automatically shifted to the nearest free window outside any busy Timetable event!
  3. Recomputed automatically whenever the global Timetable changes.
  4. When due, produces a structured PassiveTriggerPayload submitted to MessageGateway —
     NEVER bypasses Message Gateway!
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ..models import (
    FollowUpIntent,
    FollowUpStatus,
    PassiveTriggerPayload,
    TimetableEntry,
    utc_now,
)
from ..slow_loop.bm25_store import parse_iso_datetime
from ..storage import Storage

logger = logging.getLogger(__name__)


class FollowUpTimingCalculator:
    """Dynamically computes optimal follow-up check windows against the global Timetable."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    @staticmethod
    def _event_span(entry: TimetableEntry, ref: datetime) -> tuple[datetime, datetime] | None:
        s_dt = parse_iso_datetime(entry.start_time, ref)
        if s_dt is None:
            return None
        e_dt = parse_iso_datetime(entry.end_time, ref) if entry.end_time else (s_dt + timedelta(minutes=45))
        return s_dt, e_dt

    def _find_free_slot_near(
        self,
        *,
        desired: datetime,
        deadline: datetime | None,
        busy_spans: list[tuple[datetime, datetime]],
        now: datetime,
    ) -> datetime:
        """Align desired check time outside any busy Timetable event while preserving safety margin."""
        candidate = max(desired, now + timedelta(minutes=2))
        for _ in range(6):
            blocking = None
            for s_dt, e_dt in busy_spans:
                if s_dt <= candidate < e_dt:
                    blocking = (s_dt, e_dt)
                    break
            if blocking is None:
                break
            b_start, b_end = blocking
            after_class = b_end + timedelta(minutes=5)
            before_class = b_start - timedelta(minutes=10)
            # Prefer right after the blocking class ends if we still have >= 15 min before deadline
            if deadline is None or after_class <= (deadline - timedelta(minutes=15)):
                candidate = after_class
            elif before_class >= now + timedelta(minutes=2):
                candidate = before_class
            else:
                # Tight deadline inside class: place right after class or 10m before deadline
                candidate = min(after_class, deadline - timedelta(minutes=5)) if deadline else after_class
                break

        if deadline is not None and candidate > deadline - timedelta(minutes=5):
            candidate = max(now + timedelta(minutes=1), deadline - timedelta(minutes=5))
        return candidate

    async def compute_check_time(
        self,
        fu: FollowUpIntent,
        *,
        now: datetime | None = None,
        timetable: list[TimetableEntry] | None = None,
    ) -> str | None:
        """Compute optimal `computed_check_at` ISO timestamp for a single FollowUpIntent."""
        ref = now or utc_now()
        entries = (
            timetable
            if timetable is not None
            else await self.storage.list_timetable(status="scheduled", limit=80)
        )
        busy_spans: list[tuple[datetime, datetime]] = []
        by_id: dict[str, TimetableEntry] = {}
        for e in entries:
            by_id[e.id] = e
            span = self._event_span(e, ref)
            if span is not None:
                busy_spans.append(span)
        busy_spans.sort(key=lambda sp: sp[0])

        deadline_dt = parse_iso_datetime(fu.deadline_at, ref)

        # Case 1: Event-anchored follow-up (e.g. right after a specific class ends)
        if fu.anchor_event_id and fu.anchor_event_id in by_id:
            span = self._event_span(by_id[fu.anchor_event_id], ref)
            if span is not None:
                desired = span[1] + timedelta(minutes=5)
                chosen = self._find_free_slot_near(
                    desired=desired,
                    deadline=deadline_dt,
                    busy_spans=busy_spans,
                    now=ref,
                )
                return chosen.isoformat()

        # Case 2: Deadline-backtracked follow-up
        if deadline_dt is not None:
            if deadline_dt < ref - timedelta(minutes=15):
                return None  # Expired past deadline
            margin = timedelta(minutes=max(15, fu.safety_margin_minutes))
            desired = deadline_dt - margin
            chosen = self._find_free_slot_near(
                desired=desired,
                deadline=deadline_dt,
                busy_spans=busy_spans,
                now=ref,
            )
            return chosen.isoformat()

        # Case 3: Existing computed_check_at or default +45m delay shifted outside busy spans
        existing_dt = parse_iso_datetime(fu.computed_check_at, ref)
        desired = existing_dt if existing_dt is not None else (ref + timedelta(minutes=45))
        chosen = self._find_free_slot_near(
            desired=desired,
            deadline=None,
            busy_spans=busy_spans,
            now=ref,
        )
        return chosen.isoformat()

    async def recalculate_all_open(self, *, now: datetime | None = None) -> int:
        """Short-beat Backward Pass: recompute check windows for all open follow-ups."""
        ref = now or utc_now()
        open_fus = await self.storage.list_followups(status=FollowUpStatus.OPEN.value, limit=100)
        if not open_fus:
            return 0
        timetable = await self.storage.list_timetable(status="scheduled", limit=80)
        updated = 0

        busy_spans = [
            sp for e in timetable if (sp := self._event_span(e, ref)) is not None
        ]

        for fu in open_fus:
            deadline_dt = parse_iso_datetime(fu.deadline_at, ref)
            if deadline_dt is not None and deadline_dt < ref - timedelta(minutes=15):
                fu.status = FollowUpStatus.EXPIRED
                await self.storage.upsert_followup(fu)
                updated += 1
                continue

            # If already due within the active firing window and NOT blocked by a busy event,
            # do NOT bump it forward into the future — let `_check_due_followups` submit it to Gateway!
            existing_dt = parse_iso_datetime(fu.computed_check_at, ref)
            if existing_dt is not None:
                delta_s = (ref - existing_dt).total_seconds()
                in_busy = any(s_dt <= ref < e_dt for s_dt, e_dt in busy_spans)
                if 0 <= delta_s <= 1800 and not in_busy:
                    continue

            new_check = await self.compute_check_time(fu, now=ref, timetable=timetable)
            if new_check and new_check != fu.computed_check_at:
                fu.computed_check_at = new_check
                await self.storage.upsert_followup(fu)
                updated += 1
        return updated

    async def register_followup(
        self,
        fu: FollowUpIntent,
        *,
        now: datetime | None = None,
    ) -> FollowUpIntent:
        """Save a new/updated follow-up intent and immediately calculate its free check window."""
        ref = now or utc_now()
        fu.computed_check_at = await self.compute_check_time(fu, now=ref)
        return await self.storage.upsert_followup(fu)

    async def defer_followup(
        self,
        followup_id: str,
        *,
        delay_minutes: int = 25,
        now: datetime | None = None,
    ) -> FollowUpIntent | None:
        """Called when Message Gateway defers a follow-up due to a higher-priority conflict."""
        fu = await self.storage.get_followup(followup_id)
        if fu is None or fu.status != FollowUpStatus.OPEN:
            return None
        ref = now or utc_now()
        entries = await self.storage.list_timetable(status="scheduled", limit=80)
        busy_spans = [
            sp for e in entries if (sp := self._event_span(e, ref)) is not None
        ]
        busy_spans.sort(key=lambda sp: sp[0])
        deadline_dt = parse_iso_datetime(fu.deadline_at, ref)
        next_slot = self._find_free_slot_near(
            desired=ref + timedelta(minutes=delay_minutes),
            deadline=deadline_dt,
            busy_spans=busy_spans,
            now=ref,
        )
        fu.computed_check_at = next_slot.isoformat()
        return await self.storage.upsert_followup(fu)

    @staticmethod
    def build_gateway_payload(fu: FollowUpIntent, *, now: datetime | None = None) -> PassiveTriggerPayload:
        """Package a due follow-up as a structured request to Message Gateway (100% Gateway-arbitrated)."""
        ref = now or utc_now()
        dl_part = f"（截止时间：{fu.deadline_at}）" if fu.deadline_at else ""
        hint_part = f"；背景：{fu.context_hint}" if fu.context_hint else ""
        directive = (
            f"主动跟进请求：「{fu.topic}」{dl_part}{hint_part}。"
            f"这是第 {fu.check_count + 1}/{fu.max_checks} 次跟进，请确认现在是否适合温和询问用户进展或提醒出发。"
        )
        urgency = fu.priority
        if fu.deadline_at:
            dl_dt = parse_iso_datetime(fu.deadline_at, ref)
            if dl_dt is not None and (dl_dt - ref) <= timedelta(minutes=45):
                urgency = max(urgency, 9.0)
        return PassiveTriggerPayload(
            trigger_name=f"followup_{fu.id}",
            directive=directive,
            category="followup_check",
            priority=fu.priority,
            urgency=urgency,
            related_followup_id=fu.id,
            fired_at=ref,
        )
