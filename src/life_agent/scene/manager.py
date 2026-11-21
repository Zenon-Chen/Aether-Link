"""Tiered Scene State Manager (T0–T3 TTL) — eliminates cross-day & domain crosstalk.

Tiers:
  - T0 (t0_ephemeral):  Event-bound or 2–3h (temporary ambient notes, live location). Silent cleanup on expiry.
  - T1 (t1_daily):      Resets every day at 04:00 (awake, meals today, campus run today).
  - T2 (t2_multiday):   Default 3–7d (illness/injury, exam week). NEVER silently deleted; emits gentle
                        renewal confirmation check via Message Gateway before expiry!
  - T3 (t3_persistent): Durable living_profile & stable preferences. Never expires automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..config import AppConfig
from ..models import (
    ActivityMode,
    GatingRule,
    GatingRuleStatus,
    Interruptibility,
    PassiveTriggerPayload,
    StateTier,
    TieredSceneEntry,
    TimetableEntry,
    utc_now,
)
from ..slow_loop.bm25_store import parse_iso_datetime
from ..storage import Storage

logger = logging.getLogger(__name__)


@dataclass
class ActivitySnapshot:
    mode: ActivityMode
    interruptibility: Interruptibility
    current_event: TimetableEntry | None = None
    next_event: TimetableEntry | None = None
    minutes_until_next: int | None = None
    free_gap_minutes: int | None = None
    active_override_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "interruptibility": self.interruptibility.value,
            "current_event": (
                {
                    "id": self.current_event.id,
                    "title": self.current_event.title,
                    "start_time": self.current_event.start_time,
                    "end_time": self.current_event.end_time,
                    "location": self.current_event.location,
                }
                if self.current_event
                else None
            ),
            "next_event": (
                {
                    "id": self.next_event.id,
                    "title": self.next_event.title,
                    "start_time": self.next_event.start_time,
                    "location": self.next_event.location,
                }
                if self.next_event
                else None
            ),
            "minutes_until_next": self.minutes_until_next,
            "free_gap_minutes": self.free_gap_minutes,
            "active_override_reason": self.active_override_reason,
        }


class SceneStateManager:
    """Manages T0–T3 scoped scene entries, expiry cleanup, T2 renewal checks, and ActivitySnapshot."""

    def __init__(self, config: AppConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage

    def _next_t1_reset(self, now: datetime) -> datetime:
        reset_hour = self.config.gating_evolution.t1_daily_reset_hour
        candidate = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    async def sync_living_profile(self) -> None:
        """Persist T3 living_profile baseline from config into T3 tier (never expires)."""
        lp = self.config.owner.living_profile
        await self.set_state(
            key="living_profile",
            value=lp.model_dump(mode="json"),
            tier=StateTier.T3_PERSISTENT,
            source="config:living_profile",
        )

    async def set_state(
        self,
        *,
        key: str,
        value: Any,
        tier: StateTier = StateTier.T0_EPHEMERAL,
        source: str = "system",
        ttl_hours: float | None = None,
        ttl_days: float | None = None,
        now: datetime | None = None,
    ) -> TieredSceneEntry:
        ref = now or utc_now()
        expires_at: datetime | None = None

        if tier == StateTier.T0_EPHEMERAL:
            hrs = ttl_hours if ttl_hours is not None else 3.0
            expires_at = ref + timedelta(hours=hrs)
            await self._enforce_t0_cap(exclude_key=key)
        elif tier == StateTier.T1_DAILY:
            expires_at = self._next_t1_reset(ref)
        elif tier == StateTier.T2_MULTIDAY:
            days = ttl_days if ttl_days is not None else float(self.config.gating_evolution.t2_default_ttl_days)
            expires_at = ref + timedelta(days=days)
        elif tier == StateTier.T3_PERSISTENT:
            expires_at = None

        entry = TieredSceneEntry(
            key=key,
            tier=tier,
            value=value,
            source=source,
            expires_at=expires_at,
            renewal_prompted=False,
            updated_at=ref,
        )
        return await self.storage.upsert_tiered_scene(entry)

    async def clear_state(self, key: str) -> bool:
        return await self.storage.delete_tiered_scene(key)

    async def renew_t2_state(
        self,
        key: str,
        *,
        extend_days: float | None = None,
        now: datetime | None = None,
    ) -> TieredSceneEntry | None:
        entry = await self.storage.get_tiered_scene(key)
        if entry is None or entry.tier != StateTier.T2_MULTIDAY:
            return None
        ref = now or utc_now()
        days = extend_days if extend_days is not None else float(self.config.gating_evolution.t2_default_ttl_days)
        entry.expires_at = ref + timedelta(days=days)
        entry.renewal_prompted = False
        return await self.storage.upsert_tiered_scene(entry)

    async def _enforce_t0_cap(self, *, exclude_key: str) -> None:
        cap = self.config.gating_evolution.max_t0_ambient_notes
        t0_entries = await self.storage.list_tiered_scene(tier=StateTier.T0_EPHEMERAL)
        others = [e for e in t0_entries if e.key != exclude_key]
        if len(others) >= cap:
            # Evict oldest T0 entries beyond cap
            others.sort(key=lambda e: e.updated_at)
            for victim in others[: len(others) - cap + 1]:
                await self.storage.delete_tiered_scene(victim.key)

    async def run_short_beat_maintenance(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Short-beat Backward Pass: clean expired T0/T1 entries and collect T2 renewal confirmations.

        T2 multi-day states (e.g. knee injury / illness / exam week) are NEVER silently dropped!
        """
        ref = now or utc_now()
        entries = await self.storage.list_tiered_scene()
        cleaned_t0 = 0
        cleaned_t1 = 0
        renewal_payloads: list[PassiveTriggerPayload] = []
        lead = timedelta(hours=self.config.gating_evolution.t2_renewal_lead_hours)

        for entry in entries:
            if entry.tier == StateTier.T3_PERSISTENT:
                continue
            if entry.expires_at is None:
                continue

            if entry.tier in (StateTier.T0_EPHEMERAL, StateTier.T1_DAILY):
                if ref >= entry.expires_at:
                    await self.storage.delete_tiered_scene(entry.key)
                    if entry.tier == StateTier.T0_EPHEMERAL:
                        cleaned_t0 += 1
                    else:
                        cleaned_t1 += 1
            elif entry.tier == StateTier.T2_MULTIDAY:
                # Within renewal lead window (or past expiry) & not yet prompted -> ask user gently via Gateway!
                if (ref + lead) >= entry.expires_at and not entry.renewal_prompted:
                    entry.renewal_prompted = True
                    await self.storage.upsert_tiered_scene(entry)
                    renewal_payloads.append(
                        PassiveTriggerPayload(
                            trigger_name=f"t2_renewal_{entry.key}",
                            directive=(
                                f"持续状态续期确认：你之前记录的跨日状态「{entry.key}: {entry.value}」快到预设周期了。"
                                "请在用户空闲时顺带温和关心一句现在情况好转/结束了吗？"
                                "如果好了就清除该状态，还没好则继续保留续期。"
                            ),
                            category="t2_state_renewal",
                            priority=5.5,
                            urgency=4.0,
                            related_scene_key=entry.key,
                            fired_at=ref,
                        )
                    )

        return {
            "cleaned_t0": cleaned_t0,
            "cleaned_t1": cleaned_t1,
            "t2_renewals": renewal_payloads,
        }

    async def compute_activity_snapshot(
        self,
        *,
        now: datetime | None = None,
        active_rules: list[GatingRule] | None = None,
    ) -> ActivitySnapshot:
        """Deterministically compute ActivityMode, Interruptibility, and exact free gap from global Timetable."""
        ref = now or utc_now()
        rules = active_rules if active_rules is not None else await self.storage.list_gating_rules(status="active")

        # 1. Check active temporary_override DND switches first
        for rule in rules:
            if rule.scope_type == "temporary_override" and rule.status == GatingRuleStatus.ACTIVE:
                if rule.expires_at is not None and ref < rule.expires_at:
                    if rule.effect in ("suppress", "scene_only", "timing_policy"):
                        return ActivitySnapshot(
                            mode=ActivityMode.DND,
                            interruptibility=Interruptibility.URGENT_ONLY,
                            active_override_reason=rule.instruction,
                        )

        today_str = (ref - timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow_str = (ref + timedelta(days=1)).strftime("%Y-%m-%d")
        entries = await self.storage.list_timetable(
            date_from=today_str,
            date_to=tomorrow_str,
            status="scheduled",
            limit=60,
        )

        current_event: TimetableEntry | None = None
        next_event: TimetableEntry | None = None
        prev_ended_recently = False

        for entry in entries:
            s_dt = parse_iso_datetime(entry.start_time, ref)
            if s_dt is None:
                continue
            e_dt = parse_iso_datetime(entry.end_time, ref) if entry.end_time else (s_dt + timedelta(minutes=45))
            if s_dt <= ref < e_dt:
                current_event = entry
            elif e_dt <= ref and (ref - e_dt) <= timedelta(minutes=25):
                prev_ended_recently = True
            elif s_dt > ref and next_event is None:
                next_event = entry

        minutes_until_next: int | None = None
        if next_event is not None:
            ns_dt = parse_iso_datetime(next_event.start_time, ref)
            if ns_dt is not None:
                minutes_until_next = max(0, int((ns_dt - ref).total_seconds() // 60))

        # 2. Currently inside a scheduled event
        if current_event is not None:
            # Check if an explicit active timing_policy rule marks this course as chill/low-friction
            is_chill = False
            title_low = current_event.title.lower()
            for rule in rules:
                if (
                    rule.status == GatingRuleStatus.ACTIVE
                    and rule.scope_type == "course_or_time"
                    and rule.effect == "timing_policy"
                ):
                    if rule.expires_at is not None and ref >= rule.expires_at:
                        continue
                    key_clean = rule.scope_key.lower().replace("class:", "").strip()
                    if key_clean and (key_clean in title_low or title_low in key_clean):
                        if any(k in rule.instruction for k in ("摸鱼", "允许", "可以推", "low_friction", "看新闻")):
                            is_chill = True
                            break

            if current_event.category == "class":
                return ActivitySnapshot(
                    mode=ActivityMode.IN_CLASS_CHILL if is_chill else ActivityMode.IN_CLASS,
                    interruptibility=(
                        Interruptibility.LOW_FRICTION if is_chill else Interruptibility.URGENT_ONLY
                    ),
                    current_event=current_event,
                    next_event=next_event,
                    minutes_until_next=minutes_until_next,
                    free_gap_minutes=0,
                )
            return ActivitySnapshot(
                mode=ActivityMode.BUSY_EVENT,
                interruptibility=Interruptibility.URGENT_ONLY,
                current_event=current_event,
                next_event=next_event,
                minutes_until_next=minutes_until_next,
                free_gap_minutes=0,
            )

        # 3. Sleeping quiet hours (e.g. 23:30..07:00)
        lp = self.config.owner.living_profile
        in_sleep_hours = (
            (ref.hour > lp.sleep_start_hour or (ref.hour == lp.sleep_start_hour and ref.minute >= 30))
            or (ref.hour < lp.sleep_end_hour)
        )
        if in_sleep_hours:
            return ActivitySnapshot(
                mode=ActivityMode.SLEEPING,
                interruptibility=Interruptibility.URGENT_ONLY,
                next_event=next_event,
                minutes_until_next=minutes_until_next,
                free_gap_minutes=minutes_until_next,
            )

        # 4. Open window: short class break vs free_short vs free_long
        gap = minutes_until_next if minutes_until_next is not None else 180
        if prev_ended_recently and gap <= 35:
            mode = ActivityMode.BREAK
        elif gap < 45:
            mode = ActivityMode.FREE_SHORT
        else:
            mode = ActivityMode.FREE_LONG

        return ActivitySnapshot(
            mode=mode,
            interruptibility=Interruptibility.OPEN,
            next_event=next_event,
            minutes_until_next=minutes_until_next,
            free_gap_minutes=gap,
        )

    async def get_structured_view(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Return clean tiered Scene State dictionary excluding expired T0/T1 items."""
        ref = now or utc_now()
        entries = await self.storage.list_tiered_scene()
        grouped: dict[str, dict[str, Any]] = {
            "T0_瞬时态": {},
            "T1_当日态": {},
            "T2_跨日持续态": {},
            "T3_长期基线": {},
        }
        tier_labels = {
            StateTier.T0_EPHEMERAL: "T0_瞬时态",
            StateTier.T1_DAILY: "T1_当日态",
            StateTier.T2_MULTIDAY: "T2_跨日持续态",
            StateTier.T3_PERSISTENT: "T3_长期基线",
        }
        for e in entries:
            # Hide already-expired T0/T1 even before short-beat cleanup tick runs
            if e.tier in (StateTier.T0_EPHEMERAL, StateTier.T1_DAILY) and e.expires_at and ref >= e.expires_at:
                continue
            bucket = grouped[tier_labels[e.tier]]
            if e.tier == StateTier.T2_MULTIDAY and e.expires_at:
                bucket[e.key] = {
                    "value": e.value,
                    "expires_at": e.expires_at.strftime("%Y-%m-%d"),
                    "renewal_prompted": e.renewal_prompted,
                }
            else:
                bucket[e.key] = e.value

        activity = await self.compute_activity_snapshot(now=ref)
        grouped["实时活动态"] = activity.to_dict()
        return grouped
