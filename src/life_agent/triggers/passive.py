"""Passive Trigger Engine (v2: Forward Triggers + Two-Phase Backward Pass Maintenance).

Strictly separates:
  1. Forward Triggers:
     - Dynamic Wake-up (computed daily from actual first morning event + weekday/weekend policy)
     - Static/configured routine slots (lunch/nap, afternoon class, evening campus run, night wrap-up)
     - Explicit dated `scheduled_triggers` (from Schedule/Shift Auto-Importer & Slow Loop promotions)
     - Fallback T-N min Timetable reminders
     - Transition-driven Break / Free-gap digest (`dispatch_window_digest` when an event just ended)
  2. Short-Beat Backward Pass (deterministic maintenance — NEVER mutates `grade`):
     - T0/T1 expiry cleanup & T2 multi-day state renewal confirmation (via Message Gateway)
     - Slow Loop 7-day rolling horizon promotion into global Timetable + Passive Triggers
     - Dynamic Follow-up timing recalculation against global Timetable & Gateway-arbitrated dispatch
     - Weighted Buffer sweep
  3. Long-Beat Nightly Backward Pass (23:30 once per day):
     - Runs `NightlyGradeReflector` to evolve & atomically publish `grade_v(N+1)`
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from ..config import AppConfig, PassiveTriggerSlot
from ..gateway.gateway import MessageGateway
from ..gateway.policy import GatingPolicyManager
from ..models import FollowUpStatus, PassiveTriggerPayload, utc_now
from ..scene.manager import SceneStateManager
from ..slow_loop.bm25_store import SlowLoopBM25Store, parse_iso_datetime
from ..storage import Storage
from .followup import FollowUpTimingCalculator

logger = logging.getLogger(__name__)


class PassiveTriggerEngine:
    """Orchestrates dynamic forward triggers, short-beat calibration, and nightly grade evolution."""

    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        gateway: MessageGateway,
        slow_store: SlowLoopBM25Store | None = None,
        scene_manager: SceneStateManager | None = None,
        policy: GatingPolicyManager | None = None,
        followup_calc: FollowUpTimingCalculator | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.gateway = gateway
        self.slow_store = slow_store
        self.scene_manager = scene_manager or gateway.scene_manager
        self.policy = policy or gateway.policy
        self.followup_calc = followup_calc or FollowUpTimingCalculator(storage)

        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_sweep_at: datetime | None = None
        self._last_promote_at: datetime | None = None

    async def start(self) -> None:
        if not self.config.passive_triggers.enabled:
            logger.info("PassiveTriggerEngine disabled in config.")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="passive_trigger_engine")
        logger.info(
            "PassiveTriggerEngine started (dynamic_wakeup=%s, transition_digest=%s, nightly=%s).",
            self.config.passive_triggers.dynamic_wakeup_enabled,
            self.config.passive_triggers.transition_digest_enabled,
            self.config.gating_evolution.nightly_reflection_time,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error inside PassiveTriggerEngine tick")
            await asyncio.sleep(self.config.passive_triggers.tick_interval_seconds)

    async def tick(self, now: datetime | None = None) -> dict[str, Any]:
        """Execute one full tick: Short-Beat Backward -> Forward Triggers -> Long-Beat Nightly."""
        current = now or utc_now()

        # 1. Short-Beat Backward Pass (deterministic calibration; NEVER mutates grade!)
        backward_stats = await self.run_short_beat_backward(now=current)

        # 2. Forward Triggers
        wakeup_fired = await self._check_dynamic_wakeup(current)
        slots_fired = await self._check_daily_slots(current)
        scheduled_fired = await self._check_scheduled_triggers(current)
        reminders_fired = await self._check_timetable_reminders(current)
        transition_pushed = await self._check_transition_digest(current)

        # 3. Buffer Sweep
        swept = 0
        sweep_interval = self.config.gateway.buffer_sweep_interval_seconds
        if (
            self._last_sweep_at is None
            or (current - self._last_sweep_at).total_seconds() >= sweep_interval
        ):
            self._last_sweep_at = current
            swept = await self.gateway.sweep_due_buffer_items(now=current)

        # 4. Long-Beat Nightly Backward Pass (23:30 once per day)
        nightly_ran = await self._check_nightly_reflection(current)

        return {
            "wakeup_fired": wakeup_fired,
            "slots_fired": slots_fired,
            "scheduled_fired": scheduled_fired,
            "reminders_fired": reminders_fired,
            "transition_pushed": transition_pushed,
            "buffer_swept": swept,
            "nightly_ran": nightly_ran,
            **backward_stats,
        }

    async def run_short_beat_backward(self, *, now: datetime | None = None) -> dict[str, int]:
        """Short-beat deterministic Backward Pass (post-persistence calibration, zero rule mutation)."""
        ref = now or utc_now()

        # 1a. T0/T1 expiry cleanup + T2 renewal confirmation (sent via Message Gateway)
        scene_res = await self.scene_manager.run_short_beat_maintenance(now=ref)
        t2_renewals_sent = 0
        for renewal_payload in scene_res["t2_renewals"]:
            await self.gateway.evaluate_passive_trigger(renewal_payload, now=ref)
            t2_renewals_sent += 1

        # 1b. Slow Loop 7-day rolling horizon promotion into global Timetable
        promoted_count = 0
        if self.slow_store is not None and self.config.slow_loop.enabled:
            promote_interval = self.config.slow_loop.auto_promote_interval_seconds
            if (
                self._last_promote_at is None
                or (ref - self._last_promote_at).total_seconds() >= promote_interval
            ):
                self._last_promote_at = ref
                promoted = await self.slow_store.promote_due_items(
                    now=ref,
                    horizon_days=self.config.slow_loop.horizon_days,
                    default_lead_minutes=self.config.schedule_importer.default_reminder_lead_minutes,
                    default_priority=self.config.schedule_importer.default_priority,
                )
                promoted_count = len(promoted)

        # 1c. Recalculate open Follow-Up timings against global Timetable + dispatch due ones TO GATEWAY
        recalced = await self.followup_calc.recalculate_all_open(now=ref)
        followups_submitted = await self._check_due_followups(ref)

        return {
            "cleaned_t0": scene_res["cleaned_t0"],
            "cleaned_t1": scene_res["cleaned_t1"],
            "t2_renewals_sent": t2_renewals_sent,
            "slow_promoted": promoted_count,
            "followups_recalculated": recalced,
            "followups_submitted": followups_submitted,
        }

    async def _check_due_followups(self, now: datetime) -> int:
        open_fus = await self.storage.list_followups(status=FollowUpStatus.OPEN.value, limit=50)
        submitted = 0
        for fu in open_fus:
            if fu.check_count >= fu.max_checks or not fu.computed_check_at:
                continue
            check_dt = parse_iso_datetime(fu.computed_check_at, now)
            if check_dt is None:
                continue
            delta_s = (now - check_dt).total_seconds()
            if 0 <= delta_s <= 1800:
                payload = self.followup_calc.build_gateway_payload(fu, now=now)
                await self.gateway.evaluate_followup_request(payload, now=now)
                submitted += 1
        return submitted

    async def compute_dynamic_wakeup_time(self, now: datetime) -> datetime:
        """Compute today's real wake-up time from global Timetable + weekday/weekend policy."""
        lp = self.config.owner.living_profile
        today_str = now.strftime("%Y-%m-%d")
        entries = await self.storage.list_timetable(
            date_from=today_str,
            date_to=today_str,
            status="scheduled",
            limit=30,
        )
        first_morning: datetime | None = None
        for e in entries:
            s_dt = parse_iso_datetime(e.start_time, now)
            if s_dt is not None and 5 <= s_dt.hour < 12:
                if first_morning is None or s_dt < first_morning:
                    first_morning = s_dt

        is_weekend = now.weekday() >= 5
        fallback_hhmm = lp.weekend_wakeup if is_weekend else lp.weekday_latest_wakeup
        fh, fm = (int(x) for x in fallback_hhmm.split(":")[:2])
        fallback_dt = now.replace(hour=fh, minute=fm, second=0, microsecond=0)

        if first_morning is not None:
            event_wake = first_morning - timedelta(minutes=lp.wakeup_lead_minutes)
            return min(event_wake, fallback_dt)
        return fallback_dt

    async def _check_dynamic_wakeup(self, now: datetime) -> int:
        if not self.config.passive_triggers.dynamic_wakeup_enabled:
            return 0
        date_str = now.strftime("%Y-%m-%d")
        key = f"dynamic_wakeup:{date_str}"
        if await self.storage.has_trigger_fired(key):
            return 0
        wake_dt = await self.compute_dynamic_wakeup_time(now)
        delta_s = (now - wake_dt).total_seconds()
        if not (0 <= delta_s <= 900):
            return 0

        await self.storage.mark_trigger_fired(key)
        payload = PassiveTriggerPayload(
            trigger_name="dynamic_morning_wakeup",
            directive=(
                f"动态起床提醒（根据今日实际日程算出的起床时间 {wake_dt.strftime('%H:%M')}）："
                "叫用户起床，结合 T3 早餐偏好建议今天早饭吃什么，并结合全局时间表告知上午第一节课/安排的时间和教学楼地点。"
            ),
            category="daily_wakeup",
            priority=9.5,
            urgency=9.5,
            fired_at=now,
        )
        await self.gateway.evaluate_passive_trigger(payload, now=now)
        return 1

    async def _check_transition_digest(self, now: datetime) -> int:
        """Detect class/event-end transitions -> trigger bundled Break / Free-Gap digest!"""
        if not self.config.passive_triggers.transition_digest_enabled:
            return 0
        today_str = now.strftime("%Y-%m-%d")
        entries = await self.storage.list_timetable(
            date_from=today_str,
            date_to=today_str,
            status="scheduled",
            limit=40,
        )
        pushed_count = 0
        for entry in entries:
            s_dt = parse_iso_datetime(entry.start_time, now)
            if s_dt is None:
                continue
            e_dt = parse_iso_datetime(entry.end_time, now) if entry.end_time else (s_dt + timedelta(minutes=45))
            elapsed_since_end = (now - e_dt).total_seconds()
            if not (0 <= elapsed_since_end <= 600):
                continue

            trig_key = f"transition_end:{entry.id}:{e_dt.strftime('%Y%m%d%H%M')}"
            if await self.storage.has_trigger_fired(trig_key):
                continue
            await self.storage.mark_trigger_fired(trig_key)

            snap = await self.scene_manager.compute_activity_snapshot(now=now)
            gap = snap.free_gap_minutes if snap.free_gap_minutes is not None else 60
            window_kind = "break" if gap <= 35 else "free_long"
            res = await self.gateway.dispatch_window_digest(
                window_kind=window_kind,
                free_gap_minutes=gap,
                now=now,
            )
            if res.get("pushed"):
                pushed_count += 1
        return pushed_count

    async def _check_nightly_reflection(self, now: datetime) -> bool:
        """Long-beat Backward Pass: run once per night at configured reflection time (23:30)."""
        target_str = self.config.gating_evolution.nightly_reflection_time.strip()
        try:
            hh, mm = (int(x) for x in target_str.split(":")[:2])
        except ValueError:
            return False
        target_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta_s = (now - target_dt).total_seconds()
        if not (0 <= delta_s <= 900):
            return False

        key = f"nightly_reflection:{now.strftime('%Y-%m-%d')}"
        if await self.storage.has_trigger_fired(key):
            return False
        await self.storage.mark_trigger_fired(key)
        await self.policy.run_nightly_reflection(now=now)
        return True

    async def _check_daily_slots(self, now: datetime) -> int:
        fired = 0
        date_str = now.strftime("%Y-%m-%d")
        weekday = now.weekday()

        for slot in self.config.passive_triggers.slots:
            if not slot.enabled:
                continue
            if weekday not in slot.weekdays:
                continue
            if not self._is_slot_due(slot, now):
                continue

            trigger_key = f"slot:{date_str}:{slot.name}"
            if await self.storage.has_trigger_fired(trigger_key):
                continue

            await self.storage.mark_trigger_fired(trigger_key)
            await self.fire_slot(slot, now=now)
            fired += 1
        return fired

    @staticmethod
    def _is_slot_due(slot: PassiveTriggerSlot, now: datetime) -> bool:
        try:
            hh_str, mm_str = slot.time.strip().split(":")
            slot_dt = now.replace(hour=int(hh_str), minute=int(mm_str), second=0, microsecond=0)
        except (ValueError, AttributeError):
            return False
        delta_s = (now - slot_dt).total_seconds()
        return 0 <= delta_s <= 900

    async def fire_slot(self, slot: PassiveTriggerSlot, *, now: datetime | None = None) -> None:
        ref = now or utc_now()
        payload = PassiveTriggerPayload(
            trigger_name=slot.name,
            directive=slot.directive,
            category="daily_routine",
            priority=slot.priority,
            urgency=slot.priority,
            fired_at=ref,
        )
        logger.info("PassiveTrigger firing daily slot [%s] (%s)", slot.name, slot.time)
        await self.gateway.evaluate_passive_trigger(payload, now=ref)

    async def _check_scheduled_triggers(self, now: datetime) -> int:
        pending = await self.storage.list_scheduled_triggers(include_fired=False, limit=100)
        fired = 0
        for trig in pending:
            trig_dt = parse_iso_datetime(trig.trigger_at, now)
            if trig_dt is None:
                continue
            delta_s = (now - trig_dt).total_seconds()
            if delta_s < 0:
                continue
            if delta_s > 7200:
                await self.storage.mark_scheduled_trigger_fired(trig.id)
                continue

            await self.storage.mark_scheduled_trigger_fired(trig.id)
            if trig.related_timetable_id:
                await self.storage.mark_timetable_reminded(trig.related_timetable_id)

            payload = PassiveTriggerPayload(
                trigger_name=trig.trigger_name,
                directive=trig.directive,
                category=trig.category,
                priority=trig.priority,
                urgency=max(trig.priority, 9.0),
                related_timetable_id=trig.related_timetable_id,
                fired_at=now,
            )
            logger.info(
                "PassiveTrigger firing scheduled trigger [%s] (scheduled %s)",
                trig.trigger_name,
                trig.trigger_at,
            )
            await self.gateway.evaluate_passive_trigger(payload, now=now)
            fired += 1
        return fired

    async def _check_timetable_reminders(self, now: datetime) -> int:
        lead = timedelta(minutes=self.config.passive_triggers.timetable_reminder_lead_minutes)
        entries = await self.storage.list_timetable(status="scheduled", limit=50)
        fired = 0

        for entry in entries:
            if entry.reminded:
                continue
            if await self.storage.has_scheduled_trigger_for_timetable(entry.id):
                continue

            start_dt = parse_iso_datetime(entry.start_time, now)
            if start_dt is None:
                continue
            delta = start_dt - now
            if timedelta(minutes=-2) <= delta <= lead:
                await self.storage.mark_timetable_reminded(entry.id)
                mins = max(0, int(delta.total_seconds() // 60))
                loc_str = f"，地点：{entry.location}" if entry.location else ""
                notes_str = f"，备注：{entry.notes}" if entry.notes else ""
                directive = (
                    f"时间表临近提醒：「{entry.title}」将在约 {mins} 分钟后"
                    f"（{entry.start_time}）开始{loc_str}{notes_str}。"
                    "请主动提醒用户准备出发或处理。"
                )
                payload = PassiveTriggerPayload(
                    trigger_name=f"timetable_{entry.id}",
                    directive=directive,
                    category="timetable_reminder",
                    priority=9.5 if entry.is_mandatory else 7.5,
                    urgency=9.5 if entry.is_mandatory else 7.5,
                    related_timetable_id=entry.id,
                    fired_at=now,
                )
                logger.info("PassiveTrigger firing timetable reminder for %s (%s)", entry.title, entry.start_time)
                await self.gateway.evaluate_passive_trigger(payload, now=now)
                fired += 1
        return fired
