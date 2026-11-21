"""RTIR Post-Conversation Parser (Step 5 & 5.1, v2 Forward-Pass Single-Direction Persistence).

Triggered after 3 minutes of user silence following conversation activity.
Strictly adheres to Forward/Backward separation:
  - NEVER mutates `grade` directly during daytime Forward Pass!
  - Explicit gating preferences & push attitude feedback -> APPEND-ONLY to `feedback_log`
    (to be evolved cleanly during the 23:30 Nightly Backward Pass).
  - Body/daily/multiday statuses -> written to T0 / T1 / T2 tiered Scene State!
  - Follow-up commitments -> registered into `follow_ups` with dynamic timing calculation!
  - URTI items -> written with immediate `next_review_at` so they are NEVER dead-lettered!
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from ..config import AppConfig
from ..gateway.policy import GatingPolicyManager
from ..llm import LLMClient
from ..models import (
    BufferItem,
    BufferStatus,
    BufferZone,
    ChatTurn,
    FollowUpIntent,
    FollowUpStatus,
    SlowMemoryItem,
    SlowMemoryKind,
    SlowMemoryStatus,
    StateTier,
    TimetableEntry,
    TimetableStatus,
    TrustTier,
    utc_now,
)
from ..scene.manager import SceneStateManager
from ..slow_loop.bm25_store import SlowLoopBM25Store, is_beyond_horizon, parse_iso_datetime
from ..storage import Storage
from ..triggers.followup import FollowUpTimingCalculator

logger = logging.getLogger(__name__)


RTIR_SYSTEM_PROMPT = """你是对话后解析器（RTIR - Real-Time Info Resolver，下行单向落盘模块）。
在用户结束对话并静默 3 分钟后，你会收到本轮新增的对话内容，以及当前已有的全局时间表快照与分级场景状态。
你的任务是从新增对话中提取有价值的结构化信息，按严格分层单向写回存储（绝不直接修改网关 grade 规则）：

1. `timetable_operations`（全局时间表，7天内）：对话中提到但尚未修改、或隐含的近期日程增删改与完成状态。
2. `urti_items`（快回路缓冲区 URTI 实时信息区）：影响近期的用户实时线索与待关注事项。
3. `slow_memory_items`（慢回路 BM25）：无确定时间的想法/空闲活动（kind="idea"）、以及超过 7 天的远期安排/规划。
4. `tiered_scene_updates`（四级保质期状态，严禁跨级混放！）：
   - `t0_ephemeral`（2–3小时瞬时态）：当前所在位置、临时心情、刚发生的小事；
   - `t1_daily`（次日 04:00 自动清零的当日态）：今日早午晚餐记录、今天是否已起床、今天是否已校园跑；
   - `t2_multiday`（3–7天跨日持续态，到期前系统会先温和确认不会静默删掉）：**身体不适（感冒/发烧/膝盖拉伤/胃痛）**、考试周、项目冲刺期、家里有事。
   - 若用户明确表示某项 T2 状态已痊愈/结束，将 `clear: true` 即可清除。
5. `followup_intents`（主动跟进承诺）：用户说“待会下课去交作业”“晚上8点前拿快递”“歇会儿去校园跑”等需要后续回访确认的承诺；若用户说“作业交完了/快递拿了”，用 `close_followup_topics` 结案。
6. `feedback_entries`（只追加写入反馈账本 `feedback_log`，供夜间 23:30 统一演化 `grade`）：
   - 用户明确表达的长期推送偏好（`is_explicit=true`，如“以后别推 CV 论文”“高数课可以摸鱼看新闻”“班长发的消息直接推”）；
   - 用户对刚才主动推送的态度（`sentiment`: positive / negative）。
7. `temporary_overrides`（仅限有时效的临时勿扰开关，如“今晚10点前别打扰我”，必须给精确 `expires_at`）。

必须严格输出 JSON 对象：
{
  "timetable_operations": [],
  "urti_items": [],
  "slow_memory_items": [],
  "tiered_scene_updates": [
    {"key": "knee_status", "tier": "t2_multiday", "value": "膝盖拉伤暂不宜跑步", "ttl_days": 5, "clear": false}
  ],
  "followup_intents": [
    {"topic": "去菜鸟驿站取快递", "context_hint": "驿站20:00关门", "intent_type": "deadline_backtrack", "deadline_at": "2026-11-21T20:00:00+08:00", "safety_margin_minutes": 60, "priority": 8.5}
  ],
  "close_followup_topics": [],
  "feedback_entries": [
    {"event_type": "explicit_preference", "source_key": "topic:CV视觉论文", "tags": ["CV", "论文"], "sentiment": "negative", "signal_value": -2.0, "raw_text": "以后别推CV论文了", "is_explicit": true}
  ],
  "temporary_overrides": [],
  "scene_state_updates": {}
}
"""


class RTIRParser:
    """Parses newly added conversation turns after silence (strictly Forward-Pass write-only)."""

    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        llm: LLMClient,
        slow_store: SlowLoopBM25Store | None = None,
        scene_manager: SceneStateManager | None = None,
        policy: GatingPolicyManager | None = None,
        followup_calc: FollowUpTimingCalculator | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.llm = llm
        self.slow_store = slow_store
        self.scene_manager = scene_manager or SceneStateManager(config, storage)
        self.policy = policy
        self.followup_calc = followup_calc or FollowUpTimingCalculator(storage)

    async def parse_unparsed_turns(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Parse all unparsed chat turns if there is at least one unparsed user turn."""
        ref = now or utc_now()
        turns = await self.storage.get_unparsed_chat_turns()
        if not turns:
            return {
                "parsed_turns": 0,
                "timetable_ops": 0,
                "urti_added": 0,
                "slow_added": 0,
                "followups_added": 0,
                "feedback_logged": 0,
            }

        has_user_turn = any(t.role == "user" for t in turns)
        max_id = max(t.id for t in turns if t.id is not None)
        if not has_user_turn:
            return {
                "parsed_turns": 0,
                "timetable_ops": 0,
                "urti_added": 0,
                "slow_added": 0,
                "followups_added": 0,
                "feedback_logged": 0,
                "skipped": "no_user_turn",
            }

        timetable = await self.storage.list_timetable(status="scheduled", limit=60)
        scene_view = await self.scene_manager.get_structured_view(now=ref)
        open_fus = await self.storage.list_followups(status=FollowUpStatus.OPEN.value, limit=20)

        transcript = self._render_turns(turns)
        tt_snapshot = [
            {
                "id": e.id,
                "title": e.title,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "location": e.location,
                "status": e.status.value,
                "notes": e.notes,
            }
            for e in timetable
        ]
        fu_snapshot = [
            {"id": f.id, "topic": f.topic, "deadline_at": f.deadline_at, "computed_check_at": f.computed_check_at}
            for f in open_fus
        ]
        periods = "\n".join(f"  - {k}: {v}" for k, v in self.config.owner.school_periods.items())
        user_prompt = (
            f"【当前时间】{ref.isoformat()}\n"
            f"【快/慢回路分界】{self.config.slow_loop.horizon_days} 天内写时间表，超过 {self.config.slow_loop.horizon_days} 天或无确定时间的想法写 slow_memory_items\n"
            f"【课节时间表】\n{periods}\n\n"
            f"【当前全局时间表快照】\n{json.dumps(tt_snapshot, ensure_ascii=False, indent=2)}\n\n"
            f"【当前分级 Scene State】\n{json.dumps(scene_view, ensure_ascii=False, indent=2)}\n\n"
            f"【当前未结案跟进项】\n{json.dumps(fu_snapshot, ensure_ascii=False, indent=2)}\n\n"
            f"【新增对话内容】\n{transcript}"
        )

        result = await self.llm.complete_json(
            role="rtir",
            system_prompt=RTIR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        tt_count, diverted_slow = await self._apply_timetable_ops(
            result.get("timetable_operations") or [],
            now=ref,
        )
        urti_count = await self._apply_urti_items(result.get("urti_items") or [], now=ref)
        slow_count = diverted_slow + await self._apply_slow_memory_items(
            result.get("slow_memory_items") or []
        )
        scene_count = await self._apply_tiered_scene_updates(
            tiered=result.get("tiered_scene_updates") or [],
            legacy=result.get("scene_state_updates") or {},
            now=ref,
        )
        fu_count = await self._apply_followups(
            new_intents=result.get("followup_intents") or [],
            close_topics=result.get("close_followup_topics") or [],
            now=ref,
        )
        fb_count = await self._apply_feedback_and_overrides(
            feedback_entries=result.get("feedback_entries") or [],
            temp_overrides=result.get("temporary_overrides") or [],
            now=ref,
        )

        # If Timetable changed, recalculate open follow-ups immediately!
        if tt_count > 0:
            await self.followup_calc.recalculate_all_open(now=ref)

        await self.storage.mark_chat_turns_parsed(max_id)
        logger.info(
            "RTIR parsed %d turn(s): tt=%d, urti=%d, slow=%d, scene=%d, fu=%d, fb=%d.",
            len(turns),
            tt_count,
            urti_count,
            slow_count,
            scene_count,
            fu_count,
            fb_count,
        )
        return {
            "parsed_turns": len(turns),
            "timetable_ops": tt_count,
            "urti_added": urti_count,
            "slow_added": slow_count,
            "scene_updates_count": scene_count,
            "followups_added": fu_count,
            "feedback_logged": fb_count,
        }

    @staticmethod
    def _render_turns(turns: list[ChatTurn]) -> str:
        lines: list[str] = []
        for t in turns:
            ts = t.created_at.strftime("%Y-%m-%d %H:%M")
            speaker = "用户" if t.role == "user" else "助手"
            lines.append(f"[{ts}] ({t.channel}) {speaker}: {t.content}")
        return "\n".join(lines)

    async def _apply_timetable_ops(self, ops: list[Any], *, now: datetime) -> tuple[int, int]:
        applied = 0
        diverted_slow = 0
        horizon = self.config.slow_loop.horizon_days

        for raw in ops:
            if not isinstance(raw, dict):
                continue
            op = str(raw.get("op") or "").lower().strip()
            entry_id = str(raw.get("entry_id") or "").strip()

            if op == "delete" and entry_id:
                if await self.storage.delete_timetable(entry_id):
                    applied += 1
            elif op == "update" and entry_id:
                existing = await self.storage.get_timetable_entry(entry_id)
                if existing is not None:
                    if raw.get("title"):
                        existing.title = str(raw["title"]).strip()
                    if raw.get("start_time"):
                        existing.start_time = str(raw["start_time"]).strip()
                    if raw.get("end_time") is not None:
                        existing.end_time = str(raw["end_time"]).strip() or None
                    if raw.get("location") is not None:
                        existing.location = str(raw["location"]).strip()
                    if raw.get("category"):
                        existing.category = str(raw["category"]).strip()
                    if raw.get("status"):
                        try:
                            existing.status = TimetableStatus(str(raw["status"]).strip())
                        except ValueError:
                            pass
                    if raw.get("notes") is not None:
                        existing.notes = str(raw["notes"]).strip()
                    existing.source = "rtir"
                    await self.storage.upsert_timetable(existing)
                    applied += 1
            elif op == "add":
                title = str(raw.get("title") or "").strip()
                start_time = str(raw.get("start_time") or "").strip()
                if not title or not start_time:
                    continue
                if (
                    self.slow_store is not None
                    and self.config.slow_loop.enabled
                    and is_beyond_horizon(start_time, now=now, horizon_days=horizon)
                ):
                    slow_item = SlowMemoryItem(
                        kind=SlowMemoryKind.FUTURE_EVENT,
                        title=title,
                        summary=f"{start_time} {str(raw.get('location') or '').strip()} {str(raw.get('notes') or '').strip()}".strip(),
                        details=str(raw.get("notes") or "").strip(),
                        tags=["远期日程", str(raw.get("category") or "other")],
                        planned_start=start_time,
                        planned_end=str(raw["end_time"]).strip() if raw.get("end_time") else None,
                        location=str(raw.get("location") or "").strip(),
                        category=str(raw.get("category") or "other").strip(),
                        is_mandatory=True,
                        source="rtir:conversation",
                        status=SlowMemoryStatus.ACTIVE,
                    )
                    await self.slow_store.add_or_update(slow_item)
                    diverted_slow += 1
                    continue

                status_val = TimetableStatus.SCHEDULED
                if raw.get("status"):
                    try:
                        status_val = TimetableStatus(str(raw["status"]).strip())
                    except ValueError:
                        pass
                entry = TimetableEntry(
                    title=title,
                    start_time=start_time,
                    end_time=str(raw["end_time"]).strip() if raw.get("end_time") else None,
                    location=str(raw.get("location") or "").strip(),
                    category=str(raw.get("category") or "other").strip(),
                    status=status_val,
                    notes=str(raw.get("notes") or "").strip(),
                    source="rtir",
                )
                await self.storage.upsert_timetable(entry)
                applied += 1
        return applied, diverted_slow

    async def _apply_urti_items(self, items: list[Any], *, now: datetime) -> int:
        """Write active URTI items with non-null next_review_at so they are NEVER dead-lettered!"""
        added = 0
        for raw in items:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            summary = str(raw.get("summary") or "").strip()
            if not title and not summary:
                continue
            try:
                weight = float(raw.get("weight", 5.5))
            except (TypeError, ValueError):
                weight = 5.5
            clamped = max(0.5, min(self.config.gateway.max_weight, weight))
            tags = raw.get("tags") if isinstance(raw.get("tags"), list) else []
            item = BufferItem(
                zone=BufferZone.URTI,
                source="rtir:conversation",
                title=title or summary[:24],
                summary=summary or title,
                details=str(raw.get("details") or "").strip(),
                tags=[str(t).strip() for t in tags if str(t).strip()],
                weight=clamped,
                intrinsic_value=clamped,
                urgency=min(clamped, 7.0),
                trust_tier=TrustTier.HIGH,
                status=BufferStatus.ACTIVE,
                # Eliminate dead-letter bug: URTI becomes immediately eligible for next sweep/window!
                next_review_at=now + timedelta(minutes=5),
            )
            await self.storage.add_buffer_item(item)
            added += 1
        return added

    async def _apply_slow_memory_items(self, items: list[Any]) -> int:
        if self.slow_store is None or not self.config.slow_loop.enabled:
            return 0
        added = 0
        for raw in items:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            summary = str(raw.get("summary") or title).strip()
            if not title and not summary:
                continue
            planned_start = str(raw["planned_start"]).strip() if raw.get("planned_start") else None
            default_kind = "future_event" if planned_start else "idea"
            try:
                kind = SlowMemoryKind(str(raw.get("kind") or default_kind).lower().strip())
            except ValueError:
                kind = SlowMemoryKind.IDEA
            tags = raw.get("tags") if isinstance(raw.get("tags"), list) else []
            est = raw.get("estimated_minutes")
            item = SlowMemoryItem(
                kind=kind,
                title=title or summary[:24],
                summary=summary,
                details=str(raw.get("details") or "").strip(),
                tags=[str(t).strip() for t in tags if str(t).strip()],
                planned_start=planned_start,
                planned_end=str(raw["planned_end"]).strip() if raw.get("planned_end") else None,
                location=str(raw.get("location") or "").strip(),
                category=str(raw.get("category") or ("idea" if kind == SlowMemoryKind.IDEA else "other")).strip(),
                estimated_minutes=int(est) if est is not None else None,
                energy_level=str(raw.get("energy_level") or "").strip(),
                is_mandatory=kind in (SlowMemoryKind.FUTURE_EVENT, SlowMemoryKind.FUTURE_TASK),
                source="rtir:conversation",
                status=SlowMemoryStatus.ACTIVE,
            )
            await self.slow_store.add_or_update(item)
            added += 1
        return added

    async def _apply_tiered_scene_updates(
        self,
        *,
        tiered: list[Any],
        legacy: Any,
        now: datetime,
    ) -> int:
        count = 0
        for raw in tiered:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("key") or "").strip()
            if not key:
                continue
            if raw.get("clear") is True or raw.get("value") is None:
                await self.scene_manager.clear_state(key)
                count += 1
                continue
            tier_str = str(raw.get("tier") or "t1_daily").lower().strip()
            try:
                tier = StateTier(tier_str)
            except ValueError:
                tier = StateTier.T1_DAILY
            if tier == StateTier.T3_PERSISTENT:
                # RTIR does not silently overwrite T3 persistent profile
                tier = StateTier.T2_MULTIDAY
            await self.scene_manager.set_state(
                key=key,
                value=raw["value"],
                tier=tier,
                source="rtir:conversation",
                ttl_hours=float(raw["ttl_hours"]) if raw.get("ttl_hours") is not None else None,
                ttl_days=float(raw["ttl_days"]) if raw.get("ttl_days") is not None else None,
                now=now,
            )
            count += 1

        if isinstance(legacy, dict):
            for k, v in legacy.items():
                if v is None:
                    await self.scene_manager.clear_state(k)
                else:
                    # Smart default tier for legacy dict keys (health/injury -> T2, daily -> T1)
                    low_k = f"{k} {v}".lower()
                    inferred_tier = (
                        StateTier.T2_MULTIDAY
                        if any(w in low_k for w in ("伤", "痛", "感冒", "发烧", "生病", "考试周", "knee", "sick"))
                        else StateTier.T1_DAILY
                    )
                    await self.scene_manager.set_state(
                        key=k,
                        value=v,
                        tier=inferred_tier,
                        source="rtir:legacy_scene",
                        now=now,
                    )
                count += 1
        return count

    async def _apply_followups(
        self,
        *,
        new_intents: list[Any],
        close_topics: list[Any],
        now: datetime,
    ) -> int:
        added = 0
        if close_topics:
            open_fus = await self.storage.list_followups(status=FollowUpStatus.OPEN.value, limit=50)
            for phrase in close_topics:
                p_low = str(phrase).strip().lower()
                if not p_low:
                    continue
                for fu in open_fus:
                    if p_low in fu.topic.lower() or fu.topic.lower() in p_low:
                        fu.status = FollowUpStatus.DONE
                        await self.storage.upsert_followup(fu)

        for raw in new_intents:
            if not isinstance(raw, dict):
                continue
            topic = str(raw.get("topic") or "").strip()
            if not topic:
                continue
            fu = FollowUpIntent(
                topic=topic,
                context_hint=str(raw.get("context_hint") or "").strip(),
                intent_type=str(raw.get("intent_type") or "deadline_backtrack").strip(),
                deadline_at=str(raw["deadline_at"]).strip() if raw.get("deadline_at") else None,
                anchor_event_id=str(raw["anchor_event_id"]).strip() if raw.get("anchor_event_id") else None,
                safety_margin_minutes=int(raw.get("safety_margin_minutes") or 60),
                priority=float(raw.get("priority") or 8.0),
                source="rtir:conversation",
            )
            await self.followup_calc.register_followup(fu, now=now)
            added += 1
        return added

    async def _apply_feedback_and_overrides(
        self,
        *,
        feedback_entries: list[Any],
        temp_overrides: list[Any],
        now: datetime,
    ) -> int:
        if self.policy is None:
            return 0
        count = 0
        for raw in feedback_entries:
            if not isinstance(raw, dict):
                continue
            raw_text = str(raw.get("raw_text") or "").strip()
            if not raw_text and not raw.get("source_key"):
                continue
            tags = raw.get("tags") if isinstance(raw.get("tags"), list) else []
            await self.policy.append_feedback(
                event_type=str(raw.get("event_type") or "explicit_preference").strip(),
                source_key=str(raw.get("source_key") or "").strip(),
                tags=[str(t).strip() for t in tags if str(t).strip()],
                category=str(raw.get("category") or "").strip(),
                sentiment=str(raw.get("sentiment") or "neutral").strip(),
                signal_value=float(raw.get("signal_value") or 0.0),
                raw_text=raw_text,
                is_explicit=bool(raw.get("is_explicit", True)),
                now=now,
            )
            count += 1

        for raw in temp_overrides:
            if not isinstance(raw, dict):
                continue
            exp = parse_iso_datetime(str(raw.get("expires_at") or ""), now)
            instruction = str(raw.get("instruction") or "").strip()
            if exp is not None and exp > now and instruction:
                await self.policy.set_temporary_override(
                    scope_key=str(raw.get("scope_key") or "override:dnd").strip(),
                    instruction=instruction,
                    expires_at=exp,
                    effect=str(raw.get("effect") or "suppress").strip(),
                )
        return count
