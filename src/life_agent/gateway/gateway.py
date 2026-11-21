"""Message Gateway (Box 2, v2 Two-Phase Architecture).

Strict 3-block Gateway prompt:
  1) `system`     (permanently fixed, NEVER mutated by Backward Pass)
  2) `grade`      (frozen read-only published snapshot `grade_vN`, evolved ONLY nightly at 23:30)
  3) `time table` (global single source of truth)

Features:
  - Layer A deterministic hard guardrails (urgent_only intercepts non-urgent items at zero LLM cost).
  - Dual-axis Buffer scoring (intrinsic_value + urgency + capped starvation_boost) preventing spam avalanches.
  - Break / Free-gap coalesced digest pushing (`dispatch_window_digest`) with cooldown budget.
  - 100% unified arbitration for dynamic Follow-Up intents (`evaluate_followup_request`).
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from ..config import AppConfig
from ..llm import LLMClient
from ..models import (
    BufferItem,
    BufferStatus,
    FollowUpStatus,
    GatewayAction,
    GatewayDecision,
    Interruptibility,
    PassiveTriggerPayload,
    SlowMemoryItem,
    SlowMemoryKind,
    SlowMemoryStatus,
    StateTier,
    utc_now,
)
from ..scene.manager import SceneStateManager
from ..slow_loop.bm25_store import SlowLoopBM25Store
from ..storage import Storage
from .policy import GatingPolicyManager

if TYPE_CHECKING:
    from ..triggers.followup import FollowUpTimingCalculator

logger = logging.getLogger(__name__)

# Supports both 2-arg (instruction, trigger_source) and 3-arg (instruction, trigger_source, meta)
ActivePushCallback = Callable[..., Awaitable[None]]


class MessageGateway:
    """Single proactive exit gate arbitrating Buffer items, Passive Triggers, and Follow-ups."""

    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        llm: LLMClient,
        on_active_push: ActivePushCallback | None = None,
        policy: GatingPolicyManager | None = None,
        scene_manager: SceneStateManager | None = None,
        slow_store: SlowLoopBM25Store | None = None,
        followup_calc: FollowUpTimingCalculator | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.llm = llm
        self.on_active_push = on_active_push
        self.policy = policy or GatingPolicyManager(config, storage, llm)
        self.scene_manager = scene_manager or SceneStateManager(config, storage)
        self.slow_store = slow_store
        self.followup_calc = followup_calc
        self._last_non_urgent_push_at: datetime | None = None

    async def _emit_active_push(
        self,
        instruction: str,
        trigger_source: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if self.on_active_push is None:
            return
        sig = inspect.signature(self.on_active_push)
        if len(sig.parameters) >= 3:
            await self.on_active_push(instruction, trigger_source, meta or {})
        else:
            await self.on_active_push(instruction, trigger_source)

    async def _build_system_prompt(self) -> str:
        """Strictly 2 fixed+frozen blocks in system message: [fixed system] + [frozen grade_vN]."""
        snap = await self.policy.get_frozen_grade()
        return (
            f"{self.config.gateway.system_prompt}\n\n"
            f"【当前冻结版分级器 grade_v{snap.version}（日间只读）】\n"
            f"{snap.grade_text}\n\n"
            "必须严格返回如下 JSON 格式：\n"
            "{\n"
            '  "action": "ACTIVE_PUSH" | "SCENE_STATE" | "DEFER_TO_WINDOW" | "ARCHIVE_OR_SLOW",\n'
            '  "reason": "简短决策理由",\n'
            '  "push_instruction": "当 action=ACTIVE_PUSH 时，给工作区对话模型的明确主动发言指导",\n'
            '  "scene_updates": {"键": "值"},\n'
            '  "scene_tier": "t0_ephemeral" | "t1_daily" | "t2_multiday",\n'
            '  "archive_to_slow": false\n'
            "}"
        )

    async def _build_dynamic_context(self, *, now: datetime | None = None) -> str:
        """Third block: [global single-source-of-truth time table] + tiered scene view."""
        ref = now or utc_now()
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        today = ref.strftime("%Y-%m-%d")
        tomorrow = (ref + timedelta(days=1)).strftime("%Y-%m-%d")

        timetable = await self.storage.list_timetable(
            date_from=today,
            date_to=tomorrow,
            status="scheduled",
            limit=40,
        )
        scene_view = await self.scene_manager.get_structured_view(now=ref)
        top_buffer = await self.storage.list_buffer_items(status="active", limit=8)

        tt_lines = [
            f"- [{e.id}] {e.start_time}"
            + (f" ~ {e.end_time}" if e.end_time else "")
            + f" | {e.title}"
            + (f" @ {e.location}" if e.location else "")
            + (f" ({e.notes})" if e.notes else "")
            for e in timetable
        ]
        buf_lines = [
            f"- [{b.id}] ({b.zone.value}, val={b.intrinsic_value:.1f}, urg={b.urgency:.1f}, w={b.weight:.1f}, ret={b.return_count}) {b.title}: {b.summary}"
            for b in top_buffer
        ]

        return (
            f"【当前精确时间】{ref.isoformat()} ({weekdays[ref.weekday()]})\n"
            f"【全局唯一动态时间表 (time table: 今日与明日)】\n"
            + ("\n".join(tt_lines) if tt_lines else "(暂无安排)")
            + "\n\n【分级场景状态 (T0–T3 Scene state)】\n"
            + json.dumps(scene_view, ensure_ascii=False, indent=2)
            + "\n\n【当前缓冲区高权重参考条目】\n"
            + ("\n".join(buf_lines) if buf_lines else "(空)")
        )

    async def evaluate_buffer_item(
        self,
        item: BufferItem,
        *,
        now: datetime | None = None,
    ) -> GatewayDecision:
        """Evaluate a structured BufferItem through Layer-A guardrails -> Frozen Grade LLM."""
        ref = now or utc_now()
        activity = await self.scene_manager.compute_activity_snapshot(now=ref)
        scores = await self.policy.compute_effective_scores(item, now=ref)

        # Layer A Hard Guardrail 1: Explicit frozen suppress rule -> silent archive/defer
        if scores["force_suppress"] and not item.needs_confirmation and item.urgency < 8.5:
            decision = GatewayDecision(
                action=GatewayAction.ARCHIVE_OR_SLOW,
                reason=f"命中当前冻结版 grade_v{scores['grade_version']} 的屏蔽规则: {scores['matched_rules']}",
                archive_to_slow=False,
            )
            await self._apply_item_decision(item, decision, now=ref)
            return decision

        # Layer A Hard Guardrail 2: During urgent_only (in_class/sleeping/DND), non-urgent items
        # are deferred cleanly to the next open window at ZERO LLM token cost!
        pierce = self.config.gating_evolution.urgent_pierce_threshold
        if (
            activity.interruptibility == Interruptibility.URGENT_ONLY
            and item.urgency < pierce
            and not scores["force_always_push"]
            and not item.needs_confirmation
        ):
            decision = GatewayDecision(
                action=GatewayAction.DEFER_TO_WINDOW,
                reason=(
                    f"硬护栏拦截：当前处于 {activity.mode.value} (urgent_only)，"
                    f"条目紧迫度 {item.urgency:.1f} < {pierce:.1f}，暂存等待下次空闲跃迁窗。"
                ),
            )
            await self._apply_item_decision(item, decision, now=ref)
            return decision

        # Layer A Hard Guardrail 3: Non-urgent push cooldown check -> defer for bundling!
        if (
            item.urgency < pierce
            and not scores["force_always_push"]
            and not item.needs_confirmation
            and self._is_in_non_urgent_cooldown(ref)
        ):
            decision = GatewayDecision(
                action=GatewayAction.DEFER_TO_WINDOW,
                reason="处于非紧急推送冷却期内，暂存等待下一窗口合并打包推送。",
            )
            await self._apply_item_decision(item, decision, now=ref)
            return decision

        dyn_ctx = await self._build_dynamic_context(now=ref)
        confirm_note = (
            "【注意：来自中等可信群聊的疑似日程变动，尚未写入时间表，需向主人确认！】\n"
            if item.needs_confirmation
            else ""
        )
        user_prompt = (
            f"{dyn_ctx}\n\n"
            f"{confirm_note}"
            "【待评估的结构化消息】\n"
            f"- ID: {item.id}\n"
            f"- 缓冲区分区: {item.zone.value} | 信任级别: {item.trust_tier.value}\n"
            f"- 来源: {item.source}\n"
            f"- 标题: {item.title}\n"
            f"- 摘要: {item.summary}\n"
            f"- 细节: {item.details}\n"
            f"- 标签: {', '.join(item.tags)}\n"
            f"- 双轴评分: intrinsic_value={item.intrinsic_value:.1f}, urgency={item.urgency:.1f}, "
            f"starvation_boost=+{scores['starvation_boost']:.1f}, affinity={scores['affinity_bonus']:+.2f}, "
            f"rule_delta={scores['rule_delta']:+.1f} => effective={scores['effective_value']:.1f} / "
            f"weight={item.weight:.1f} (已退回次数: {item.return_count})\n"
            f"- 命中规则: {', '.join(scores['matched_rules']) or '无'}\n"
            f"- 有明确时间: {item.has_explicit_time} | 必须执行任务: {item.is_mandatory_task}\n"
            "请判断应执行 ACTIVE_PUSH、SCENE_STATE、DEFER_TO_WINDOW 还是 ARCHIVE_OR_SLOW。"
        )
        decision = await self._decide(user_prompt)
        await self._apply_item_decision(item, decision, now=ref)
        return decision

    async def evaluate_passive_trigger(
        self,
        payload: PassiveTriggerPayload,
        *,
        now: datetime | None = None,
    ) -> GatewayDecision:
        """Evaluate a fixed/dynamic trigger directive through the frozen grade Gateway."""
        ref = now or utc_now()
        dyn_ctx = await self._build_dynamic_context(now=ref)
        user_prompt = (
            f"{dyn_ctx}\n\n"
            "【被动触发器指令 (Passive Trigger)】\n"
            f"- 触发器名称: {payload.trigger_name}\n"
            f"- 类别: {payload.category}\n"
            f"- 优先级: {payload.priority:.1f} | 紧迫度: {payload.urgency:.1f}\n"
            f"- 指令: {payload.directive}\n"
            "请结合当前全局时间表、分级场景状态与冻结版分级器，决定是否立刻 ACTIVE_PUSH。"
        )
        decision = await self._decide(user_prompt)
        await self._apply_scene_updates(decision, source=f"trigger:{payload.trigger_name}", now=ref)

        if decision.action == GatewayAction.ACTIVE_PUSH:
            instruction = decision.push_instruction or payload.directive
            logger.info(
                "Gateway ACTIVE_PUSH from trigger [%s]: %s",
                payload.trigger_name,
                decision.reason,
            )
            if payload.urgency < self.config.gating_evolution.urgent_pierce_threshold:
                self._last_non_urgent_push_at = ref
            await self._emit_active_push(
                instruction,
                f"passive_trigger:{payload.trigger_name}",
                meta={
                    "source_key": f"trigger:{payload.category}",
                    "tags": [payload.category],
                    "category": payload.category,
                    "urgency": payload.urgency,
                },
            )
        elif decision.action == GatewayAction.SCENE_STATE:
            logger.info("Gateway SCENE_STATE from trigger [%s]: %s", payload.trigger_name, decision.reason)
        else:
            logger.info("Gateway deferred trigger [%s]: %s", payload.trigger_name, decision.reason)
        return decision

    async def evaluate_followup_request(
        self,
        payload: PassiveTriggerPayload,
        *,
        now: datetime | None = None,
    ) -> GatewayDecision:
        """100% Gateway-arbitrated Follow-up evaluation (never bypasses Gateway!)."""
        ref = now or utc_now()
        fu_id = payload.related_followup_id
        activity = await self.scene_manager.compute_activity_snapshot(now=ref)

        # Layer A Hard Guardrail: if user is currently in a busy/urgent-only slot and follow-up isn't critical
        pierce = self.config.gating_evolution.urgent_pierce_threshold
        if activity.interruptibility == Interruptibility.URGENT_ONLY and payload.urgency < pierce:
            decision = GatewayDecision(
                action=GatewayAction.DEFER_TO_WINDOW,
                reason=f"当前处于 {activity.mode.value}，跟进申请暂缓并重算下一空闲窗。",
            )
            if fu_id and self.followup_calc is not None:
                await self.followup_calc.defer_followup(fu_id, delay_minutes=25, now=ref)
            return decision

        decision = await self.evaluate_passive_trigger(payload, now=ref)
        if fu_id:
            fu = await self.storage.get_followup(fu_id)
            if fu is not None and fu.status == FollowUpStatus.OPEN:
                if decision.action == GatewayAction.ACTIVE_PUSH:
                    fu.check_count += 1
                    if fu.check_count >= fu.max_checks:
                        fu.computed_check_at = None
                    elif self.followup_calc is not None:
                        await self.storage.upsert_followup(fu)
                        await self.followup_calc.defer_followup(fu.id, delay_minutes=60, now=ref)
                        return decision
                    await self.storage.upsert_followup(fu)
                else:
                    # Deferred by Gateway (e.g. higher-priority conflict) -> recalculate next free window!
                    if self.followup_calc is not None:
                        await self.followup_calc.defer_followup(fu.id, delay_minutes=25, now=ref)
        return decision

    async def dispatch_window_digest(
        self,
        *,
        window_kind: str = "break",
        free_gap_minutes: int = 20,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Coalesce Top 1..N active buffered items (+ Slow Loop ideas on long gaps) into ONE push."""
        ref = now or utc_now()
        if self._is_in_non_urgent_cooldown(ref):
            return {"pushed": False, "reason": "cooldown_active", "bundled_count": 0}

        activity = await self.scene_manager.compute_activity_snapshot(now=ref)
        if activity.interruptibility == Interruptibility.URGENT_ONLY:
            return {"pushed": False, "reason": f"interruptibility={activity.mode.value}", "bundled_count": 0}

        active_items = await self.storage.list_buffer_items(status=BufferStatus.ACTIVE.value, limit=30)
        scored_items: list[tuple[float, BufferItem]] = []
        for item in active_items:
            sc = await self.policy.compute_effective_scores(item, now=ref)
            if sc["force_suppress"]:
                continue
            if sc["effective_value"] >= 4.5 or item.needs_confirmation:
                scored_items.append((sc["effective_value"], item))
        scored_items.sort(key=lambda pair: (-pair[0], -pair[1].urgency))

        max_n = self.config.gating_evolution.digest_max_items
        chosen = [it for _, it in scored_items[:max_n]]

        # On long free windows (>= 45 min), also fetch 1–2 duration-fitting Slow Loop BM25 ideas!
        slow_suggestions: list[dict[str, Any]] = []
        if free_gap_minutes >= 45 and self.slow_store is not None and self.config.slow_loop.enabled:
            slow_suggestions = await self.slow_store.search(
                query="空闲 活动 想法 学习",
                kind="idea",
                max_minutes=max(15, free_gap_minutes - 15),
                top_k=2,
            )

        if not chosen and not slow_suggestions:
            return {"pushed": False, "reason": "nothing_notable", "bundled_count": 0}

        # Mark bundled buffer items as PUSHED so they aren't pushed again individually
        for it in chosen:
            it.status = BufferStatus.PUSHED
            it.next_review_at = None
            await self.storage.update_buffer_item(it)

        bullets: list[str] = []
        all_tags: list[str] = [f"window:{window_kind}"]
        for idx, it in enumerate(chosen, 1):
            all_tags.extend(it.tags[:2])
            bullets.append(f"{idx}. [{it.zone.value}/{it.source}] {it.title}：{it.summary}")
        if slow_suggestions:
            for s in slow_suggestions:
                est = f"（约{s['estimated_minutes']}分钟）" if s.get("estimated_minutes") else ""
                bullets.append(f"• [慢回路空闲建议] {s['title']}{est}：{s['summary']}")

        gap_label = f"约 {free_gap_minutes} 分钟" if free_gap_minutes < 180 else "较长"
        instruction = (
            f"现在刚进入「{window_kind}」空闲窗口（距下一项安排 {gap_label}）。"
            "请把以下精选内容自然合并为**一条**简短清爽的消息告诉用户（不要拆成多条），"
            "挑最有趣的顺带问用户要不要参与讨论或安排做一件事：\n"
            + "\n".join(bullets)
        )
        self._last_non_urgent_push_at = ref
        source_key = chosen[0].source if len(chosen) == 1 else f"digest:{window_kind}"
        await self._emit_active_push(
            instruction,
            f"window_digest:{window_kind}",
            meta={
                "source_key": source_key,
                "tags": list(dict.fromkeys(all_tags))[:6],
                "category": f"digest_{window_kind}",
                "urgency": max((it.urgency for it in chosen), default=4.5),
            },
        )
        return {
            "pushed": True,
            "bundled_count": len(chosen),
            "slow_suggestions": len(slow_suggestions),
        }

    def _is_in_non_urgent_cooldown(self, now: datetime) -> bool:
        if self._last_non_urgent_push_at is None:
            return False
        cd = timedelta(minutes=self.config.gating_evolution.non_urgent_push_cooldown_minutes)
        return (now - self._last_non_urgent_push_at) < cd

    async def _decide(self, user_prompt: str) -> GatewayDecision:
        raw = await self.llm.complete_json(
            role="gateway",
            system_prompt=await self._build_system_prompt(),
            user_prompt=user_prompt,
        )
        action_raw = str(raw.get("action") or "DEFER_TO_WINDOW").upper().strip()
        try:
            action = GatewayAction(action_raw)
        except ValueError:
            action = GatewayAction.DEFER_TO_WINDOW

        scene_updates = raw.get("scene_updates")
        if not isinstance(scene_updates, dict):
            scene_updates = {}

        tier_raw = str(raw.get("scene_tier") or "t0_ephemeral").lower().strip()
        try:
            scene_tier = StateTier(tier_raw)
        except ValueError:
            scene_tier = StateTier.T0_EPHEMERAL

        return GatewayDecision(
            action=action,
            reason=str(raw.get("reason") or "").strip(),
            push_instruction=str(raw.get("push_instruction") or "").strip(),
            scene_updates=scene_updates,
            scene_tier=scene_tier,
            archive_to_slow=bool(raw.get("archive_to_slow", False)),
        )

    async def _apply_scene_updates(
        self,
        decision: GatewayDecision,
        *,
        source: str,
        now: datetime,
    ) -> None:
        if not decision.scene_updates:
            return
        for k, v in decision.scene_updates.items():
            if v is None:
                await self.scene_manager.clear_state(k)
            else:
                await self.scene_manager.set_state(
                    key=k,
                    value=v,
                    tier=decision.scene_tier,
                    source=source,
                    now=now,
                )

    async def _apply_item_decision(
        self,
        item: BufferItem,
        decision: GatewayDecision,
        *,
        now: datetime,
    ) -> None:
        if decision.action == GatewayAction.ACTIVE_PUSH:
            item.status = BufferStatus.PUSHED
            item.next_review_at = None
            await self.storage.update_buffer_item(item)
            await self._apply_scene_updates(decision, source=f"gateway:{item.id}", now=now)

            instruction = decision.push_instruction or (
                f"请主动向用户同步这条信息（来源：{item.source}）：\n"
                f"标题：{item.title}\n摘要：{item.summary}\n细节：{item.details}"
            )
            if item.urgency < self.config.gating_evolution.urgent_pierce_threshold:
                self._last_non_urgent_push_at = now
            logger.info(
                "Gateway -> ACTIVE_PUSH item [%s] (%s, val=%.1f, urg=%.1f): %s",
                item.id,
                item.title,
                item.intrinsic_value,
                item.urgency,
                decision.reason,
            )
            await self._emit_active_push(
                instruction,
                f"buffer:{item.id}",
                meta={
                    "source_key": item.source,
                    "tags": item.tags,
                    "category": item.zone.value,
                    "urgency": item.urgency,
                },
            )

        elif decision.action == GatewayAction.SCENE_STATE:
            item.status = BufferStatus.MERGED_SCENE
            item.next_review_at = None
            await self.storage.update_buffer_item(item)
            updates = dict(decision.scene_updates)
            if not updates:
                updates[f"note_{item.id}"] = f"{item.title}: {item.summary}"
            decision.scene_updates = updates
            await self._apply_scene_updates(decision, source=f"gateway:{item.id}", now=now)
            logger.info("Gateway -> SCENE_STATE item [%s] (%s): %s", item.id, item.title, decision.reason)

        elif decision.action == GatewayAction.ARCHIVE_OR_SLOW:
            item.status = BufferStatus.DISMISSED
            item.next_review_at = None
            await self.storage.update_buffer_item(item)
            if decision.archive_to_slow and self.slow_store is not None and self.config.slow_loop.enabled:
                await self.slow_store.add_or_update(
                    SlowMemoryItem(
                        kind=SlowMemoryKind.NOTE,
                        title=item.title,
                        summary=item.summary,
                        details=item.details,
                        tags=item.tags,
                        category=item.zone.value,
                        source=f"gateway_archived:{item.source}",
                        status=SlowMemoryStatus.ACTIVE,
                    )
                )
                logger.info("Gateway -> ARCHIVE_OR_SLOW moved [%s] (%s) to Slow Loop BM25.", item.id, item.title)
            else:
                logger.info("Gateway -> ARCHIVE_OR_SLOW dismissed [%s] (%s).", item.id, item.title)

        else:
            # DEFER_TO_WINDOW (or legacy RETURN_TO_BUFFER): capped starvation boost!
            item.return_count += 1
            delta = (
                decision.weight_delta
                if decision.weight_delta is not None
                else self.config.gateway.weight_boost_on_return
            )
            max_cap = self.config.gating_evolution.max_starvation_boost
            item.starvation_boost = min(max_cap, round(item.starvation_boost + max(0.0, delta), 2))
            item.weight = min(
                self.config.gateway.max_weight,
                round(item.intrinsic_value + item.starvation_boost, 2),
            )
            if item.return_count >= self.config.gateway.max_returns:
                item.status = BufferStatus.DISMISSED
                item.next_review_at = None
                logger.info("Gateway -> Buffer item [%s] reached max_returns (%d), archiving.", item.id, item.return_count)
            else:
                item.status = BufferStatus.ACTIVE
                item.next_review_at = now + timedelta(minutes=self.config.gateway.return_retry_minutes)
                logger.info(
                    "Gateway -> DEFER_TO_WINDOW item [%s] (%s): weight=%.1f (starv=+%.1f/%.1f, ret=%d)",
                    item.id,
                    item.title,
                    item.weight,
                    item.starvation_boost,
                    max_cap,
                    item.return_count,
                )
            await self.storage.update_buffer_item(item)

    async def sweep_due_buffer_items(self, *, now: datetime | None = None) -> int:
        """Re-evaluate due deferred buffer items (respects Layer-A guardrails)."""
        ref = now or utc_now()
        await self.storage.expire_stale_buffer_items(self.config.gateway.buffer_ttl_hours)
        due = await self.storage.list_due_buffer_items(
            now=ref,
            limit=self.config.gateway.sweep_batch_size,
        )
        for item in due:
            try:
                await self.evaluate_buffer_item(item, now=ref)
            except Exception:
                logger.exception("Error re-evaluating buffered item %s", item.id)
        return len(due)
