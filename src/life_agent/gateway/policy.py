"""Two-Phase Dynamic Gating Policy Manager & Nightly Grade Reflector (v2).

Strict Forward/Backward Separation:
  - Forward Pass (Daytime):
      * Gateway ONLY reads the frozen active `grade_vN` snapshot (NEVER mutates `grade` live!).
      * Replies, ignores, rejections, and explicit preference statements are ONLY appended
        to `feedback_log` (append-only ledger).
      * Only time-bounded `temporary_override` hard switches (e.g. "今晚 23:00 前勿扰") take
        immediate Layer-A effect and auto-expire.
  - Backward Pass (Nightly 23:30 `NightlyGradeReflector`):
      * Processes today's `feedback_log` in batch.
      * Updates smoothed Bayesian EMA `source_affinity` (bounded [-3.0, +3.0]).
      * Auto-merges EXPLICIT user preferences into active `gating_rules` (capped at 25).
      * Inserts purely system-inferred behavioral rules as `status=candidate` (Option A) so
        the user can confirm/reject them without silent drift!
      * Atomically publishes `grade_v(N+1)` snapshot for next day — `system` prompt is NEVER modified!
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from ..config import AppConfig
from ..llm import LLMClient
from ..models import (
    BufferItem,
    FeedbackLogEntry,
    GatingRule,
    GatingRuleStatus,
    GradeSnapshot,
    SourceAffinity,
    utc_now,
)
from ..storage import Storage

logger = logging.getLogger(__name__)


REFLECTOR_SYSTEM_PROMPT = """你是消息网关的夜间分级器复盘专家（Nightly Grade Reflector，上行长节拍批处理）。
你只在每晚 23:30 运行一次，任务是根据【当日原始反馈账本 feedback_log】、【当前已生效规则】与【统计亲和度 source_affinity】，
演化下一版的动态分级器规则（`grade`）。注意：绝不触碰固定系统协议 `system`。

严格遵守以下原则：
1. 对于 `is_explicit=true` 的用户明示偏好（例如用户亲口说“以后别推 CV 论文”“高数课可以摸鱼看新闻”“班长消息直接推”）：
   - 放入 `explicit_rule_updates`，会自动以 `active` 状态合入新版 `grade`。
2. 对于你从用户回复/忽略行为规律中推测出的新规则（用户未明说，例如“连续多天上午课间忽略某源，下午爱看”）：
   - 放入 `inferred_candidate_rules`，会以 `candidate`（待确认）入库，等待用户同意后再启用，防止系统擅自跑偏！
3. 如果某条旧规则与用户新明示偏好冲突、已过时或可合并精简，把它的 ID 放入 `disable_rule_ids`（总活跃规则上限 25 条）。
4. 必须严格输出如下 JSON 格式：
{
  "summary_note": "一句话概括本次夜间复盘变化",
  "explicit_rule_updates": [
    {
      "scope_type": "source" | "topic" | "course_or_time" | "routine",
      "scope_key": "简短作用域键（如 topic:CV视觉论文 / class:高等数学 / source:计科通知群）",
      "effect": "always_push" | "boost" | "suppress" | "scene_only" | "timing_policy",
      "instruction": "一句话可执行分级规则（<=60字）",
      "score_delta": -2.5
    }
  ],
  "inferred_candidate_rules": [
    {
      "scope_type": "source" | "topic" | "course_or_time" | "routine",
      "scope_key": "...",
      "effect": "boost" | "suppress" | "timing_policy",
      "instruction": "建议规则（<=60字）",
      "score_delta": 1.0,
      "confidence": 0.65
    }
  ],
  "disable_rule_ids": []
}
"""


class GatingPolicyManager:
    """Manages frozen grade_vN reads, daytime append-only feedback ledger, and nightly grade evolution."""

    def __init__(self, config: AppConfig, storage: Storage, llm: LLMClient) -> None:
        self.config = config
        self.storage = storage
        self.llm = llm

    async def ensure_initialized(self) -> GradeSnapshot:
        """Ensure grade_v1 snapshot exists from seed grading_criteria."""
        active = await self.storage.get_active_grade_snapshot()
        if active is not None:
            return active
        return await self.storage.publish_grade_snapshot(
            grade_text=self.config.gateway.grading_criteria,
            rule_ids=[],
            summary_note="Initial seed grade_v1",
        )

    async def get_frozen_grade(self) -> GradeSnapshot:
        """Forward Pass read-only accessor for the currently frozen published grade_vN."""
        return await self.ensure_initialized()

    async def append_feedback(
        self,
        *,
        event_type: str,
        source_key: str = "",
        tags: list[str] | None = None,
        category: str = "",
        activity_mode: str = "",
        sentiment: str = "neutral",
        signal_value: float = 0.0,
        raw_text: str = "",
        is_explicit: bool = False,
        now: datetime | None = None,
    ) -> FeedbackLogEntry:
        """Forward Pass append-only write to feedback_log (NEVER mutates grade_vN mid-day!)."""
        ref = now or utc_now()
        entry = FeedbackLogEntry(
            event_type=event_type,
            source_key=source_key,
            tags=tags or [],
            category=category,
            activity_mode=activity_mode,
            hour_bucket=ref.hour,
            sentiment=sentiment,
            signal_value=signal_value,
            raw_text=raw_text,
            is_explicit=is_explicit,
            created_at=ref,
        )
        return await self.storage.append_feedback_log(entry)

    async def set_temporary_override(
        self,
        *,
        scope_key: str,
        instruction: str,
        expires_at: datetime,
        effect: str = "suppress",
    ) -> GatingRule:
        """Daytime hard switch (strictly requires expires_at; never touches published grade_vN)."""
        rule = GatingRule(
            scope_type="temporary_override",
            scope_key=scope_key,
            effect=effect,
            instruction=instruction,
            score_delta=-3.0 if effect == "suppress" else 0.0,
            confidence=1.0,
            origin="explicit_user",
            expires_at=expires_at,
            status=GatingRuleStatus.ACTIVE,
        )
        return await self.storage.upsert_gating_rule(rule)

    async def compute_effective_scores(
        self,
        item: BufferItem,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Deterministically compute effective_score using frozen grade rules + persisted source_affinity."""
        ref = now or utc_now()
        snap = await self.ensure_initialized()

        # Only read rules belonging to the currently published frozen grade snapshot!
        all_active = await self.storage.list_gating_rules(status=GatingRuleStatus.ACTIVE.value)
        published_id_set = set(snap.rule_ids)
        applicable_rules = [
            r
            for r in all_active
            if (r.id in published_id_set or r.scope_type == "temporary_override")
            and (r.expires_at is None or ref < r.expires_at)
        ]

        # Match rules against item's source, tags, title, summary
        haystack = f"{item.source} {item.title} {item.summary} {' '.join(item.tags)}".lower()
        rule_delta = 0.0
        matched_rules: list[str] = []
        force_always_push = False
        force_suppress = False

        for r in applicable_rules:
            if r.scope_type in ("source", "topic"):
                needle = r.scope_key.lower()
                for prefix in ("source:", "group:", "topic:", "tag:", "sender:"):
                    if needle.startswith(prefix):
                        needle = needle[len(prefix) :].strip()
                if needle and needle in haystack:
                    rule_delta += r.score_delta
                    matched_rules.append(f"{r.scope_key}({r.effect})")
                    if r.effect == "always_push":
                        force_always_push = True
                    elif r.effect == "suppress":
                        force_suppress = True

        # Source & tag affinity bonus from persisted EMA table
        affinity_keys = [f"source:{item.source}"] + [f"tag:{t}" for t in item.tags[:4]]
        aff_values: list[float] = []
        for k in affinity_keys:
            aff = await self.storage.get_source_affinity(k)
            if aff is not None and aff.sample_count > 0:
                aff_values.append(aff.affinity)
        affinity_bonus = round(sum(aff_values) / len(aff_values), 2) if aff_values else 0.0

        capped_starvation = min(
            self.config.gating_evolution.max_starvation_boost,
            max(0.0, item.starvation_boost),
        )
        effective_value = max(
            0.5,
            min(
                self.config.gateway.max_weight,
                round(item.intrinsic_value + capped_starvation + rule_delta + affinity_bonus, 2),
            ),
        )
        return {
            "effective_value": effective_value,
            "intrinsic_value": item.intrinsic_value,
            "urgency": item.urgency,
            "starvation_boost": capped_starvation,
            "affinity_bonus": affinity_bonus,
            "rule_delta": round(rule_delta, 2),
            "matched_rules": matched_rules,
            "force_always_push": force_always_push,
            "force_suppress": force_suppress,
            "grade_version": snap.version,
        }

    def _compile_grade_text(
        self,
        active_rules: list[GatingRule],
        affinities: list[SourceAffinity],
    ) -> str:
        """Render the versioned `grade` block from seed criteria + active atomic rules + top affinities."""
        parts = [self.config.gateway.grading_criteria.strip()]
        durable_rules = [r for r in active_rules if r.scope_type != "temporary_override"]
        if durable_rules:
            lines = ["\n【个性化动态门控规则（演化生效中）】"]
            for r in durable_rules[: self.config.gating_evolution.max_active_rules]:
                lines.append(
                    f"- [{r.scope_type}:{r.scope_key}] ({r.effect}, delta={r.score_delta:+.1f}) {r.instruction}"
                )
            parts.append("\n".join(lines))

        notable_aff = [a for a in affinities if a.sample_count >= 2 and abs(a.affinity) >= 0.4][:10]
        if notable_aff:
            lines = ["\n【历史互动亲和度参考（统计自动校准）】"]
            for a in notable_aff:
                trend = "用户偏好" if a.affinity > 0 else "用户常忽略"
                lines.append(f"- {a.key}: {a.affinity:+.2f} ({trend}, n={a.sample_count})")
            parts.append("\n".join(lines))

        return "\n".join(parts)

    async def _update_affinities_from_feedback(self, entries: list[FeedbackLogEntry]) -> int:
        """Update bounded Bayesian-smoothed EMA source_affinity from unprocessed feedback."""
        updated_keys: set[str] = set()
        alpha = 0.30
        for fb in entries:
            if fb.signal_value == 0.0:
                continue
            keys: list[str] = []
            if fb.source_key:
                keys.append(f"source:{fb.source_key}")
            for tag in fb.tags[:4]:
                if tag:
                    keys.append(f"tag:{tag}")

            for k in keys:
                cur = await self.storage.get_source_affinity(k)
                if cur is None:
                    # Bayesian shrinkage: single sample starts gently scaled toward 0
                    new_aff = round(max(-3.0, min(3.0, fb.signal_value * 0.45)), 3)
                    cur = SourceAffinity(key=k, affinity=new_aff, sample_count=1)
                else:
                    blended = (1.0 - alpha) * cur.affinity + alpha * (fb.signal_value * 1.5)
                    cur.affinity = round(max(-3.0, min(3.0, blended)), 3)
                    cur.sample_count += 1
                await self.storage.upsert_source_affinity(cur)
                updated_keys.add(k)
        return len(updated_keys)

    async def run_nightly_reflection(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Long-beat Backward Pass (23:30): consume feedback_log and atomically publish grade_v(N+1).

        Guarantees:
          - `config.gateway.system_prompt` is NEVER mutated.
          - Explicit user preferences -> auto-merged into active rules.
          - Purely system-inferred rules -> inserted as `candidate` (Option A) unless config overrides.
          - Total active rules strictly capped at `max_active_rules` (25).
        """
        ref = now or utc_now()
        current_snap = await self.ensure_initialized()

        # 1. Clean expired temporary overrides first
        all_rules = await self.storage.list_gating_rules(status="all")
        expired_removed = 0
        for r in all_rules:
            if r.expires_at is not None and ref >= r.expires_at:
                await self.storage.delete_gating_rule(r.id)
                expired_removed += 1

        unprocessed = await self.storage.list_unprocessed_feedback(limit=500)
        if not unprocessed and expired_removed == 0:
            return {
                "published": False,
                "version": current_snap.version,
                "reason": "no_new_feedback",
                "explicit_added": 0,
                "candidates_added": 0,
            }

        # 2. Update statistical source_affinity (pure math)
        aff_updated = await self._update_affinities_from_feedback(unprocessed)
        affinities = await self.storage.list_source_affinities(limit=50)
        active_rules = await self.storage.list_gating_rules(status=GatingRuleStatus.ACTIVE.value)
        durable_active = [r for r in active_rules if r.scope_type != "temporary_override"]

        # 3. Ask Reflector LLM to synthesize explicit rules & candidate behavioral rules
        fb_payload = [
            {
                "event_type": f.event_type,
                "source_key": f.source_key,
                "tags": f.tags,
                "category": f.category,
                "activity_mode": f.activity_mode,
                "hour_bucket": f.hour_bucket,
                "sentiment": f.sentiment,
                "signal_value": f.signal_value,
                "raw_text": f.raw_text,
                "is_explicit": f.is_explicit,
            }
            for f in unprocessed
        ]
        rules_payload = [
            {
                "id": r.id,
                "scope_type": r.scope_type,
                "scope_key": r.scope_key,
                "effect": r.effect,
                "instruction": r.instruction,
                "score_delta": r.score_delta,
                "origin": r.origin,
            }
            for r in durable_active
        ]
        aff_payload = [
            {"key": a.key, "affinity": a.affinity, "samples": a.sample_count}
            for a in affinities[:20]
        ]

        user_prompt = (
            f"【复盘时间】{ref.isoformat()}\n"
            f"【当前分级器版本】grade_v{current_snap.version} (活跃规则上限 {self.config.gating_evolution.max_active_rules} 条)\n\n"
            f"【当前生效规则】\n{json.dumps(rules_payload, ensure_ascii=False, indent=2)}\n\n"
            f"【最新来源/标签统计亲和度】\n{json.dumps(aff_payload, ensure_ascii=False, indent=2)}\n\n"
            f"【今日原始反馈账本 (feedback_log)】\n{json.dumps(fb_payload, ensure_ascii=False, indent=2)}"
        )
        llm_out = await self.llm.complete_json(
            role="reflector",
            system_prompt=REFLECTOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # 4. Disable obsolete/conflicting rules
        disabled_count = 0
        for rid in llm_out.get("disable_rule_ids") or []:
            r = await self.storage.get_gating_rule(str(rid))
            if r is not None:
                r.status = GatingRuleStatus.DISABLED
                await self.storage.upsert_gating_rule(r)
                disabled_count += 1

        # 5. Apply explicit user rules (auto-active!)
        explicit_added = 0
        for raw in llm_out.get("explicit_rule_updates") or []:
            if not isinstance(raw, dict) or not raw.get("instruction"):
                continue
            delta = float(raw.get("score_delta", 0.0))
            rule = GatingRule(
                scope_type=str(raw.get("scope_type") or "topic").strip(),
                scope_key=str(raw.get("scope_key") or "").strip(),
                effect=str(raw.get("effect") or "boost").strip(),
                instruction=str(raw["instruction"]).strip()[:120],
                score_delta=max(-3.0, min(3.0, delta)),
                confidence=1.0,
                origin="explicit_user",
                status=GatingRuleStatus.ACTIVE,
            )
            await self.storage.upsert_gating_rule(rule)
            explicit_added += 1

        # 6. Apply system-inferred rules (Option A: `candidate` awaiting user confirmation!)
        candidates_added = 0
        inferred_status = (
            GatingRuleStatus.ACTIVE
            if self.config.gating_evolution.auto_publish_inferred_rules
            else GatingRuleStatus.CANDIDATE
        )
        for raw in llm_out.get("inferred_candidate_rules") or []:
            if not isinstance(raw, dict) or not raw.get("instruction"):
                continue
            delta = float(raw.get("score_delta", 0.0))
            conf = float(raw.get("confidence", 0.65))
            rule = GatingRule(
                scope_type=str(raw.get("scope_type") or "topic").strip(),
                scope_key=str(raw.get("scope_key") or "").strip(),
                effect=str(raw.get("effect") or "boost").strip(),
                instruction=str(raw["instruction"]).strip()[:120],
                score_delta=max(-3.0, min(3.0, delta)),
                confidence=max(0.1, min(0.95, conf)),
                origin="nightly_reflection",
                status=inferred_status,
            )
            await self.storage.upsert_gating_rule(rule)
            candidates_added += 1

        # 7. Enforce active rule cap (<= 25) and atomically publish grade_v(N+1)
        refreshed_active = [
            r
            for r in await self.storage.list_gating_rules(status=GatingRuleStatus.ACTIVE.value)
            if r.scope_type != "temporary_override"
        ]
        max_cap = self.config.gating_evolution.max_active_rules
        if len(refreshed_active) > max_cap:
            # Keep highest-confidence & most recently updated rules
            refreshed_active.sort(key=lambda r: (r.confidence, r.updated_at.timestamp()), reverse=True)
            for overflow in refreshed_active[max_cap:]:
                overflow.status = GatingRuleStatus.DISABLED
                await self.storage.upsert_gating_rule(overflow)
            refreshed_active = refreshed_active[:max_cap]

        new_grade_text = self._compile_grade_text(refreshed_active, affinities)
        summary_note = str(llm_out.get("summary_note") or f"Nightly reflection ({len(unprocessed)} feedback entries)")
        new_snap = await self.storage.publish_grade_snapshot(
            grade_text=new_grade_text,
            rule_ids=[r.id for r in refreshed_active],
            summary_note=summary_note,
        )

        # 8. Mark today's feedback_log entries processed
        await self.storage.mark_feedback_processed([f.id for f in unprocessed], processed_at=ref)
        logger.info(
            "NightlyGradeReflector published grade_v%d (explicit=%d, candidates=%d, disabled=%d, aff_updated=%d).",
            new_snap.version,
            explicit_added,
            candidates_added,
            disabled_count,
            aff_updated,
        )
        return {
            "published": True,
            "version": new_snap.version,
            "summary_note": summary_note,
            "explicit_added": explicit_added,
            "candidates_added": candidates_added,
            "disabled_count": disabled_count,
            "affinities_updated": aff_updated,
            "processed_feedback": len(unprocessed),
        }

    async def approve_candidate_rule(self, rule_id: str) -> GradeSnapshot | None:
        """User explicitly approves a candidate rule -> activates it & publishes updated grade snapshot."""
        rule = await self.storage.get_gating_rule(rule_id)
        if rule is None:
            return None
        rule.status = GatingRuleStatus.ACTIVE
        rule.confidence = 1.0
        await self.storage.upsert_gating_rule(rule)

        active_rules = [
            r
            for r in await self.storage.list_gating_rules(status=GatingRuleStatus.ACTIVE.value)
            if r.scope_type != "temporary_override"
        ]
        affinities = await self.storage.list_source_affinities(limit=50)
        new_text = self._compile_grade_text(active_rules, affinities)
        return await self.storage.publish_grade_snapshot(
            grade_text=new_text,
            rule_ids=[r.id for r in active_rules],
            summary_note=f"Approved candidate rule {rule.id} ({rule.scope_key})",
        )

    async def reject_candidate_rule(self, rule_id: str) -> bool:
        rule = await self.storage.get_gating_rule(rule_id)
        if rule is None:
            return False
        rule.status = GatingRuleStatus.DISABLED
        await self.storage.upsert_gating_rule(rule)
        return True
