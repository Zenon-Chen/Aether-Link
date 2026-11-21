"""Comprehensive v2 tests: Two-Phase Forward/Backward Gating, T0–T3 TTL, Dynamic FollowUps, Trust Firewall, Bundled Transitions & Live Planning."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from life_agent.config import AppConfig
from life_agent.llm import LLMClient, LLMResponse
from life_agent.models import (
    BufferItem,
    BufferStatus,
    BufferZone,
    ChannelType,
    FollowUpIntent,
    GatingRuleStatus,
    GatewayAction,
    RawIncomingEvent,
    SlowMemoryItem,
    SlowMemoryKind,
    StateTier,
    TimetableEntry,
    TrustTier,
    utc_now,
)
from life_agent.orchestrator import LifeAgentApp
from life_agent.web import create_web_app


class ScriptedV2LLM(LLMClient):
    def __init__(self) -> None:
        self.json_handlers: dict[str, Any] = {}
        self.chat_handler: Any = None
        self.role_calls: dict[str, int] = {}

    async def complete_json(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        self.role_calls[role] = self.role_calls.get(role, 0) + 1
        handler = self.json_handlers.get(role)
        if callable(handler):
            return handler(system_prompt, user_prompt)
        return {}

    async def chat(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.role_calls[role] = self.role_calls.get(role, 0) + 1
        if callable(self.chat_handler):
            return self.chat_handler(messages, tools)
        return LLMResponse(content="收到，已整理妥当！")

    async def close(self) -> None:
        return None


def _make_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.db_path = str(tmp_path / "v2_test.db")
    cfg.plugins_dir = str(tmp_path / "plugins")
    cfg.astrbot.enabled = False
    cfg.rss.enabled = False
    cfg.arxiv.enabled = False
    cfg.passive_triggers.enabled = False
    cfg.owner.qq_id = "88886666"
    cfg.owner.weixin_user_id = "wx_v2@im.wechat"
    cfg.delivery.failover_min_urgency = 0.0  # Option B: full failover on bundled active pushes
    cfg.gating_evolution.auto_publish_inferred_rules = False  # Option A: inferred -> candidate
    return cfg


async def test_forward_backward_isolation_and_nightly_grade_evolution(tmp_path: Path) -> None:
    """Iron Laws 1 & 2: Daytime Forward Pass NEVER mutates grade_v1; 23:30 Nightly Reflector evolves ONLY grade -> grade_v2."""
    llm = ScriptedV2LLM()
    llm.json_handlers["reflector"] = lambda _sys, prompt: {
        "summary_note": "合入用户明示的 CV 论文屏蔽规则，并提出一条下午看论文的候选建议",
        "explicit_rule_updates": [
            {
                "scope_type": "topic",
                "scope_key": "topic:CV视觉论文",
                "effect": "suppress",
                "instruction": "用户明确表示不再推送 CV 视觉方向论文",
                "score_delta": -3.0,
            }
        ],
        "inferred_candidate_rules": [
            {
                "scope_type": "routine",
                "scope_key": "routine:afternoon_papers",
                "effect": "boost",
                "instruction": "检测到下午空闲时用户更常回复论文推送，建议优先下午推送",
                "score_delta": 1.0,
                "confidence": 0.7,
            }
        ],
        "disable_rule_ids": [],
    }

    cfg = _make_cfg(tmp_path)
    original_system_prompt = cfg.gateway.system_prompt
    app = LifeAgentApp(cfg, base_dir=tmp_path, llm_client=llm)
    await app.start()
    try:
        v1 = await app.policy.get_frozen_grade()
        assert v1.version == 1
        v1_text = v1.grade_text

        # Forward Pass daytime: user records explicit preference + push ignored/engaged telemetry
        res_tool = await app.working_state.tool_executor.execute(
            "view_and_manage_grade",
            {
                "action": "record_preference_feedback",
                "preference_text": "以后别推 CV 视觉论文了",
                "source_key": "topic:CV视觉论文",
                "sentiment": "negative",
            },
        )
        assert "feedback_log" in res_tool or "反馈账本" in res_tool
        await app.policy.append_feedback(
            event_type="push_ignored",
            source_key="rss/Hacker News",
            tags=["CV"],
            signal_value=-0.6,
            is_explicit=False,
        )

        # STRICT CHECK: During daytime Forward Pass, active grade_v1 MUST remain completely frozen!
        still_v1 = await app.policy.get_frozen_grade()
        assert still_v1.version == 1
        assert still_v1.grade_text == v1_text

        # Now run Long-Beat Nightly Backward Pass (23:30)
        night_now = datetime.fromisoformat("2026-11-21T23:31:00+08:00")
        refl = await app.policy.run_nightly_reflection(now=night_now)
        assert refl["published"] is True
        assert refl["version"] == 2
        assert refl["explicit_added"] == 1
        assert refl["candidates_added"] == 1

        # Verify immutable `system` prompt was NEVER touched!
        assert app.config.gateway.system_prompt == original_system_prompt

        # Verify active grade_v2 includes EXPLICIT rule, while inferred rule is still `candidate` (Option A)!
        v2 = await app.policy.get_frozen_grade()
        assert v2.version == 2
        assert "CV视觉论文" in v2.grade_text
        assert "下午推送" not in v2.grade_text

        candidates = await app.storage.list_gating_rules(status=GatingRuleStatus.CANDIDATE.value)
        assert len(candidates) == 1
        assert candidates[0].scope_key == "routine:afternoon_papers"

        # Approve candidate rule -> publishes grade_v3, then verify rollback to v1 works!
        v3 = await app.policy.approve_candidate_rule(candidates[0].id)
        assert v3 is not None and v3.version == 3
        assert "下午推送" in v3.grade_text

        restored = await app.storage.activate_grade_version(1)
        assert restored is not None and restored.version == 1
        assert (await app.policy.get_frozen_grade()).version == 1
    finally:
        await app.stop()


async def test_tiered_scene_state_ttl_and_t2_gentle_renewal(tmp_path: Path) -> None:
    """Iron Law 3: T0/T1 expire cleanly at TTL / 04:00; T2 survives multi-day and prompts renewal before expiry."""
    pushed: list[str] = []

    async def mock_sender(channel: str, target: str, text: str) -> bool:
        pushed.append(f"{channel}:{text}")
        return True

    llm = ScriptedV2LLM()
    llm.json_handlers["gateway"] = lambda _s, p: {
        "action": "ACTIVE_PUSH",
        "reason": "T2 状态临期温和确认",
        "push_instruction": "顺带关心用户膝盖拉伤好点了吗",
    }
    llm.chat_handler = lambda _m, _t: LLMResponse(content="对了，前几天你说膝盖有点拉伤，现在恢复得怎么样啦？")

    app = LifeAgentApp(_make_cfg(tmp_path), base_dir=tmp_path, llm_client=llm, custom_sender=mock_sender)
    await app.start()
    try:
        day1 = datetime.fromisoformat("2026-11-21T12:30:00+08:00")
        await app.scene_manager.set_state(key="temp_note", value="在图书馆三楼", tier=StateTier.T0_EPHEMERAL, ttl_hours=2, now=day1)
        await app.scene_manager.set_state(key="lunch_today", value="二食堂麻辣香锅", tier=StateTier.T1_DAILY, now=day1)
        await app.scene_manager.set_state(key="knee_status", value="膝盖拉伤暂不宜校园跑", tier=StateTier.T2_MULTIDAY, ttl_days=5, now=day1)

        # Next morning at 04:05 -> T0 & T1 MUST be cleared; T2 & T3 MUST remain intact!
        day2_morning = datetime.fromisoformat("2026-11-22T04:05:00+08:00")
        m1 = await app.passive_triggers.run_short_beat_backward(now=day2_morning)
        assert m1["cleaned_t0"] == 1
        assert m1["cleaned_t1"] == 1
        assert m1["t2_renewals_sent"] == 0

        assert await app.storage.get_tiered_scene("temp_note") is None
        assert await app.storage.get_tiered_scene("lunch_today") is None
        t2_entry = await app.storage.get_tiered_scene("knee_status")
        assert t2_entry is not None and t2_entry.value == "膝盖拉伤暂不宜校园跑"
        assert await app.storage.get_tiered_scene("living_profile") is not None  # T3 persistent

        # Advance to Day 5.5 (within 24h renewal window of 5-day TTL: Nov 26 12:30)
        near_expiry = datetime.fromisoformat("2026-11-25T18:00:00+08:00")
        m2 = await app.passive_triggers.run_short_beat_backward(now=near_expiry)
        assert m2["t2_renewals_sent"] == 1
        assert len(pushed) == 1
        assert "膝盖" in pushed[0]
        # T2 was NOT silently deleted!
        still_t2 = await app.storage.get_tiered_scene("knee_status")
        assert still_t2 is not None and still_t2.renewal_prompted is True

        # Renew T2 if user says not yet recovered
        renewed = await app.scene_manager.renew_t2_state("knee_status", extend_days=4, now=near_expiry)
        assert renewed is not None and renewed.renewal_prompted is False
    finally:
        await app.stop()


async def test_dynamic_followup_timing_recalc_and_mandatory_gateway_arbitration(tmp_path: Path) -> None:
    """Iron Law 4: Follow-up computes free window against global Timetable, recalculates on schedule edit, and 100% goes through Gateway."""
    pushed: list[str] = []

    async def mock_sender(channel: str, target: str, text: str) -> bool:
        pushed.append(f"{channel}:{text}")
        return True

    llm = ScriptedV2LLM()
    # Gateway simulates a higher-priority conflict and defers the follow-up!
    llm.json_handlers["gateway"] = lambda _s, _p: {
        "action": "DEFER_TO_WINDOW",
        "reason": "此刻有更紧急事务，将取快递跟进暂缓至下一空档",
    }

    app = LifeAgentApp(_make_cfg(tmp_path), base_dir=tmp_path, llm_client=llm, custom_sender=mock_sender)
    await app.start()
    try:
        ref = datetime.fromisoformat("2026-11-21T15:00:00+08:00")
        cls = await app.storage.upsert_timetable(
            TimetableEntry(
                title="编译原理",
                start_time="2026-11-21T18:30:00+08:00",
                end_time="2026-11-21T19:30:00+08:00",
                location="三教105",
                category="class",
            )
        )

        # Register deadline-backtracked follow-up: deadline 20:00, margin 60m -> naive 19:00 is INSIDE class!
        fu = await app.followup_calc.register_followup(
            FollowUpIntent(
                topic="去菜鸟驿站取快递",
                context_hint="驿站20:00关门",
                intent_type="deadline_backtrack",
                deadline_at="2026-11-21T20:00:00+08:00",
                safety_margin_minutes=60,
                priority=8.0,
            ),
            now=ref,
        )
        # Must automatically shift right after class ends -> 19:35!
        assert fu.computed_check_at == "2026-11-21T19:35:00+08:00"

        # Now reschedule the class to 18:00-18:45 -> recalculation shifts check back to 19:00 (now free!)
        cls.start_time = "2026-11-21T18:00:00+08:00"
        cls.end_time = "2026-11-21T18:45:00+08:00"
        await app.storage.upsert_timetable(cls)
        recalced = await app.followup_calc.recalculate_all_open(now=ref)
        assert recalced == 1
        fu_after = await app.storage.get_followup(fu.id)
        assert fu_after is not None and fu_after.computed_check_at == "2026-11-21T19:00:00+08:00"

        # At 19:01, short-beat backward submits follow-up to Gateway -> Gateway DEFERS it!
        at_1901 = datetime.fromisoformat("2026-11-21T19:01:00+08:00")
        res = await app.passive_triggers.run_short_beat_backward(now=at_1901)
        assert res["followups_submitted"] == 1
        # MUST NOT bypass Gateway! Zero pushes sent, and follow-up was deferred to a later slot!
        assert len(pushed) == 0
        fu_deferred = await app.storage.get_followup(fu.id)
        assert fu_deferred is not None and fu_deferred.computed_check_at is not None
        assert fu_deferred.computed_check_at > "2026-11-21T19:01:00+08:00"
    finally:
        await app.stop()


async def test_trust_firewall_anti_avalanche_and_bundled_transition_digest(tmp_path: Path) -> None:
    """Verify Trust Firewall (TRUST_LOW blocked from Timetable), Layer-A zero-LLM class deferral, capped starvation, and 1-message break bundle."""
    pushed: list[str] = []

    async def mock_sender(channel: str, target: str, text: str) -> bool:
        pushed.append(f"{channel}:{text}")
        return True

    llm = ScriptedV2LLM()
    llm.json_handlers["structurer"] = lambda _s, prompt: {
        "items": [
            {
                "zone": "message",
                "title": "疑似日程事件",
                "summary": "提到了明天上午9点的安排",
                "details": "原文内容",
                "tags": ["测试"],
                "intrinsic_value": 6.0,
                "urgency": 9.0,
                "has_explicit_time": True,
                "is_mandatory_task": True,
                "timetable_entry": {
                    "title": "外部发布会或疑似调课",
                    "start_time": "2026-11-22T09:00:00+08:00",
                    "location": "未知",
                    "category": "other",
                },
            }
        ]
    }
    llm.chat_handler = lambda _m, _t: LLMResponse(content="刚下高数课，趁课间给你打包三条有趣资讯与群聊热点！")

    app = LifeAgentApp(_make_cfg(tmp_path), base_dir=tmp_path, llm_client=llm, custom_sender=mock_sender)
    await app.start()
    try:
        # 1. TRUST_LOW (RSS news) MUST NEVER write to global Timetable even if LLM outputs timetable_entry!
        rss_items = await app.structurer.process_event(
            RawIncomingEvent(
                source_plugin="rss_news",
                platform_type="rss",
                platform_id="Hacker News",
                channel_type=ChannelType.NEWS,
                channel_id="hn",
                text="OpenAI announces conference tomorrow at 9am",
            )
        )
        assert len(await app.storage.list_timetable()) == 0
        assert rss_items[0].trust_tier == TrustTier.LOW
        assert rss_items[0].urgency <= 6.0  # Low trust cannot forge emergency urgency!

        # 2. TRUST_MEDIUM (ordinary group chatter) MUST NOT silently write Timetable; flagged needs_confirmation!
        med_items = await app.structurer.process_event(
            RawIncomingEvent(
                source_plugin="astrbot",
                platform_type="aiocqhttp",
                platform_id="qq_onebot",
                channel_type=ChannelType.PRIVATE,
                channel_id="u_random",
                sender_name="路人甲",
                text="听说好像明天上午9点要调课？",
            )
        )
        assert len(await app.storage.list_timetable()) == 0
        assert med_items[0].trust_tier == TrustTier.MEDIUM
        assert med_items[0].needs_confirmation is True

        # Clean up earlier items for clean class-break bundling test
        for old in await app.storage.list_buffer_items(status="all"):
            await app.storage.delete_buffer_item(old.id)

        # 3. During active class (08:00-09:40), ingest 5 news items at 08:15 -> Layer-A zero-LLM intercept!
        await app.storage.upsert_timetable(
            TimetableEntry(
                title="高等数学",
                start_time="2026-11-21T08:00:00+08:00",
                end_time="2026-11-21T09:40:00+08:00",
                location="一教101",
                category="class",
            )
        )
        in_class_time = datetime.fromisoformat("2026-11-21T08:15:00+08:00")
        llm.role_calls["gateway"] = 0
        pushed.clear()

        for i in range(5):
            it = await app.storage.add_buffer_item(
                BufferItem(
                    zone=BufferZone.NEWS,
                    source="arxiv/cs.AI",
                    title=f"新论文 #{i+1}",
                    summary=f"有趣论文摘要 {i+1}",
                    intrinsic_value=5.0 + i * 0.2,
                    weight=5.0 + i * 0.2,
                    urgency=3.5,
                    trust_tier=TrustTier.LOW,
                )
            )
            dec = await app.gateway.evaluate_buffer_item(it, now=in_class_time)
            assert dec.action == GatewayAction.DEFER_TO_WINDOW
            # Starvation boost strictly capped (never inflates to emergency!)
            updated = await app.storage.get_buffer_item(it.id)
            assert updated is not None and updated.starvation_boost <= 2.0

        # Zero Gateway LLM calls wasted while user is in class!
        assert llm.role_calls.get("gateway", 0) == 0
        assert len(pushed) == 0

        # 4. At 09:42 (class just ended!), transition digest bundles Top 3 into ONE single message!
        after_class = datetime.fromisoformat("2026-11-21T09:42:00+08:00")
        tick_res = await app.passive_triggers.tick(now=after_class)
        assert tick_res["transition_pushed"] == 1
        assert len(pushed) == 1  # Bundled into EXACTLY 1 push instead of 5 spam pushes!
    finally:
        await app.stop()


async def test_dynamic_wakeup_live_planning_and_v2_web_api(tmp_path: Path) -> None:
    """Verify dynamic wake-up computation, plan_free_slots + Slow Loop BM25, and v2 Web API."""
    llm = ScriptedV2LLM()
    app = LifeAgentApp(_make_cfg(tmp_path), base_dir=tmp_path, llm_client=llm)
    await app.start()
    try:
        # Weekday (Monday 2026-11-23) with 08:00 class -> dynamic wakeup = 07:20 (40m lead)
        mon = datetime.fromisoformat("2026-11-23T06:00:00+08:00")
        await app.storage.upsert_timetable(
            TimetableEntry(
                title="早八高数",
                start_time="2026-11-23T08:00:00+08:00",
                end_time="2026-11-23T09:40:00+08:00",
                category="class",
            )
        )
        wake_mon = await app.passive_triggers.compute_dynamic_wakeup_time(mon)
        assert wake_mon.strftime("%H:%M") == "07:20"

        # Weekday (Tuesday 2026-11-24) with NO morning class -> weekday_latest_wakeup = 08:30
        tue = datetime.fromisoformat("2026-11-24T06:00:00+08:00")
        wake_tue = await app.passive_triggers.compute_dynamic_wakeup_time(tue)
        assert wake_tue.strftime("%H:%M") == "08:30"

        # Weekend (Saturday 2026-11-21) with NO morning class -> weekend_wakeup = 09:30
        sat = datetime.fromisoformat("2026-11-21T06:00:00+08:00")
        wake_sat = await app.passive_triggers.compute_dynamic_wakeup_time(sat)
        assert wake_sat.strftime("%H:%M") == "09:30"

        # Live Planning tool (`plan_free_slots`) computes exact gaps on Monday + pulls Slow Loop BM25 activity
        await app.slow_store.add_or_update(
            SlowMemoryItem(
                kind=SlowMemoryKind.IDEA,
                title="阅读论文《Attention Is All You Need》笔记整理",
                summary="空闲 60 分钟整理精读笔记",
                tags=["空闲", "论文", "学习"],
                estimated_minutes=60,
            )
        )
        import json

        plan_raw = await app.working_state.tool_executor.execute(
            "plan_free_slots",
            {"date": "2026-11-23", "days": 1, "min_slot_minutes": 45, "activity_query": "论文 学习 空闲"},
        )
        plan = json.loads(plan_raw)
        assert plan["ok"] is True
        assert any(slot["start"].startswith("2026-11-23T09:40") for slot in plan["free_slots"])
        assert any("Attention" in s["title"] for s in plan["slow_loop_candidates"])

        # Web API v2 endpoints check
        web_app = create_web_app(app)
        client = TestClient(web_app)
        state = client.get("/api/state").json()
        assert "grade" in state and state["grade"]["active"]["version"] == 1
        assert "follow_ups" in state
        assert "feedback_log" in state
        assert "T3_长期基线" in state["scene_state"]
    finally:
        await app.stop()
