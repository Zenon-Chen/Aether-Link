"""End-to-end tests covering every single box and arrow of the Life Agent architecture."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from life_agent.config import AppConfig, PassiveTriggerSlot
from life_agent.llm import LLMClient, LLMResponse, ToolCallRequest
from life_agent.models import (
    BufferStatus,
    BufferZone,
    ChannelType,
    PushStatus,
    RawIncomingEvent,
    TimetableEntry,
    utc_now,
)
from life_agent.orchestrator import LifeAgentApp
from life_agent.web import create_web_app


class ScriptedLLM(LLMClient):
    """Deterministic programmable LLM double for verifying every agent & tool-call loop."""

    def __init__(self) -> None:
        self.json_handlers: dict[str, Any] = {}
        self.chat_handler: Any = None
        self.calls: list[dict[str, Any]] = []

    async def complete_json(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        self.calls.append({"type": "json", "role": role, "user_prompt": user_prompt})
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
        self.calls.append({"type": "chat", "role": role, "messages": messages, "tools": tools})
        if callable(self.chat_handler):
            return self.chat_handler(messages, tools)
        return LLMResponse(content="好的，明白！")

    async def close(self) -> None:
        return None


def _make_test_config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.db_path = str(tmp_path / "test_life.db")
    cfg.plugins_dir = str(tmp_path / "plugins")
    cfg.astrbot.enabled = False
    cfg.rss.enabled = False
    cfg.arxiv.enabled = False
    cfg.passive_triggers.enabled = False
    cfg.owner.qq_id = "99887766"
    cfg.owner.weixin_user_id = "wx_owner@im.wechat"
    cfg.group_filter.silence_seconds = 0.1
    cfg.delivery.failover_timeout_seconds = 0.2
    cfg.working_state.rtir_silence_seconds = 0.15
    cfg.gateway.weight_boost_on_return = 2.0
    cfg.gateway.return_retry_minutes = 0.0  # Eligible immediately on sweep
    # Keep legacy pipeline E2E tests wall-clock independent even if executed after midnight
    cfg.owner.living_profile.sleep_start_hour = 24
    cfg.owner.living_profile.sleep_end_hour = 0
    return cfg


async def test_full_pipeline_group_structurer_timetable_gateway_qq_failover_wechat(tmp_path: Path) -> None:
    """Verify: Group -> 5m silence -> Structurer -> Step 1.1 Timetable -> Gateway ACTIVE_PUSH -> QQ -> 5m timeout -> WeChat."""
    sent_log: list[tuple[str, str, str]] = []

    async def mock_sender(channel: str, target: str, text: str) -> bool:
        sent_log.append((channel, target, text))
        return True

    llm = ScriptedLLM()
    llm.json_handlers["structurer"] = lambda _sys, prompt: {
        "items": [
            {
                "zone": "message",
                "title": "高数课调至三教201",
                "summary": "班长通知明天上午高数课调到三教201",
                "details": "需要携带作业本",
                "tags": ["调课", "必修"],
                "initial_weight": 9.0,
                "has_explicit_time": True,
                "is_mandatory_task": True,
                "timetable_entry": {
                    "title": "高等数学",
                    "start_time": "2026-11-19T10:00:00+08:00",
                    "end_time": "2026-11-19T11:40:00+08:00",
                    "location": "三教201",
                    "category": "class",
                    "notes": "带作业本",
                },
            }
        ]
    }
    llm.json_handlers["gateway"] = lambda _sys, _prompt: {
        "action": "ACTIVE_PUSH",
        "reason": "紧急调课通知",
        "push_instruction": "立刻通知用户明天高数课改到三教201，提醒带作业本",
        "scene_updates": {"latest_notice": "明天高数调至三教201"},
    }
    llm.chat_handler = lambda _msgs, _tools: LLMResponse(
        content="提醒你一下！刚看到班长通知，明天10点的高数课换到三教201了，记得带作业本~"
    )

    app = LifeAgentApp(_make_test_config(tmp_path), base_dir=tmp_path, llm_client=llm, custom_sender=mock_sender)
    await app.start()
    try:
        # Ingest group messages (1 meaningful + 1 echo noise)
        await app.router.handle_event(
            RawIncomingEvent(
                source_plugin="astrbot",
                platform_type="aiocqhttp",
                platform_id="qq_onebot",
                channel_type=ChannelType.GROUP,
                channel_id="group_cs",
                channel_name="计科通知群",
                sender_id="1001",
                sender_name="班长",
                text="通知：明天上午10点高数课改到三教201，带作业本",
            )
        )
        await app.router.handle_event(
            RawIncomingEvent(
                source_plugin="astrbot",
                platform_type="aiocqhttp",
                platform_id="qq_onebot",
                channel_type=ChannelType.GROUP,
                channel_id="group_cs",
                channel_name="计科通知群",
                sender_id="1002",
                sender_name="路人",
                text="收到",
            )
        )

        # Wait for group silence flush (0.1s) -> Structurer -> Timetable -> Gateway -> QQ push
        await asyncio.sleep(0.18)

        # 1.1: Timetable auto-written
        tt = await app.storage.list_timetable()
        assert len(tt) == 1
        assert tt[0].title == "高等数学"
        assert tt[0].location == "三教201"

        # Scene state updated
        scene = await app.storage.get_scene_state()
        assert scene.get("latest_notice") == "明天高数调至三教201"

        # Step 4: Sent to QQ first!
        assert len(sent_log) == 1
        assert sent_log[0][0] == "qq"
        assert sent_log[0][1] == "99887766"
        assert "三教201" in sent_log[0][2]

        # Step 4.1: User does NOT reply within failover timeout (0.2s) -> auto-resend via WeChat!
        await asyncio.sleep(0.25)
        assert len(sent_log) == 2
        assert sent_log[1][0] == "weixin_oc"
        assert sent_log[1][1] == "wx_owner@im.wechat"
        assert "三教201" in sent_log[1][2]

        pushes = await app.storage.list_pending_pushes()
        assert pushes[0].status == PushStatus.ESCALATED_WECHAT
    finally:
        await app.stop()


async def test_qq_reply_cancels_wechat_failover_and_triggers_rtir_after_silence(tmp_path: Path) -> None:
    """Verify: QQ reply cancels 5m WeChat failover AND 3m silence runs RTIR back to Timetable & URTI."""
    sent_log: list[tuple[str, str, str]] = []

    async def mock_sender(channel: str, target: str, text: str) -> bool:
        sent_log.append((channel, target, text))
        return True

    llm = ScriptedLLM()
    llm.chat_handler = lambda _msgs, _tools: LLMResponse(content="好嘞，记下来了，今晚好好休息别硬跑！")
    llm.json_handlers["rtir"] = lambda _sys, prompt: {
        "timetable_operations": [
            {
                "op": "add",
                "title": "和导师开组会",
                "start_time": "2026-11-19T19:00:00+08:00",
                "location": "科研楼405",
                "category": "meeting",
                "notes": "带周报",
            }
        ],
        "urti_items": [
            {
                "title": "膝盖酸痛不想校园跑",
                "summary": "用户反馈今天膝盖有点酸，今晚跳过校园跑",
                "details": "晚间推送时不要催促校园跑",
                "tags": ["身体状态", "校园跑"],
                "weight": 7.5,
            }
        ],
        "scene_state_updates": {"knee_status": "有点酸，今晚不跑"},
    }

    app = LifeAgentApp(_make_test_config(tmp_path), base_dir=tmp_path, llm_client=llm, custom_sender=mock_sender)
    await app.start()
    try:
        # Send proactive QQ message (arms 0.2s WeChat failover)
        await app.delivery.send_proactive("今晚要不要去校园跑？", trigger_source="evening_campus_run")
        assert len(sent_log) == 1
        assert sent_log[0][0] == "qq"

        # User replies on QQ within 0.05s (< 0.2s failover timeout)
        await asyncio.sleep(0.05)
        await app.router.handle_event(
            RawIncomingEvent(
                source_plugin="astrbot",
                platform_type="aiocqhttp",
                platform_id="qq_onebot",
                channel_type=ChannelType.PRIVATE,
                channel_id="99887766",
                sender_id="99887766",
                sender_name="主人",
                text="我今天膝盖有点酸就不跑了，对了明天晚上7点科研楼405要和导师开组会",
                is_from_owner=True,
            )
        )

        # Wait past BOTH failover timeout (0.2s) and RTIR silence window (0.15s)
        await asyncio.sleep(0.3)

        # Verify WeChat failover was NEVER triggered!
        channels = [c for c, _, _ in sent_log]
        assert "weixin_oc" not in channels
        pushes = await app.storage.list_pending_pushes()
        assert pushes[0].status == PushStatus.ACKNOWLEDGED

        # Step 5 & 5.1: RTIR ran automatically after 3-min (0.15s) silence -> wrote to Timetable + URTI!
        tt = await app.storage.list_timetable()
        assert any(e.title == "和导师开组会" and e.location == "科研楼405" for e in tt)

        urti = await app.storage.list_buffer_items(zone="urti")
        assert len(urti) == 1
        assert "膝盖" in urti[0].title

        # Verify chat turns marked parsed so RTIR won't duplicate on a second pass
        res2 = await app.rtir.parse_unparsed_turns()
        assert res2["parsed_turns"] == 0
    finally:
        await app.stop()


async def test_gateway_return_to_buffer_weighting_and_escalation(tmp_path: Path) -> None:
    """Verify Step 2.1.1: Returned buffer items receive weight boost and escalate on re-review."""
    llm = ScriptedLLM()
    llm.json_handlers["structurer"] = lambda _s, _p: {
        "items": [
            {
                "zone": "news",
                "title": "DeepSeek 新架构论文",
                "summary": "一篇关于长上下文高效注意力的新 arXiv 论文",
                "details": "https://arxiv.org/abs/2611.00001",
                "tags": ["arXiv", "LLM"],
                "initial_weight": 4.0,
                "has_explicit_time": False,
                "is_mandatory_task": False,
            }
        ]
    }

    gateway_calls = 0

    def gateway_logic(_s: str, prompt: str) -> dict[str, Any]:
        nonlocal gateway_calls
        gateway_calls += 1
        if gateway_calls == 1:
            return {"action": "RETURN_TO_BUFFER", "reason": "用户正在上课，先退回缓冲加权"}
        # Second sweep sees boosted weight 6.0 -> moves to SCENE_STATE!
        assert "6.0" in prompt
        return {
            "action": "SCENE_STATE",
            "reason": "加权后达到状态更新阈值，写入课间推荐列表",
            "scene_updates": {"recommended_paper": "DeepSeek 新架构论文"},
        }

    llm.json_handlers["gateway"] = gateway_logic

    app = LifeAgentApp(_make_test_config(tmp_path), base_dir=tmp_path, llm_client=llm)
    await app.start()
    try:
        items = await app.structurer.process_event(
            RawIncomingEvent(
                source_plugin="arxiv_papers",
                platform_type="arxiv",
                platform_id="arxiv",
                channel_type=ChannelType.PAPER,
                channel_id="2611.00001",
                text="New paper on efficient attention",
            )
        )
        assert len(items) == 1
        item = await app.storage.get_buffer_item(items[0].id)
        assert item is not None
        assert item.status == BufferStatus.ACTIVE
        assert item.return_count == 1
        assert item.weight == 6.0  # 4.0 + 2.0 weight boost!

        # Trigger buffer re-evaluation sweep
        swept = await app.gateway.sweep_due_buffer_items()
        assert swept == 1
        item_after = await app.storage.get_buffer_item(items[0].id)
        assert item_after is not None
        assert item_after.status == BufferStatus.MERGED_SCENE
        scene = await app.storage.get_scene_state()
        assert scene["recommended_paper"] == "DeepSeek 新架构论文"
    finally:
        await app.stop()


async def test_four_direct_tools_and_ambiguous_time_clarification(tmp_path: Path) -> None:
    """Verify the 4 direct LLM Function Call tools including follow-up question for vague time."""
    llm = ScriptedLLM()

    # Seed timetable & buffer
    app = LifeAgentApp(_make_test_config(tmp_path), base_dir=tmp_path, llm_client=llm)
    await app.start()
    try:
        chinese_cls = await app.storage.upsert_timetable(
            TimetableEntry(
                title="语文课",
                start_time="2026-11-18T14:00:00+08:00",
                end_time="2026-11-18T14:45:00+08:00",
                location="一教102",
                category="class",
            )
        )
        english_cls = await app.storage.upsert_timetable(
            TimetableEntry(
                title="英语课",
                start_time="2026-11-18T16:00:00+08:00",
                end_time="2026-11-18T16:45:00+08:00",
                location="二教305",
                category="class",
            )
        )

        # Scenario 1: Direct natural language reschedule with exact period ("改成第四节")
        def chat_script(messages: list[dict[str, Any]], tools: Any) -> LLMResponse:
            last = messages[-1]
            if last["role"] == "user" and "语文课时间改成第四节" in last["content"]:
                return LLMResponse(
                    tool_calls=[ToolCallRequest(id="tc1", name="view_timetable", arguments={"keyword": "语文"})]
                )
            if last["role"] == "tool" and last["name"] == "view_timetable" and "语文课" in last["content"]:
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="tc2",
                            name="modify_timetable",
                            arguments={
                                "action": "update",
                                "entry_id": chinese_cls.id,
                                "start_time": "2026-11-18T10:55:00+08:00",
                                "end_time": "2026-11-18T11:40:00+08:00",
                            },
                        )
                    ]
                )
            if last["role"] == "tool" and last["name"] == "modify_timetable":
                return LLMResponse(content="已经帮你把时间表修改好啦！")

            # Scenario 2: Ambiguous time ("把今天下午英语课改到晚上") -> asks clarification first!
            if last["role"] == "user" and "英语课的时间改到晚上" in last["content"]:
                return LLMResponse(content="好的，请问你想把今天下午的英语课改到晚上几点钟呢？")

            # User supplies exact evening hour -> calls modify_timetable!
            if last["role"] == "user" and "晚上7点半" in last["content"]:
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="tc3",
                            name="modify_timetable",
                            arguments={
                                "action": "update",
                                "entry_id": english_cls.id,
                                "start_time": "2026-11-18T19:30:00+08:00",
                                "end_time": "2026-11-18T20:15:00+08:00",
                            },
                        )
                    ]
                )

            # Scenario 3: Buffer add + view direct tools
            if last["role"] == "user" and "帮我记到缓冲区" in last["content"]:
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="tc4",
                            name="modify_buffer",
                            arguments={
                                "action": "add",
                                "zone": "urti",
                                "title": "周末买雨伞",
                                "summary": "用户提到周末记得买一把折叠伞",
                                "weight": 6.0,
                            },
                        )
                    ]
                )
            if last["role"] == "tool" and last["name"] == "modify_buffer":
                return LLMResponse(
                    tool_calls=[ToolCallRequest(id="tc5", name="view_buffer", arguments={"zone": "urti"})]
                )
            if last["role"] == "tool" and last["name"] == "view_buffer":
                return LLMResponse(content="已写入缓冲区并确认存在！")
            return LLMResponse(content="OK")

        llm.chat_handler = chat_script

        # Execute Scenario 1
        r1 = await app.chat_local("把今天下午的语文课时间改成第四节")
        assert [t["name"] for t in r1["tools_used"]] == ["view_timetable", "modify_timetable"]
        updated_cn = await app.storage.get_timetable_entry(chinese_cls.id)
        assert updated_cn is not None and updated_cn.start_time == "2026-11-18T10:55:00+08:00"

        # Execute Scenario 2 (Clarification -> Confirmation -> Modify)
        r2a = await app.chat_local("把今天下午英语课的时间改到晚上")
        assert r2a["tools_used"] == []
        assert "几点" in r2a["reply"]
        still_old = await app.storage.get_timetable_entry(english_cls.id)
        assert still_old is not None and still_old.start_time == "2026-11-18T16:00:00+08:00"

        r2b = await app.chat_local("改到晚上7点半吧")
        assert [t["name"] for t in r2b["tools_used"]] == ["modify_timetable"]
        updated_en = await app.storage.get_timetable_entry(english_cls.id)
        assert updated_en is not None and updated_en.start_time == "2026-11-18T19:30:00+08:00"

        # Execute Scenario 3 (Buffer direct tools)
        r3 = await app.chat_local("帮我记到缓冲区：周末买雨伞")
        assert [t["name"] for t in r3["tools_used"]] == ["modify_buffer", "view_buffer"]
        buf = await app.storage.list_buffer_items(zone="urti")
        assert any(b.title == "周末买雨伞" for b in buf)
    finally:
        await app.stop()


async def test_passive_triggers_custom_plugin_and_web_api(tmp_path: Path) -> None:
    """Verify Passive Trigger daily slot + timetable lead reminder + dynamic custom plugin + FastAPI."""
    pushed: list[str] = []

    async def mock_sender(channel: str, target: str, text: str) -> bool:
        pushed.append(f"{channel}:{text}")
        return True

    # Create a custom external plugin outside AstrBot in tmp_path/plugins/
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    (plugins_dir / "my_custom_feed.py").write_text(
        """
from life_agent.router.base import SourcePlugin

class MyCustomFeed(SourcePlugin):
    name = "my_custom_feed"
    description = "Custom test plugin outside AstrBot"
    async def start(self, emit):
        self.started = True
    async def stop(self):
        self.started = False

def create_plugin(config, storage):
    return MyCustomFeed()
""",
        encoding="utf-8",
    )

    llm = ScriptedLLM()
    llm.json_handlers["gateway"] = lambda _s, p: {
        "action": "ACTIVE_PUSH",
        "reason": "触发推送",
        "push_instruction": "发送提醒",
    }
    llm.chat_handler = lambda _m, _t: LLMResponse(content="早上好！该起床吃早饭去一教上课啦！")

    cfg = _make_test_config(tmp_path)
    now = utc_now()
    cfg.passive_triggers.slots = [
        PassiveTriggerSlot(
            name="test_wakeup",
            time=now.strftime("%H:%M"),
            directive="早上叫起床",
            priority=9.5,
        )
    ]

    app = LifeAgentApp(cfg, base_dir=tmp_path, llm_client=llm, custom_sender=mock_sender)
    await app.start()
    try:
        # Custom non-AstrBot plugin was dynamically loaded!
        assert "my_custom_feed" in app.router.plugins

        # Add an upcoming timetable entry 10 minutes from now (< 15 min lead)
        soon = (now + timedelta(minutes=10)).isoformat()
        await app.storage.upsert_timetable(
            TimetableEntry(title="操作系统课", start_time=soon, location="四教101")
        )

        # Tick passive trigger engine once -> fires daily slot AND timetable reminder
        res1 = await app.passive_triggers.tick(now=now)
        assert res1["slots_fired"] == 1
        assert res1["reminders_fired"] == 1
        assert len(pushed) == 2

        # Second tick same minute -> must NOT double-fire!
        res2 = await app.passive_triggers.tick(now=now)
        assert res2["slots_fired"] == 0
        assert res2["reminders_fired"] == 0

        # Check FastAPI control panel endpoints
        web_app = create_web_app(app)
        client = TestClient(web_app)
        resp = client.get("/api/state")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["timetable"]) == 1
        assert any(p["name"] == "my_custom_feed" for p in body["router"]["plugins"])
    finally:
        await app.stop()
