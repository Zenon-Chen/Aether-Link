"""Comprehensive tests for the BM25 Slow Loop & Excel/Image Schedule/Shift Auto-Importer."""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import openpyxl
from fastapi.testclient import TestClient
from PIL import Image

from life_agent.config import AppConfig
from life_agent.importers.schedule_importer import extract_spreadsheet_text
from life_agent.llm import LLMClient, LLMResponse, ToolCallRequest
from life_agent.models import (
    ChannelType,
    RawIncomingEvent,
    SlowMemoryItem,
    SlowMemoryKind,
    SlowMemoryStatus,
    utc_now,
)
from life_agent.orchestrator import LifeAgentApp
from life_agent.web import create_web_app


class ScriptedVisionLLM(LLMClient):
    """Programmable LLM double supporting JSON, multimodal vision JSON, and tool calls."""

    def __init__(self) -> None:
        self.json_handlers: dict[str, Any] = {}
        self.vision_handler: Any = None
        self.chat_handler: Any = None
        self.vision_calls: list[dict[str, Any]] = []
        self.chat_calls: list[list[dict[str, Any]]] = []

    async def complete_json(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        handler = self.json_handlers.get(role)
        if callable(handler):
            return handler(system_prompt, user_prompt)
        return {}

    async def complete_vision_json(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> dict[str, Any]:
        self.vision_calls.append(
            {
                "role": role,
                "user_prompt": user_prompt,
                "size": len(image_bytes),
                "mime_type": mime_type,
            }
        )
        if callable(self.vision_handler):
            return self.vision_handler(system_prompt, user_prompt, image_bytes, mime_type)
        return {}

    async def chat(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.chat_calls.append(messages)
        if callable(self.chat_handler):
            return self.chat_handler(messages, tools)
        return LLMResponse(content="OK")

    async def close(self) -> None:
        return None


def _make_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.db_path = str(tmp_path / "slow_test.db")
    cfg.plugins_dir = str(tmp_path / "plugins")
    cfg.astrbot.enabled = False
    cfg.rss.enabled = False
    cfg.arxiv.enabled = False
    cfg.passive_triggers.enabled = False
    cfg.owner.qq_id = "11223344"
    cfg.slow_loop.horizon_days = 7
    cfg.schedule_importer.default_reminder_lead_minutes = 20
    return cfg


def _build_sample_xlsx() -> bytes:
    """Create a realistic university timetable .xlsx with merged cells."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "2026秋季课表"
    ws["A1"] = "节次"
    ws["B1"] = "周一"
    ws["C1"] = "周三"
    ws["A2"] = "第1节"
    ws["A3"] = "第2节"
    ws["B2"] = "高等数学\n三教201\n第12-14周"
    ws.merge_cells("B2:B3")  # Multi-period merged cell!
    ws["C2"] = "实验室早班值班 08:00-10:00"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_sample_png() -> bytes:
    img = Image.new("RGB", (120, 60), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_excel_merged_cell_extraction() -> None:
    xlsx_bytes = _build_sample_xlsx()
    kind, text = extract_spreadsheet_text(xlsx_bytes, "timetable.xlsx")
    assert kind == "excel"
    assert "2026秋季课表" in text
    # Merged cell B2:B3 should appear in BOTH row 2 and row 3 so the LLM knows it spans 第1节 and 第2节
    lines = [ln for ln in text.splitlines() if "高等数学" in ln]
    assert len(lines) == 2


async def test_slow_loop_bm25_retrieval_and_working_state_tools(tmp_path: Path) -> None:
    """Verify BM25 retrieval works cleanly, does NOT pollute Fast Loop context, and supports LLM tool calls + promotion."""
    llm = ScriptedVisionLLM()
    app = LifeAgentApp(_make_cfg(tmp_path), base_dir=tmp_path, llm_client=llm)
    await app.start()
    try:
        # Seed Slow Loop with undated ideas and >7-day future plans
        rust_idea = await app.slow_store.add_or_update(
            SlowMemoryItem(
                kind=SlowMemoryKind.IDEA,
                title="读《Rust 编程之道》并练手所有权",
                summary="空闲时抽 40 分钟学习 Rust 语言的所有权与生命周期",
                tags=["空闲活动", "编程", "Rust", "学习"],
                category="idea",
                estimated_minutes=40,
                energy_level="medium",
            )
        )
        await app.slow_store.add_or_update(
            SlowMemoryItem(
                kind=SlowMemoryKind.IDEA,
                title="周末去香山爬山看红叶",
                summary="天气好且有半天空闲时去户外爬山放松",
                tags=["空闲活动", "户外", "爬山"],
                category="activity",
                estimated_minutes=240,
                energy_level="high",
            )
        )
        await app.slow_store.add_or_update(
            SlowMemoryItem(
                kind=SlowMemoryKind.FUTURE_EVENT,
                title="操作系统期末考试",
                summary="下个月15号上午在二教401考操作系统",
                tags=["考试", "期末", "操作系统"],
                planned_start="2026-12-15T09:00:00+08:00",
                location="二教401",
                category="exam",
                is_mandatory=True,
            )
        )

        # Direct BM25 search checks
        hits_rust = await app.slow_store.search("Rust 所有权 学习")
        assert len(hits_rust) >= 1
        assert hits_rust[0]["id"] == rust_idea.id
        assert hits_rust[0]["bm25_score"] > 0.0

        # Filter by max_minutes=60 for a 45-min free slot -> excludes 240-min hiking!
        hits_short = await app.slow_store.search("空闲 活动", kind="idea", max_minutes=60)
        assert all(h["title"] != "周末去香山爬山看红叶" for h in hits_short)
        assert any(h["id"] == rust_idea.id for h in hits_short)

        # Simulate LLM chat retrieving from Slow Loop via search_slow_memory and promoting via modify_slow_memory
        def chat_script(messages: list[dict[str, Any]], tools: Any) -> LLMResponse:
            # Verify Fast Loop system prompt is NOT polluted with raw slow-loop content!
            sys_prompt = messages[0]["content"]
            assert "香山爬山看红叶" not in sys_prompt
            assert "操作系统期末考试" not in sys_prompt

            last = messages[-1]
            if last["role"] == "user" and "空闲 45 分钟" in last["content"]:
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="s1",
                            name="search_slow_memory",
                            arguments={"query": "空闲 学习 Rust 活动", "kind": "idea", "max_minutes": 45},
                        )
                    ]
                )
            if last["role"] == "tool" and last["name"] == "search_slow_memory":
                return LLMResponse(
                    content="帮你从慢回路里检索到了：你之前想抽 40 分钟读《Rust 编程之道》练手所有权，刚好契合这 45 分钟空闲！"
                )
            if last["role"] == "user" and "定在明天下午4点" in last["content"]:
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="s2",
                            name="modify_slow_memory",
                            arguments={
                                "action": "promote_to_timetable",
                                "item_id": rust_idea.id,
                                "planned_start": "2026-11-21T16:00:00+08:00",
                                "planned_end": "2026-11-21T16:40:00+08:00",
                            },
                        )
                    ]
                )
            if last["role"] == "tool" and last["name"] == "modify_slow_memory":
                return LLMResponse(content="已将《Rust 编程之道》升入快回路时间表并设好提前提醒！")
            return LLMResponse(content="好")

        llm.chat_handler = chat_script

        r1 = await app.chat_local("我待会空闲 45 分钟，之前有没有记过什么可做的事？")
        assert [t["name"] for t in r1["tools_used"]] == ["search_slow_memory"]
        assert "Rust" in r1["reply"]

        r2 = await app.chat_local("行，那就把它定在明天下午4点做吧")
        assert [t["name"] for t in r2["tools_used"]] == ["modify_slow_memory"]

        # Verify promoted into Fast Loop timetable + passive trigger + slow status=promoted
        tt = await app.storage.list_timetable()
        assert any("Rust" in e.title for e in tt)
        trigs = await app.storage.list_scheduled_triggers()
        assert any("Rust" in t.directive for t in trigs)
        slow_after = await app.slow_store.get(rust_idea.id)
        assert slow_after is not None and slow_after.status == SlowMemoryStatus.PROMOTED
    finally:
        await app.stop()


async def test_structurer_and_rtir_route_beyond_week_and_ideas_to_slow_loop(tmp_path: Path) -> None:
    """Verify Structurer & RTIR automatically divert >7-day items & undated ideas to Slow Loop BM25."""
    llm = ScriptedVisionLLM()
    far_date = (utc_now() + timedelta(days=15)).replace(hour=14, minute=0, second=0, microsecond=0).isoformat()

    llm.json_handlers["structurer"] = lambda _s, _p: {
        "items": [
            {
                "zone": "message",
                "title": "人工智能前沿讲座",
                "summary": "两周后在学术报告厅举办的讲座",
                "details": "需提前报名",
                "tags": ["讲座", "AI"],
                "initial_weight": 7.0,
                "has_explicit_time": True,
                "is_mandatory_task": True,
                "timetable_entry": {
                    "title": "人工智能前沿讲座",
                    "start_time": far_date,
                    "location": "学术报告厅",
                    "category": "meeting",
                    "notes": "提前报名",
                },
            }
        ]
    }
    llm.json_handlers["rtir"] = lambda _s, _p: {
        "timetable_operations": [
            {
                "op": "add",
                "title": "概率论期末考",
                "start_time": far_date,
                "location": "一教301",
                "category": "exam",
                "notes": "开卷",
            }
        ],
        "urti_items": [],
        "slow_memory_items": [
            {
                "kind": "idea",
                "title": "抽空学习视频剪辑",
                "summary": "用户随口提到以后有空想学 DaVinci Resolve 剪辑",
                "tags": ["空闲活动", "剪辑"],
                "category": "idea",
                "estimated_minutes": 60,
            }
        ],
        "scene_state_updates": {},
    }

    app = LifeAgentApp(_make_cfg(tmp_path), base_dir=tmp_path, llm_client=llm)
    await app.start()
    try:
        # Structurer receives 15-day-away event -> must go to Slow Loop, NOT Fast Loop Timetable!
        await app.structurer.process_event(
            RawIncomingEvent(
                source_plugin="webhook",
                platform_type="webhook",
                platform_id="wh",
                channel_type=ChannelType.PRIVATE,
                channel_id="notice",
                text="两周后人工智能前沿讲座",
            )
        )
        assert len(await app.storage.list_timetable()) == 0
        hits = await app.slow_store.search("人工智能 讲座")
        assert len(hits) == 1
        assert hits[0]["title"] == "人工智能前沿讲座"

        # RTIR receives user conversation -> >7d exam + undated idea BOTH go to Slow Loop!
        await app.chat_local("对了下个月考概率论，另外以后有空想学一下视频剪辑")
        res = await app.working_state.flush_rtir_now()
        assert res["timetable_ops"] == 0
        assert res["slow_added"] == 2
        assert len(await app.storage.list_timetable()) == 0

        hits_clip = await app.slow_store.search("视频 剪辑")
        assert any("剪辑" in h["title"] for h in hits_clip)
        hits_exam = await app.slow_store.search("概率论 期末考")
        assert any("概率论" in h["title"] for h in hits_exam)
    finally:
        await app.stop()


async def test_schedule_importer_excel_and_image_split_passive_triggers_and_promotion(tmp_path: Path) -> None:
    """Verify Excel & Image Schedule Importer: <=7d -> Timetable + Passive Trigger; >7d -> Slow Loop BM25 -> Auto Promotion."""
    pushed: list[str] = []

    async def mock_sender(channel: str, target: str, text: str) -> bool:
        pushed.append(f"{channel}:{text}")
        return True

    # Fixed reference date: Friday 2026-11-20 10:00+08:00 (Week 12)
    ref_now = datetime.fromisoformat("2026-11-20T10:00:00+08:00")
    within_week_shift = "2026-11-23T08:00:00+08:00"  # +3 days (Monday of Week 13) -> <= 7 days!
    beyond_week_shift = "2026-12-03T18:00:00+08:00"  # +13 days -> > 7 days!

    llm = ScriptedVisionLLM()
    llm.json_handlers["structurer"] = lambda _s, prompt: {
        "schedule_type": "mixed",
        "dated_events": [
            {
                "title": "实验室早班值班",
                "start_time": within_week_shift,
                "end_time": "2026-11-23T10:00:00+08:00",
                "location": "信息楼A302",
                "category": "shift",
                "notes": "检查服务器温度",
                "reminder_lead_minutes": 20,
            },
            {
                "title": "实验室晚班排班",
                "start_time": beyond_week_shift,
                "end_time": "2026-12-03T21:00:00+08:00",
                "location": "信息楼A302",
                "category": "shift",
                "notes": "关门前断电",
                "reminder_lead_minutes": 30,
            },
        ],
        "recurring_rules": [
            {
                "title": "高等数学",
                "weekday": 0,  # Monday
                "start_hhmm": "10:00",
                "end_hhmm": "11:40",
                "location": "三教201",
                "category": "class",
                "week_start": 13,  # Mon Nov 23 (<= 7 days!)
                "week_end": 14,    # Mon Nov 30 (> 7 days!)
                "week_parity": "all",
                "notes": "王老师",
                "reminder_lead_minutes": 15,
            }
        ],
        "warnings": [],
    }
    llm.vision_handler = lambda _s, _p, _b, _m: {
        "schedule_type": "shift",
        "dated_events": [
            {
                "title": "社团迎新值班",
                "start_time": "2026-11-24T14:00:00+08:00",  # +4 days -> Fast Loop
                "end_time": "2026-11-24T16:00:00+08:00",
                "location": "学生活动中心",
                "category": "shift",
                "notes": "带工牌",
                "reminder_lead_minutes": 20,
            },
            {
                "title": "社团期末值班",
                "start_time": "2026-12-10T14:00:00+08:00",  # +20 days -> Slow Loop
                "end_time": "2026-12-10T16:00:00+08:00",
                "location": "学生活动中心",
                "category": "shift",
                "notes": "整理器材",
                "reminder_lead_minutes": 20,
            },
        ],
        "recurring_rules": [],
        "warnings": [],
    }
    llm.json_handlers["gateway"] = lambda _s, _p: {
        "action": "ACTIVE_PUSH",
        "reason": "导入课表/排班到达提前提醒时刻",
        "push_instruction": "提醒用户按时去值班/上课",
    }
    llm.chat_handler = lambda _m, _t: LLMResponse(content="提醒你：还有20分钟要去信息楼A302值早班啦！")

    app = LifeAgentApp(_make_cfg(tmp_path), base_dir=tmp_path, llm_client=llm, custom_sender=mock_sender)
    await app.start()
    try:
        # 1. Import Excel (.xlsx) with current_week=12
        xlsx_path = tmp_path / "my_schedule.xlsx"
        xlsx_path.write_bytes(_build_sample_xlsx())
        res_xl = await app.schedule_importer.import_file(
            xlsx_path,
            current_week=12,
            now=ref_now,
        )
        assert res_xl.input_kind == "excel"
        # Within 7 days: Nov 23 08:00 实验室早班值班 + Nov 23 10:00 高等数学(第13周) = 2
        assert res_xl.fast_loop_timetable_count == 2
        assert res_xl.passive_triggers_written == 2
        # Beyond 7 days: Nov 30 10:00 高等数学(第14周) + Dec 3 18:00 实验室晚班排班 = 2
        assert res_xl.slow_loop_bm25_count == 2

        # Verify passive trigger exact timestamp (08:00 - 20min = 07:40; 10:00 - 15min = 09:45)
        triggers = await app.storage.list_scheduled_triggers()
        trig_times = {t.trigger_at for t in triggers}
        assert "2026-11-23T07:40:00+08:00" in trig_times
        assert "2026-11-23T09:45:00+08:00" in trig_times

        # Verify PassiveTriggerEngine fires the imported trigger at 07:40 (and doesn't double-fire!)
        tick_res = await app.passive_triggers.tick(
            now=datetime.fromisoformat("2026-11-23T07:41:00+08:00")
        )
        assert tick_res["scheduled_fired"] == 1
        assert tick_res["reminders_fired"] == 0
        assert len(pushed) == 1

        # Verify Slow Loop BM25 finds the >7-day shift & Week-14 class
        slow_hits = await app.slow_store.search("实验室 晚班 排班")
        assert len(slow_hits) >= 1
        assert slow_hits[0]["title"] == "实验室晚班排班"

        # 2. Import Image (.png) via multimodal vision path
        res_img = await app.schedule_importer.import_bytes(
            data=_build_sample_png(),
            filename="shift_screenshot.png",
            schedule_type="shift",
            now=ref_now,
        )
        assert res_img.input_kind == "image"
        assert len(llm.vision_calls) == 1
        assert llm.vision_calls[0]["role"] == "vision"
        assert llm.vision_calls[0]["mime_type"] == "image/png"
        assert res_img.fast_loop_timetable_count == 1
        assert res_img.passive_triggers_written == 1
        assert res_img.slow_loop_bm25_count == 1

        # 3. Automatic Rolling Horizon Promotion: advance time to 2026-11-27 (now Nov 30 & Dec 3 are <= 7 days!)
        future_now = datetime.fromisoformat("2026-11-27T09:00:00+08:00")
        promoted = await app.slow_store.promote_due_items(now=future_now, horizon_days=7)
        promoted_titles = {p.title for p in promoted}
        assert "高等数学" in promoted_titles
        assert "实验室晚班排班" in promoted_titles
        # Dec 10 is still 13 days away from Nov 27 -> remains active in Slow Loop!
        remaining_slow = await app.slow_store.list_all(status="active")
        assert [s.title for s in remaining_slow] == ["社团期末值班"]

        # 4. Verify FastAPI upload & BM25 search endpoints
        web_app = create_web_app(app)
        client = TestClient(web_app)
        search_resp = client.get("/api/slow-memory/search", params={"q": "社团 期末 值班"})
        assert search_resp.status_code == 200
        assert search_resp.json()["count"] == 1

        upload_resp = client.post(
            "/api/import/schedule",
            files={"file": ("shifts.png", _build_sample_png(), "image/png")},
            data={"schedule_type": "shift"},
        )
        assert upload_resp.status_code == 200
        assert upload_resp.json()["ok"] is True
    finally:
        await app.stop()
