"""Top-level independent-process orchestrator (v2: Two-Phase Forward/Backward Architecture)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from .agents.rtir import RTIRParser
from .agents.structurer import StructurerSubAgent
from .config import AppConfig
from .delivery.manager import DeliveryManager, TransportSender
from .gateway.gateway import MessageGateway
from .gateway.policy import GatingPolicyManager
from .importers.schedule_importer import (
    EXCEL_EXTENSIONS,
    IMAGE_EXTENSIONS,
    ScheduleImporter,
)
from .llm import LLMClient
from .models import GatingRuleStatus, GroupChatBatch, RawIncomingEvent, ScheduleImportResult
from .router.plugins.arxiv_papers import ArxivPapersPlugin
from .router.plugins.astrbot_bridge import AstrBotBridgePlugin
from .router.plugins.rss_news import RSSNewsPlugin
from .router.plugins.webhook_source import WebhookSourcePlugin
from .router.router import InformationRouter
from .scene.manager import SceneStateManager
from .slow_loop.bm25_store import SlowLoopBM25Store
from .storage import Storage
from .triggers.followup import FollowUpTimingCalculator
from .triggers.passive import PassiveTriggerEngine
from .working_state.session import WorkingStateSession

logger = logging.getLogger(__name__)

_IMPORT_KEYWORDS = ("导入课表", "导入排班", "课表导入", "排班导入", "识别课表", "识别排班")


class LifeAgentApp:
    """Independent process hosting the complete v2 Forward/Backward Life Agent architecture."""

    def __init__(
        self,
        config: AppConfig,
        *,
        base_dir: Path | None = None,
        llm_client: LLMClient | None = None,
        custom_sender: TransportSender | None = None,
    ) -> None:
        self.config = config
        self.base_dir = (base_dir or Path.cwd()).resolve()

        db_raw = Path(config.db_path)
        db_path = (
            db_raw
            if str(db_raw) == ":memory:" or db_raw.is_absolute()
            else (self.base_dir / db_raw).resolve()
        )
        self.storage = Storage(db_path)
        self.llm = llm_client or LLMClient(config.llm)

        # v2 Shared Core Managers
        self.slow_store = SlowLoopBM25Store(self.storage)
        self.scene_manager = SceneStateManager(config, self.storage)
        self.policy = GatingPolicyManager(config, self.storage, self.llm)
        self.followup_calc = FollowUpTimingCalculator(self.storage)
        self.schedule_importer = ScheduleImporter(
            config=config,
            storage=self.storage,
            llm=self.llm,
            slow_store=self.slow_store,
        )

        # Built-in router plugins
        self.astrbot_bridge = AstrBotBridgePlugin(
            config=config.astrbot,
            owner=config.owner,
            base_dir=self.base_dir,
        )
        self.rss_plugin = RSSNewsPlugin(config=config.rss, storage=self.storage)
        self.arxiv_plugin = ArxivPapersPlugin(config=config.arxiv, storage=self.storage)
        self.webhook_plugin = WebhookSourcePlugin()

        # Core architectural components
        self.delivery = DeliveryManager(
            config=config,
            storage=self.storage,
            astrbot_bridge=self.astrbot_bridge,
            custom_sender=custom_sender,
            policy=self.policy,
        )
        self.rtir = RTIRParser(
            config=config,
            storage=self.storage,
            llm=self.llm,
            slow_store=self.slow_store,
            scene_manager=self.scene_manager,
            policy=self.policy,
            followup_calc=self.followup_calc,
        )
        self.working_state = WorkingStateSession(
            config=config,
            storage=self.storage,
            llm=self.llm,
            rtir=self.rtir,
            slow_store=self.slow_store,
            scene_manager=self.scene_manager,
            policy=self.policy,
            followup_calc=self.followup_calc,
        )
        self.gateway = MessageGateway(
            config=config,
            storage=self.storage,
            llm=self.llm,
            on_active_push=self._handle_gateway_active_push,
            policy=self.policy,
            scene_manager=self.scene_manager,
            slow_store=self.slow_store,
            followup_calc=self.followup_calc,
        )
        self.structurer = StructurerSubAgent(
            config=config,
            storage=self.storage,
            llm=self.llm,
            on_structured_item=self.gateway.evaluate_buffer_item,
            slow_store=self.slow_store,
        )
        self.router = InformationRouter(
            config=config,
            storage=self.storage,
            on_owner_chat=self._handle_owner_chat,
            on_direct_event=self._handle_direct_event,
            on_group_batch=self._handle_group_batch,
            base_dir=self.base_dir,
        )
        self.router.register_plugin(self.astrbot_bridge)
        self.router.register_plugin(self.rss_plugin)
        self.router.register_plugin(self.arxiv_plugin)
        self.router.register_plugin(self.webhook_plugin)

        self.passive_triggers = PassiveTriggerEngine(
            config=config,
            storage=self.storage,
            gateway=self.gateway,
            slow_store=self.slow_store,
            scene_manager=self.scene_manager,
            policy=self.policy,
            followup_calc=self.followup_calc,
        )

    async def start(self) -> None:
        await self.storage.init()
        await self.policy.ensure_initialized()
        await self.scene_manager.sync_living_profile()
        await self.delivery.start()
        await self.router.start()
        await self.passive_triggers.start()
        logger.info("LifeAgentApp (v2 Two-Phase) started cleanly as an independent process.")

    async def stop(self) -> None:
        await self.passive_triggers.stop()
        await self.router.stop()
        await self.working_state.stop()
        await self.delivery.stop()
        await self.llm.close()
        await self.storage.close()
        logger.info("LifeAgentApp stopped.")

    async def _try_import_chat_attachment(self, event: RawIncomingEvent) -> str | None:
        attachments = event.raw_meta.get("attachments")
        if not isinstance(attachments, list) or not attachments:
            return None
        if not any(kw in event.text for kw in _IMPORT_KEYWORDS):
            return None

        schedule_type = (
            "shift" if "排班" in event.text else ("timetable" if "课表" in event.text else "auto")
        )
        for att in attachments:
            if not isinstance(att, dict):
                continue
            ref = str(att.get("path_or_url") or "").strip()
            name = str(att.get("name") or "").strip()
            if not ref:
                continue
            try:
                if ref.startswith(("http://", "https://")):
                    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                        resp = await client.get(ref)
                        resp.raise_for_status()
                        data = resp.content
                    filename = name or (
                        "schedule.png" if att.get("type") == "image" else Path(ref.split("?")[0]).name
                    )
                else:
                    p = Path(ref)
                    if not p.exists():
                        continue
                    data = p.read_bytes()
                    filename = name or p.name
                ext = Path(filename).suffix.lower()
                if ext not in (EXCEL_EXTENSIONS | IMAGE_EXTENSIONS):
                    if att.get("type") == "image":
                        filename = f"{filename or 'schedule'}.png"
                    else:
                        continue
                res = await self.schedule_importer.import_bytes(
                    data=data,
                    filename=filename,
                    schedule_type=schedule_type,
                    extra_instructions=event.text,
                )
                await self.followup_calc.recalculate_all_open()
                return (
                    f"已自动解析并导入「{res.source_filename}」：\n"
                    f"• 一周之内：{res.fast_loop_timetable_count} 条写入全局时间表，并写入 {res.passive_triggers_written} 条被动触发器；\n"
                    f"• 一周之外：{res.slow_loop_bm25_count} 条已录入慢回路 BM25 索引（进入 7 天视野后会自动升入时间表）。"
                )
            except Exception as exc:
                logger.exception("Auto-importing chat attachment failed")
                return f"解析附件时遇到问题：{exc}"
        return None

    async def _handle_owner_chat(self, event: RawIncomingEvent) -> None:
        await self.delivery.on_user_replied(channel=event.platform_type)

        import_reply = await self._try_import_chat_attachment(event)
        if import_reply is not None:
            await self.delivery.send_reply(
                platform_id=event.platform_id,
                platform_type=event.platform_type,
                session_id=event.channel_id or event.sender_id,
                text=import_reply,
            )
            return

        reply, _ = await self.working_state.handle_user_message(
            event.text,
            channel=event.platform_type,
        )
        await self.delivery.send_reply(
            platform_id=event.platform_id,
            platform_type=event.platform_type,
            session_id=event.channel_id or event.sender_id,
            text=reply,
        )

    async def _handle_direct_event(self, event: RawIncomingEvent) -> None:
        await self.structurer.process_event(event)

    async def _handle_group_batch(self, batch: GroupChatBatch) -> None:
        await self.structurer.process_group_batch(batch)

    async def _handle_gateway_active_push(
        self,
        instruction: str,
        trigger_source: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        message_text, _ = await self.working_state.generate_active_push(
            instruction,
            trigger_source=trigger_source,
        )
        m = meta or {}
        await self.delivery.send_proactive(
            message_text,
            trigger_source=trigger_source,
            source_key=str(m.get("source_key") or trigger_source),
            tags=m.get("tags") if isinstance(m.get("tags"), list) else [],
            category=str(m.get("category") or "general"),
            urgency=float(m.get("urgency") if m.get("urgency") is not None else 8.0),
        )

    async def import_schedule_file(
        self,
        file_path: str | Path,
        *,
        schedule_type: str = "auto",
        current_week: int | None = None,
        semester_start_date: str | None = None,
        extra_instructions: str = "",
    ) -> ScheduleImportResult:
        res = await self.schedule_importer.import_file(
            file_path,
            schedule_type=schedule_type,
            current_week=current_week,
            semester_start_date=semester_start_date,
            extra_instructions=extra_instructions,
        )
        await self.followup_calc.recalculate_all_open()
        return res

    async def chat_local(self, text: str, channel: str = "web") -> dict[str, Any]:
        await self.delivery.on_user_replied(channel=channel)
        reply, tools_used = await self.working_state.handle_user_message(text, channel=channel)
        return {"reply": reply, "tools_used": tools_used}

    async def get_state_snapshot(self) -> dict[str, Any]:
        timetable = await self.storage.list_timetable(status="all", limit=100)
        scheduled_triggers = await self.storage.list_scheduled_triggers(include_fired=True, limit=100)
        buffer_items = await self.storage.list_buffer_items(status="all", limit=100)
        slow_items = await self.slow_store.list_all(status="all", limit=200)
        tiered_scene = await self.scene_manager.get_structured_view()
        active_grade = await self.policy.get_frozen_grade()
        grade_history = await self.storage.list_grade_snapshots(limit=10)
        gating_rules = await self.storage.list_gating_rules(status="all")
        feedback_log = await self.storage.list_recent_feedback(limit=30)
        affinities = await self.storage.list_source_affinities(limit=30)
        followups = await self.storage.list_followups(status="all", limit=50)
        chat_turns = await self.storage.get_recent_chat_turns(limit=50)
        pending_pushes = await self.storage.list_pending_pushes(limit=30)

        return {
            "timetable": [e.model_dump(mode="json") for e in timetable],
            "scheduled_triggers": [t.model_dump(mode="json") for t in scheduled_triggers],
            "buffer": {
                "message": [
                    b.model_dump(mode="json") for b in buffer_items if b.zone.value == "message"
                ],
                "urti": [
                    b.model_dump(mode="json") for b in buffer_items if b.zone.value == "urti"
                ],
                "news": [
                    b.model_dump(mode="json") for b in buffer_items if b.zone.value == "news"
                ],
            },
            "slow_memory": [s.model_dump(mode="json") for s in slow_items],
            "scene_state": tiered_scene,
            "grade": {
                "active": active_grade.model_dump(mode="json"),
                "history": [g.model_dump(mode="json") for g in grade_history],
                "rules": [r.model_dump(mode="json") for r in gating_rules],
                "candidates": [
                    r.model_dump(mode="json")
                    for r in gating_rules
                    if r.status == GatingRuleStatus.CANDIDATE
                ],
                "affinities": [a.model_dump(mode="json") for a in affinities],
            },
            "feedback_log": [f.model_dump(mode="json") for f in feedback_log],
            "follow_ups": [f.model_dump(mode="json") for f in followups],
            "context": [c.model_dump(mode="json") for c in chat_turns],
            "pending_pushes": [p.model_dump(mode="json") for p in pending_pushes],
            "router": self.router.status(),
            "passive_slots": [s.model_dump(mode="json") for s in self.config.passive_triggers.slots],
        }
