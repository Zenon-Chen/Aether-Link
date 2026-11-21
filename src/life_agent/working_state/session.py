"""Working State Area (工作状态区, Steps 3, 4, 5, v2 Tiered Scene State & 9 Structured Tools)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..agents.rtir import RTIRParser
from ..config import AppConfig
from ..gateway.policy import GatingPolicyManager
from ..llm import LLMClient
from ..models import ChatTurn, utc_now
from ..scene.manager import SceneStateManager
from ..slow_loop.bm25_store import SlowLoopBM25Store
from ..storage import Storage
from ..triggers.followup import FollowUpTimingCalculator
from .tools import DIRECT_TOOL_SCHEMAS, DirectToolExecutor

logger = logging.getLogger(__name__)


class WorkingStateSession:
    """Manages Working State context (system + T0–T3 Scene state + Context), tools, and 3-min RTIR timer."""

    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        llm: LLMClient,
        rtir: RTIRParser,
        slow_store: SlowLoopBM25Store | None = None,
        scene_manager: SceneStateManager | None = None,
        policy: GatingPolicyManager | None = None,
        followup_calc: FollowUpTimingCalculator | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.llm = llm
        self.rtir = rtir
        self.slow_store = slow_store or SlowLoopBM25Store(storage)
        self.scene_manager = scene_manager or SceneStateManager(config, storage)
        self.policy = policy
        self.followup_calc = followup_calc or FollowUpTimingCalculator(storage)
        self.tool_executor = DirectToolExecutor(
            storage,
            self.slow_store,
            policy=self.policy,
            followup_calc=self.followup_calc,
            default_reminder_lead_minutes=config.schedule_importer.default_reminder_lead_minutes,
            sleep_start_hour=config.owner.living_profile.sleep_start_hour,
            sleep_end_hour=config.owner.living_profile.sleep_end_hour,
        )

        self._chat_lock = asyncio.Lock()
        self._rtir_timer: asyncio.Task[None] | None = None

    async def _build_system_message(self) -> str:
        """Assemble Working State = system + T0–T3 Scoped Scene State (Fast Loop stays clean!)."""
        now = utc_now()
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        periods = "\n".join(
            f"  - {k}: {v}" for k, v in self.config.owner.school_periods.items()
        )
        scene_view = await self.scene_manager.get_structured_view(now=now)
        scene_json = json.dumps(scene_view, ensure_ascii=False, indent=2)

        return (
            f"{self.config.working_state.system_prompt}\n\n"
            f"【用户画像与偏好】\n{self.config.owner.persona_notes}\n\n"
            f"【当前精确时间】{now.isoformat()} ({weekdays[now.weekday()]})\n"
            f"【课节时间对照表】\n{periods}\n\n"
            f"【当前分级场景状态 (T0–T3 Scene state)】\n{scene_json}"
        )

    async def _build_messages(self, extra_instruction: str | None = None) -> list[dict[str, Any]]:
        system_content = await self._build_system_message()
        history = await self.storage.get_recent_chat_turns(
            limit=self.config.working_state.max_context_messages
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        for turn in history:
            if turn.role in ("user", "assistant"):
                messages.append({"role": turn.role, "content": turn.content})

        if extra_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "【主动推送触发指令 (Active push)】\n"
                        f"{extra_instruction}\n"
                        "如需参考今日精确时间表、缓冲区、慢回路空闲想法或跟进项，"
                        "请先调用对应工具再生成简洁自然的一条主动消息。"
                    ),
                }
            )
        return messages

    async def _run_llm_with_tools(
        self,
        messages: list[dict[str, Any]],
        max_rounds: int = 6,
    ) -> tuple[str, list[dict[str, Any]]]:
        executed_tools: list[dict[str, Any]] = []
        working_messages = list(messages)

        for _ in range(max_rounds):
            resp = await self.llm.chat(
                role="chat",
                messages=working_messages,
                tools=DIRECT_TOOL_SCHEMAS,
            )
            if not resp.tool_calls:
                return resp.content.strip(), executed_tools

            working_messages.append(
                {
                    "role": "assistant",
                    "content": resp.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in resp.tool_calls
                    ],
                }
            )
            for tc in resp.tool_calls:
                tool_output = await self.tool_executor.execute(tc.name, tc.arguments)
                executed_tools.append(
                    {"name": tc.name, "arguments": tc.arguments, "result": json.loads(tool_output)}
                )
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": tool_output,
                    }
                )

        final_resp = await self.llm.chat(role="chat", messages=working_messages, tools=None)
        return final_resp.content.strip(), executed_tools

    async def handle_user_message(
        self,
        text: str,
        *,
        channel: str = "qq",
    ) -> tuple[str, list[dict[str, Any]]]:
        async with self._chat_lock:
            await self.storage.append_chat_turn(
                ChatTurn(role="user", content=text, channel=channel)
            )
            messages = await self._build_messages()
            reply, tools_used = await self._run_llm_with_tools(messages)
            if not reply:
                reply = "收到啦，我已经处理好了。"
            await self.storage.append_chat_turn(
                ChatTurn(role="assistant", content=reply, channel=channel)
            )
            self._reset_rtir_timer()
            return reply, tools_used

    async def generate_active_push(
        self,
        instruction: str,
        *,
        trigger_source: str = "gateway",
    ) -> tuple[str, list[dict[str, Any]]]:
        async with self._chat_lock:
            messages = await self._build_messages(extra_instruction=instruction)
            text, tools_used = await self._run_llm_with_tools(messages)
            if not text:
                text = instruction
            await self.storage.append_chat_turn(
                ChatTurn(role="assistant", content=text, channel=f"active_push:{trigger_source}")
            )
            return text, tools_used

    def _reset_rtir_timer(self) -> None:
        if self._rtir_timer is not None and not self._rtir_timer.done():
            self._rtir_timer.cancel()
        delay = self.config.working_state.rtir_silence_seconds
        self._rtir_timer = asyncio.create_task(
            self._rtir_countdown(delay),
            name="rtir_silence_timer",
        )

    async def _rtir_countdown(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            await self.rtir.parse_unparsed_turns()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("RTIR post-conversation parser failed")

    async def flush_rtir_now(self) -> dict[str, Any]:
        if self._rtir_timer is not None and not self._rtir_timer.done():
            self._rtir_timer.cancel()
        return await self.rtir.parse_unparsed_turns()

    async def stop(self) -> None:
        if self._rtir_timer is not None and not self._rtir_timer.done():
            self._rtir_timer.cancel()
            try:
                await self._rtir_timer
            except asyncio.CancelledError:
                pass
