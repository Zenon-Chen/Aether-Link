"""Proactive Delivery & Failover Manager (Step 4.1 + Append-Only Feedback Ledger Telemetry).

User-confirmed Option B (Full Failover on coalesced pushes):
  - `delivery.failover_min_urgency = 0.0`: every proactive push (already bundled cleanly by
    MessageGateway) sends to QQ first, and if unacknowledged for 5 minutes, escalates to
    Official Personal WeChat direct connection (`weixin_oc`).
  - Strictly Forward-Pass telemetry: user replies -> appends `push_engaged` to `feedback_log`;
    5-min timeouts -> appends `push_ignored` to `feedback_log`. NEVER mutates `grade` live!
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta

from ..config import AppConfig
from ..gateway.policy import GatingPolicyManager
from ..models import PendingPush, PushStatus, utc_now
from ..router.plugins.astrbot_bridge import AstrBotBridgePlugin
from ..storage import Storage

logger = logging.getLogger(__name__)

TransportSender = Callable[[str, str, str], Awaitable[bool]]


class DeliveryManager:
    """Manages QQ primary delivery, 5-minute WeChat direct failover, and feedback_log telemetry."""

    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        astrbot_bridge: AstrBotBridgePlugin | None = None,
        custom_sender: TransportSender | None = None,
        policy: GatingPolicyManager | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.astrbot_bridge = astrbot_bridge
        self.custom_sender = custom_sender
        self.policy = policy

        self._timers: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        waiting = await self.storage.list_pending_pushes(status=PushStatus.WAITING_QQ_REPLY.value)
        now = utc_now()
        for push in waiting:
            remaining = max(0.0, (push.failover_at - now).total_seconds())
            logger.info("Recovered pending QQ push %s (failover in %.1fs)", push.id, remaining)
            self._schedule_failover_timer(push.id, remaining)

    async def stop(self) -> None:
        async with self._lock:
            for task in self._timers.values():
                if not task.done():
                    task.cancel()
            self._timers.clear()

    async def _send_qq(self, qq_target: str, text: str) -> bool:
        if self.custom_sender is not None:
            return await self.custom_sender("qq", qq_target, text)
        if self.astrbot_bridge is None or not qq_target:
            logger.warning("Cannot send QQ message: astrbot_bridge or owner.qq_id missing.")
            return False
        return await self.astrbot_bridge.send_text(
            platform_id=self.config.owner.qq_platform_id,
            platform_type="aiocqhttp",
            session_id=qq_target,
            text=text,
            is_group=False,
        )

    async def _send_weixin(self, weixin_target: str, text: str) -> bool:
        if self.custom_sender is not None:
            return await self.custom_sender("weixin_oc", weixin_target, text)
        if self.astrbot_bridge is None:
            logger.warning("Cannot send WeChat message: astrbot_bridge missing.")
            return False
        effective_target = weixin_target or self.astrbot_bridge.get_effective_weixin_user_id()
        if not effective_target:
            logger.warning("Cannot send WeChat failover: owner.weixin_user_id is empty.")
            return False
        return await self.astrbot_bridge.send_text(
            platform_id=self.config.owner.weixin_platform_id,
            platform_type="weixin_oc",
            session_id=effective_target,
            text=text,
            is_group=False,
        )

    async def send_reply(self, platform_id: str, platform_type: str, session_id: str, text: str) -> bool:
        if self.custom_sender is not None:
            ch = "weixin_oc" if platform_type == "weixin_oc" else "qq"
            return await self.custom_sender(ch, session_id, text)
        if self.astrbot_bridge is None:
            return False
        return await self.astrbot_bridge.send_text(
            platform_id=platform_id,
            platform_type=platform_type,
            session_id=session_id,
            text=text,
            is_group=False,
        )

    async def send_proactive(
        self,
        content: str,
        *,
        trigger_source: str = "gateway",
        source_key: str = "",
        tags: list[str] | None = None,
        category: str = "general",
        urgency: float = 8.0,
    ) -> PendingPush:
        now = utc_now()
        timeout_s = self.config.delivery.failover_timeout_seconds
        qq_target = str(self.config.owner.qq_id or "")
        wx_target = str(
            self.config.owner.weixin_user_id
            or (self.astrbot_bridge.get_effective_weixin_user_id() if self.astrbot_bridge else "")
        )

        push = PendingPush(
            content=content,
            trigger_source=trigger_source,
            source_key=source_key or trigger_source,
            tags=tags or [],
            category=category,
            urgency=urgency,
            qq_target=qq_target,
            weixin_target=wx_target,
            status=PushStatus.WAITING_QQ_REPLY,
            sent_at=now,
            failover_at=now + timedelta(seconds=timeout_s),
        )
        await self.storage.create_pending_push(push)

        qq_ok = False
        qq_err = ""
        try:
            qq_ok = await self._send_qq(qq_target, content)
        except Exception as exc:
            qq_err = str(exc)
            logger.exception("Proactive QQ send raised exception for push %s", push.id)

        should_failover = urgency >= self.config.delivery.failover_min_urgency

        if qq_ok:
            logger.info(
                "Proactive push [%s] sent to QQ (%s, urg=%.1f). Armed %.0fs watchdog (wechat_failover=%s).",
                push.id,
                qq_target,
                urgency,
                timeout_s,
                should_failover,
            )
            self._schedule_failover_timer(push.id, timeout_s)
        elif should_failover and self.config.delivery.failover_immediately_on_qq_error:
            logger.warning(
                "Proactive QQ send failed for [%s] (%s); immediately escalating to WeChat.",
                push.id,
                qq_err or "adapter unavailable",
            )
            await self._execute_wechat_failover(push.id, reason=qq_err or "qq_send_failed")
        else:
            self._schedule_failover_timer(push.id, timeout_s)

        return (await self.storage.get_pending_push(push.id)) or push

    def _schedule_failover_timer(self, push_id: str, delay_seconds: float) -> None:
        existing = self._timers.get(push_id)
        if existing is not None and not existing.done():
            existing.cancel()
        self._timers[push_id] = asyncio.create_task(
            self._failover_watchdog(push_id, delay_seconds),
            name=f"qq_failover_{push_id}",
        )

    async def _failover_watchdog(self, push_id: str, delay_seconds: float) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            await self._execute_wechat_failover(push_id, reason="qq_5min_timeout")
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Error inside WeChat failover watchdog for push %s", push_id)
        finally:
            self._timers.pop(push_id, None)

    async def _execute_wechat_failover(self, push_id: str, *, reason: str) -> None:
        async with self._lock:
            push = await self.storage.get_pending_push(push_id)
            if push is None or push.status != PushStatus.WAITING_QQ_REPLY:
                return

            if self.policy is not None and reason == "qq_5min_timeout":
                await self.policy.append_feedback(
                    event_type="push_ignored",
                    source_key=push.source_key,
                    tags=push.tags,
                    category=push.category,
                    sentiment="neutral",
                    signal_value=-0.6,
                    raw_text=f"QQ 5分钟未回复: {push.content[:80]}",
                    is_explicit=False,
                )

            if push.urgency < self.config.delivery.failover_min_urgency:
                await self.storage.update_pending_push(
                    push_id,
                    status=PushStatus.ACKNOWLEDGED,
                    resolved_at=utc_now(),
                    error_message="below_failover_urgency_logged_only",
                )
                return

            wechat_text = f"{self.config.delivery.failover_prefix}{push.content}"
            wx_ok = False
            wx_err = ""
            try:
                wx_ok = await self._send_weixin(push.weixin_target, wechat_text)
            except Exception as exc:
                wx_err = str(exc)
                logger.exception("WeChat direct failover failed for push %s", push_id)

            if wx_ok:
                await self.storage.update_pending_push(
                    push_id,
                    status=PushStatus.ESCALATED_WECHAT,
                    resolved_at=utc_now(),
                    error_message=reason,
                )
                logger.info(
                    "Failover SUCCESS [%s] (%s): resent proactive message on official personal WeChat.",
                    push.id,
                    reason,
                )
            else:
                await self.storage.update_pending_push(
                    push_id,
                    status=PushStatus.FAILED,
                    resolved_at=utc_now(),
                    error_message=f"{reason}; wechat_error={wx_err or 'send_returned_false'}",
                )
                logger.warning("Failover FAILED [%s]: could not deliver via WeChat.", push_id)

    async def on_user_replied(self, channel: str = "qq") -> int:
        async with self._lock:
            waiting = await self.storage.list_pending_pushes(
                status=PushStatus.WAITING_QQ_REPLY.value
            )
            now = utc_now()
            cancelled = 0
            for push in waiting:
                task = self._timers.pop(push.id, None)
                if task is not None and not task.done():
                    task.cancel()
                await self.storage.update_pending_push(
                    push.id,
                    status=PushStatus.ACKNOWLEDGED,
                    resolved_at=now,
                    error_message=f"replied_on_{channel}",
                )
                cancelled += 1

                if self.policy is not None:
                    latency_s = max(0.0, (now - push.sent_at).total_seconds())
                    sig = 1.0 if latency_s <= 180 else 0.5
                    await self.policy.append_feedback(
                        event_type="push_engaged",
                        source_key=push.source_key,
                        tags=push.tags,
                        category=push.category,
                        sentiment="positive",
                        signal_value=sig,
                        raw_text=f"用户在 {int(latency_s)}s 内回复 ({channel}): {push.content[:80]}",
                        is_explicit=False,
                        now=now,
                    )

            if cancelled:
                logger.info(
                    "User replied on %s: acknowledged and cancelled %d pending WeChat failover(s).",
                    channel,
                    cancelled,
                )
            return cancelled
