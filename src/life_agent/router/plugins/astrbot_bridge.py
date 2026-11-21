"""AstrBot multi-platform I/O bridge embedded as a standalone module (NOT an AstrBot plugin).

Reuses AstrBotDevs/AstrBot's PlatformManager & platform adapters inside Life Agent's
own independent process without launching AstrBot's LLM/Pipeline/Star/WebUI runtime.
Supports all AstrBot platforms including QQ OneBot v11 (aiocqhttp), official personal
WeChat direct connection (weixin_oc via ilinkai.weixin.qq.com), Telegram, Discord,
Lark/Feishu, DingTalk, WeCom, Slack, KOOK, Line, Satori, Misskey, Mattermost, etc.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

from ...config import AstrBotModuleConfig, OwnerConfig
from ...models import ChannelType, RawIncomingEvent, utc_now
from ..base import EventCallback, SourcePlugin

logger = logging.getLogger(__name__)

_ASTRBOT_BOOTSTRAPPED = False


def bootstrap_astrbot_environment(repo_path: str | Path, data_root: str | Path) -> Path:
    """Prepare sys.path and ASTRBOT_ROOT before importing AstrBot platform modules.

    Stubs AstrBot's optional RAG/KnowledgeBase module so platform adapters load cleanly
    without pulling heavy RAG dependencies (FAISS/pypdf/etc.), keeping our architecture
    strictly RAG-free.
    """
    global _ASTRBOT_BOOTSTRAPPED
    resolved_repo = Path(repo_path).resolve()
    resolved_root = Path(data_root).resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)

    if not _ASTRBOT_BOOTSTRAPPED:
        os.environ["ASTRBOT_ROOT"] = str(resolved_root)
        if str(resolved_repo) not in sys.path:
            sys.path.insert(0, str(resolved_repo))

        # Stub heavy KnowledgeBase/RAG package that AstrBot's Star context eagerly imports
        for mod_name in (
            "astrbot.core.knowledge_base",
            "astrbot.core.knowledge_base.kb_mgr",
        ):
            if mod_name not in sys.modules:
                stub = types.ModuleType(mod_name)
                stub.KnowledgeBaseManager = object  # type: ignore[attr-defined]
                sys.modules[mod_name] = stub

        _ASTRBOT_BOOTSTRAPPED = True
    return resolved_repo


class AstrBotBridgePlugin(SourcePlugin):
    """Information Router plugin + Outbound Messenger wrapping AstrBot's PlatformManager."""

    name = "astrbot"
    description = (
        "Embeds AstrBotDevs/AstrBot platform adapters (QQ OneBot v11, official personal "
        "WeChat weixin_oc, Telegram, Discord, Lark, DingTalk, WeCom, Slack, etc.) as an "
        "in-process messaging module."
    )

    # Fields owned/persisted dynamically by AstrBot adapters (never clobbered by empty YAML values)
    _PRESERVE_DYNAMIC_KEYS = (
        "weixin_oc_token",
        "weixin_oc_account_id",
        "weixin_oc_sync_buf",
        "weixin_oc_context_tokens",
        "weixin_oc_base_url",
        "webhook_uuid",
    )

    def __init__(
        self,
        config: AstrBotModuleConfig,
        owner: OwnerConfig,
        base_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.owner = owner
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self._event_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._platform_manager: Any = None
        self._astrbot_config: Any = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._emit: EventCallback | None = None
        self._running = False
        self._discovered_weixin_user_id: str | None = None
        self._load_error: str | None = None

    def _resolve_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def _merge_platform_configs(
        self,
        desired: list[dict[str, Any]],
        existing: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge Life Agent YAML platform entries with persisted AstrBot runtime state."""
        existing_by_id = {
            str(p.get("id")): p for p in existing if isinstance(p, dict) and p.get("id")
        }
        merged: list[dict[str, Any]] = []
        for item in desired:
            entry = dict(item)
            pid = str(entry.get("id", ""))
            prev = existing_by_id.get(pid)
            if prev:
                for key in self._PRESERVE_DYNAMIC_KEYS:
                    if not entry.get(key) and prev.get(key):
                        entry[key] = prev[key]
            merged.append(entry)
        return merged

    async def start(self, emit: EventCallback) -> None:
        self._emit = emit
        if not self.config.enabled:
            logger.info("AstrBotBridgePlugin disabled in config.")
            return

        repo_path = self._resolve_path(self.config.astrbot_repo_path)
        data_root = self._resolve_path(self.config.astrbot_data_root)
        if not (repo_path / "astrbot").exists():
            self._load_error = (
                f"AstrBot repository not found at {repo_path}. "
                "Clone https://github.com/AstrBotDevs/AstrBot into vendor/AstrBot."
            )
            logger.warning(self._load_error)
            return

        try:
            bootstrap_astrbot_environment(repo_path, data_root)
            from astrbot.core import astrbot_config
            from astrbot.core.platform.manager import PlatformManager

            self._astrbot_config = astrbot_config
            existing_platforms = astrbot_config.get("platform", [])
            if not isinstance(existing_platforms, list):
                existing_platforms = []

            merged = self._merge_platform_configs(self.config.platforms, existing_platforms)
            astrbot_config["platform"] = merged
            settings = astrbot_config.get("platform_settings", {})
            if isinstance(settings, dict):
                settings.update(self.config.platform_settings)
                astrbot_config["platform_settings"] = settings
            astrbot_config.save_config()

            self._platform_manager = PlatformManager(astrbot_config, self._event_queue)
            for platform_cfg in merged:
                if platform_cfg.get("enable"):
                    try:
                        await self._platform_manager.load_platform(platform_cfg)
                    except Exception as exc:
                        logger.exception(
                            "Failed to start AstrBot platform %s(%s): %s",
                            platform_cfg.get("type"),
                            platform_cfg.get("id"),
                            exc,
                        )

            self._running = True
            self._consumer_task = asyncio.create_task(
                self._consume_events(),
                name="astrbot_event_consumer",
            )
            logger.info(
                "AstrBotBridgePlugin started in standalone mode with %d active platform adapter(s).",
                len(self._platform_manager.platform_insts),
            )
        except Exception as exc:
            self._load_error = str(exc)
            logger.exception("Failed to initialize AstrBotBridgePlugin: %s", exc)

    async def stop(self) -> None:
        self._running = False
        if self._consumer_task is not None and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        if self._platform_manager is not None:
            for inst in list(self._platform_manager.platform_insts):
                try:
                    await self._platform_manager._terminate_inst_and_tasks(inst)
                except Exception:
                    logger.exception("Error stopping AstrBot adapter %s", inst)
            self._platform_manager.platform_insts.clear()

    def _is_owner_sender(self, platform_id: str, platform_type: str, sender_id: str) -> bool:
        if not sender_id:
            return False
        if (
            self.owner.qq_id
            and platform_id == self.owner.qq_platform_id
            and sender_id == str(self.owner.qq_id)
        ):
            return True
        if (
            self.owner.qq_id
            and platform_type in ("aiocqhttp", "qq_official", "qq_official_webhook")
            and sender_id == str(self.owner.qq_id)
        ):
            return True

        wx_id = self.owner.weixin_user_id or self._discovered_weixin_user_id or ""
        if (
            platform_type == "weixin_oc"
            and (
                (wx_id and sender_id == wx_id)
                # Personal WeChat iLink is a 1:1 bot bound to the owner's personal WeChat
                or (not self.owner.weixin_user_id and platform_id == self.owner.weixin_platform_id)
            )
        ):
            return True

        composite = f"{platform_id}:{sender_id}"
        return composite in self.owner.extra_owner_identities

    def _convert_event(self, astr_event: Any) -> RawIncomingEvent | None:
        """Normalize an AstrMessageEvent into Life Agent's RawIncomingEvent."""
        from astrbot.core.platform.message_type import MessageType

        platform_meta = astr_event.platform_meta
        platform_type = str(getattr(platform_meta, "name", "unknown"))
        platform_id = str(getattr(platform_meta, "id", platform_type))
        msg_obj = astr_event.message_obj
        msg_type = astr_event.get_message_type()

        sender_id = str(astr_event.get_sender_id() or "")
        sender_name = str(astr_event.get_sender_name() or sender_id)

        # Auto-discover owner's weixin_oc user_id (e.g. xxx@im.wechat) on first personal WeChat message
        if platform_type == "weixin_oc" and sender_id and not self.owner.weixin_user_id:
            if self._discovered_weixin_user_id != sender_id:
                self._discovered_weixin_user_id = sender_id
                logger.info(
                    "Auto-discovered personal WeChat (weixin_oc) owner user_id=%s",
                    sender_id,
                )

        # Ignore messages echoed by the bot itself
        self_id = str(getattr(msg_obj, "self_id", "") or "")
        if self_id and sender_id and sender_id == self_id:
            return None

        text = str(astr_event.message_str or "").strip()
        components = getattr(msg_obj, "message", None) or []
        attachments: list[dict[str, str]] = []
        placeholders: list[str] = []
        for comp in components:
            cname = type(comp).__name__
            if cname == "Plain":
                continue
            placeholders.append(f"[{cname}]")
            if cname in ("Image", "File"):
                file_ref = str(getattr(comp, "file", "") or getattr(comp, "url", "") or "").strip()
                name_ref = str(getattr(comp, "name", "") or "").strip()
                if file_ref:
                    attachments.append(
                        {
                            "type": cname.lower(),
                            "path_or_url": file_ref,
                            "name": name_ref,
                        }
                    )
        if not text:
            text = " ".join(placeholders).strip()
        if not text:
            return None

        is_group = msg_type == MessageType.GROUP_MESSAGE
        group_id = str(astr_event.get_group_id() or "") if is_group else ""
        group_obj = getattr(msg_obj, "group", None)
        group_name = str(getattr(group_obj, "group_name", "") or group_id) if group_obj else group_id

        channel_type = ChannelType.GROUP if is_group else ChannelType.PRIVATE
        channel_id = group_id if is_group else str(astr_event.session_id or sender_id)
        channel_name = group_name if is_group else sender_name
        is_owner = (not is_group) and self._is_owner_sender(platform_id, platform_type, sender_id)

        return RawIncomingEvent(
            source_plugin=self.name,
            platform_type=platform_type,
            platform_id=platform_id,
            channel_type=channel_type,
            channel_id=channel_id,
            channel_name=channel_name,
            sender_id=sender_id,
            sender_name=sender_name,
            text=text,
            timestamp=utc_now(),
            is_from_owner=is_owner,
            raw_meta={
                "unified_msg_origin": str(getattr(astr_event, "unified_msg_origin", "")),
                "message_id": str(getattr(msg_obj, "message_id", "")),
                "attachments": attachments,
            },
        )

    async def _consume_events(self) -> None:
        while self._running:
            try:
                astr_event = await self._event_queue.get()
                try:
                    raw_event = self._convert_event(astr_event)
                    if raw_event is not None and self._emit is not None:
                        await self._emit(raw_event)
                finally:
                    self._event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error consuming event from AstrBot platform queue")

    def get_adapter(self, platform_id: str | None = None, platform_type: str | None = None) -> Any | None:
        """Locate a running AstrBot Platform adapter instance by id or type."""
        if self._platform_manager is None:
            return None
        for inst in self._platform_manager.platform_insts:
            meta = inst.meta()
            if platform_id and (meta.id == platform_id or inst.config.get("id") == platform_id):
                return inst
        if platform_type:
            for inst in self._platform_manager.platform_insts:
                if inst.meta().name == platform_type:
                    return inst
        return None

    def get_effective_weixin_user_id(self) -> str:
        """Return configured or auto-discovered WeChat iLink user_id."""
        if self.owner.weixin_user_id:
            return self.owner.weixin_user_id
        if self._discovered_weixin_user_id:
            return self._discovered_weixin_user_id
        # Inspect cached context_tokens in running WeixinOCAdapter if available
        adapter = self.get_adapter(self.owner.weixin_platform_id, "weixin_oc")
        if adapter is not None:
            tokens = getattr(adapter, "_context_tokens", None) or {}
            if tokens:
                return next(iter(tokens.keys()))
        return ""

    async def send_text(
        self,
        *,
        platform_id: str | None = None,
        platform_type: str | None = None,
        session_id: str,
        text: str,
        is_group: bool = False,
    ) -> bool:
        """Send a proactive or reply text message via an embedded AstrBot platform adapter."""
        adapter = self.get_adapter(platform_id=platform_id, platform_type=platform_type)
        if adapter is None:
            logger.warning(
                "No running AstrBot adapter found for platform_id=%s platform_type=%s",
                platform_id,
                platform_type,
            )
            return False

        from astrbot.api.event import MessageChain
        from astrbot.api.message_components import Plain
        from astrbot.core.platform.astr_message_event import MessageSesion
        from astrbot.core.platform.message_type import MessageType

        msg_type = MessageType.GROUP_MESSAGE if is_group else MessageType.FRIEND_MESSAGE
        session = MessageSesion(
            platform_name=adapter.meta().id,
            message_type=msg_type,
            session_id=str(session_id),
        )
        chain = MessageChain([Plain(text)])

        # Guard for weixin_oc missing context_token so caller gets clear False status
        if adapter.meta().name == "weixin_oc":
            ctx_tokens = getattr(adapter, "_context_tokens", {})
            if not getattr(adapter, "token", None):
                logger.warning("weixin_oc adapter is not logged in yet (missing token).")
                return False
            if str(session_id) not in ctx_tokens:
                logger.warning(
                    "weixin_oc missing context_token for %s. Send one message to the WeChat bot first to initialize context_token.",
                    session_id,
                )
                return False

        await adapter.send_by_session(session, chain)
        return True

    async def request_weixin_qr(self, platform_id: str | None = None) -> dict[str, Any]:
        """Request an official personal WeChat (iLink / weixin_oc) login QR code."""
        repo_path = self._resolve_path(self.config.astrbot_repo_path)
        data_root = self._resolve_path(self.config.astrbot_data_root)
        bootstrap_astrbot_environment(repo_path, data_root)

        from astrbot.core.platform.sources.weixin_oc.login_registration import (
            request_weixin_oc_login_qr,
        )

        target_id = platform_id or self.owner.weixin_platform_id
        pcfg = next(
            (p for p in self.config.platforms if p.get("id") == target_id or p.get("type") == "weixin_oc"),
            {"id": target_id, "type": "weixin_oc"},
        )
        reg = await request_weixin_oc_login_qr(pcfg)
        return {
            "platform_id": target_id,
            "qrcode": reg.qrcode,
            "qrcode_img_content": reg.qrcode_img_content,
            "interval": reg.interval,
        }

    async def poll_weixin_qr(self, qrcode: str, platform_id: str | None = None) -> dict[str, Any]:
        """Poll official personal WeChat QR login status and persist credentials on confirmation."""
        repo_path = self._resolve_path(self.config.astrbot_repo_path)
        data_root = self._resolve_path(self.config.astrbot_data_root)
        bootstrap_astrbot_environment(repo_path, data_root)

        from astrbot.core.platform.sources.weixin_oc.login_registration import (
            poll_weixin_oc_login_once,
        )

        target_id = platform_id or self.owner.weixin_platform_id
        pcfg = next(
            (p for p in self.config.platforms if p.get("id") == target_id or p.get("type") == "weixin_oc"),
            {"id": target_id, "type": "weixin_oc"},
        )
        result = await poll_weixin_oc_login_once(platform_config=pcfg, qrcode=qrcode)
        if result.get("status") == "created":
            # Persist credentials into AstrBot config & hot-update running adapter if present
            if result.get("weixin_oc_user_id"):
                self._discovered_weixin_user_id = str(result["weixin_oc_user_id"])
            if self._astrbot_config is not None:
                platforms = self._astrbot_config.get("platform", [])
                for p in platforms:
                    if p.get("id") == target_id or p.get("type") == "weixin_oc":
                        p["weixin_oc_token"] = result.get("weixin_oc_token", "")
                        p["weixin_oc_account_id"] = result.get("weixin_oc_account_id", "")
                        p["weixin_oc_base_url"] = result.get("weixin_oc_base_url", p.get("weixin_oc_base_url", ""))
                        break
                self._astrbot_config.save_config()

            adapter = self.get_adapter(target_id, "weixin_oc")
            if adapter is not None:
                adapter.token = result.get("weixin_oc_token") or None
                adapter.account_id = result.get("weixin_oc_account_id") or None
                if result.get("weixin_oc_base_url"):
                    adapter.base_url = str(result["weixin_oc_base_url"]).rstrip("/")
                if hasattr(adapter, "_sync_client_state"):
                    adapter._sync_client_state()
        return result

    def status(self) -> dict[str, Any]:
        adapters = []
        if self._platform_manager is not None:
            for inst in self._platform_manager.platform_insts:
                stat = inst.get_stats()
                if inst.meta().name == "weixin_oc":
                    stat["weixin_logged_in"] = bool(getattr(inst, "token", None))
                    stat["weixin_cached_users"] = list(getattr(inst, "_context_tokens", {}).keys())
                    login_sess = getattr(inst, "_login_session", None)
                    if login_sess is not None and not stat["weixin_logged_in"]:
                        stat["weixin_qr_content"] = getattr(login_sess, "qrcode_img_content", "")
                adapters.append(stat)
        return {
            "name": self.name,
            "description": self.description,
            "running": self._running,
            "error": self._load_error,
            "effective_weixin_user_id": self.get_effective_weixin_user_id(),
            "adapters": adapters,
        }
