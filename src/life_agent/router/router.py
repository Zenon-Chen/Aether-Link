"""Information Router (Box 0): pluggable ingestion, dynamic plugin loading, and group debounce."""

from __future__ import annotations

import importlib.util
import inspect
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..models import ChannelType, GroupChatBatch, RawIncomingEvent
from ..storage import Storage
from .base import SourcePlugin
from .group_buffer import GroupChatBufferManager

logger = logging.getLogger(__name__)

OwnerChatHandler = Callable[[RawIncomingEvent], Awaitable[None]]
DirectEventHandler = Callable[[RawIncomingEvent], Awaitable[None]]
GroupBatchHandler = Callable[[GroupChatBatch], Awaitable[None]]


class InformationRouter:
    """Multi-platform Information Router supporting AstrBot + arbitrary custom SourcePlugins."""

    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        *,
        on_owner_chat: OwnerChatHandler,
        on_direct_event: DirectEventHandler,
        on_group_batch: GroupBatchHandler,
        base_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.on_owner_chat = on_owner_chat
        self.on_direct_event = on_direct_event
        self.on_group_batch = on_group_batch

        self.plugins: dict[str, SourcePlugin] = {}
        self.group_buffer = GroupChatBufferManager(
            config=config.group_filter,
            on_flush=self.on_group_batch,
        )

    def register_plugin(self, plugin: SourcePlugin) -> None:
        if plugin.name in self.plugins:
            logger.warning("Overwriting already registered router plugin: %s", plugin.name)
        self.plugins[plugin.name] = plugin
        logger.info("Registered Information Router plugin: %s", plugin.name)

    def load_external_plugins(self, plugins_dir: str | Path | None = None) -> list[str]:
        """Dynamically discover and register custom SourcePlugin files from plugins/."""
        raw_dir = Path(plugins_dir or self.config.plugins_dir)
        target_dir = raw_dir if raw_dir.is_absolute() else (self.base_dir / raw_dir).resolve()
        if not target_dir.exists():
            return []

        loaded: list[str] = []
        for py_file in sorted(target_dir.glob("*.py")):
            if py_file.name.startswith("_") or py_file.name.startswith("example_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"life_agent_ext_plugin_{py_file.stem}",
                    py_file,
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                plugin_inst: SourcePlugin | None = None
                factory = getattr(module, "create_plugin", None)
                if callable(factory):
                    sig = inspect.signature(factory)
                    if len(sig.parameters) >= 2:
                        plugin_inst = factory(self.config, self.storage)
                    elif len(sig.parameters) == 1:
                        plugin_inst = factory(self.config)
                    else:
                        plugin_inst = factory()
                else:
                    for _, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, SourcePlugin) and obj is not SourcePlugin:
                            plugin_inst = obj()
                            break

                if isinstance(plugin_inst, SourcePlugin):
                    self.register_plugin(plugin_inst)
                    loaded.append(plugin_inst.name)
            except Exception:
                logger.exception("Failed to load external router plugin from %s", py_file)
        return loaded

    async def start(self) -> None:
        self.load_external_plugins()
        for plugin in self.plugins.values():
            try:
                await plugin.start(self.handle_event)
            except Exception:
                logger.exception("Failed to start router plugin %s", plugin.name)

    async def stop(self) -> None:
        await self.group_buffer.stop()
        for plugin in self.plugins.values():
            try:
                await plugin.stop()
            except Exception:
                logger.exception("Failed to stop router plugin %s", plugin.name)

    async def handle_event(self, event: RawIncomingEvent) -> None:
        """Central routing rule for every event entering Information Router."""
        if event.is_from_owner and event.channel_type == ChannelType.PRIVATE:
            logger.debug(
                "Routing owner direct chat (%s/%s): %s",
                event.platform_type,
                event.sender_id,
                event.text[:80],
            )
            await self.on_owner_chat(event)
            return

        if event.channel_type == ChannelType.GROUP:
            await self.group_buffer.ingest(event)
            return

        await self.on_direct_event(event)

    def status(self) -> dict[str, Any]:
        return {
            "plugins": [p.status() for p in self.plugins.values()],
            "pending_groups": self.group_buffer.pending_summary(),
        }
