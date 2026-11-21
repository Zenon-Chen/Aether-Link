"""Generic Webhook / Programmatic Push plugin for integrating any external platform."""

from __future__ import annotations

from typing import Any

from ...models import ChannelType, RawIncomingEvent, utc_now
from ..base import EventCallback, SourcePlugin


class WebhookSourcePlugin(SourcePlugin):
    name = "webhook"
    description = "Accepts events pushed from arbitrary external scripts, campus scrapers, or HTTP webhooks."

    def __init__(self) -> None:
        self._emit: EventCallback | None = None
        self._running = False
        self._ingested_count = 0

    async def start(self, emit: EventCallback) -> None:
        self._emit = emit
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def push(
        self,
        *,
        text: str,
        platform_type: str = "webhook",
        platform_id: str = "webhook",
        channel_type: ChannelType = ChannelType.WEBHOOK,
        channel_id: str = "default",
        channel_name: str = "外部Webhook",
        sender_id: str = "external",
        sender_name: str = "外部平台",
        is_from_owner: bool = False,
        raw_meta: dict[str, Any] | None = None,
    ) -> RawIncomingEvent:
        event = RawIncomingEvent(
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
            is_from_owner=is_from_owner,
            raw_meta=raw_meta or {},
        )
        self._ingested_count += 1
        if self._emit is not None:
            await self._emit(event)
        return event

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "running": self._running,
            "ingested_count": self._ingested_count,
        }
