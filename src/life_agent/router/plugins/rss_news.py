"""RSS / Atom news polling plugin for Information Router."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import feedparser
import httpx

from ...config import RSSPluginConfig
from ...models import ChannelType, RawIncomingEvent, utc_now
from ...storage import Storage
from ..base import EventCallback, SourcePlugin

logger = logging.getLogger(__name__)


class RSSNewsPlugin(SourcePlugin):
    name = "rss_news"
    description = "Polls configured RSS/Atom news feeds and feeds new articles to the Structurer."

    def __init__(self, config: RSSPluginConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage
        self._task: asyncio.Task[None] | None = None
        self._emit: EventCallback | None = None
        self._running = False

    async def start(self, emit: EventCallback) -> None:
        self._emit = emit
        if not self.config.enabled or not self.config.feeds:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="rss_news_poller")
        logger.info("RSSNewsPlugin started with %d feed(s).", len(self.config.feeds))

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def poll_once(self) -> int:
        if self._emit is None:
            return 0
        emitted = 0
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            for feed_cfg in self.config.feeds:
                try:
                    resp = await client.get(feed_cfg.url)
                    resp.raise_for_status()
                    parsed = await asyncio.to_thread(feedparser.parse, resp.text)
                    entries = (parsed.entries or [])[: self.config.max_items_per_feed]
                    for entry in entries:
                        ext_id = str(
                            entry.get("id") or entry.get("link") or entry.get("title") or ""
                        ).strip()
                        if not ext_id:
                            continue
                        if await self.storage.is_seen("rss_news", ext_id):
                            continue
                        await self.storage.mark_seen("rss_news", ext_id)
                        title = str(entry.get("title", "")).strip()
                        summary = str(entry.get("summary", "") or entry.get("description", "")).strip()
                        link = str(entry.get("link", "")).strip()
                        text = f"标题：{title}\n来源：{feed_cfg.name}\n链接：{link}\n摘要：{summary[:800]}"
                        await self._emit(
                            RawIncomingEvent(
                                source_plugin=self.name,
                                platform_type="rss",
                                platform_id=feed_cfg.name,
                                channel_type=ChannelType.NEWS,
                                channel_id=feed_cfg.url,
                                channel_name=feed_cfg.name,
                                sender_name=feed_cfg.name,
                                text=text,
                                timestamp=utc_now(),
                                raw_meta={"title": title, "link": link},
                            )
                        )
                        emitted += 1
                except Exception as exc:
                    logger.warning("RSS poll failed for %s (%s): %s", feed_cfg.name, feed_cfg.url, exc)
        return emitted

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unexpected error in RSSNewsPlugin poll loop")
            await asyncio.sleep(self.config.poll_interval_seconds)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "running": self._running,
            "feeds": [f.name for f in self.config.feeds],
        }
