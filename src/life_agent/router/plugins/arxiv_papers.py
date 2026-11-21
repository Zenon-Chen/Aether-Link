"""arXiv new papers polling plugin for Information Router."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import quote_plus

import feedparser
import httpx

from ...config import ArxivPluginConfig
from ...models import ChannelType, RawIncomingEvent, utc_now
from ...storage import Storage
from ..base import EventCallback, SourcePlugin

logger = logging.getLogger(__name__)


class ArxivPapersPlugin(SourcePlugin):
    name = "arxiv_papers"
    description = "Polls arXiv API for new papers matching configured categories and keywords."

    def __init__(self, config: ArxivPluginConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage
        self._task: asyncio.Task[None] | None = None
        self._emit: EventCallback | None = None
        self._running = False

    async def start(self, emit: EventCallback) -> None:
        self._emit = emit
        if not self.config.enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="arxiv_poller")
        logger.info("ArxivPapersPlugin started for categories=%s", self.config.categories)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _build_query_url(self) -> str:
        cat_query = " OR ".join(f"cat:{c}" for c in self.config.categories)
        if self.config.keywords:
            kw_query = " OR ".join(f'all:"{kw}"' for kw in self.config.keywords)
            full_query = f"({cat_query}) AND ({kw_query})" if cat_query else kw_query
        else:
            full_query = cat_query or "cat:cs.AI"
        encoded = quote_plus(full_query)
        return (
            f"https://export.arxiv.org/api/query?search_query={encoded}"
            f"&sortBy=submittedDate&sortOrder=descending&max_results={self.config.max_results}"
        )

    async def poll_once(self) -> int:
        if self._emit is None:
            return 0
        url = self._build_query_url()
        emitted = 0
        try:
            async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            parsed = await asyncio.to_thread(feedparser.parse, resp.text)
            for entry in parsed.entries or []:
                paper_id = str(entry.get("id", "")).strip()
                if not paper_id or await self.storage.is_seen("arxiv", paper_id):
                    continue
                await self.storage.mark_seen("arxiv", paper_id)
                title = " ".join(str(entry.get("title", "")).split())
                abstract = " ".join(str(entry.get("summary", "")).split())
                authors = ", ".join(a.get("name", "") for a in (entry.get("authors") or [])[:5])
                link = str(entry.get("link", paper_id)).strip()
                text = (
                    f"[arXiv 新论文]\n标题：{title}\n作者：{authors}\n"
                    f"链接：{link}\n摘要：{abstract[:900]}"
                )
                await self._emit(
                    RawIncomingEvent(
                        source_plugin=self.name,
                        platform_type="arxiv",
                        platform_id="arxiv",
                        channel_type=ChannelType.PAPER,
                        channel_id=paper_id,
                        channel_name="arXiv",
                        sender_name=authors or "arXiv",
                        text=text,
                        timestamp=utc_now(),
                        raw_meta={"title": title, "link": link, "paper_id": paper_id},
                    )
                )
                emitted += 1
        except Exception as exc:
            logger.warning("arXiv poll failed: %s", exc)
        return emitted

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unexpected error in ArxivPapersPlugin loop")
            await asyncio.sleep(self.config.poll_interval_seconds)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "running": self._running,
            "categories": self.config.categories,
            "keywords": self.config.keywords,
        }
