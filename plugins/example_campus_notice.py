"""Example custom Information Router plugin for platforms OUTSIDE AstrBot.

How to activate:
  1. Copy this file to `plugins/campus_notice.py` (files prefixed with `example_` are skipped).
  2. Implement `start()` to poll or subscribe to your platform (e.g., university JWXT portal,
     IMAP email, Bilibili dynamic feed, GitHub notifications, Weibo, Xiaohongshu, etc.).
  3. Call `await emit(RawIncomingEvent(...))` whenever new content arrives.
  Life Agent's InformationRouter will auto-discover and start it on launch!
"""

from __future__ import annotations

import asyncio

from life_agent.config import AppConfig
from life_agent.models import ChannelType, RawIncomingEvent
from life_agent.router.base import EventCallback, SourcePlugin
from life_agent.storage import Storage


class CampusNoticePlugin(SourcePlugin):
    name = "campus_notice"
    description = "Example plugin polling a university academic affairs notice board."

    def __init__(self, config: AppConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self, emit: EventCallback) -> None:
        self._running = True
        self._task = asyncio.create_task(self._poll(emit))

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _poll(self, emit: EventCallback) -> None:
        while self._running:
            # Replace with your real HTTP fetch / parser
            # notice_id = "2026-11-18-01"
            # if not await self.storage.is_seen(self.name, notice_id):
            #     await self.storage.mark_seen(self.name, notice_id)
            #     await emit(RawIncomingEvent(
            #         source_plugin=self.name,
            #         platform_type="campus_portal",
            #         platform_id="jwxt",
            #         channel_type=ChannelType.NEWS,
            #         channel_id="jwxt_notices",
            #         channel_name="教务处通知",
            #         sender_name="教务处",
            #         text="关于本周四下午全校停课半天举行运动会的通知...",
            #     ))
            await asyncio.sleep(1800)


def create_plugin(config: AppConfig, storage: Storage) -> SourcePlugin:
    return CampusNoticePlugin(config, storage)
