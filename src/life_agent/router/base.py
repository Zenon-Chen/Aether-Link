"""Plugin contract for Information Router sources (AstrBot + arbitrary external platforms)."""

from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable
from typing import Any

from ..models import RawIncomingEvent

EventCallback = Callable[[RawIncomingEvent], Awaitable[None]]


class SourcePlugin(abc.ABC):
    """Base class for all Information Router source plugins.

    Any platform outside AstrBot can be integrated by subclassing ``SourcePlugin``
    and placing a ``.py`` file with a ``create_plugin(config, storage)`` factory
    function (or a ``SourcePlugin`` subclass) inside the configured ``plugins/`` directory.
    """

    name: str = "unnamed_plugin"
    description: str = ""

    @abc.abstractmethod
    async def start(self, emit: EventCallback) -> None:
        """Start listening/polling and call ``await emit(event)`` for every incoming message."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Gracefully stop background tasks and release resources."""

    def status(self) -> dict[str, Any]:
        """Return diagnostic status for dashboard/CLI inspection."""
        return {"name": self.name, "description": self.description, "running": True}
