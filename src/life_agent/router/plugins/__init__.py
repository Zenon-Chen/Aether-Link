"""Built-in Information Router source plugins."""

from .arxiv_papers import ArxivPapersPlugin
from .astrbot_bridge import AstrBotBridgePlugin
from .rss_news import RSSNewsPlugin
from .webhook_source import WebhookSourcePlugin

__all__ = [
    "ArxivPapersPlugin",
    "AstrBotBridgePlugin",
    "RSSNewsPlugin",
    "WebhookSourcePlugin",
]
