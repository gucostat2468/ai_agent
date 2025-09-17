"""Tiny web-scraper plugin (synchronous) for demonstration/testing only."""

from typing import Dict, Any
from ..base_plugin import BasePlugin, PluginMeta
import requests


class WebScraperPlugin(BasePlugin):
    def __init__(self, meta: PluginMeta = None):
        super().__init__(meta or PluginMeta(name="web_scraper", description="Basic web scraper"))

    def activate(self, context: Dict[str, Any]) -> None:
        return

    def deactivate(self) -> None:
        return

    def fetch(self, url: str, timeout: int = 5) -> Dict[str, Any]:
        """Fetch URL and return a simple payload. In production, respect robots.txt and rate limits."""
        r = requests.get(url, timeout=timeout)
        return {
            "status_code": r.status_code,
            "url": r.url,
            "content": r.text[:4000],  # limit size in examples
            "headers": dict(r.headers),
        }
