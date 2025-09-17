"""Plugin system small exports."""

from .base_plugin import BasePlugin, PluginMeta
from .examples.file_manager import FileManagerPlugin
from .examples.web_scraper import WebScraperPlugin

__all__ = ["BasePlugin", "PluginMeta", "FileManagerPlugin", "WebScraperPlugin"]
