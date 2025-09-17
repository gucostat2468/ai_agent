"""Base classes for plugins."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PluginMeta:
    name: str
    version: str = "0.0.1"
    author: str = "unknown"
    description: str = ""


class BasePlugin(ABC):
    """Minimal plugin interface used by examples/tests."""

    meta: PluginMeta

    def __init__(self, meta: PluginMeta):
        self.meta = meta

    @abstractmethod
    def activate(self, context: Dict[str, Any]) -> None:
        """Activate plugin with provided context."""
        raise NotImplementedError

    @abstractmethod
    def deactivate(self) -> None:
        """Cleanup resources."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Run the plugin's primary action."""
        raise NotImplementedError
