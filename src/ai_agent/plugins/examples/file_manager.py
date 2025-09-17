"""Simple file-manager plugin used in examples/tests."""

from typing import Dict, Any, List
from pathlib import Path
from ..base_plugin import BasePlugin, PluginMeta


class FileManagerPlugin(BasePlugin):
    """Provides a safe subset of file operations under a given root directory."""

    def __init__(self, root: str = ".", meta: PluginMeta = None):
        super().__init__(meta or PluginMeta(name="file_manager", description="Basic file manager plugin"))
        self.root = Path(root).resolve()

    def _safe_path(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if not str(candidate).startswith(str(self.root)):
            raise PermissionError("Access outside of plugin root is not allowed")
        return candidate

    def activate(self, context: Dict[str, Any]) -> None:
        # no-op for now
        return

    def deactivate(self) -> None:
        return

    def list_files(self, path: str = "") -> List[str]:
        p = self._safe_path(path)
        if not p.exists():
            return []
        return [str(x.relative_to(self.root)) for x in p.iterdir() if x.is_file()]

    def read_file(self, path: str) -> str:
        p = self._safe_path(path)
        return p.read_text(encoding='utf-8')
