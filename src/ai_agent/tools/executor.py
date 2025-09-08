"""
Executor de ferramentas.
Permite invocar funções registradas no catálogo.
"""

from typing import Any
from .tool_registry import ToolRegistry


class ToolExecutor:
    """
    Classe para executar ferramentas registradas.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, name: str, *args, **kwargs) -> Any:
        """
        Executa uma ferramenta pelo nome.
        """
        tool = self.registry.get(name)
        if not tool:
            raise ValueError(f"Ferramenta '{name}' não encontrada.")
        return tool(*args, **kwargs)
