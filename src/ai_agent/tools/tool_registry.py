"""
Registro de ferramentas disponíveis para o agente.
"""

from typing import Callable, Dict


class ToolRegistry:
    """
    Mantém um catálogo de ferramentas registradas.
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        """
        Registra uma nova ferramenta no catálogo.
        """
        self._tools[name] = func

    def get(self, name: str) -> Callable:
        """
        Recupera uma ferramenta pelo nome.
        """
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, Callable]:
        """
        Retorna todas as ferramentas registradas.
        """
        return self._tools
