"""
Interface para integração de ferramentas externas.
"""

from abc import ABC, abstractmethod
from typing import Any


class ToolInterface(ABC):
    """
    Contrato para ferramentas que o agente pode executar.
    """

    @abstractmethod
    def run(self, command: str, **kwargs) -> Any:
        """
        Executa a ferramenta com um comando específico.
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """
        Retorna metadados sobre a ferramenta.
        """
        pass
