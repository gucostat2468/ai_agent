"""
Interface para integração com fontes de dados.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class DataInterface(ABC):
    """
    Contrato para provedores de dados (APIs, bancos, arquivos, etc.)
    """

    @abstractmethod
    def fetch(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Recupera dados com base em uma consulta.
        """
        pass

    @abstractmethod
    def save(self, data: Dict[str, Any], **kwargs) -> bool:
        """
        Salva dados na fonte de dados.
        """
        pass

    @abstractmethod
    def delete(self, identifier: str, **kwargs) -> bool:
        """
        Remove dados da fonte de dados.
        """
        pass
