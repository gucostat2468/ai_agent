"""
Interface para integração com LLMs (Large Language Models).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMInterface(ABC):
    """
    Contrato para conectores de modelos de linguagem.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera uma resposta a partir de um prompt.
        """
        pass

    @abstractmethod
    def embed(self, text: str, **kwargs) -> Any:
        """
        Gera embeddings para um texto.
        """
        pass

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo.
        """
        pass
