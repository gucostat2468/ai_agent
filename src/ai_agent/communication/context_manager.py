"""
Gerenciador de contexto.
Mantém o estado da conversa para continuidade e coerência.
"""

from typing import Dict, Any


class ContextManager:
    """
    Classe que mantém e atualiza o contexto da interação.
    """

    def __init__(self):
        self.context: Dict[str, Any] = {}

    def update(self, key: str, value: Any) -> None:
        """
        Atualiza ou adiciona uma informação ao contexto.
        """
        self.context[key] = value

    def get(self, key: str) -> Any:
        """
        Recupera uma informação do contexto.
        """
        return self.context.get(key)

    def reset(self) -> None:
        """
        Limpa o contexto atual.
        """
        self.context = {}
