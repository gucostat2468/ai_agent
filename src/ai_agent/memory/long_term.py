"""
Memória de Longo Prazo.
Armazena conhecimento persistente do agente.
"""

from typing import List, Any


class LongTermMemory:
    """
    Implementação básica de memória persistente.
    """

    def __init__(self):
        self.storage: List[Any] = []

    def save(self, item: Any) -> None:
        """
        Salva um item na memória de longo prazo.
        """
        self.storage.append(item)

    def search(self, keyword: str) -> List[Any]:
        """
        Busca itens contendo uma palavra-chave.
        """
        return [item for item in self.storage if keyword in str(item)]

    def get_all(self) -> List[Any]:
        """
        Retorna todo o conhecimento armazenado.
        """
        return self.storage
