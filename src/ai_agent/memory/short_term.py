"""
Memória de Curto Prazo.
Usada para armazenar informações relevantes no contexto imediato.
"""

from typing import List, Any


class ShortTermMemory:
    """
    Armazena e recupera informações de curto prazo.
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.buffer: List[Any] = []

    def add(self, item: Any) -> None:
        """
        Adiciona um item à memória.
        Remove o mais antigo se ultrapassar a capacidade.
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)

    def get_all(self) -> List[Any]:
        """
        Retorna todos os itens armazenados.
        """
        return self.buffer

    def clear(self) -> None:
        """
        Limpa a memória.
        """
        self.buffer = []
