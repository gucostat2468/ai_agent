"""
Auto-reflexão do agente.
Permite avaliar ações passadas e aprender com elas.
"""

from typing import List


class Reflection:
    """
    Implementação básica de auto-reflexão.
    """

    def __init__(self):
        self.journal: List[str] = []

    def record(self, experience: str) -> None:
        """
        Registra uma experiência para reflexão.
        """
        self.journal.append(experience)

    def review(self) -> List[str]:
        """
        Retorna todas as reflexões registradas.
        """
        return self.journal

    def latest(self) -> str:
        """
        Retorna a última reflexão registrada.
        """
        return self.journal[-1] if self.journal else ""
