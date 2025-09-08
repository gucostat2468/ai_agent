"""
Raciocínio sequencial (Chain of Thought).
Permite que o agente desenvolva respostas passo a passo.
"""

from typing import List


class ChainOfThought:
    """
    Implementação simples de raciocínio em cadeia.
    """

    def __init__(self):
        self.steps: List[str] = []

    def add_step(self, thought: str) -> None:
        """
        Adiciona um passo ao raciocínio.
        """
        self.steps.append(thought)

    def get_full_reasoning(self) -> str:
        """
        Retorna o raciocínio completo concatenado.
        """
        return " -> ".join(self.steps)

    def clear(self) -> None:
        """
        Limpa a cadeia de raciocínio.
        """
        self.steps = []
