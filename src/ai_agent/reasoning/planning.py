"""
Módulo de planejamento.
Permite que o agente defina objetivos e planos de ação.
"""

from typing import List, Dict


class Planner:
    """
    Planejador básico de tarefas.
    """

    def __init__(self):
        self.goals: List[str] = []
        self.plan: List[Dict[str, str]] = []

    def add_goal(self, goal: str) -> None:
        """
        Adiciona um novo objetivo.
        """
        self.goals.append(goal)

    def create_plan(self, steps: List[str]) -> None:
        """
        Gera um plano a partir de uma lista de passos.
        """
        self.plan = [{"step": step, "status": "pending"} for step in steps]

    def mark_done(self, step: str) -> None:
        """
        Marca um passo do plano como concluído.
        """
        for task in self.plan:
            if task["step"] == step:
                task["status"] = "done"

    def get_plan(self) -> List[Dict[str, str]]:
        """
        Retorna o plano atual.
        """
        return self.plan
