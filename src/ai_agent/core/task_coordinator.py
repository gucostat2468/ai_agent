"""
Task Coordinator - Coordenação de tarefas do agente.
Responsável por organizar, priorizar e despachar tarefas para execução.
"""

from typing import List, Dict


class TaskCoordinator:
    """
    Classe que coordena as tarefas do agente.
    """

    def __init__(self):
        self.tasks: List[Dict] = []

    def add_task(self, task: Dict) -> None:
        """
        Adiciona uma nova tarefa à fila.

        Args:
            task (Dict): Representação da tarefa (id, descrição, prioridade, etc.)
        """
        self.tasks.append(task)

    def get_next_task(self) -> Dict:
        """
        Retorna a próxima tarefa a ser executada, considerando prioridade.
        """
        if not self.tasks:
            return {}

        # TODO: Implementar lógica de prioridade
        return self.tasks.pop(0)

    def has_tasks(self) -> bool:
        """
        Verifica se ainda há tarefas na fila.
        """
        return len(self.tasks) > 0
