"""
Processador de mensagens do agente.
Responsável por interpretar entradas e transformá-las em ações ou consultas.
"""

from typing import Dict, Any


class MessageHandler:
    """
    Classe para lidar com mensagens recebidas.
    """

    def __init__(self):
        self.history: list[Dict[str, Any]] = []

    def process(self, message: str, sender: str = "user") -> Dict[str, Any]:
        """
        Processa uma mensagem e retorna um objeto estruturado.
        """
        msg_obj = {"sender": sender, "content": message}
        self.history.append(msg_obj)
        return msg_obj

    def get_history(self) -> list[Dict[str, Any]]:
        """
        Retorna o histórico de mensagens.
        """
        return self.history
