"""
Formatador de respostas do agente.
Responsável por estruturar as saídas em diferentes formatos.
"""

from typing import Dict


class ResponseFormatter:
    """
    Classe para formatar respostas em múltiplos estilos.
    """

    def format_text(self, message: str) -> str:
        """
        Formata a resposta como texto simples.
        """
        return message.strip()

    def format_json(self, message: str) -> Dict[str, str]:
        """
        Formata a resposta como JSON.
        """
        return {"response": message.strip()}

    def format_markdown(self, message: str) -> str:
        """
        Formata a resposta em Markdown.
        """
        return f"**Resposta:** {message.strip()}"
