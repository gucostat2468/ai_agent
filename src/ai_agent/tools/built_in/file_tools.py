"""
Ferramentas internas para manipulação de arquivos locais.
"""

from pathlib import Path


def read_file(path: str) -> str:
    """
    Lê o conteúdo de um arquivo de texto.
    """
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8") if file_path.exists() else ""


def write_file(path: str, content: str) -> None:
    """
    Escreve conteúdo em um arquivo de texto.
    """
    file_path = Path(path)
    file_path.write_text(content, encoding="utf-8")
