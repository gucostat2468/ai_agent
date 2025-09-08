"""
Ferramentas internas relacionadas ao sistema operacional.
"""

import platform
import os


def system_info() -> dict:
    """
    Retorna informações básicas do sistema.
    """
    return {
        "os": platform.system(),
        "version": platform.version(),
        "release": platform.release(),
        "cpu": platform.processor(),
    }


def list_directory(path: str = ".") -> list[str]:
    """
    Lista arquivos e diretórios em um caminho.
    """
    return os.listdir(path)
