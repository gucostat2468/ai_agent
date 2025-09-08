"""
Ferramentas internas para interações com a Web.
"""

import requests


def fetch_url(url: str) -> str:
    """
    Faz uma requisição HTTP GET e retorna o conteúdo como texto.
    """
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.text
