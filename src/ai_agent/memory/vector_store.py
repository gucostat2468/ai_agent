"""
Armazenamento Vetorial.
Permite salvar e consultar embeddings para raciocínio semântico.
"""

from typing import Any, Dict, List, Tuple


class VectorStore:
    """
    Estrutura para armazenamento vetorial simples.
    """

    def __init__(self):
        self.vectors: Dict[str, Tuple[List[float], Any]] = {}

    def add(self, key: str, vector: List[float], metadata: Any = None) -> None:
        """
        Adiciona um vetor ao armazenamento.
        """
        self.vectors[key] = (vector, metadata)

    def get(self, key: str) -> Tuple[List[float], Any]:
        """
        Recupera um vetor pelo identificador.
        """
        return self.vectors.get(key, ([], None))

    def all_keys(self) -> List[str]:
        """
        Retorna todos os identificadores armazenados.
        """
        return list(self.vectors.keys())
