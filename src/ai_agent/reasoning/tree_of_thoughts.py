"""
Raciocínio em árvore (Tree of Thoughts).
Permite explorar múltiplos caminhos de raciocínio.
"""

from typing import Dict, List


class TreeOfThoughts:
    """
    Implementação básica de raciocínio em árvore.
    """

    def __init__(self):
        self.tree: Dict[str, List[str]] = {}

    def add_branch(self, parent: str, child: str) -> None:
        """
        Adiciona um novo ramo à árvore.
        """
        if parent not in self.tree:
            self.tree[parent] = []
        self.tree[parent].append(child)

    def get_children(self, node: str) -> List[str]:
        """
        Retorna os filhos de um nó.
        """
        return self.tree.get(node, [])

    def get_tree(self) -> Dict[str, List[str]]:
        """
        Retorna a árvore completa.
        """
        return self.tree
