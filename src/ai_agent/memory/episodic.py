"""
Memória Episódica.
Registra eventos e experiências organizados em episódios.
"""

from typing import Dict, List, Any


class EpisodicMemory:
    """
    Estrutura para armazenamento de episódios da experiência do agente.
    """

    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []

    def add_episode(self, episode: Dict[str, Any]) -> None:
        """
        Adiciona um novo episódio à memória.
        """
        self.episodes.append(episode)

    def get_latest(self) -> Dict[str, Any]:
        """
        Retorna o episódio mais recente.
        """
        return self.episodes[-1] if self.episodes else {}

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Retorna todos os episódios registrados.
        """
        return self.episodes
