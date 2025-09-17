"""A simple in-memory experience buffer for storing agent experiences."""

from typing import Any, List, Dict, Optional
import random
import threading
import time


class ExperienceBuffer:
    """Thread-safe ring-buffer-like experience storage for small-scale experiments."""

    def __init__(self, capacity: int = 1000):
        self.capacity = int(capacity)
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def add(self, experience: Dict[str, Any]) -> None:
        with self._lock:
            if len(self._buffer) >= self.capacity:
                # drop oldest
                self._buffer.pop(0)
            experience.setdefault("timestamp", time.time())
            self._buffer.append(experience)

    def sample(self, n: int = 1) -> List[Dict[str, Any]]:
        with self._lock:
            if n <= 0:
                return []
            n = min(n, len(self._buffer))
            return random.sample(self._buffer, n)

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._buffer)

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)
