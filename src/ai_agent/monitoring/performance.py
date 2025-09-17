"""Performance measurement helpers (timers, decorators)."""

import time
import functools
from typing import Callable, Any, Dict
import logging

logger = logging.getLogger(__name__)


def measure_time(func: Callable) -> Callable:
    """Decorator to measure execution time of functions (sync or async)."""
    if hasattr(func, "__call__"):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            try:
                logger.debug("%s executed in %.4fs", func.__name__, elapsed)
            except Exception:
                pass
            return result
        return _wrapper
    return func


class PerformanceTracker:
    def __init__(self):
        self.records = []

    def record(self, name: str, duration: float, metadata: Dict[str, Any] = None):
        self.records.append({"name": name, "duration": duration, "metadata": metadata or {}})
