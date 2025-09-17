"""Prometheus metrics helpers with graceful fallback if prometheus_client isn't installed."""

from typing import Optional, Callable, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST, start_http_server
    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False


_metrics = {}


def setup_prometheus_metrics(port: int = 8001) -> Optional[Dict[str, object]]:
    """Start a prometheus exposition server (best-effort). Returns dict of created metrics or None."""
    if not _PROM_AVAILABLE:
        logger.debug("prometheus_client not available; metrics disabled")
        return None
    # minimal metrics used by the framework
    _metrics['requests_total'] = Counter('ai_agent_requests_total', 'Total requests to the agent')
    _metrics['active_tasks'] = Gauge('ai_agent_active_tasks', 'Number of active tasks')
    try:
        start_http_server(port)
    except Exception:
        logger.debug("failed to start prometheus http server on port %s", port)
    return _metrics


def get_metrics_handler() -> Callable[[], bytes]:
    """Return a handler function that returns Prometheus metrics in bytes.
    If prometheus_client is not present, returns a no-op handler.
    """
    if not _PROM_AVAILABLE:
        def _noop():
            return b"# metrics disabled (prometheus_client not installed)\n"
        return _noop

    def _handler():
        # generate_latest() returns bytes
        return generate_latest()
    return _handler
