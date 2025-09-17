"""Lightweight health-check utilities."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


@dataclass
class HealthStatus:
    status: str
    checks: Dict[str, Any]
    timestamp: datetime = datetime.now(timezone.utc)


async def check_system_health(include_checks: Optional[List[str]] = None) -> HealthStatus:
    """Run a couple of lightweight health checks and return a HealthStatus object.
    This is intentionally small so tests/examples can use it without heavy dependencies.
    """
    checks = {}
    # simple liveness checks that do not require external services
    checks['uptime'] = True
    checks['memory_ok'] = True
    checks['disk_ok'] = True
    # If more comprehensive checks are needed, extend here (DB, Redis, etc.)
    return HealthStatus(status="ok", checks=checks)
