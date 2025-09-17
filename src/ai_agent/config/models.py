"""Pydantic configuration models used across the project."""

from pydantic import BaseModel, AnyUrl
from typing import Optional


class DBConfig(BaseModel):
    """Database configuration placeholder."""
    url: AnyUrl = "sqlite+aiosqlite:///./ai_agent.db"
    pool_min: int = 1
    pool_max: int = 5
    connect_timeout: int = 10


class ServiceConfig(BaseModel):
    """Service-level configuration."""
    name: str = "ai-agent"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True


class BaseConfig(BaseModel):
    """Aggregate configuration used by simple examples/tests."""
    db: DBConfig = DBConfig()
    service: ServiceConfig = ServiceConfig()
