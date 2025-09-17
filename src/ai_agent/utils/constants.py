"""Project-wide constants used in examples and tests."""

VERSION = "0.1.0-dev"

DEFAULT_CONFIG = {
    "service": {
        "name": "ai-agent",
        "host": "127.0.0.1",
        "port": 8000
    },
    "db": {
        "url": "sqlite+aiosqlite:///./ai_agent.db"
    }
}
