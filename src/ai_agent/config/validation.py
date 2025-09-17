"""Simple validation helpers for configuration."""

from typing import Type, Dict, Any
from pydantic import BaseModel, ValidationError


def validate_config(model: Type[BaseModel], data: Dict[str, Any]) -> BaseModel:
    """Validate a dict against a pydantic model and return parsed instance.
    Raises ValueError with details if validation fails.
    """
    try:
        return model.parse_obj(data)
    except ValidationError as e:
        raise ValueError(f"Configuration validation failed: {e}") from e
