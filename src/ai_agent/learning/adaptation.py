"""Adaptation module with lightweight parameter tuning and logging."""

from typing import Dict, Any
import logging


class AdaptationModule:
    """Simple adaptation module that tweaks parameters based on feedback."""

    def __init__(self, initial_params: Dict[str, Any] = None):
        self.params = initial_params or {"temperature": 0.7, "max_tokens": 512}
        self.logger = logging.getLogger(__name__)

    def adapt(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt parameters using a naive strategy based on feedback dict.
        Example feedback: {"too_long": True, "low_quality": False}
        """
        if feedback.get("too_long"):
            self.params["max_tokens"] = max(64, int(self.params.get("max_tokens", 512) * 0.8))
            self.logger.debug("Adaptation: reduced max_tokens to %s", self.params["max_tokens"])
        if feedback.get("low_quality"):
            # increase temperature slightly to encourage more diverse outputs
            self.params["temperature"] = min(1.0, float(self.params.get("temperature", 0.7)) + 0.05)
            self.logger.debug("Adaptation: increased temperature to %s", self.params["temperature"])
        return self.params
