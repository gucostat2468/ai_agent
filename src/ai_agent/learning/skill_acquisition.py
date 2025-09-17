"""Very small skill acquisition utilities used by examples and tests."""

from typing import List, Dict, Any


class SkillAcquisition:
    """Prototype of a skill-acquisition component. Uses frequency-based learning for examples."""

    def __init__(self):
        self.skills = {}

    def learn_from_experiences(self, experiences: List[Dict[str, Any]]) -> Dict[str, int]:
        """Update internal skill counts from a batch of experiences.
        Each experience can include a 'skill' key identifying the skill involved.
        Returns the updated skill frequency mapping.
        """
        for e in experiences:
            skill = e.get("skill") or e.get("intent") or "generic"
            self.skills.setdefault(skill, 0)
            self.skills[skill] += 1
        return dict(self.skills)

    def get_skills(self) -> Dict[str, int]:
        return dict(self.skills)
