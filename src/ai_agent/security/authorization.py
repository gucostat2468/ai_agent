"""Very small authorization helpers (policy stubs)."""

from typing import Dict, Any


def has_permission(identity: Dict[str, Any], action: str, resource: str) -> bool:
    """Basic permission check.
    identity can be a dict with roles e.g. {'roles': ['admin']}.
    This is a stub that grants permission to 'admin' roles or for 'read' actions.
    """
    if not identity:
        return False
    roles = identity.get("roles", []) or []
    if "admin" in roles:
        return True
    if action in ("read", "list"):
        return True
    # extend with real RBAC in production
    return False
