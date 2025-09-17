"""Input validation and sanitization helpers."""

import html
from typing import Any


def sanitize_input(text: Any) -> str:
    """A minimal sanitizer suitable for examples/tests.
    Note: this is NOT secure for untrusted HTML/JS; use a robust library (bleach) in production.
    """
    if text is None:
        return ""
    # coerce to str and escape HTML to avoid simple injection in demos
    return html.escape(str(text))
