"""Public interface for ECHO and its paper ablations."""

from .proposal import (
    ECHOWrapper,
    wrap_adapter,
    find_echo,
    configure_from_note,
)


__all__ = [
    "ECHOWrapper",
    "wrap_adapter",
    "find_echo",
    "configure_from_note",
]
