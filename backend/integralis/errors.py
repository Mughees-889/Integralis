"""Exception hierarchy for INTEGRALIS."""
from __future__ import annotations

from typing import Optional


class IntegralisError(Exception):
    """Base class for all errors raised deliberately by the engine."""


class ParseError(IntegralisError):
    """User input could not be parsed. Carries a position for precise feedback."""

    def __init__(self, message: str, position: Optional[int] = None, hint: Optional[str] = None,
                 source: Optional[str] = None, length: int = 1):
        super().__init__(message)
        self.message = message
        self.position = position
        self.hint = hint
        self.source = source
        self.length = max(1, length)

    def to_dict(self) -> dict:
        return {"message": self.message, "position": self.position, "length": self.length,
                "hint": self.hint, "source": self.source}

    def __str__(self) -> str:
        out = self.message
        if self.source is not None and self.position is not None:
            out += f"\n  {self.source}\n  {' ' * self.position}{'^' * self.length}"
        if self.hint:
            out += f"\n  hint: {self.hint}"
        return out


class UnjustifiedStepError(IntegralisError):
    """A derivation step could not be verified, so it must never be shown to the user."""


class TechniqueFailure(IntegralisError):
    """A technique recognised the problem shape but could not complete it."""


class EngineTimeout(IntegralisError):
    """A soft time limit expired."""
