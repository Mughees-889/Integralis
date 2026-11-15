"""Technique interface and the services the engine offers to techniques."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set, Tuple

import sympy as sp

from ..config import EngineOptions
from ..models import Solution
from ..strategy.features import Features

if TYPE_CHECKING:  # pragma: no cover
    from ..engine import Engine


class Technique:
    """One integration technique.

    ``suitability`` inspects structural features only and returns ``(score, reason)`` or ``None``.
    ``attempt`` builds a *verified* derivation: it returns a :class:`Solution`, ``None`` when the technique does
    not apply after all, or raises ``TechniqueFailure`` / ``UnjustifiedStepError`` when it cannot be completed.
    To add a technique, subclass this and register it in ``techniques/__init__.py`` (see docs/EXTENDING.md).
    """

    name: str = "technique"
    title: str = ""
    family: str = ""
    description: str = ""

    def suitability(self, feats: Features) -> Optional[Tuple[float, str]]:  # pragma: no cover - interface
        return None

    def attempt(self, f: sp.Expr, x: sp.Symbol, ctx: "Context") -> Optional[Solution]:  # pragma: no cover
        raise NotImplementedError


_NAMES = ["u", "v", "w", "t", "s", "r", "p", "q"]


class Context:
    """Services for techniques: recursive sub-integration, fresh variable names, options."""

    def __init__(self, engine: "Engine", depth: int):
        self.engine = engine
        self.depth = depth

    @property
    def options(self) -> EngineOptions:
        return self.engine.options

    def solve_sub(self, f: sp.Expr, x: sp.Symbol) -> Optional[Solution]:
        """Integrate a simpler integral produced by this technique (cycle- and depth-protected)."""
        return self.engine.integrate(f, x, self.depth + 1)

    def fresh_symbol(self, *avoid: sp.Basic) -> sp.Symbol:
        taken: Set[str] = set()
        for a in avoid:
            taken |= {s.name for s in sp.sympify(a).free_symbols}
            if isinstance(a, sp.Symbol):
                taken.add(a.name)
        for n in _NAMES:
            if n not in taken:
                return sp.Symbol(n, real=True)
        k = 1
        while f"u_{k}" in taken:
            k += 1
        return sp.Symbol(f"u_{k}", real=True)
