"""Construction of derivation steps.

``make_step`` is the single gate through which every displayed step passes. All of the step's claims are
verified first; a failed claim raises :class:`UnjustifiedStepError`, so an invented or wrong transformation can
never reach the user.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import sympy as sp

from .errors import UnjustifiedStepError
from .models import CheckResult, Claim, Solution, Step, Verification
from .render import Msg, as_msg
from .verify import check_claim, combine


def Int(f, x) -> sp.Integral:
    """An unevaluated indefinite integral (used to display the state of a derivation)."""
    return sp.Integral(f, x)


def make_step(rule: str, title, before, after, explanation, *, conditions: Iterable = (),
              claims: Sequence[Claim] = (), substeps: Iterable[Step] = (), table: Optional[dict] = None,
              formula: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
              trust: Optional[CheckResult] = None, symbolic_timeout: float = 1.2) -> Step:
    """Create a step after verifying every claim it makes."""
    results: List[CheckResult] = []
    for c in claims:
        r = check_claim(c, symbolic_timeout)
        if r.status is Verification.FAILED:
            raise UnjustifiedStepError(f"step '{rule}' could not be justified "
                                       f"({c.description or c.kind}): {r.detail}")
        results.append(r)
    if trust is not None:
        results.append(trust)
    agg = combine(results)
    return Step(rule=rule, title=as_msg(title), before=before, after=after, explanation=as_msg(explanation),
                conditions=[as_msg(c) for c in conditions], verification=agg.status, check_detail=agg.detail,
                claims=list(claims), substeps=list(substeps), table=table, formula=formula, meta=meta or {})


def wrap_solution_steps(f, x, sol: Solution, title: Optional[Any] = None) -> List[Step]:
    """Present the steps of a sub-integral: inline when it is a single step, nested otherwise."""
    if len(sol.steps) == 1:
        return list(sol.steps)
    ttl = title if title is not None else Msg("Evaluate the integral {I}", I=Int(f, x))
    return [make_step("sub_integral", ttl, Int(f, x), sol.F,
                      Msg("This integral is evaluated in the sub-steps below."),
                      substeps=sol.steps, trust=sol.verification or CheckResult(Verification.UNCHECKED))]


def merge_conditions(*groups: Iterable[Msg]) -> List[Msg]:
    seen, out = set(), []
    for g in groups:
        for c in g:
            key = c.text()
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out


def merge_levels(*sols: Solution) -> str:
    levels = [s.derivation_level for s in sols]
    if all(l == "full" for l in levels):
        return "full"
    if all(l == "result_only" for l in levels):
        return "result_only"
    return "partial"


def split_constant(expr, var):
    """Split ``expr = c * rest`` with ``c`` free of ``var``."""
    c, rest = sp.sympify(expr).as_independent(var, as_Add=False)
    return c, rest
