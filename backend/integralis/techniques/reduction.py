"""Reduction formulae for  ∫ f(θ)^n dx  with f in {sin, cos, tan, cot, sec, csc} and θ linear in x.

For every use, the instance of the recurrence is *verified* by checking  d/dx(closed part) + coef·(next integrand)
== integrand, so the displayed formula is never taken on trust. Repeated application is the engine's recursion.
"""
from __future__ import annotations

from typing import Optional

import sympy as sp

from ..derivation import Int, make_step, split_constant, wrap_solution_steps
from ..errors import TechniqueFailure
from ..models import Claim, Solution
from ..render import Msg
from ..strategy.features import Features
from .base import Context, Technique
from .table import linear_form

_LATEX = {
    sp.sin: r"\int\sin^{n}\theta\,d\theta=-\frac{\sin^{n-1}\theta\cos\theta}{n}+\frac{n-1}{n}\int\sin^{n-2}\theta\,d\theta",
    sp.cos: r"\int\cos^{n}\theta\,d\theta=\frac{\cos^{n-1}\theta\sin\theta}{n}+\frac{n-1}{n}\int\cos^{n-2}\theta\,d\theta",
    sp.tan: r"\int\tan^{n}\theta\,d\theta=\frac{\tan^{n-1}\theta}{n-1}-\int\tan^{n-2}\theta\,d\theta",
    sp.cot: r"\int\cot^{n}\theta\,d\theta=-\frac{\cot^{n-1}\theta}{n-1}-\int\cot^{n-2}\theta\,d\theta",
    sp.sec: r"\int\sec^{n}\theta\,d\theta=\frac{\sec^{n-2}\theta\tan\theta}{n-1}+\frac{n-2}{n-1}\int\sec^{n-2}\theta\,d\theta",
    sp.csc: r"\int\csc^{n}\theta\,d\theta=-\frac{\csc^{n-2}\theta\cot\theta}{n-1}+\frac{n-2}{n-1}\int\csc^{n-2}\theta\,d\theta",
}


def _build(fn, th, a, n):
    """Return (closed, coef, next_integrand) with  ∫ fn(th)^n dx = closed + coef * ∫ next dx."""
    if fn is sp.sin:
        return -sp.sin(th) ** (n - 1) * sp.cos(th) / (a * n), sp.Rational(n - 1, n), sp.sin(th) ** (n - 2)
    if fn is sp.cos:
        return sp.cos(th) ** (n - 1) * sp.sin(th) / (a * n), sp.Rational(n - 1, n), sp.cos(th) ** (n - 2)
    if fn is sp.tan:
        return sp.tan(th) ** (n - 1) / (a * (n - 1)), sp.Integer(-1), sp.tan(th) ** (n - 2)
    if fn is sp.cot:
        return -sp.cot(th) ** (n - 1) / (a * (n - 1)), sp.Integer(-1), sp.cot(th) ** (n - 2)
    if fn is sp.sec:
        return sp.sec(th) ** (n - 2) * sp.tan(th) / (a * (n - 1)), sp.Rational(n - 2, n - 1), sp.sec(th) ** (n - 2)
    if fn is sp.csc:
        return -sp.csc(th) ** (n - 2) * sp.cot(th) / (a * (n - 1)), sp.Rational(n - 2, n - 1), sp.csc(th) ** (n - 2)
    raise ValueError(fn)


class ReductionFormulae(Technique):
    name = "reduction"
    title = "Reduction formulae"
    family = "reduction"
    description = ("Recurrences for ∫sinⁿθ, ∫cosⁿθ, ∫tanⁿθ, ∫cotⁿθ, ∫secⁿθ and ∫cscⁿθ that lower the power by one or two "
                   "at each application until a standard integral remains. Each instance of the recurrence is verified "
                   "by differentiation before it is shown.")

    @staticmethod
    def _match(f, x):
        c, rest = split_constant(f, x)
        if c != 1:
            return None
        base, e = rest.as_base_exp()
        if e.is_Integer and e <= -3 and isinstance(base, (sp.cos, sp.sin)):
            # 1/cos^n and 1/sin^n are sec^n and csc^n
            base, e = (sp.sec if isinstance(base, sp.cos) else sp.csc)(base.args[0]), -e
        if not (e.is_Integer and e >= 2):
            return None
        fn = type(base)
        if fn not in _LATEX:
            return None
        th = base.args[0]
        lf = linear_form(th, x)
        if lf is None:
            return None
        n = int(e)
        if fn in (sp.sec, sp.csc) and n < 3:
            return None
        return fn, th, lf[0], n

    def suitability(self, feats: Features):
        m = self._match(feats.expr, feats.x)
        if m is None:
            return None
        fn, _, _, n = m
        if fn in (sp.sec, sp.csc):
            return 0.9, f"A power of {fn.__name__}: use the reduction formula."
        if fn in (sp.sin, sp.cos) and n % 2 == 0 and n >= 4:
            return 0.86, f"An even power of {fn.__name__}: the reduction formula lowers the power by 2 each time."
        return 0.7, f"A power of {fn.__name__}: a reduction formula applies."

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        m = self._match(f, x)
        if m is None:
            return None
        fn, th, a, n = m
        closed, coef, nxt = _build(fn, th, a, n)
        after = closed + coef * Int(nxt, x)
        step = make_step(
            "reduction_formula", Msg("Apply the reduction formula (n = {n})", n=str(n)), Int(f, x), after,
            Msg("With n = {n} the reduction formula expresses the integral through the same kind of integral with "
                "power {m}.", n=str(n), m=str(n - 2)),
            claims=[Claim.identity(sp.diff(closed, x) + coef * nxt, f, x,
                                   "differentiating the closed part and adding coef·(next integrand) gives the integrand")],
            formula=_LATEX[fn])
        sub = ctx.solve_sub(nxt, x)
        if sub is None:
            raise TechniqueFailure("the reduced integral could not be solved")
        F = closed + coef * sub.F
        steps = [step] + wrap_solution_steps(nxt, x, sub)
        steps.append(make_step("reduction_combine", Msg("Combine"), after, F,
                               Msg("Insert the value of the reduced integral."),
                               claims=[Claim.derivative(F, f, x, "the result differentiates to the integrand")]))
        return Solution(F, steps, self.name, sub.conditions, derivation_level=sub.derivation_level)
