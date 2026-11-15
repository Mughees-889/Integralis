"""Linearity: split sums and pull constant factors out of the integral."""
from __future__ import annotations

from typing import Optional

import sympy as sp

from ..derivation import Int, make_step, merge_conditions, merge_levels, wrap_solution_steps
from ..errors import TechniqueFailure
from ..models import Claim, Solution
from ..render import Msg
from ..strategy.features import Features
from .base import Context, Technique


class Linearity(Technique):
    name = "linearity"
    title = "Linearity (sum rule and constant multiples)"
    family = "basic"
    description = ("Integrate a sum term by term and take constant factors outside the integral. "
                   "The resulting integrals are solved recursively by the other techniques.")

    def suitability(self, feats: Features):
        f, x = feats.expr, feats.x
        if f.is_Add and len(f.args) >= 2:
            return 1.0, "The integrand is a sum, so it can be integrated term by term."
        c, rest = f.as_independent(x, as_Add=False)
        if f.has(x) and c != 1 and rest != 1:
            return 1.0, "A constant factor can be taken outside the integral."
        return None

    def attempt(self, f: sp.Expr, x: sp.Symbol, ctx: Context) -> Optional[Solution]:
        if f.is_Add:
            return self._sum(f, x, ctx)
        c, rest = f.as_independent(x, as_Add=False)
        if c == 1 or rest == 1 or not f.has(x):
            return None
        return self._constant(f, c, rest, x, ctx)

    # ------------------------------------------------------------------------------------------------
    def _sum(self, f, x, ctx) -> Solution:
        terms = list(f.args)
        pieces = [Int(t, x) for t in terms]
        steps = [make_step(
            "sum_rule", Msg("Sum rule"), Int(f, x), sp.Add(*pieces),
            Msg("The integral of a sum is the sum of the integrals, so integrate each of the {n} terms separately.",
                n=str(len(terms))),
            claims=[Claim.identity(f, sp.Add(*terms), x, "the terms add up to the integrand")],
            formula=r"\int (g+h)\,dx=\int g\,dx+\int h\,dx")]
        subs = []
        for t in terms:
            sub = ctx.solve_sub(t, x)
            if sub is None:
                raise TechniqueFailure(f"the term {t} could not be integrated")
            subs.append(sub)
            steps.extend(wrap_solution_steps(t, x, sub))
        F = sp.Add(*[s.F for s in subs])
        steps.append(make_step(
            "combine", Msg("Combine the results"), sp.Add(*pieces), F,
            Msg("Add the antiderivatives found for the individual terms."),
            claims=[Claim.derivative(F, f, x, "the sum of the antiderivatives differentiates to the integrand")]))
        return Solution(F, steps, self.name, merge_conditions(*[s.conditions for s in subs]),
                        derivation_level=merge_levels(*subs))

    def _constant(self, f, c, rest, x, ctx) -> Solution:
        steps = [make_step(
            "constant_multiple", Msg("Constant multiple rule"), Int(f, x), c * Int(rest, x),
            Msg("A constant factor can be taken outside the integral. Here the constant is {c}.", c=c),
            claims=[Claim.identity(f, c * rest, x, "the constant times the rest is the integrand")],
            formula=r"\int c\,g(x)\,dx=c\int g(x)\,dx")]
        sub = ctx.solve_sub(rest, x)
        if sub is None:
            raise TechniqueFailure(f"{rest} could not be integrated")
        steps.extend(wrap_solution_steps(rest, x, sub))
        F = c * sub.F
        steps.append(make_step(
            "constant_multiple_result", Msg("Restore the constant factor"), c * Int(rest, x), F,
            Msg("Multiply the antiderivative by the constant {c}.", c=c),
            claims=[Claim.derivative(F, f, x, "the scaled antiderivative differentiates to the integrand")]))
        return Solution(F, steps, self.name, sub.conditions, derivation_level=sub.derivation_level)
