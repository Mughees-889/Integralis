"""Integration by parts: ordinary (LIATE), tabular, and cyclic (e^{ax}·sin/cos).

Every use of the formula  ∫u dv = uv − ∫v du  is justified by three verified facts: u·dv equals the integrand,
dv integrates to v (v' = dv), and du = u'. Repeated application arises naturally through recursion.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import sympy as sp

from ..derivation import Int, make_step, merge_conditions, split_constant, wrap_solution_steps
from ..errors import TechniqueFailure
from ..models import Claim, Solution
from ..render import Msg, latex
from ..strategy.features import Features, factor_kind
from .base import Context, Technique
from .table import linear_form

RANK = {"log": 0, "invtrig": 1, "algebraic": 2, "trig": 3, "hyp": 3, "exp": 4}


def split_options(f, x) -> List[Tuple[sp.Expr, sp.Expr]]:
    """Candidate (u, dv) pairs ordered by the LIATE rule."""
    c, rest = split_constant(f, x)
    factors = [a for a in sp.Mul.make_args(rest) if a.has(x)]
    if not factors:
        return []
    # a negative power of a log / inverse-trig function (e.g. 1/ln x) never becomes simpler under differentiation
    if any(a.is_Pow and a.exp.is_negative and factor_kind(a, x) in ("log", "invtrig") for a in factors):
        return []
    if len(factors) == 1:
        k = factor_kind(factors[0], x)
        if k in ("log", "invtrig"):
            return [(c * factors[0], sp.Integer(1))]
        return []
    kinds = [(factor_kind(a, x), a) for a in factors]
    if any(k not in RANK for k, _ in kinds):
        return []
    if all(k == "algebraic" for k, _ in kinds):  # polynomial × polynomial: expand instead of integrating by parts
        return []
    ranked = sorted(kinds, key=lambda t: RANK[t[0]])
    u = ranked[0][1]
    dv = c * sp.Mul(*[a for k, a in ranked[1:]])
    out = [(u, dv)]
    if len(ranked) == 2 and RANK[ranked[0][0]] != RANK[ranked[1][0]]:
        # the swapped choice is a legitimate alternative (e.g. polynomial times exponential)
        out.append((ranked[1][1] * c, ranked[0][1]))
    return out


class IntegrationByParts(Technique):
    name = "parts"
    title = "Integration by parts"
    family = "parts"
    description = ("∫u dv = uv − ∫v du with u chosen by the LIATE guideline (logarithmic, inverse-trigonometric, "
                   "algebraic, trigonometric, exponential). A lone ln x or arctan x uses dv = dx. Repeated "
                   "application happens automatically when the new integral again needs parts.")

    def suitability(self, feats: Features):
        opts = split_options(feats.expr, feats.x)
        if not opts:
            return None
        u = opts[0][0]
        k = factor_kind(split_constant(u, feats.x)[1], feats.x)
        if k in ("log", "invtrig"):
            return 0.86, f"A {k} factor appears: integrate by parts with u = {split_constant(u, feats.x)[1]}."
        if k == "algebraic" and u.is_polynomial(feats.x):
            return 0.8, "A polynomial times an exponential/trigonometric function: integrate by parts."
        return None

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        last: Optional[str] = None
        for u, dv in split_options(f, x):
            try:
                sol = self._apply(f, x, u, dv, ctx)
            except TechniqueFailure as exc:
                last = str(exc)
                continue
            if sol is not None:
                return sol
        if last:
            raise TechniqueFailure(last)
        return None

    def _apply(self, f, x, u, dv, ctx: Context) -> Optional[Solution]:
        du = sp.diff(u, x)
        vsol = ctx.solve_sub(dv, x)
        if vsol is None:
            raise TechniqueFailure(f"{dv} could not be integrated, so u = {u} is not workable")
        v = vsol.F
        new = v * du
        if new == f:
            return None
        rest = ctx.solve_sub(new, x)
        if rest is None:
            raise TechniqueFailure(f"the integral of v·du = {new} could not be solved")
        F = u * v - rest.F
        table = {"kind": "key_value", "rows": [["u", latex(u)], ["dv", latex(dv) + r"\,d" + x.name],
                                               ["du", latex(du) + r"\,d" + x.name], ["v", latex(v)]]}
        u_kind = factor_kind(split_constant(u, x)[1], x)
        why = {"log": "a logarithm becomes algebraic when differentiated",
               "invtrig": "an inverse trigonometric function becomes algebraic when differentiated",
               "algebraic": "a polynomial loses a degree each time it is differentiated"}.get(
            u_kind, "it is the factor that simplifies or stays manageable on differentiation")
        claims_choice = [Claim.identity(u * dv, f, x, "u·dv equals the integrand")]
        steps = [
            make_step("parts_choice", Msg("Choose u and dv"), Int(f, x), Int(u * dv, x),
                      Msg("Take u = {u} ({why}) and dv = {dv} d{x}, a factor we can integrate (LIATE guideline).",
                          u=u, why=why, dv=dv, x=x),
                      claims=claims_choice, formula=r"\int u\,dv=uv-\int v\,du"),
            make_step("parts_derivatives", Msg("Find du and v"), Int(f, x), Int(f, x),
                      Msg("Differentiate u: d{u_} = {du} d{x}. Integrate dv: v = {v}.", u_="u", u=u, du=du, v=v, x=x),
                      claims=[Claim.derivative(v, dv, x, "v is an antiderivative of dv")],
                      substeps=vsol.steps, table=table),
            make_step("parts_formula", Msg("Apply the integration by parts formula"), Int(f, x),
                      u * v - Int(new, x),
                      Msg("Substitute into ∫u dv = uv − ∫v du. The new integral is {new}.", new=new),
                      claims=claims_choice + [Claim.derivative(v, dv, x, "v' = dv")]),
        ]
        steps.extend(wrap_solution_steps(new, x, rest))
        steps.append(make_step("parts_combine", Msg("Combine"), u * v - Int(new, x),
                               F, Msg("Subtract the value of the new integral from uv."),
                               claims=[Claim.derivative(F, f, x, "the result differentiates to the integrand")]))
        return Solution(F, steps, self.name, merge_conditions(vsol.conditions, rest.conditions),
                        derivation_level=_level(vsol, rest))


def _level(*sols) -> str:
    levels = [s.derivation_level for s in sols]
    if all(l == "full" for l in levels):
        return "full"
    return "partial"


# ---------------------------------------------------------------------------------------------------------
_EXP_LIKE = (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh)


class TabularParts(Technique):
    name = "tabular_parts"
    title = "Tabular integration by parts"
    family = "parts"
    description = ("For P(x)·e^{ax+b}, P(x)·sin(ax+b), P(x)·cos(ax+b) (and hyperbolic analogues) with a polynomial "
                   "P: differentiate P down to zero in one column, integrate the other factor repeatedly in a second "
                   "column, and add the diagonal products with alternating signs.")

    @staticmethod
    def _parse(f, x):
        c, rest = split_constant(f, x)
        fs = [a for a in sp.Mul.make_args(rest) if a.has(x)]
        if len(fs) != 2:
            return None
        poly = [a for a in fs if a.is_polynomial(x)]
        other = [a for a in fs if not a.is_polynomial(x)]
        if len(poly) != 1 or len(other) != 1:
            return None
        P, E = poly[0], other[0]
        if sp.degree(P, x) < 2 or not isinstance(E, _EXP_LIKE) or linear_form(E.args[0], x) is None:
            return None
        return c, P, E

    def suitability(self, feats: Features):
        if self._parse(feats.expr, feats.x) is None:
            return None
        return 0.84, "A polynomial of degree ≥ 2 times e^(ax+b), sin or cos: the tabular method is efficient."

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        parsed = self._parse(f, x)
        if parsed is None:
            return None
        c, P, E = parsed
        d = sp.degree(P, x)
        D = [P]
        for _ in range(d):
            D.append(sp.diff(D[-1], x))
        D.append(sp.Integer(0))  # P^(d+1) = 0
        I_col = [E]
        subs = []
        for k in range(d + 1):
            s = ctx.solve_sub(I_col[-1], x)
            if s is None:
                raise TechniqueFailure("a repeated integral of the exponential/trigonometric factor failed")
            subs.append(s)
            I_col.append(s.F)
        terms = [(-1) ** k * D[k] * I_col[k + 1] for k in range(d + 1)]
        F = c * sp.Add(*terms)
        rows = []
        for k in range(d + 2):
            rows.append([("+" if k % 2 == 0 else "−") if k <= d else "", latex(D[k]),
                         latex(I_col[k]) if k <= d + 1 else ""])
        table = {"kind": "tabular", "columns": ["sign", "differentiate", "integrate"], "rows": rows,
                 "note": "Multiply each entry of the middle column by the entry one row below in the right column, "
                         "attach the sign, and add."}
        for k in range(d + 1):
            if not subs[k].verification or not subs[k].verification.ok:
                raise TechniqueFailure("an intermediate antiderivative did not verify")
        step = make_step(
            "tabular_parts", Msg("Tabular integration by parts"), Int(f, x), F,
            Msg("Differentiate {P} repeatedly until it reaches 0 and integrate {E} the same number of times. Each "
                "entry of the derivative column is multiplied by the entry diagonally below it in the integral column, "
                "with alternating signs.", P=P, E=E),
            claims=[Claim.derivative(F, f, x, "the tabular sum differentiates to the integrand")],
            table=table, formula=r"\int P\,E\,dx=\sum_{k\ge0}(-1)^kP^{(k)}\,E^{(-k-1)}")
        return Solution(F, [step], self.name, merge_conditions(*[s.conditions for s in subs]),
                        derivation_level=_level(*subs))


# ---------------------------------------------------------------------------------------------------------
class CyclicParts(Technique):
    name = "cyclic_parts"
    title = "Repeated integration by parts returning to the original integral"
    family = "parts"
    description = ("For e^{ax+b}·sin(cx+d) and e^{ax+b}·cos(cx+d): two applications of integration by parts bring back "
                   "a multiple of the original integral I, giving an equation I = A + kI that is solved for I.")

    @staticmethod
    def _parse(f, x):
        c, rest = split_constant(f, x)
        fs = [a for a in sp.Mul.make_args(rest) if a.has(x)]
        if len(fs) != 2:
            return None
        E = [a for a in fs if isinstance(a, sp.exp) and linear_form(a.args[0], x)]
        T = [a for a in fs if isinstance(a, (sp.sin, sp.cos)) and linear_form(a.args[0], x)]
        if len(E) != 1 or len(T) != 1:
            return None
        return c, E[0], T[0]

    def suitability(self, feats: Features):
        if self._parse(feats.expr, feats.x) is None:
            return None
        return 0.91, "An exponential times sin/cos: apply parts twice and solve for the integral."

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        parsed = self._parse(f, x)
        if parsed is None:
            return None
        c, E, T = parsed
        base = E * T
        I = Int(base, x)
        # first application: u1 = T, dv1 = E dx
        v1sol = ctx.solve_sub(E, x)
        u1, du1, v1 = T, sp.diff(T, x), v1sol.F if v1sol else None
        if v1 is None:
            raise TechniqueFailure("the exponential factor could not be integrated")
        g1 = v1 * du1
        # second application on g1: u2 = (trig factor of g1), dv2 = exp factor
        c1, rest1 = split_constant(g1, x)
        fs1 = [a for a in sp.Mul.make_args(rest1) if a.has(x)]
        T2c = [a for a in fs1 if isinstance(a, (sp.sin, sp.cos))]
        if len(T2c) != 1:
            raise TechniqueFailure("the integrand after the first application is not of exponential×trig form")
        u2 = T2c[0]
        du2 = sp.diff(u2, x)
        dv2 = g1 / u2
        v2sol = ctx.solve_sub(dv2, x)
        if v2sol is None:
            raise TechniqueFailure("the second application could not integrate dv")
        v2 = v2sol.F
        g2 = v2 * du2
        ratio = sp.simplify(g2 / base)
        if ratio.has(x) or ratio == 1:
            raise TechniqueFailure("the original integral does not reappear as a constant multiple")
        k = ratio
        A = u1 * v1 - (u2 * v2)  # I = u1 v1 - [u2 v2 - ∫g2] = A + k·I   (g1 = u2·dv2)
        # I = A + k I  ->  I = A/(1-k)
        F = c * A / (1 - k)
        Isym = sp.Symbol("I", real=True)
        steps = [
            make_step("parts_choice", Msg("First application: choose u and dv"), Int(f, x), Int(u1 * E * c, x),
                      Msg("Take u = {u} and dv = {dv} d{x}.", u=u1, dv=E, x=x),
                      claims=[Claim.identity(u1 * E * c, f, x, "u·dv equals the integrand")],
                      table={"kind": "key_value", "rows": [["u", latex(u1)], ["dv", latex(E) + r"\,dx"],
                                                           ["du", latex(du1) + r"\,dx"], ["v", latex(v1)]]}),
            make_step("parts_formula", Msg("Apply the formula once"), Int(f, x), c * (u1 * v1 - Int(g1, x)),
                      Msg("∫u dv = uv − ∫v du gives a second integral {g}.", g=g1),
                      claims=[Claim.derivative(v1, E, x, "v' = dv"), Claim.identity(du1, sp.diff(u1, x), x)]),
            make_step("parts_choice_2", Msg("Second application: choose u and dv again"), Int(g1, x), Int(u2 * dv2, x),
                      Msg("Apply parts to the new integral with the same pattern: u = {u}, dv = {dv} d{x}.",
                          u=u2, dv=dv2, x=x),
                      claims=[Claim.identity(u2 * dv2, g1, x, "u·dv equals the new integrand")],
                      table={"kind": "key_value", "rows": [["u", latex(u2)], ["dv", latex(dv2) + r"\,dx"],
                                                           ["du", latex(du2) + r"\,dx"], ["v", latex(v2)]]}),
            make_step("parts_formula_2", Msg("Apply the formula again"), Int(g1, x), u2 * v2 - Int(g2, x),
                      Msg("The integral on the right is {k}·I: the original integral has reappeared.", k=k),
                      claims=[Claim.derivative(v2, dv2, x, "v' = dv"), Claim.identity(g2, k * base, x,
                                                                                      "the new integral is a multiple of the original")]),
            make_step("solve_for_integral", Msg("Solve for the integral"),
                      sp.Eq(Isym, c * (A + k * Isym)), sp.Eq(Isym, F),
                      Msg("Collecting the two applications gives I = {A} + ({k})·I with I the original integral "
                          "(times the constant {c}); solve this linear equation for I.", A=c * A, k=k, c=c),
                      claims=[Claim.derivative(F, f, x, "the solution differentiates to the integrand")]),
        ]
        return Solution(F, steps, self.name, merge_conditions(v1sol.conditions, v2sol.conditions),
                        derivation_level="full")
