"""Integration by substitution (reverse chain rule and changes of variable).

Candidate inner functions ``g(x)`` are discovered from the structure of the integrand (arguments of functions,
bases and exponents of powers, the function nodes themselves). For each candidate the transformed integrand ``h(u)``
is built in two ways:

  method A (structural):  h(u) = (f / g')  with g replaced by u, accepted only if no x remains;
  method B (inverse):     for a g that can be inverted uniquely, x = phi(u) and h(u) = f(phi(u)) phi'(u).

Whatever is built, the identity  h(g(x)) * g'(x) == f(x)  is verified before the step is shown, so the derivation
can never contain an invalid substitution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sympy as sp

from ..derivation import Int, make_step, merge_conditions, split_constant, wrap_solution_steps
from ..errors import EngineTimeout, TechniqueFailure
from ..models import Claim, Solution
from ..render import Msg
from ..strategy.features import Features
from ..utils.timeouts import time_limit
from .base import Context, Technique
from .table import linear_form

_U = sp.Symbol("_U_", real=True)  # neutral placeholder; renamed to a friendly symbol when a plan is used


@dataclass
class Plan:
    g: sp.Expr
    h: sp.Expr          # in terms of _U
    method: str         # "A" (clean structural), "A2" (needed x^k replacement) or "B" (inverse)
    kind: str           # "inner" or "outer"
    phi: Optional[sp.Expr] = None  # x as a function of _U (method B)

    def sort_key(self):
        return (0 if self.kind == "inner" else 1, {"A": 0, "A2": 1, "B": 2}[self.method], sp.count_ops(self.h))


def candidates(f: sp.Expr, x: sp.Symbol) -> List[Tuple[sp.Expr, str]]:
    seen, out = set(), []

    def add(g, kind):
        if not g.has(x) or g == x or g in seen:
            return
        seen.add(g)
        out.append((g, kind))

    for node in sp.preorder_traversal(f):
        if isinstance(node, sp.Function) and node.has(x):
            for a in node.args:
                add(a, "inner")
            add(node, "outer")
        elif node.is_Pow and node.has(x):
            b, e = node.as_base_exp()
            if e.has(x):
                add(e, "inner")
            if b.has(x):
                add(b, "inner")
                if e.is_Rational and not e.is_Integer and linear_form(b, x) is not None:
                    add(b ** sp.Rational(1, e.q), "inner")  # the q-th root of a linear expression
            add(node, "outer")
    # inner candidates first, keeping discovery order
    return [c for c in out if c[1] == "inner"] + [c for c in out if c[1] == "outer"]


def _binomial_power(g, x) -> Optional[Tuple[int, sp.Expr, sp.Expr]]:
    """If g = alpha*x**k + beta, return (k, alpha, beta)."""
    try:
        p = sp.Poly(g, x)
    except sp.PolynomialError:
        return None
    terms = p.terms()
    k = p.degree()
    if k < 1 or len(terms) > 2:
        return None
    alpha = p.coeff_monomial(x ** k)
    beta = g - alpha * x ** k
    if beta.has(x):
        return None
    return k, alpha, beta


def _poly_in_g(P, g, x) -> Optional[sp.Expr]:
    """Write the polynomial P(x) as a polynomial in g(x) (g-adic expansion); None if impossible."""
    try:
        Pp, Gp = sp.Poly(P, x), sp.Poly(g, x)
    except sp.PolynomialError:
        return None
    if Gp.degree() < 1:
        return None
    out, j = 0, 0
    while not Pp.is_zero:
        q, r = sp.div(Pp, Gp)
        if r.degree() > 0:
            return None
        out += r.as_expr() * _U ** j
        Pp, j = q, j + 1
        if j > 60:
            return None
    return out


def _method_a3(f, x, g) -> Optional[sp.Expr]:
    """For a rational f and polynomial g: f/g' is a rational function of g when numerator and denominator are
    polynomials in g."""
    if not (g.is_polynomial(x) and f.is_rational_function(x)):
        return None
    gp = sp.diff(g, x)
    if gp == 0:
        return None
    try:
        with time_limit(1.5):
            num, den = sp.fraction(sp.cancel(sp.together(f / gp)))
    except Exception:
        return None
    n, d = _poly_in_g(num, g, x), _poly_in_g(den, g, x)
    if n is None or d is None or d == 0:
        return None
    return n / d


def _method_a(f, x, g) -> Optional[Tuple[sp.Expr, bool]]:
    """Structural substitution. Returns (h, needed_leftover_replacement) or None."""
    gp = sp.diff(g, x)
    if gp == 0:
        return None
    r0 = f / gp
    prepared = [lambda: r0, lambda: sp.together(r0), lambda: sp.cancel(r0)]
    if not r0.has(sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc, sp.sinh, sp.cosh, sp.tanh):
        prepared.append(lambda: sp.simplify(r0))
    binom = _binomial_power(g, x)
    for prep in prepared:
        try:
            with time_limit(0.7):
                r = prep()
        except EngineTimeout:
            continue
        except Exception:
            continue
        h = r.subs(g, _U)
        if not h.has(x):
            return h, False
        if binom is not None:
            k, alpha, beta = binom
            h2 = h.subs(x ** k, (_U - beta) / alpha)
            if not h2.has(x):
                return h2, True
    h3 = _method_a3(f, x, g)
    if h3 is not None and not h3.has(x):
        return h3, False
    return None


def b_eligible(g, x) -> bool:
    """Monotone, cheaply invertible inner functions: radicals / exp / log of a linear expression."""
    if isinstance(g, sp.exp) or isinstance(g, sp.log):
        return linear_form(g.args[0], x) is not None
    if g.is_Pow:
        b, e = g.as_base_exp()
        if not e.has(x) and e.is_Rational and not e.is_Integer and linear_form(b, x) is not None:
            return True
        if not b.has(x) and linear_form(e, x) is not None:
            return True
    return False


def _method_b(f, x, g) -> Optional[Tuple[sp.Expr, sp.Expr]]:
    if not b_eligible(g, x):
        return None
    even_root = g.is_Pow and g.exp.is_Rational and not g.exp.is_Integer and g.exp.q % 2 == 0
    U = sp.Dummy("U", positive=True) if even_root else _U
    try:
        with time_limit(1.5):
            sols = sp.solve(sp.Eq(g, U), x)
    except EngineTimeout:
        return None
    except Exception:
        return None
    sols = [s for s in sols if not s.has(sp.I) and U in s.free_symbols]
    if len(sols) != 1:
        return None
    phi = sols[0]
    try:
        with time_limit(1.5):
            h = sp.powsimp(sp.cancel(f.subs(x, phi) * sp.diff(phi, U)))
    except EngineTimeout:
        return None
    except Exception:
        return None
    if h.has(x):
        return None
    if even_root:
        h, phi = h.xreplace({U: _U}), phi.xreplace({U: _U})
    return h, phi


def build_plans(f: sp.Expr, x: sp.Symbol, method_b: bool = False, limit: int = 12) -> List[Plan]:
    plans: List[Plan] = []
    f_canon = sp.srepr(f.subs(x, _U))
    for g, kind in candidates(f, x)[:limit]:
        plan = None
        if not method_b:
            ma = _method_a(f, x, g)
            if ma is not None and ma[0] != 0:
                plan = Plan(g, ma[0], "A2" if ma[1] else "A", kind)
        else:
            b = _method_b(f, x, g)
            if b is not None and b[0] != 0:
                plan = Plan(g, b[0], "B", kind, b[1])
        if plan is None:
            continue
        if sp.srepr(plan.h) == f_canon:  # identity substitution: useless
            continue
        plans.append(plan)
    plans.sort(key=Plan.sort_key)
    return plans


class Substitution(Technique):
    name = "substitution"
    title = "Substitution (reverse chain rule and change of variable)"
    family = "substitution"
    description = ("Finds an inner function g(x) whose derivative appears in the integrand (or that can be inverted "
                   "uniquely), rewrites the integral in u = g(x), integrates, and substitutes back. Every rewriting "
                   "is verified by checking h(g(x))·g'(x) = f(x).")

    def __init__(self):
        self._plans: Dict[tuple, List[Plan]] = {}
        self._bplans: Dict[tuple, List[Plan]] = {}

    def _get_plans(self, f, x) -> List[Plan]:
        key = (sp.srepr(f), x.name)
        if key not in self._plans:
            self._plans[key] = build_plans(f, x, method_b=False)
        return self._plans[key]

    def _get_b_plans(self, f, x) -> List[Plan]:
        key = (sp.srepr(f), x.name)
        if key not in self._bplans:
            self._bplans[key] = build_plans(f, x, method_b=True)
        return self._bplans[key]

    def suitability(self, feats: Features):
        plans = self._get_plans(feats.expr, feats.x)
        if not plans:
            for g, _ in candidates(feats.expr, feats.x):
                if b_eligible(g, feats.x):
                    return 0.5, f"The change of variable u = {g} (solved for {feats.x}) may simplify the integrand."
            return None
        best = plans[0]
        linear = len(sp.Poly(best.g, feats.x).all_coeffs()) == 2 if best.g.is_polynomial(feats.x) else False
        if best.method == "A" and best.kind == "inner" and not linear:
            return 0.93, f"The inner function {best.g} has its derivative in the integrand: substitute u = {best.g}."
        if best.method == "A" and best.kind == "outer":
            return 0.85, f"The derivative of {best.g} appears as a factor: substitute u = {best.g}."
        if best.method == "A":
            return 0.7, f"Substitute u = {best.g} (linear inner function)."
        return 0.55, f"Substitute u = {best.g} and express the remaining {feats.x} through u."

    def attempt(self, f: sp.Expr, x: sp.Symbol, ctx: Context) -> Optional[Solution]:
        last_error: Optional[str] = None
        for pass_b in (False, True):
            plans = self._get_b_plans(f, x) if pass_b else self._get_plans(f, x)
            try:
                sol = self._run_plans(f, x, plans, ctx)
            except TechniqueFailure as exc:
                last_error = str(exc)
                continue
            if sol is not None:
                return sol
        if last_error:
            raise TechniqueFailure(last_error)
        return None

    def _run_plans(self, f, x, plans, ctx: Context) -> Optional[Solution]:
        last_error: Optional[str] = None
        for plan in plans[: ctx.options.max_sub_candidates]:
            ctx.engine.check_deadline()
            try:
                sol = self._try_plan(f, x, plan, ctx)
            except TechniqueFailure as exc:
                last_error = str(exc)
                continue
            if sol is not None:
                return sol
        if last_error:
            raise TechniqueFailure(last_error)
        return None

    def _try_plan(self, f, x, plan: Plan, ctx: Context) -> Optional[Solution]:
        g = plan.g
        u = ctx.fresh_symbol(f, x)
        h = plan.h.xreplace({_U: u})
        gp = sp.diff(g, x)
        c, h0 = split_constant(h, u)
        # 1. the rewriting is checked before anything is shown
        rewrite_claim = Claim.identity(h.xreplace({u: g}) * gp, f, x, "h(g(x))·g'(x) equals the integrand")
        sub = ctx.solve_sub(h0, u)
        if sub is None:
            raise TechniqueFailure(f"the integral in {u} obtained from u = {g} could not be solved")
        F = (c * sub.F).xreplace({u: g})

        if plan.method in ("A", "A2"):
            why = Msg("The expression {g} appears inside the integrand, and its derivative {gp} accounts for the "
                      "remaining factor (up to a constant), so {g} is a good choice for the substitution.",
                      g=g, gp=gp) if plan.method == "A" else Msg(
                "Substituting {u} = {g} (with d{u} = {gp} d{x}) leaves powers of {x} that can be written through {u}.",
                u=u, g=g, gp=gp, x=x)
            relation = sp.Eq(u, g)
            diff_note = Msg("Let {u} = {g}. Then d{u} = {gp} d{x}.", u=u, g=g, gp=gp, x=x)
        else:
            why = Msg("Writing {x} as a function of {u} = {g} turns the integrand into a simpler function of {u}.",
                      x=x, u=u, g=g)
            relation = sp.Eq(x, plan.phi.xreplace({_U: u}))
            diff_note = Msg("Let {u} = {g}, so {x} = {phi} and d{x} = {dphi} d{u}.", u=u, g=g, x=x,
                            phi=plan.phi.xreplace({_U: u}), dphi=sp.diff(plan.phi.xreplace({_U: u}), u))
        conds = []
        if gp.free_symbols - {x}:
            conds.append(Msg("{gp} ≠ 0", gp=gp))
        steps = [
            make_step("identify_substitution", Msg("Choose the substitution"), Int(f, x), Int(f, x), why),
            make_step("substitution", Msg("Introduce the new variable"), Int(f, x), relation, diff_note,
                      conditions=conds),
            make_step("rewrite_in_new_variable", Msg("Rewrite the integral in terms of {u}", u=u), Int(f, x),
                      c * Int(h0, u),
                      Msg("Replace everything by {u} and d{u}: the integrand becomes {h}.", u=u, h=h0)
                      if c == 1 else
                      Msg("Replace everything by {u} and d{u}; the constant {c} is taken outside.", u=u, c=c),
                      claims=[rewrite_claim], meta={"u": u, "g": g}),
        ]
        steps.extend(wrap_solution_steps(h0, u, sub))
        steps.append(make_step(
            "back_substitute", Msg("Substitute back"), c * sub.F, F,
            Msg("Replace {u} by {g} to return to the variable {x}.", u=u, g=g, x=x),
            claims=[Claim.derivative(F, f, x, "the result differentiates to the integrand")]))
        meta = {"substitution": {"u": u, "g": g, "h": h, "c": c, "h0": h0, "method": plan.method, "x": x}}
        return Solution(F, steps, self.name, merge_conditions(conds, sub.conditions), meta=meta,
                        derivation_level=sub.derivation_level)
