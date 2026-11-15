"""Standard antiderivatives, generalised to a linear inner function by the reverse chain rule."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import sympy as sp

from ..derivation import Int, make_step
from ..models import Claim, Solution
from ..render import Msg
from ..strategy.features import Features
from .base import Context, Technique


def linear_form(expr, x) -> Optional[Tuple[sp.Expr, sp.Expr]]:
    """Return ``(a, b)`` with ``expr == a*x + b`` (``a`` non-zero, both free of ``x``), else ``None``."""
    if not expr.has(x):
        return None
    try:
        p = sp.Poly(expr, x)
    except sp.PolynomialError:
        return None
    if p.degree() != 1:
        return None
    a, b = p.all_coeffs()
    return a, b


@dataclass
class Hit:
    F: sp.Expr
    title: str
    formula: str
    explanation: Msg
    conditions: List[Msg] = field(default_factory=list)
    linear: bool = False


def _a_cond(a) -> List[Msg]:
    return [Msg("{a} ≠ 0", a=a)] if a.free_symbols else []


def _expl(L, a, x) -> Msg:
    if L == x:
        return Msg("This is a standard integral.")
    return Msg("The inner function {L} is linear in {x} with derivative {a}, so the reverse chain rule divides "
               "the standard result by {a}.", L=L, x=x, a=a)


def _title(L, x, base: str) -> str:
    return base if L == x else base + " (linear inner function)"


# ---------------------------------------------------------------------------------------------------------
def _constant(f, x) -> Optional[Hit]:
    if f.has(x):
        return None
    return Hit(f * x, "Constant rule", r"\int c\,dx=cx+C", Msg("The integrand does not depend on {x}.", x=x))


def _power(f, x) -> Optional[Hit]:
    base, e = f.as_base_exp()
    if e.has(x):
        return None
    lf = linear_form(base, x)
    if lf is None:
        return None
    a, _ = lf
    L = base
    conds = _a_cond(a)
    if e == -1:
        F = sp.log(sp.Abs(L)) / a
        return Hit(F, _title(L, x, "Reciprocal rule"), r"\int\frac{1}{u}\,du=\ln|u|+C", _expl(L, a, x),
                   conds + [Msg("{L} ≠ 0", L=L)], L != x)
    F = L ** (e + 1) / (a * (e + 1))
    if e.free_symbols:
        conds.append(Msg("{e} ≠ -1", e=e))
    return Hit(F, _title(L, x, "Power rule"), r"\int u^{n}\,du=\frac{u^{n+1}}{n+1}+C\quad(n\neq-1)",
               _expl(L, a, x), conds, L != x)


def _exponential(f, x) -> Optional[Hit]:
    if isinstance(f, sp.exp):
        L = f.args[0]
        lf = linear_form(L, x)
        if lf is None:
            return None
        a = lf[0]
        return Hit(sp.exp(L) / a, _title(L, x, "Exponential rule"), r"\int e^{u}\,du=e^{u}+C", _expl(L, a, x),
                   _a_cond(a), L != x)
    if f.is_Pow and not f.base.has(x) and f.exp.has(x):
        lf = linear_form(f.exp, x)
        if lf is None:
            return None
        a, c, L = lf[0], f.base, f.exp
        conds = _a_cond(a) + ([Msg("{c} > 0 and {c} ≠ 1", c=c)] if c.free_symbols else [])
        return Hit(f / (a * sp.log(c)), _title(L, x, "Exponential rule"),
                   r"\int c^{u}\,du=\frac{c^{u}}{\ln c}+C", _expl(L, a, x), conds, L != x)
    return None


_SINGLE = {
    sp.sin: (lambda t: -sp.cos(t), r"\int\sin u\,du=-\cos u+C"),
    sp.cos: (lambda t: sp.sin(t), r"\int\cos u\,du=\sin u+C"),
    sp.tan: (lambda t: -sp.log(sp.Abs(sp.cos(t))), r"\int\tan u\,du=-\ln|\cos u|+C"),
    sp.cot: (lambda t: sp.log(sp.Abs(sp.sin(t))), r"\int\cot u\,du=\ln|\sin u|+C"),
    sp.sec: (lambda t: sp.log(sp.Abs(sp.sec(t) + sp.tan(t))), r"\int\sec u\,du=\ln|\sec u+\tan u|+C"),
    sp.csc: (lambda t: -sp.log(sp.Abs(sp.csc(t) + sp.cot(t))), r"\int\csc u\,du=-\ln|\csc u+\cot u|+C"),
    sp.sinh: (lambda t: sp.cosh(t), r"\int\sinh u\,du=\cosh u+C"),
    sp.cosh: (lambda t: sp.sinh(t), r"\int\cosh u\,du=\sinh u+C"),
    sp.tanh: (lambda t: sp.log(sp.cosh(t)), r"\int\tanh u\,du=\ln\cosh u+C"),
    sp.coth: (lambda t: sp.log(sp.Abs(sp.sinh(t))), r"\int\coth u\,du=\ln|\sinh u|+C"),
    sp.sech: (lambda t: sp.atan(sp.sinh(t)), r"\int\operatorname{sech} u\,du=\arctan(\sinh u)+C"),
    sp.csch: (lambda t: sp.log(sp.Abs(sp.tanh(t / 2))), r"\int\operatorname{csch} u\,du=\ln|\tanh(u/2)|+C"),
}


def _single_function(f, x) -> Optional[Hit]:
    spec = _SINGLE.get(type(f))
    if spec is None:
        return None
    L = f.args[0]
    lf = linear_form(L, x)
    if lf is None:
        return None
    G, formula = spec
    a = lf[0]
    return Hit(G(L) / a, _title(L, x, "Standard integral"), formula, _expl(L, a, x), _a_cond(a), L != x)


_POW = {
    (sp.cos, -2): (lambda t: sp.tan(t), r"\int\sec^2u\,du=\tan u+C"),
    (sp.sec, 2): (lambda t: sp.tan(t), r"\int\sec^2u\,du=\tan u+C"),
    (sp.sin, -2): (lambda t: -sp.cot(t), r"\int\csc^2u\,du=-\cot u+C"),
    (sp.csc, 2): (lambda t: -sp.cot(t), r"\int\csc^2u\,du=-\cot u+C"),
    (sp.cosh, -2): (lambda t: sp.tanh(t), r"\int\operatorname{sech}^2u\,du=\tanh u+C"),
    (sp.sech, 2): (lambda t: sp.tanh(t), r"\int\operatorname{sech}^2u\,du=\tanh u+C"),
    (sp.sinh, -2): (lambda t: -sp.coth(t), r"\int\operatorname{csch}^2u\,du=-\coth u+C"),
    (sp.csch, 2): (lambda t: -sp.coth(t), r"\int\operatorname{csch}^2u\,du=-\coth u+C"),
    (sp.cos, -1): (_SINGLE[sp.sec][0], _SINGLE[sp.sec][1]),
    (sp.sin, -1): (_SINGLE[sp.csc][0], _SINGLE[sp.csc][1]),
    (sp.cosh, -1): (_SINGLE[sp.sech][0], _SINGLE[sp.sech][1]),
    (sp.sinh, -1): (_SINGLE[sp.csch][0], _SINGLE[sp.csch][1]),
    (sp.tan, -1): (_SINGLE[sp.cot][0], _SINGLE[sp.cot][1]),
    (sp.cot, -1): (_SINGLE[sp.tan][0], _SINGLE[sp.tan][1]),
}


def _trig_power(f, x) -> Optional[Hit]:
    if not f.is_Pow:
        return None
    base, e = f.as_base_exp()
    if e.has(x) or not e.is_Integer:
        return None
    spec = _POW.get((type(base), int(e)))
    if spec is None:
        return None
    L = base.args[0]
    lf = linear_form(L, x)
    if lf is None:
        return None
    G, formula = spec
    a = lf[0]
    return Hit(G(L) / a, _title(L, x, "Standard integral"), formula, _expl(L, a, x), _a_cond(a), L != x)


def _square_term(rest, x):
    c, r = rest.as_coeff_Mul()
    if c in (1, -1) and r.is_Pow and r.exp == 2:
        lf = linear_form(r.base, x)
        if lf is not None:
            return int(c), r.base, lf
    return None


def _quadratic_forms(f, x) -> Optional[Hit]:
    """1/(L²+k), 1/(k-L²), 1/sqrt(k-L²), 1/sqrt(L²+k) for a linear L (inverse trigonometric / logarithmic forms)."""
    if not f.is_Pow:
        return None
    base, e = f.as_base_exp()
    if e not in (-1, sp.Rational(-1, 2)) or not base.is_Add:
        return None
    k, rest = base.as_independent(x, as_Add=True)
    if not rest.has(x) or k == 0:
        return None
    st = _square_term(rest, x)
    if st is None:
        return None
    c, L, (a, _) = st
    conds = _a_cond(a)
    expl = _expl(L, a, x)
    if e == -1 and c == 1:
        if k.is_negative:
            m = sp.sqrt(-k)
            return Hit(sp.log(sp.Abs((L - m) / (L + m))) / (2 * a * m), "Logarithmic form",
                       r"\int\frac{du}{u^{2}-m^{2}}=\frac{1}{2m}\ln\left|\frac{u-m}{u+m}\right|+C", expl, conds, L != x)
        s = sp.sqrt(k)
        if k.is_positive is None:
            conds = conds + [Msg("{k} > 0", k=k)]
        return Hit(sp.atan(L / s) / (a * s), "Arctangent form",
                   r"\int\frac{du}{u^{2}+m^{2}}=\frac{1}{m}\arctan\frac{u}{m}+C", expl, conds, L != x)
    if e == -1 and c == -1:
        if k.is_negative:
            m = sp.sqrt(-k)
            return Hit(-sp.atan(L / m) / (a * m), "Arctangent form",
                       r"\int\frac{du}{u^{2}+m^{2}}=\frac{1}{m}\arctan\frac{u}{m}+C", expl, conds, L != x)
        s = sp.sqrt(k)
        if k.is_positive is None:
            conds = conds + [Msg("{k} > 0", k=k)]
        return Hit(sp.log(sp.Abs((s + L) / (s - L))) / (2 * a * s), "Logarithmic form",
                   r"\int\frac{du}{m^{2}-u^{2}}=\frac{1}{2m}\ln\left|\frac{m+u}{m-u}\right|+C", expl, conds, L != x)
    if e == sp.Rational(-1, 2) and c == -1:
        if k.is_positive is False:
            return None
        s = sp.sqrt(k)
        if k.is_positive is None:
            conds = conds + [Msg("{k} > 0", k=k)]
        return Hit(sp.asin(L / s) / a, "Arcsine form",
                   r"\int\frac{du}{\sqrt{m^{2}-u^{2}}}=\arcsin\frac{u}{m}+C", expl, conds, L != x)
    if e == sp.Rational(-1, 2) and c == 1:
        return Hit(sp.log(sp.Abs(L + sp.sqrt(L ** 2 + k))) / a, "Logarithmic form",
                   r"\int\frac{du}{\sqrt{u^{2}+k}}=\ln\left|u+\sqrt{u^{2}+k}\right|+C", expl, conds, L != x)
    return None


_MATCHERS = (_constant, _power, _exponential, _single_function, _trig_power, _quadratic_forms)


def match_table(f, x) -> Optional[Hit]:
    for m in _MATCHERS:
        try:
            hit = m(f, x)
        except Exception:
            hit = None
        if hit is not None:
            return hit
    return None


class StandardIntegrals(Technique):
    name = "table"
    title = "Standard integrals and the reverse chain rule"
    family = "basic"
    description = ("Direct use of standard antiderivatives: power rule, reciprocal, exponentials, trigonometric and "
                   "hyperbolic functions, and inverse-trigonometric/logarithmic forms — each generalised to a "
                   "linear inner function ax+b by the reverse chain rule.")

    def suitability(self, feats: Features):
        if match_table(feats.expr, feats.x) is not None:
            return 0.99, "A standard antiderivative applies directly."
        return None

    def attempt(self, f: sp.Expr, x: sp.Symbol, ctx: Context) -> Optional[Solution]:
        hit = match_table(f, x)
        if hit is None:
            return None
        rule = "reverse_chain_rule" if hit.linear else "standard_integral"
        step = make_step(rule, hit.title, Int(f, x), hit.F, hit.explanation, conditions=hit.conditions,
                         claims=[Claim.derivative(hit.F, f, x, "differentiating the antiderivative gives the integrand")],
                         formula=hit.formula)
        return Solution(hit.F, [step], self.name, list(hit.conditions))
