"""Structural analysis of an integrand. Strategy selection is driven only by these features."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

import sympy as sp

TRIG = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
HYP = (sp.sinh, sp.cosh, sp.tanh, sp.coth, sp.sech, sp.csch)
INVTRIG = (sp.asin, sp.acos, sp.atan, sp.acot, sp.asec, sp.acsc, sp.asinh, sp.acosh, sp.atanh, sp.acoth)


def factor_kind(e, x) -> str:
    """Classify one factor for the LIATE ordering: constant, log, invtrig, algebraic, trig, hyp, exp, other."""
    if not e.has(x):
        return "constant"
    if isinstance(e, sp.log):
        return "log"
    if isinstance(e, INVTRIG):
        return "invtrig"
    if isinstance(e, TRIG):
        return "trig"
    if isinstance(e, HYP):
        return "hyp"
    if isinstance(e, sp.exp):
        return "exp"
    if e.is_Pow:
        base, ex = e.as_base_exp()
        if ex.has(x):
            return "exp" if not base.has(x) else "other"
        if isinstance(base, (sp.log,) + INVTRIG + TRIG + HYP + (sp.exp,)):
            return factor_kind(base, x)
    try:
        if e.is_algebraic_expr(x):
            return "algebraic"
    except Exception:
        pass
    return "other"


@dataclass
class Features:
    expr: sp.Expr
    x: sp.Symbol
    params: Set[sp.Symbol]
    is_polynomial: bool
    is_rational: bool
    num_degree: Optional[int]
    den_degree: Optional[int]
    funcs: Set[str]
    has_trig: bool
    has_hyp: bool
    has_exp: bool
    has_log: bool
    has_invtrig: bool
    has_abs: bool
    has_radical: bool
    notes: List[str] = field(default_factory=list)


def analyze(expr: sp.Expr, x: sp.Symbol) -> Features:
    funcs: Set[str] = set()
    has_radical = False
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Function) and node.has(x):
            funcs.add(node.func.__name__)
        if node.is_Pow and node.base.has(x) and node.exp.is_Rational and not node.exp.is_Integer:
            has_radical = True
    params = {s for s in expr.free_symbols if s != x}
    is_poly = bool(expr.is_polynomial(x))
    is_rat = bool(expr.is_rational_function(x))
    num_deg = den_deg = None
    if is_rat:
        try:
            n, d = sp.fraction(sp.together(expr))
            num_deg, den_deg = sp.degree(n, x), sp.degree(d, x)
        except Exception:
            pass
    f = Features(
        expr=expr, x=x, params=params, is_polynomial=is_poly, is_rational=is_rat, num_degree=num_deg,
        den_degree=den_deg, funcs=funcs,
        has_trig=any(n in funcs for n in ("sin", "cos", "tan", "cot", "sec", "csc")),
        has_hyp=any(n in funcs for n in ("sinh", "cosh", "tanh", "coth", "sech", "csch")),
        has_exp="exp" in funcs or any(n.is_Pow and n.exp.has(x) for n in sp.preorder_traversal(expr)),
        has_log="log" in funcs,
        has_invtrig=any(n in funcs for n in ("asin", "acos", "atan", "acot", "asec", "acsc",
                                              "asinh", "acosh", "atanh", "acoth")),
        has_abs="Abs" in funcs, has_radical=has_radical)
    f.notes = describe(f)
    return f


def describe(ft: Features) -> List[str]:
    e, x = ft.expr, ft.x
    notes: List[str] = []
    if ft.is_polynomial:
        notes.append(f"Polynomial in {x} of degree {sp.degree(e, x) if e.has(x) else 0}.")
    elif ft.is_rational and ft.num_degree is not None:
        extra = " (improper: polynomial division comes first)" if ft.num_degree >= ft.den_degree else ""
        notes.append(f"Rational function: numerator degree {ft.num_degree}, denominator degree "
                     f"{ft.den_degree}{extra}.")
    if e.is_Add:
        notes.append(f"A sum of {len(e.args)} terms, so the sum rule applies.")
    elif e.is_Mul:
        kinds = [factor_kind(a, x) for a in e.args if a.has(x)]
        if len(kinds) >= 2:
            notes.append("A product of " + ", ".join(sorted(set(kinds))) + " factors.")
    if ft.has_exp:
        notes.append("Contains an exponential.")
    if ft.has_log:
        notes.append("Contains a logarithm.")
    if ft.has_trig:
        notes.append("Contains trigonometric functions.")
    if ft.has_hyp:
        notes.append("Contains hyperbolic functions.")
    if ft.has_invtrig:
        notes.append("Contains inverse trigonometric or hyperbolic functions.")
    if ft.has_radical:
        notes.append("Contains radicals / fractional powers.")
    if ft.has_abs:
        notes.append("Contains an absolute value.")
    if ft.params:
        notes.append("Parameters: " + ", ".join(sorted(s.name for s in ft.params)) + " (treated as real constants).")
    return notes
