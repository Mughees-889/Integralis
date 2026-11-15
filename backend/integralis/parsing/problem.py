"""Turn user input (expression text, optional limits and variable) into an :class:`IntegralProblem`."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sympy as sp

from .. import render
from ..errors import ParseError
from .parser import normalise_text, parse_expression, symbol_factory

_INTEGRAL = re.compile(r"""^\s*∫\s*
    (?:_\s*(?P<lo>\{[^{}]*\}|\([^()]*\)|[-+]?[A-Za-z0-9_.]+)\s*
       \^\s*(?P<hi>\{[^{}]*\}|\([^()]*\)|[-+]?[A-Za-z0-9_.]+))?
    \s*(?P<body>.*?)\s*\bd\s*(?P<var>[A-Za-z][A-Za-z0-9_]*)\s*$""", re.VERBOSE | re.DOTALL)


@dataclass
class IntegralProblem:
    integrand: sp.Expr
    var: sp.Symbol
    lower: Optional[sp.Expr] = None
    upper: Optional[sp.Expr] = None
    warnings: List[str] = field(default_factory=list)
    source: str = ""

    @property
    def definite(self) -> bool:
        return self.lower is not None and self.upper is not None

    def latex(self) -> str:
        f = render.latex(self.integrand)
        if self.integrand.is_Add:
            f = r"\left(" + f + r"\right)"
        v = render.latex(self.var)
        if self.definite:
            return rf"\int_{{{render.latex(self.lower)}}}^{{{render.latex(self.upper)}}} {f}\,d{v}"
        return rf"\int {f}\,d{v}"

    def text(self) -> str:
        f = render.text(self.integrand)
        v = render.text(self.var)
        if self.definite:
            return f"∫_{{{render.text(self.lower)}}}^{{{render.text(self.upper)}}} ({f}) d{v}"
        return f"∫ ({f}) d{v}"


def _strip_wrapper(s: str) -> str:
    return s.strip()


def parse_problem(expression: str, variable: Optional[str] = None, lower: Optional[str] = None,
                  upper: Optional[str] = None, mode: Optional[str] = None,
                  assumptions: Optional[Dict[str, str]] = None, natural_log: bool = True) -> IntegralProblem:
    """Parse an integration request.

    ``expression`` may be a bare integrand (``x^2*exp(x)``) or full notation (``∫_0^1 x e^(x²) dx``).
    ``lower``/``upper`` are optional limit strings (``oo``, ``-inf``, ``pi/2`` ... are accepted).
    """
    if expression is None or not str(expression).strip():
        raise ParseError("Empty expression", 0, "Enter an integrand such as sin(x) or ∫ x e^(x²) dx", "")
    warnings: List[str] = []
    text = normalise_text(str(expression), brackets=False).strip()
    nb_lo = nb_hi = nb_var = None
    body = text
    if text.startswith("∫"):
        m = _INTEGRAL.match(text)
        if not m:
            raise ParseError("Could not read the ∫ notation", None,
                             "Write ∫ f(x) dx or ∫_0^1 f(x) dx — the differential (dx) must come last", text)
        body, nb_var = m.group("body"), m.group("var")
        nb_lo, nb_hi = m.group("lo"), m.group("hi")
        if not body.strip():
            raise ParseError("Nothing to integrate between ∫ and d" + nb_var, None, None, text)
    elif re.search(r"\bd\s*[A-Za-z]\s*$", text):
        raise ParseError("The expression ends with a differential (dx), but no ∫ sign was given", None,
                         "Either enter only the integrand (e.g. x^2) or write the full ∫ x^2 dx", text)

    variable = variable.strip() if variable and variable.strip() else None
    if nb_var and variable and nb_var != variable:
        raise ParseError(f"The expression integrates with respect to {nb_var} (d{nb_var}) but the variable field "
                         f"says {variable}", None, "Make the two agree", text)

    symbols: Dict[str, sp.Symbol] = {}
    make = symbol_factory(assumptions, symbols)
    outcome = parse_expression(body, assumptions=assumptions, natural_log=natural_log, symbols=symbols)
    warnings.extend(outcome.warnings)
    integrand = outcome.expr

    name = variable or nb_var
    if not name:
        free = sorted(s.name for s in integrand.free_symbols)
        if not free:
            name = "x"
        elif "x" in free:
            name = "x"
        elif len(free) == 1:
            name = free[0]
        else:
            raise ParseError("Several symbols appear (" + ", ".join(free) + "); choose the integration variable",
                             None, "Fill in the variable field or end the integral with d<variable>", text)
    var = make(name)

    lo_s = nb_lo if nb_lo is not None else (lower if lower not in (None, "") else None)
    hi_s = nb_hi if nb_hi is not None else (upper if upper not in (None, "") else None)
    if (nb_lo is not None) and (lower not in (None, "") or upper not in (None, "")):
        raise ParseError("Limits were given both in the expression and in the limit fields", None,
                         "Use one or the other", text)
    if mode == "indefinite" and (lo_s is not None or hi_s is not None):
        warnings.append("The limits were ignored because indefinite mode is selected.")
        lo_s = hi_s = None
    if (lo_s is None) != (hi_s is None):
        raise ParseError("A definite integral needs both a lower and an upper limit", None, None, text)
    if mode == "definite" and lo_s is None:
        raise ParseError("Definite mode needs a lower and an upper limit", None, None, text)

    lo = hi = None
    if lo_s is not None:
        lo_o = parse_expression(lo_s, assumptions=assumptions, allow_infinity=True, natural_log=natural_log,
                                symbols=symbols)
        hi_o = parse_expression(hi_s, assumptions=assumptions, allow_infinity=True, natural_log=natural_log,
                                symbols=symbols)
        lo, hi = lo_o.expr, hi_o.expr
        warnings.extend(w for w in lo_o.warnings + hi_o.warnings if w not in warnings)
        if var in lo.free_symbols or var in hi.free_symbols:
            raise ParseError(f"The limits must not depend on the integration variable {var.name}", None, None, text)
    return IntegralProblem(integrand, var, lo, hi, warnings, text)
