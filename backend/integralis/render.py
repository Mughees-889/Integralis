"""Presentation helpers: LaTeX and plain-text rendering of SymPy objects, and bilingual message templates."""
from __future__ import annotations

import re
from typing import Any, Dict

import sympy as sp
from sympy.printing.str import StrPrinter

_LATEX_FIXES = [
    (re.compile(r"\\operatorname\{asinh\}"), r"\\operatorname{arsinh}"),
    (re.compile(r"\\operatorname\{acosh\}"), r"\\operatorname{arcosh}"),
    (re.compile(r"\\operatorname\{atanh\}"), r"\\operatorname{artanh}"),
    (re.compile(r"\\operatorname\{acoth\}"), r"\\operatorname{arcoth}"),
]


def latex(expr: Any) -> str:
    """LaTeX for a SymPy object (uses ln, arcsin-style names, KaTeX-compatible output)."""
    if isinstance(expr, str):
        return expr
    expr = sp.sympify(expr)
    s = sp.latex(expr, ln_notation=True, inv_trig_style="full")
    for pat, rep in _LATEX_FIXES:
        s = pat.sub(rep, s)
    return s


_NAME_MAP = {"log": "ln", "atan": "arctan", "asin": "arcsin", "acos": "arccos", "acot": "arccot",
             "asec": "arcsec", "acsc": "arccsc", "asinh": "arsinh", "acosh": "arcosh",
             "atanh": "artanh", "acoth": "arcoth", "Abs": "abs"}


class _TextPrinter(StrPrinter):
    """Plain text that our own parser can read back: x^2, ln(x), arctan(x), ∫_{a}^{b} (f) dx."""

    def _print_Function(self, expr, **kw):
        name = _NAME_MAP.get(expr.func.__name__, expr.func.__name__)
        return name + "(" + ", ".join(self._print(a) for a in expr.args) + ")"

    def _print_log(self, expr, **kw):
        return "ln(" + self._print(expr.args[0]) + ")"

    def _print_Abs(self, expr, **kw):
        return "abs(" + self._print(expr.args[0]) + ")"

    def _print_Exp1(self, expr, **kw):
        return "e"

    def _print_Integral(self, expr, **kw):
        f = self._print(expr.function)
        lim = expr.limits[0]
        v = self._print(lim[0])
        if len(lim) == 3:
            return f"∫_{{{self._print(lim[1])}}}^{{{self._print(lim[2])}}} ({f}) d{v}"
        return f"∫ ({f}) d{v}"


_PRINTER = _TextPrinter({"order": "none"})


def text(expr: Any) -> str:
    """Plain-text rendering of a SymPy object."""
    if isinstance(expr, str):
        return expr
    expr = sp.sympify(expr)
    return _PRINTER.doprint(expr).replace("**", "^")


def is_exprlike(v: Any) -> bool:
    return isinstance(v, (sp.Basic, int)) and not isinstance(v, bool)


class Msg:
    """A message template with ``{name}`` placeholders that renders to LaTeX ($...$ inline math) or plain text.

    Only ``{identifier}`` placeholders are substituted, so LaTeX braces such as ``\\frac{1}{2}`` are safe.
    """

    _PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z_0-9]*)\}")

    def __init__(self, template: str, **values: Any):
        self.template = template
        self.values: Dict[str, Any] = values

    def _render(self, fn_expr) -> str:
        def sub(m):
            key = m.group(1)
            if key not in self.values:
                return m.group(0)
            v = self.values[key]
            return fn_expr(v) if is_exprlike(v) else str(v)

        return self._PLACEHOLDER.sub(sub, self.template)

    def latex(self) -> str:
        return self._render(lambda v: "$" + latex(v) + "$")

    def text(self) -> str:
        return self._render(text)

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        return f"Msg({self.text()!r})"


def as_msg(m: Any) -> Msg:
    return m if isinstance(m, Msg) else Msg(str(m))
