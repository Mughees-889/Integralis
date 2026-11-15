"""A safe expression parser for calculator-style and mathematical notation.

No ``eval``/``exec`` is ever used: input is tokenised with a regular expression and parsed by a small
recursive-descent parser that builds SymPy objects directly from a whitelist of functions and constants.
Unknown characters, attribute access, calls to unknown names, etc. therefore simply fail to parse.

Conventions (documented in the README):
  * ``^`` and ``**`` are powers; ``-x^2`` means ``-(x^2)``; ``a^b^c`` is right associative.
  * implicit multiplication: ``2x``, ``x sin x``, ``(x+1)(x-1)``, ``xe^x`` (= x*e^x).
  * ``sin x`` takes the next operand; ``sin 2x`` means ``sin(2x)``; ``sin^2 x`` means ``(sin x)^2``.
  * ``log(x)`` is the natural logarithm (university convention); ``log10`` and ``log(x, b)`` give other bases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import sympy as sp

from ..errors import ParseError

GREEK_NAMES = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
               "lambda", "mu", "nu", "xi", "omicron", "rho", "sigma", "tau", "upsilon", "phi", "chi",
               "psi", "omega"}
GREEK_UNICODE = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "θ": "theta",
                 "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ρ": "rho", "σ": "sigma", "τ": "tau",
                 "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega"}

_SUPER = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8",
          "⁹": "9", "⁺": "+", "⁻": "-"}
_SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}
_REPL = {"−": "-", "–": "-", "—": "-", "×": "*", "·": "*", "⋅": "*", "∙": "*", "÷": "/", "π": "pi",
         "∞": "oo", "ℯ": "e", "√": " sqrt ", "“": '"', "”": '"'}


def normalise_text(s: str, brackets: bool = True) -> str:
    """Map unicode mathematics to plain ASCII notation."""
    s = re.sub("[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+", lambda m: "^(" + "".join(_SUPER[c] for c in m.group(0)) + ")", s)
    s = re.sub("[₀₁₂₃₄₅₆₇₈₉]+", lambda m: "_" + "".join(_SUB[c] for c in m.group(0)), s)
    for k, v in _REPL.items():
        s = s.replace(k, v)
    for k, v in GREEK_UNICODE.items():
        s = s.replace(k, " " + v + " ")
    if brackets:
        s = s.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    return s


def _log_n(base):
    return lambda a: sp.log(a) / sp.log(base)


def _root(a, n):
    return sp.Pow(a, sp.Integer(1) / n)


# name -> (callable, min_args, max_args)
FUNCS: Dict[str, Tuple[Callable, int, int]] = {
    "sin": (sp.sin, 1, 1), "cos": (sp.cos, 1, 1), "tan": (sp.tan, 1, 1), "cot": (sp.cot, 1, 1),
    "sec": (sp.sec, 1, 1), "csc": (sp.csc, 1, 1),
    "arcsin": (sp.asin, 1, 1), "asin": (sp.asin, 1, 1), "arccos": (sp.acos, 1, 1), "acos": (sp.acos, 1, 1),
    "arctan": (sp.atan, 1, 1), "atan": (sp.atan, 1, 1), "arccot": (sp.acot, 1, 1), "acot": (sp.acot, 1, 1),
    "arcsec": (sp.asec, 1, 1), "asec": (sp.asec, 1, 1), "arccsc": (sp.acsc, 1, 1), "acsc": (sp.acsc, 1, 1),
    "sinh": (sp.sinh, 1, 1), "cosh": (sp.cosh, 1, 1), "tanh": (sp.tanh, 1, 1), "coth": (sp.coth, 1, 1),
    "sech": (sp.sech, 1, 1), "csch": (sp.csch, 1, 1),
    "arsinh": (sp.asinh, 1, 1), "asinh": (sp.asinh, 1, 1), "arcsinh": (sp.asinh, 1, 1),
    "arcosh": (sp.acosh, 1, 1), "acosh": (sp.acosh, 1, 1), "arccosh": (sp.acosh, 1, 1),
    "artanh": (sp.atanh, 1, 1), "atanh": (sp.atanh, 1, 1), "arctanh": (sp.atanh, 1, 1),
    "arcoth": (sp.acoth, 1, 1), "acoth": (sp.acoth, 1, 1), "arccoth": (sp.acoth, 1, 1),
    "exp": (sp.exp, 1, 1), "ln": (sp.log, 1, 1), "log": (sp.log, 1, 2),
    "log2": (_log_n(2), 1, 1), "log10": (_log_n(10), 1, 1),
    "sqrt": (sp.sqrt, 1, 1), "cbrt": (lambda a: sp.Pow(a, sp.Rational(1, 3)), 1, 1),
    "abs": (sp.Abs, 1, 1), "root": (_root, 2, 2),
    "erf": (sp.erf, 1, 1), "erfc": (sp.erfc, 1, 1), "erfi": (sp.erfi, 1, 1),
    "Ei": (sp.Ei, 1, 1), "Si": (sp.Si, 1, 1), "Ci": (sp.Ci, 1, 1), "li": (sp.li, 1, 1),
    "Shi": (sp.Shi, 1, 1), "Chi": (sp.Chi, 1, 1),
}
CASE_SENSITIVE = {"Ei", "Si", "Ci", "Shi", "Chi"}
CONSTS = {"pi", "e", "oo", "inf", "infinity"}
_KNOWN = set(FUNCS) | CONSTS | GREEK_NAMES
_SPLIT_KNOWN = _KNOWN - {"oo", "inf", "infinity"}  # never matched inside a longer run of letters

_TOKEN = re.compile(r"""
    (?P<space>\s+)
  | (?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<word>log10|log2|[A-Za-z]+(?:_[A-Za-z0-9]+)?)
  | (?P<op>\*\*|[-+*/^])
  | (?P<lp>\()
  | (?P<rp>\))
  | (?P<comma>,)
  | (?P<bar>\|)
  | (?P<bang>!)
""", re.VERBOSE)


@dataclass
class Tok:
    kind: str  # num name func const lp rp comma bar op eof
    text: str
    pos: int


def split_word(run: str) -> List[str]:
    """Split a run of letters into known words / single letters using the fewest pieces."""
    n = len(run)
    INF = float("inf")
    best = [(INF, -1)] * (n + 1)
    best[0] = (0.0, -1)
    for i in range(n):
        if best[i][0] == INF:
            continue
        for j in range(i + 1, n + 1):
            piece = run[i:j]
            if piece in _SPLIT_KNOWN or (len(piece) == 1) or piece.lower() in FUNCS and piece.lower() not in CASE_SENSITIVE:
                cost = best[i][0] + 1 + (0.0 if len(piece) > 1 else 0.01)
                if cost < best[j][0]:
                    best[j] = (cost, i)
    pieces: List[str] = []
    j = n
    while j > 0:
        i = best[j][1]
        pieces.append(run[i:j])
        j = i
    return pieces[::-1]


def tokenise(src: str, warnings: List[str]) -> List[Tok]:
    toks: List[Tok] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m:
            ch = src[pos]
            raise ParseError(f"Unsupported character {ch!r}", pos, "Use digits, letters, + - * / ^ ( ) and the "
                             "functions listed in the README", src)
        kind, text = m.lastgroup, m.group(0)
        if kind == "space":
            pass
        elif kind == "num":
            toks.append(Tok("num", text, pos))
        elif kind == "word":
            if "_" in text:
                head = text.split("_", 1)[0]
                if len(head) != 1 and head not in GREEK_NAMES:
                    raise ParseError(f"Subscripted names must start with a single letter: {text!r}", pos,
                                     None, src, len(text))
                toks.append(Tok("name", text, pos))
            else:
                if text in _KNOWN or text.lower() in FUNCS and text.lower() not in CASE_SENSITIVE:
                    pieces = [text]
                else:
                    pieces = split_word(text)
                    if len(pieces) >= 3 and all(len(p) == 1 for p in pieces):
                        if src[m.end():m.end() + 1] == "(":
                            raise ParseError(f"Unknown function {text!r}", pos,
                                             "Supported functions: sin, cos, tan, sec, csc, cot, arcsin, arctan, "
                                             "sinh, cosh, exp, ln, log, sqrt, abs, erf, ... (see the README)",
                                             src, len(text))
                        warnings.append(f"'{text}' is not a known function, so it was read as the product "
                                        f"{'·'.join(pieces)}.")
                off = pos
                for p in pieces:
                    key = p if p in _KNOWN else (p.lower() if p.lower() in FUNCS and p.lower() not in CASE_SENSITIVE else p)
                    if key in FUNCS:
                        toks.append(Tok("func", key, off))
                    elif key in CONSTS:
                        toks.append(Tok("const", "oo" if key in ("oo", "inf", "infinity") else key, off))
                    else:
                        toks.append(Tok("name", p, off))
                    off += len(p)
        elif kind == "op":
            toks.append(Tok("op", "^" if text == "**" else text, pos))
        else:
            toks.append(Tok(kind, text, pos))
        pos = m.end()
    toks.append(Tok("eof", "", len(src)))
    return toks


class _Parser:
    def __init__(self, toks: List[Tok], src: str, make_symbol: Callable[[str], sp.Symbol],
                 allow_infinity: bool, natural_log: bool, warnings: List[str]):
        self.toks, self.src, self.i = toks, src, 0
        self.make_symbol = make_symbol
        self.allow_infinity = allow_infinity
        self.natural_log = natural_log
        self.warnings = warnings
        self.abs_depth = 0

    # -- helpers ---------------------------------------------------------------------------------
    def peek(self, k: int = 0) -> Tok:
        j = min(self.i + k, len(self.toks) - 1)
        return self.toks[j]

    def next(self) -> Tok:
        t = self.toks[self.i]
        if t.kind != "eof":
            self.i += 1
        return t

    def err(self, msg: str, tok: Tok, hint: Optional[str] = None) -> ParseError:
        return ParseError(msg, tok.pos, hint, self.src, max(1, len(tok.text)))

    def unexpected(self, t: Tok) -> ParseError:
        if t.kind == "eof":
            return self.err("The expression ends unexpectedly", t, "Something is missing after the last symbol")
        if t.kind == "rp":
            return self.err("Unmatched ')'", t, "Remove it or add a matching '('")
        if t.kind == "bang":
            return self.err("Factorials are not supported", t)
        if t.kind == "num":
            return self.err(f"Unexpected number {t.text!r}", t,
                            "Did you forget an operator? Write x*2 or 2x for a product and x^2 for a power")
        return self.err(f"Unexpected {t.text!r}", t)

    def expect_rp(self, opener: Tok):
        t = self.peek()
        if t.kind != "rp":
            raise self.err(f"Missing ')' to close the '(' at position {opener.pos}", t if t.kind != "eof" else opener,
                           "Check that parentheses are balanced")
        self.next()

    # -- grammar ---------------------------------------------------------------------------------
    def parse(self) -> sp.Expr:
        if self.peek().kind == "eof":
            raise ParseError("Empty expression", 0, "Enter an expression such as sin(x) or x^2*exp(x)", self.src)
        e = self.expr()
        t = self.peek()
        if t.kind != "eof":
            raise self.unexpected(t)
        return e

    def expr(self) -> sp.Expr:
        left = self.term()
        while self.peek().kind == "op" and self.peek().text in ("+", "-"):
            op = self.next().text
            right = self.term()
            left = left + right if op == "+" else left - right
        return left

    def _starts_implicit(self, t: Tok) -> bool:
        if t.kind in ("name", "func", "const", "lp"):
            return True
        return t.kind == "bar" and self.abs_depth == 0

    def term(self) -> sp.Expr:
        left = self.unary()
        after_div = False
        while True:
            t = self.peek()
            if t.kind == "op" and t.text in ("*", "/"):
                self.next()
                right = self.unary()
                if t.text == "*":
                    left = left * right
                else:
                    if right == 0:
                        raise self.err("Division by zero", t)
                    left = left / right
                    after_div = True
            elif self._starts_implicit(t):
                right = self.power()
                if after_div:
                    self.warnings.append("A quotient followed by implicit multiplication such as 'a/bc' is read as "
                                         "(a/b)·c. Write a/(bc) if you meant the product in the denominator.")
                    after_div = False
                left = left * right
            else:
                break
        return left

    def unary(self) -> sp.Expr:
        t = self.peek()
        if t.kind == "op" and t.text in ("+", "-"):
            self.next()
            v = self.unary()
            return -v if t.text == "-" else v
        return self.power()

    def power(self) -> sp.Expr:
        base = self.primary()
        t = self.peek()
        if t.kind == "op" and t.text == "^":
            self.next()
            exponent = self.unary()
            return self.make_pow(base, exponent, t)
        return base

    def make_pow(self, base, exponent, tok: Tok):
        if base == 0 and (exponent.is_negative or exponent == 0):
            raise self.err("Division by zero (0 raised to a non-positive power)", tok)
        return sp.Pow(base, exponent)

    def primary(self) -> sp.Expr:
        t = self.next()
        if t.kind == "num":
            if re.fullmatch(r"\d+", t.text):
                return sp.Integer(t.text)
            return sp.Rational(t.text)
        if t.kind == "const":
            if t.text == "pi":
                return sp.pi
            if t.text == "e":
                return sp.E
            if not self.allow_infinity:
                raise self.err("∞ is only allowed in the integration limits", t)
            return sp.oo
        if t.kind == "name":
            return self.make_symbol(t.text)
        if t.kind == "lp":
            e = self.expr()
            self.expect_rp(t)
            return e
        if t.kind == "bar":
            self.abs_depth += 1
            e = self.expr()
            self.abs_depth -= 1
            c = self.peek()
            if c.kind != "bar":
                raise self.err(f"Missing closing '|' for the '|' at position {t.pos}", c if c.kind != "eof" else t)
            self.next()
            return sp.Abs(e)
        if t.kind == "func":
            return self.function(t)
        if t.kind == "bang":
            raise self.err("Factorials are not supported", t)
        raise self.unexpected(t)

    # -- functions -------------------------------------------------------------------------------
    def function(self, t: Tok) -> sp.Expr:
        nxt = self.peek()
        if nxt.kind == "op" and nxt.text == "^":
            self.next()
            exp = self.func_exponent(t)
            args = self.function_args(t)
            return sp.Pow(self.apply(t, args), exp)
        return self.apply(t, self.function_args(t))

    def func_exponent(self, ft: Tok) -> sp.Expr:
        neg = False
        if self.peek().kind == "op" and self.peek().text == "-":
            self.next()
            neg = True
        t = self.next()
        if t.kind == "num":
            val = sp.Integer(t.text) if re.fullmatch(r"\d+", t.text) else sp.Rational(t.text)
        elif t.kind == "lp":
            val = self.expr()
            self.expect_rp(t)
        else:
            raise self.err("An exponent was expected after '^'", t)
        if neg:
            val = -val
        if val == -1:
            raise self.err(f"'{ft.text}^-1' is ambiguous", ft,
                           f"Write arc{ft.text}(x) for the inverse function or 1/{ft.text}(x) for the reciprocal")
        return val

    def function_args(self, ft: Tok) -> List[sp.Expr]:
        t = self.peek()
        if t.kind == "lp":
            self.next()
            args = [self.expr()]
            while self.peek().kind == "comma":
                self.next()
                args.append(self.expr())
            self.expect_rp(t)
            return args
        return [self.bare_argument(ft)]

    def bare_argument(self, ft: Tok) -> sp.Expr:
        t = self.peek()
        neg = False
        if t.kind == "op" and t.text in ("+", "-"):
            self.next()
            neg = t.text == "-"
            t = self.peek()
        if t.kind not in ("num", "name", "const", "lp", "bar", "func"):
            raise self.err(f"'{ft.text}' needs an argument", ft if t.kind == "eof" else t,
                           f"Write {ft.text}(x) or {ft.text} x")
        chain = t.kind == "num" and self.peek(1).text != "^"
        arg = self.power()
        if chain:
            while self.peek().kind in ("name", "const", "lp"):
                arg = arg * self.power()
        return -arg if neg else arg

    def apply(self, ft: Tok, args: List[sp.Expr]) -> sp.Expr:
        fn, lo, hi = FUNCS[ft.text]
        if not (lo <= len(args) <= hi):
            want = str(lo) if lo == hi else f"{lo} to {hi}"
            raise self.err(f"{ft.text} takes {want} argument(s), got {len(args)}", ft)
        try:
            if ft.text == "log":
                if len(args) == 2:
                    return sp.log(args[0]) / sp.log(args[1])
                if not self.natural_log:
                    return sp.log(args[0]) / sp.log(10)
                self.warnings.append("log(x) is read as the natural logarithm ln(x). Use log10(x) for base 10.")
                return sp.log(args[0])
            return fn(*args)
        except ParseError:
            raise
        except Exception as exc:
            raise self.err(f"Cannot evaluate {ft.text}(...): {exc}", ft)


@dataclass
class ParseOutcome:
    expr: sp.Expr
    symbols: Dict[str, sp.Symbol]
    warnings: List[str]
    source: str


_ASSUMPTION_KEYS = {"positive": {"positive": True}, "negative": {"negative": True},
                    "nonzero": {"nonzero": True}, "nonnegative": {"nonnegative": True},
                    "nonpositive": {"nonpositive": True}, "integer": {"integer": True}, "real": {}}


def symbol_factory(assumptions: Optional[Dict[str, str]] = None,
                   cache: Optional[Dict[str, sp.Symbol]] = None) -> Callable[[str], sp.Symbol]:
    cache = {} if cache is None else cache
    assumptions = assumptions or {}

    def make(name: str) -> sp.Symbol:
        if name not in cache:
            kw = {"real": True}
            spec = assumptions.get(name)
            if spec:
                for part in re.split(r"[,\s]+", spec.strip()):
                    if part and part not in _ASSUMPTION_KEYS:
                        raise ParseError(f"Unknown assumption {part!r} for {name}", None,
                                         "Use one of: " + ", ".join(sorted(_ASSUMPTION_KEYS)))
                    kw.update(_ASSUMPTION_KEYS.get(part, {}))
            cache[name] = sp.Symbol(name, **kw)
        return cache[name]

    make.cache = cache  # type: ignore[attr-defined]
    return make


def parse_expression(text: str, *, assumptions: Optional[Dict[str, str]] = None, allow_infinity: bool = False,
                     natural_log: bool = True, symbols: Optional[Dict[str, sp.Symbol]] = None) -> ParseOutcome:
    """Parse ``text`` into a SymPy expression. Raises :class:`ParseError` with a position on bad input."""
    if not isinstance(text, str):
        raise ParseError("Expression must be text")
    if len(text) > 2000:
        raise ParseError("Expression is too long (limit 2000 characters)")
    src = normalise_text(text).strip()
    warnings: List[str] = []
    cache: Dict[str, sp.Symbol] = {} if symbols is None else symbols
    make = symbol_factory(assumptions, cache)
    toks = tokenise(src, warnings)
    parser = _Parser(toks, src, make, allow_infinity, natural_log, warnings)
    expr = parser.parse()
    if expr.has(sp.zoo) or expr.has(sp.nan):
        raise ParseError("The expression is undefined (division by zero or log(0)?)", None, None, src)
    return ParseOutcome(expr, cache, list(dict.fromkeys(warnings)), src)
