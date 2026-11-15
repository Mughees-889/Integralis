"""Verification: exact symbolic equivalence, derivative checks and independent high-precision numerics.

Principles
----------
* A step or result is ``SYMBOLIC`` only when exact simplification proves the claim.
* If simplification is inconclusive but both sides agree to ~28 digits at many sample points, the check is
  reported as ``NUMERIC`` — useful evidence, but explicitly weaker than a proof.
* Sample points where either side is complex or non-finite are skipped, so claims are only checked on the
  real domain where they make sense.
"""
from __future__ import annotations

import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import mpmath as mp
import sympy as sp

from .errors import EngineTimeout
from .models import CheckResult, Claim, Verification
from .utils.timeouts import time_limit

_FIXED = ["0.3", "0.7", "1.3", "2.1", "3.7", "-0.4", "-1.6", "-2.9", "5.3", "0.11", "0.55", "0.95",
          "1.01", "-0.65", "7.9", "-4.7", "0.2", "1.7", "2.6", "-0.85", "9.3", "0.45", "-3.3", "1.15"]
_rng = random.Random(20240611)
_EXTRA = [f"{_rng.uniform(-9, 9):.6f}" for _ in range(24)]
_POOL = ["1.37", "2.41", "0.63", "3.17", "1.91", "0.83", "2.93", "1.13"]
_INT_POOL = [2, 3, 4, 5, 7, 6, 8, 9]


def _x_values(x: sp.Symbol) -> List[str]:
    vals = _FIXED + _EXTRA
    if x.is_positive:
        vals = [v for v in vals if float(v) > 0]
    elif x.is_negative:
        vals = [v for v in vals if float(v) < 0]
    return vals


def _param_value(sym: sp.Symbol, k: int, j: int):
    if sym.is_integer:
        v = mp.mpf(_INT_POOL[(k + j) % len(_INT_POOL)])
    else:
        v = mp.mpf(_POOL[(k + 2 * j) % len(_POOL)])
    if sym.is_negative:
        v = -v
    elif sym.is_positive or sym.is_nonnegative or sym.is_integer:
        pass
    elif (k // 3 + j) % 2 == 1:
        v = -v
    return v


def numeric_compare(lhs, rhs, x: sp.Symbol, *, filter_fn: Optional[Callable[[Dict], bool]] = None,
                    want: int = 12, min_valid: int = 4, dps: int = 40
                    ) -> Tuple[Optional[bool], int, str]:
    """Compare two expressions at many real sample points.

    Returns ``(verdict, n_valid, detail)`` where verdict is True (all valid points agree), False (a mismatch was
    found) or None (too few valid points to say anything).
    """
    lhs, rhs = sp.sympify(lhs), sp.sympify(rhs)
    syms = sorted((lhs.free_symbols | rhs.free_symbols) | {x}, key=lambda s: s.name)
    try:
        fn = sp.lambdify(syms, [lhs, rhs], modules=["mpmath"])
    except Exception as exc:  # pragma: no cover - defensive
        return None, 0, f"not numerically evaluable ({type(exc).__name__})"
    params = [s for s in syms if s != x]
    valid = 0
    with mp.workdps(dps):
        tol = mp.mpf(10) ** (-(dps - 12))
        for k, xs in enumerate(_x_values(x)):
            pt = {x: mp.mpf(xs)}
            for j, s in enumerate(params):
                pt[s] = _param_value(s, k, j)
            if filter_fn is not None:
                try:
                    if not filter_fn(pt):
                        continue
                except Exception:
                    continue
            try:
                a, b = fn(*[pt[s] for s in syms])
                a, b = mp.mpmathify(a), mp.mpmathify(b)
                if not (mp.isfinite(a) and mp.isfinite(b)):
                    continue
            except Exception:
                continue
            scale = 1 + max(abs(a), abs(b))
            if abs(mp.im(a)) > tol * scale or abs(mp.im(b)) > tol * scale:
                continue
            ar, br = mp.re(a), mp.re(b)
            valid += 1
            if abs(ar - br) > tol * scale:
                return False, valid, f"mismatch at {x}={xs}: {mp.nstr(ar, 12)} vs {mp.nstr(br, 12)}"
            if valid >= want:
                break
    if valid >= min_valid:
        return True, valid, f"agree to ~{dps - 12} digits at {valid} sample points"
    return None, valid, "too few sample points where both sides are real and finite"


def _prep(expr):
    """Prepare a difference for exact simplification: sign(g) -> g/|g| and |g|^(2k) -> g^(2k) for real g."""
    expr = sp.sympify(expr)
    try:
        expr = expr.replace(sp.sign, lambda a: a / sp.Abs(a))
        expr = expr.replace(lambda e: e.is_Pow and isinstance(e.base, sp.Abs) and e.exp.is_even
                            and e.base.args[0].is_real is not False and not e.base.args[0].has(sp.I),
                            lambda e: e.base.args[0] ** e.exp)
    except Exception:  # pragma: no cover
        pass
    return expr


_SYM_CACHE: Dict[str, Tuple[bool, float]] = {}
#: shared wall-clock budget for *non-essential* symbolic proofs during one solve (None = unlimited)
_BUDGET: Dict[str, Optional[float]] = {"remaining": None}


def set_symbolic_budget(seconds: Optional[float]) -> None:
    """Limit the total time inner steps may spend on symbolic proofs (the final result check is exempt)."""
    _BUDGET["remaining"] = seconds


def _canon_trig(e):
    """Rewrite tan/cot/sec/csc (and hyperbolic analogues) with sin/cos so simplification stays fast."""
    for fn, repl in ((sp.tan, lambda a: sp.sin(a) / sp.cos(a)), (sp.cot, lambda a: sp.cos(a) / sp.sin(a)),
                     (sp.sec, lambda a: 1 / sp.cos(a)), (sp.csc, lambda a: 1 / sp.sin(a)),
                     (sp.tanh, lambda a: sp.sinh(a) / sp.cosh(a)), (sp.coth, lambda a: sp.cosh(a) / sp.sinh(a)),
                     (sp.sech, lambda a: 1 / sp.cosh(a)), (sp.csch, lambda a: 1 / sp.sinh(a))):
        if e.has(fn):
            e = e.replace(fn, repl)
    return e


def _pythag_zero(expr) -> bool:
    """Prove expr == 0 by clearing denominators and reducing the numerator modulo sin²+cos²=1 / cosh²-sinh²=1 for
    each angle (plain polynomial arithmetic: fast and complete for identities among sin/cos of fixed angles)."""
    e = _canon_trig(sp.sympify(expr))
    if not e.has(sp.sin, sp.cos, sp.sinh, sp.cosh):
        return False
    num = sp.numer(sp.together(e))
    num = sp.expand(num)
    atoms = sorted(num.atoms(sp.sin, sp.cos, sp.sinh, sp.cosh), key=sp.default_sort_key)
    if not atoms:
        return False
    dummies = {a: sp.Dummy("t") for a in atoms}
    poly_expr = num.xreplace(dummies)
    gens = list(dummies.values())
    relations = []
    angles = {a.args[0] for a in atoms}
    for ang in angles:
        for (f1, f2, sign) in ((sp.sin, sp.cos, 1), (sp.sinh, sp.cosh, -1)):
            a1, a2 = f1(ang), f2(ang)
            if a1 in dummies or a2 in dummies:
                d1 = dummies.setdefault(a1, sp.Dummy("t"))
                d2 = dummies.setdefault(a2, sp.Dummy("t"))
                for d in (d1, d2):
                    if d not in gens:
                        gens.append(d)
                relations.append(d1 ** 2 + sign * d2 ** 2 - 1 if sign == 1 else d2 ** 2 - d1 ** 2 - 1)
    try:
        poly_expr = num.xreplace(dummies)
        sp.Poly(poly_expr, *gens)
        _, rem = sp.reduced(poly_expr, relations, *gens)
    except Exception:
        return False
    return sp.expand(rem) == 0


def _radical_zero(expr, x) -> bool:
    """Prove expr == 0 for expressions with radicals of one linear base L: substitute x = (w^q - b)/a with w > 0 so
    that every radical becomes an integer power of w, then clear denominators."""
    if x is None:
        return False
    pows = [n for n in sp.preorder_traversal(expr)
            if n.is_Pow and n.exp.is_Rational and not n.exp.is_Integer and n.base.has(x)]
    if not pows:
        return False
    bases = {n.base for n in pows}
    if len(bases) != 1:
        return False
    L = bases.pop()
    try:
        coeffs = sp.Poly(L, x).all_coeffs()
    except sp.PolynomialError:
        return False
    if len(coeffs) != 2:
        return False
    a, b = coeffs
    q = sp.ilcm(*[n.exp.q for n in pows])
    w = sp.Dummy("w", positive=True)
    e = expr.subs(x, (w ** q - b) / a)
    e = sp.together(e)
    return sp.expand(sp.numer(e)) == 0 or sp.cancel(e) == 0


def _attempts(x=None):
    return (
        lambda e: sp.expand(e),
        lambda e: sp.cancel(sp.together(e)),
        lambda e: 0 if _radical_zero(e, x) else 1,
        lambda e: 0 if _pythag_zero(e) else 1,
        lambda e: 0 if _pythag_zero(sp.expand_trig(e)) else 1,
        lambda e: sp.trigsimp(sp.expand(_canon_trig(e))),
        lambda e: sp.simplify(_canon_trig(e)),
        lambda e: sp.simplify(e),
    )


def symbolic_zero(expr, timeout: float = 4.0, force: bool = False, x=None) -> bool:
    """True if exact simplification proves ``expr == 0``. Results are cached; a failure is only retried with a
    clearly larger time budget. Unless ``force`` is set, the attempt is skipped once the solve's symbolic budget is
    spent (the caller then falls back to numerical evidence, which is labelled as such)."""
    expr = _prep(expr)
    if expr == 0:
        return True
    key = sp.srepr(expr)
    hit = _SYM_CACHE.get(key)
    if hit is not None and (hit[0] or hit[1] >= timeout * 0.75):
        return hit[0]
    rem = _BUDGET["remaining"]
    if not force and rem is not None:
        if rem <= 0:
            return False
        timeout = min(timeout, max(rem, 0.2))
    t0 = time.monotonic()
    proved = False
    per = max(timeout / 3.0, 0.4)
    try:
        with time_limit(timeout):
            for fn in _attempts(x):
                try:
                    with time_limit(per):
                        r = fn(expr)
                except EngineTimeout:
                    continue
                except Exception:
                    continue
                if r == 0:
                    proved = True
                    break
    except EngineTimeout:
        proved = False
    if not force and _BUDGET["remaining"] is not None:
        _BUDGET["remaining"] = _BUDGET["remaining"] - (time.monotonic() - t0)
    if len(_SYM_CACHE) > 4000:
        _SYM_CACHE.clear()
    _SYM_CACHE[key] = (proved, timeout)
    return proved


def check_identity(lhs, rhs, x: sp.Symbol, *, filter_fn=None, symbolic_timeout: float = 4.0,
                   force: bool = False) -> CheckResult:
    """Check ``lhs == rhs``. Numeric disagreement is fatal; symbolic proof upgrades the evidence."""
    lhs, rhs = sp.sympify(lhs), sp.sympify(rhs)
    diff = lhs - rhs
    if diff == 0:
        return CheckResult(Verification.STRUCTURAL, "identical", "both sides are the same expression")
    verdict, n, detail = numeric_compare(lhs, rhs, x, filter_fn=filter_fn)
    if verdict is False:
        return CheckResult(Verification.FAILED, "numeric", detail, n)
    if symbolic_zero(diff, symbolic_timeout, force, x):
        return CheckResult(Verification.SYMBOLIC, "simplify", "difference simplifies to 0", n)
    if verdict is True:
        return CheckResult(Verification.NUMERIC, "numeric", detail, n)
    return CheckResult(Verification.FAILED, "inconclusive",
                       "could neither prove the identity symbolically nor find enough sample points", n)


class _AbsR(sp.Function):
    """|g| for real g, with the real-variable derivative d|g| = (g/|g|) dg (avoids re/im/atan2 clutter)."""
    nargs = 1

    def fdiff(self, argindex=1):
        g = self.args[0]
        return g / _AbsR(g)


def diff_real(F, x):
    """Derivative on the real line: like sp.diff, but |g| is differentiated as sign(g) g'."""
    F = sp.sympify(F)
    if not F.has(sp.Abs):
        return sp.diff(F, x)
    F2 = F.replace(lambda e: isinstance(e, sp.Abs), lambda e: _AbsR(e.args[0]))
    d = sp.diff(F2, x)
    return d.replace(lambda e: isinstance(e, _AbsR), lambda e: sp.Abs(e.args[0]))


def check_derivative(F, f, x: sp.Symbol, *, filter_fn=None, symbolic_timeout: float = 4.0,
                     force: bool = False) -> CheckResult:
    """Check ``dF/dx == f``."""
    try:
        d = diff_real(F, x)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult(Verification.FAILED, "diff", f"cannot differentiate: {exc}")
    return check_identity(d, f, x, filter_fn=filter_fn, symbolic_timeout=symbolic_timeout, force=force)


def check_claim(c: Claim, symbolic_timeout: float = 4.0) -> CheckResult:
    flt = None
    if c.domain is not None:
        lo, hi = c.domain

        def flt(pt, _x=c.x, _lo=lo, _hi=hi):
            v = pt[_x]
            return (_lo is None or v > _lo) and (_hi is None or v < _hi)

    if c.kind == "identity":
        return check_identity(c.lhs, c.rhs, c.x, filter_fn=flt, symbolic_timeout=symbolic_timeout)
    if c.kind == "derivative":
        return check_derivative(c.lhs, c.rhs, c.x, filter_fn=flt, symbolic_timeout=symbolic_timeout)
    raise ValueError(f"unknown claim kind {c.kind!r}")


_RANK = {Verification.STRUCTURAL: 0, Verification.SYMBOLIC: 0, Verification.NUMERIC: 1,
         Verification.UNCHECKED: -1, Verification.FAILED: 3}


def combine(results: Iterable[CheckResult]) -> CheckResult:
    """Weakest-link combination of several checks."""
    results = list(results)
    if not results:
        return CheckResult(Verification.UNCHECKED, "none", "no mathematical claim to check")
    worst = max(results, key=lambda r: _RANK[r.status])
    if worst.status is Verification.FAILED:
        return worst
    if worst.status is Verification.NUMERIC:
        return CheckResult(Verification.NUMERIC, "numeric", "; ".join(r.detail for r in results if r.detail),
                           sum(r.n_points for r in results))
    return CheckResult(Verification.SYMBOLIC, "simplify", "; ".join(r.detail for r in results if r.detail),
                       sum(r.n_points for r in results))
