"""Analysis helpers for definite and improper integrals.

Nothing here returns a value merely because a numerical routine produced one: divergence and convergence are
decided from exact limits or from rigorous comparison tests, and quadrature is only used as an independent
cross-check of an exact result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mpmath as mp
import sympy as sp
from sympy.calculus.util import continuous_domain

from .errors import EngineTimeout
from .utils.timeouts import time_limit


# ------------------------------------------------------------------------------------------------------------
def is_inf(p) -> bool:
    return p in (sp.oo, -sp.oo)


def numeric_order(a, b) -> Optional[bool]:
    """True if a < b, False if a >= b, None when they cannot be compared numerically."""
    try:
        if a == b:
            return False
        if a.is_number and b.is_number or is_inf(a) or is_inf(b):
            return bool(sp.N(b - a, 30) > 0) if not (is_inf(a) or is_inf(b)) else bool(b > a)
    except Exception:
        return None
    return None


@dataclass
class Discontinuities:
    points: List[sp.Expr]          # finite points of [lo, hi] where the function is undefined or unbounded
    certified: bool                # False when the analysis fell back to a heuristic
    undefined_on_interval: bool = False


def trig_poles(expr, x, lo, hi) -> Optional[List[sp.Expr]]:
    """Real poles of tan/sec (π/2 + kπ) and cot/csc (kπ) with a linear argument inside [lo, hi]; None if they cannot be
    enumerated (non-linear argument or an infinite interval)."""
    from .techniques.table import linear_form
    pts: List[sp.Expr] = []
    for n in sp.preorder_traversal(expr):
        if isinstance(n, (sp.tan, sp.sec)):
            base = sp.pi / 2
        elif isinstance(n, (sp.cot, sp.csc)):
            base = sp.Integer(0)
        else:
            continue
        lf = linear_form(n.args[0], x)
        if lf is None or is_inf(lo) or is_inf(hi):
            return None
        a_, b_ = lf
        try:
            g1, g2 = float(sp.N(a_ * lo + b_)), float(sp.N(a_ * hi + b_))
            gmin, gmax = min(g1, g2), max(g1, g2)
            k0 = int((gmin - float(base)) // float(sp.pi)) - 1
            k1 = int((gmax - float(base)) // float(sp.pi)) + 2
            if k1 - k0 > 400:
                return None
            for k in range(k0, k1 + 1):
                p = sp.simplify((base + k * sp.pi - b_) / a_)
                if float(sp.N(lo)) - 1e-12 <= float(sp.N(p)) <= float(sp.N(hi)) + 1e-12:
                    pts.append(p)
        except Exception:
            return None
    return pts


def _algebraic_singular_points(f, x, lo, hi, timeout: float = 2.5) -> Optional[List[sp.Expr]]:
    """Zeros of denominators / bases of negative or fractional powers / log arguments inside [lo, hi]."""
    iv = sp.Interval(lo, hi)
    bases = []
    for n in sp.preorder_traversal(f):
        if n.is_Pow and n.base.has(x) and (n.exp.is_negative or (n.exp.is_Rational and not n.exp.is_Integer)):
            bases.append(n.base)
        elif isinstance(n, sp.log) and n.args[0].has(x):
            bases.append(n.args[0])
    out: List[sp.Expr] = []
    for B in set(bases):
        try:
            with time_limit(timeout):
                roots = sp.solveset(sp.Eq(B, 0), x, iv)
        except Exception:
            return None
        if isinstance(roots, sp.FiniteSet):
            out.extend(r for r in roots if r.is_real)
        elif roots != sp.S.EmptySet:
            return None
    return out


def discontinuities(f, x, lo, hi, timeout: float = 4.0) -> Discontinuities:
    """Points of the closed interval [lo, hi] at which the real function f is undefined/unbounded."""
    iv = sp.Interval(lo, hi)
    if f.has(sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc, sp.sinh, sp.cosh, sp.tanh):
        poles = trig_poles(f, x, lo, hi)
        alg = _algebraic_singular_points(f, x, lo, hi) if poles is not None else None
        if poles is not None and alg is not None:
            pts = []
            for p in poles + alg:
                if not any(_same(p, q) for q in pts):
                    pts.append(p)
            return Discontinuities(sorted(pts, key=lambda p: float(sp.N(p))), True)
        return Discontinuities([], False)
    try:
        with time_limit(timeout):
            dom = continuous_domain(f, x, iv)
            bad = iv - dom
        if bad == sp.S.EmptySet:
            return Discontinuities([], True)
        if isinstance(bad, sp.FiniteSet):
            pts = [p for p in bad if not is_inf(p)]
            return Discontinuities(sorted(pts, key=lambda p: float(sp.N(p))), True)
        return Discontinuities([], True, undefined_on_interval=True)
    except EngineTimeout:
        pass
    except Exception:
        pass
    try:
        with time_limit(timeout):
            sing = sp.singularities(f, x)
        pts = []
        for p in sing:
            if p.is_real and not is_inf(p):
                if (lo == -sp.oo or sp.N(p) >= sp.N(lo)) and (hi == sp.oo or sp.N(p) <= sp.N(hi)):
                    pts.append(p)
        return Discontinuities(sorted(pts, key=lambda p: float(sp.N(p))), False)
    except Exception:
        return Discontinuities([], False)


def _same(p, q) -> bool:
    try:
        return p == q or abs(float(sp.N(p - q))) < 1e-12
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------------------
@dataclass
class LimitInfo:
    kind: str                      # finite | infinite | dne | unknown
    value: Optional[sp.Expr] = None


def classify_limit(L) -> LimitInfo:
    if L is None or L.has(sp.Limit):
        return LimitInfo("unknown")
    if isinstance(L, sp.AccumBounds) or L is sp.nan or L == sp.nan:
        return LimitInfo("dne")
    if L in (sp.oo, -sp.oo, sp.zoo):
        return LimitInfo("infinite", L)
    if L.has(sp.oo) or L.has(-sp.oo) or L.has(sp.zoo) or L.has(sp.AccumBounds):
        return LimitInfo("unknown")
    if L.free_symbols:
        return LimitInfo("finite", L)  # parametric limit: finite expression
    try:
        v = complex(sp.N(L, 30))
        if abs(v) < 1e300:
            return LimitInfo("finite", L)
    except Exception:
        pass
    return LimitInfo("unknown")


def one_sided_limit(F, x, p, side: str = "+", timeout: float = 5.0) -> LimitInfo:
    """Limit of F as x -> p from the given side ('+' from the right, '-' from the left); at ±oo the side is moot."""
    try:
        with time_limit(timeout):
            if p == sp.oo:
                L = sp.limit(F, x, sp.oo)
            elif p == -sp.oo:
                L = sp.limit(F, x, -sp.oo)
            else:
                L = sp.limit(F, x, p, side)
    except EngineTimeout:
        return LimitInfo("unknown")
    except Exception:
        return LimitInfo("unknown")
    return classify_limit(L)


def value_or_limit(F, x, p, side: str) -> LimitInfo:
    """F at p when it is a finite, regular value; otherwise the one-sided limit."""
    if not is_inf(p):
        try:
            v = F.subs(x, p)
            if not (v.has(sp.zoo) or v.has(sp.nan) or v.has(sp.oo) or v.has(-sp.oo) or v.has(sp.AccumBounds)):
                return LimitInfo("finite", v)
        except Exception:
            pass
    return one_sided_limit(F, x, p, side)


def numeric_probe_limit(F, x, p, side: str, expected: sp.Expr) -> bool:
    """Independent sanity check of a limit by evaluating F very close to p at high precision."""
    try:
        f = sp.lambdify(x, F, modules=["mpmath"])
        with mp.workdps(50):
            exp = mp.mpmathify(sp.N(expected, 40))
            if is_inf(p):
                pts = [mp.mpf(10) ** k * (1 if p == sp.oo else -1) for k in (12, 16, 20)]
            else:
                p0 = mp.mpmathify(sp.N(p, 45))
                sgn = 1 if side == "+" else -1
                pts = [p0 + sgn * mp.mpf(10) ** (-k) for k in (14, 18, 22)]
            vals = [mp.mpmathify(f(q)) for q in pts]
            return all(abs(v - exp) <= mp.mpf("1e-5") * (1 + abs(exp)) for v in vals)
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------------------
_OSC = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)


def convergence_test(f, x, point, side: str) -> Tuple[str, str]:
    """Rigorous comparison/p-test at an improper point (an infinite limit or a singularity).

    Returns ``(verdict, reason)`` with verdict in {'converges', 'diverges', 'unknown'}. The reduction is to the
    behaviour of G(t) as t -> oo (t = 1/|x-point| at a finite point), where ∫^oo G dt converges if G = O(t^-p) with
    p > 1 and diverges for a non-oscillating G of size ≥ t^-p with p < 1 (or p = 1 with nonzero constant limit of tG).
    """
    t = sp.Symbol("t", positive=True)
    try:
        if point == sp.oo:
            G = f.subs(x, t)
            where = "as x → ∞"
        elif point == -sp.oo:
            G = f.subs(x, -t)
            where = "as x → −∞"
        else:
            sgn = 1 if side == "+" else -1
            G = f.subs(x, point + sgn / t) / t ** 2
            where = f"as x → {point}{'⁺' if side == '+' else '⁻'}"
        G = sp.simplify(G) if sp.count_ops(G) < 60 else G
    except Exception:
        return "unknown", "could not transform the integrand for the comparison test"
    oscillatory = G.has(*_OSC)
    envelope = G
    if oscillatory:
        # |sin|, |cos| <= 1 : bound a product by dropping bounded oscillating factors
        env = sp.Integer(1)
        ok = True
        for fac in sp.Mul.make_args(G):
            base = fac.as_base_exp()[0]
            if isinstance(base, (sp.sin, sp.cos)):
                continue
            if fac.has(*_OSC):
                ok = False
                break
            env *= fac
        if not ok:
            return "unknown", "oscillating integrand without a simple envelope"
        envelope = env
    try:
        with time_limit(4.0):
            e = sp.limit(sp.log(sp.Abs(envelope)) / sp.log(t), t, sp.oo)
    except Exception:
        return "unknown", "the growth/decay rate could not be determined"
    if e.has(sp.Limit) or e.has(sp.AccumBounds) or e is sp.nan:
        return "unknown", "the growth/decay rate could not be determined"
    if e == -sp.oo:
        return "converges", f"the integrand decays faster than any power {where}"
    if e == sp.oo:
        if oscillatory:
            return "unknown", "oscillating integrand that does not decay"
        return "diverges", f"the integrand grows faster than any power {where}"
    p = -e
    if not p.is_number:
        return "unknown", "the decay exponent is not a number"
    if p > 1:
        return "converges", (f"|integrand| is O(t^−{p}) with exponent {p} > 1 {where} (comparison with a convergent "
                             "p-integral)")
    if oscillatory:
        return "unknown", "oscillating integrand whose envelope does not decay fast enough for absolute convergence"
    if p < 1:
        return "diverges", f"the integrand behaves like a power with exponent {p} < 1 {where} (p-test)"
    try:
        with time_limit(4.0):
            L = sp.limit(G * t, t, sp.oo)
    except Exception:
        return "unknown", "borderline p = 1 case could not be settled"
    if L.has(sp.Limit):
        return "unknown", "borderline p = 1 case could not be settled"
    if L == 0:
        return "unknown", "borderline p = 1 case (logarithmic corrections) could not be settled"
    return "diverges", f"limit comparison with 1/t: t·G(t) → {L} ≠ 0 {where}"


# ------------------------------------------------------------------------------------------------------------
def _mp_bound(p):
    if p == sp.oo:
        return mp.inf
    if p == -sp.oo:
        return -mp.inf
    return mp.mpmathify(sp.N(p, 45))


def numeric_integral(f, x, points: List[sp.Expr], dps: int = 30) -> Optional[mp.mpf]:
    """Arbitrary-precision quadrature of ∫ f dx over consecutive ``points`` (used only as an independent check).
    Returns None when the quadrature cannot be trusted (oscillation over an infinite range, large error estimate)."""
    infinite = any(is_inf(p) for p in points)
    if infinite and f.has(sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc):
        return None
    try:
        fn = sp.lambdify(x, f, modules=["mpmath"])
        with mp.workdps(dps + 10):
            total = mp.mpf(0)
            for lo, hi in zip(points[:-1], points[1:]):
                val, err = mp.quad(fn, [_mp_bound(lo), _mp_bound(hi)], error=True)
                if err > mp.mpf(10) ** -12 * (1 + abs(val)):
                    return None
                total += val
            return total
    except Exception:
        return None


def digits_of_agreement(exact, numeric) -> Optional[int]:
    """Number of agreeing significant digits between an exact value and a quadrature value (None on mismatch)."""
    try:
        with mp.workdps(40):
            ex = mp.mpmathify(sp.N(exact, 40))
            nu = mp.mpmathify(numeric)
            if abs(mp.im(ex)) > mp.mpf("1e-20") * (1 + abs(ex)):
                return None
            err = abs(mp.re(ex) - mp.re(nu))
            scale = max(abs(ex), mp.mpf(1))
            if err == 0:
                return 30
            d = int(-mp.log10(err / scale))
            return max(d, 0)
    except Exception:
        return None
