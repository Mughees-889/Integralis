"""Independent verification helpers for the tests.

These deliberately do NOT use the engine's own checker: derivatives are taken *numerically* with mpmath, and sample
points differ from the engine's, so a bug in the engine's verifier cannot hide a wrong answer here.
"""
from __future__ import annotations

import os
from typing import List, Optional

import mpmath as mp
import sympy as sp

os.environ.setdefault("INTEGRALIS_DEBUG", "1")  # surface internal errors in tests instead of hiding them

from integralis import EngineOptions, solve  # noqa: E402
from integralis.parsing import parse_problem  # noqa: E402

PTS = [mp.mpf(s) for s in ("0.37", "0.81", "1.93", "2.71", "-0.52", "-1.31", "-2.47", "4.13", "0.093", "1.07",
                           "3.31", "-0.77", "0.61", "-3.9", "6.2")]


def _eval(fn, *args):
    try:
        v = mp.mpmathify(fn(*args))
        return v if mp.isfinite(v) and abs(mp.im(v)) < mp.mpf("1e-18") else None
    except Exception:
        return None


def indep_equal(lhs, rhs, x, need: int = 3, tol="1e-15") -> bool:
    lhs, rhs = sp.sympify(lhs), sp.sympify(rhs)
    syms = sorted((lhs.free_symbols | rhs.free_symbols) | {x}, key=lambda s: s.name)
    fl, fr = sp.lambdify(syms, lhs, "mpmath"), sp.lambdify(syms, rhs, "mpmath")
    ok = 0
    with mp.workdps(30):
        for k, p in enumerate(PTS):
            args = [p if s == x else mp.mpf(1.3 + 0.4 * (j + k % 3)) for j, s in enumerate(syms)]
            a, b = _eval(fl, *args), _eval(fr, *args)
            if a is None or b is None:
                continue
            if abs(mp.re(a) - mp.re(b)) > mp.mpf(tol) * (1 + abs(a)):
                return False
            ok += 1
    return ok >= need


def indep_antiderivative(F, f, x, need: int = 3) -> bool:
    """dF/dx == f checked with a NUMERICAL derivative (mpmath.diff), independent of SymPy's symbolic diff."""
    F, f = sp.sympify(F), sp.sympify(f)
    syms = sorted((F.free_symbols | f.free_symbols) | {x}, key=lambda s: s.name)
    fF, ff = sp.lambdify(syms, F, "mpmath"), sp.lambdify(syms, f, "mpmath")
    ok = 0
    with mp.workdps(40):
        for k, p in enumerate(PTS):
            others = [mp.mpf(1.3 + 0.4 * (j + k % 3)) for j in range(len(syms) - 1)]

            def at(t):
                it = iter(others)
                return fF(*[t if s == x else next(it) for s in syms])
            try:
                num = mp.diff(at, p)
                it = iter(others)
                ex = mp.mpmathify(ff(*[p if s == x else next(it) for s in syms]))
                num = mp.mpmathify(num)
                if not (mp.isfinite(num) and mp.isfinite(ex)) or abs(mp.im(num)) > 1e-15 or abs(mp.im(ex)) > 1e-15:
                    continue
            except Exception:
                continue
            if abs(mp.re(num) - mp.re(ex)) > mp.mpf("1e-12") * (1 + abs(ex)):
                return False
            ok += 1
    return ok >= need


def solve_ok(expr: str, timeout: float = 25.0, **kw):
    opts = EngineOptions(timeout_seconds=timeout)
    return solve(expr, options=opts, **kw)


def integrand_of(expr: str, **kw):
    p = parse_problem(expr, **kw)
    return p.integrand, p.var


def check_all_steps(result) -> List[str]:
    """Re-verify every claim attached to every displayed step with the independent numerics. Returns problems."""
    problems = []
    for st in result.all_steps():
        assert st.verification.value != "failed", f"step {st.rule} carries a failed verification"
        for c in st.claims:
            try:
                if c.kind == "identity":
                    ok = indep_equal(c.lhs, c.rhs, c.x, need=2)
                else:
                    ok = indep_antiderivative(c.lhs, c.rhs, c.x, need=2)
            except Exception as exc:  # noqa: BLE001
                ok = False
                problems.append(f"{st.rule}: independent check crashed: {exc}")
            if not ok:
                problems.append(f"{st.rule}: claim {c.kind} not confirmed independently ({c.description})")
    return problems
