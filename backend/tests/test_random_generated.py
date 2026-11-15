"""Known antiderivatives generated independently, differentiated to make integrands, then solved."""
import random

import pytest
import sympy as sp

from helpers import indep_antiderivative, solve_ok
from integralis import render

x = sp.Symbol("x", real=True)


def random_antiderivative(rng: random.Random) -> sp.Expr:
    a, b = rng.randint(1, 4), rng.randint(1, 3)
    k = rng.randint(0, 3)
    c = rng.choice([1, 2, 3, sp.Rational(1, 2)])
    templates = [
        lambda: x ** k * sp.exp(a * x), lambda: x ** k * sp.sin(a * x), lambda: x ** k * sp.cos(a * x),
        lambda: sp.exp(a * x) * sp.sin(b * x), lambda: sp.exp(a * x) * sp.cos(b * x),
        lambda: x ** (k + 1) * sp.log(x), lambda: sp.atan(a * x) * x ** k, lambda: sp.log(x ** 2 + a),
        lambda: sp.sin(x) ** a * sp.cos(x) ** b, lambda: (x ** 2 + a) ** sp.Rational(b, 2),
        lambda: sp.atan(x / a) + x / (x ** 2 + a ** 2), lambda: sp.exp(x ** 2 + a) * (x + 1),
        lambda: sp.log(sp.cosh(a * x)), lambda: x * sp.sqrt(x + a), lambda: sp.sin(x) ** 3 / (a + sp.cos(x)),
        lambda: sp.log(x + a) / (x + a),
    ]
    return c * rng.choice(templates)()


def _gen(seed, n):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        F = random_antiderivative(rng)
        f = sp.diff(F, x)
        if f != 0 and f.has(x):
            out.append((F, f))
    return out


@pytest.mark.parametrize("seed", [11, 23])
def test_random_integrands_never_produce_wrong_answers(seed):
    solved = 0
    cases = _gen(seed, 18)
    for F, f in cases:
        text = render.text(f)
        r = solve_ok(text, timeout=12)
        if r.solved:
            solved += 1
            assert indep_antiderivative(r.antiderivative, f, x), f"WRONG ANSWER for {text}: {r.antiderivative}"
            assert r.verification_status in ("verified", "verified_numeric")
        else:
            assert r.status.value == "failed" and r.antiderivative is None
    assert solved >= 0.7 * len(cases), f"only {solved}/{len(cases)} solved"


def test_random_definite_integrals_match_fundamental_theorem():
    rng = random.Random(5)
    done = 0
    for F, f in _gen(77, 10):
        lo, hi = sorted([rng.choice([1, 2, 3]), rng.choice([4, 5])])
        expected = sp.N(F.subs(x, hi) - F.subs(x, lo), 25)
        r = solve_ok(render.text(f), lower=str(lo), upper=str(hi), timeout=15)
        if r.solved:
            done += 1
            assert abs(complex(sp.N(r.value, 25) - expected)) < 1e-10, (f, r.value, expected)
    assert done >= 6
