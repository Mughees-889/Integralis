import mpmath as mp
import pytest
import sympy as sp

from helpers import check_all_steps, solve_ok

EXACT = [  # integrand, lower, upper, exact value
    ("x*exp(x^2)", "0", "1", (sp.E - 1) / 2), ("sin(x)", "0", "pi", 2), ("x^2", "0", "3", 9), ("1/(1+x^2)", "0", "1", sp.pi / 4),
    ("x*sin(x)", "0", "pi", sp.pi), ("ln(x)/x", "1", "e", sp.Rational(1, 2)), ("sqrt(1-x^2)", "0", "1", sp.pi / 4),
    ("sin(x)", "pi", "0", -2), ("exp(x)", "0", "ln(2)", 1), ("1/(x^2+4*x+5)", "-2", "0", sp.atan(2)),
    ("x*exp(-x)", "0", "1", 1 - 2 / sp.E), ("cos(x)^2", "0", "pi", sp.pi / 2),
]
IMPROPER_CONVERGENT = [
    ("1/x^2", "1", "oo", 1), ("exp(-x)", "0", "oo", 1), ("ln(x)", "0", "1", -1), ("exp(-x^2)", "0", "oo", sp.sqrt(sp.pi) / 2),
    ("1/(1+x^2)", "-oo", "oo", sp.pi), ("1/sqrt(x)", "0", "1", 2), ("1/sqrt(1-x^2)", "-1", "1", sp.pi),
    ("x^2*exp(-x)", "0", "oo", 2), ("1/(x^2+1)", "0", "oo", sp.pi / 2), ("exp(-2*x)", "0", "oo", sp.Rational(1, 2)),
    ("1/(x*ln(x)^2)", "2", "oo", 1 / sp.log(2)),
]
DIVERGENT = [
    ("1/x", "1", "oo"), ("1/x", "0", "1"), ("1/x^2", "-1", "1"), ("1/x", "-1", "1"), ("1/(x-1)", "0", "2"), ("exp(x)", "0", "oo"),
    ("tan(x)", "0", "pi/2"), ("1/sqrt(x)", "1", "oo"), ("x", "0", "oo"), ("1/x^2", "0", "1"), ("1/(x*ln(x))", "2", "oo"),
]


def _close(value, expected):
    return abs(complex(sp.N(value - expected, 30))) < 1e-12


@pytest.mark.parametrize("f,lo,hi,expected", EXACT)
def test_exact_definite_values(f, lo, hi, expected):
    r = solve_ok(f, lower=lo, upper=hi)
    assert r.solved, r.message
    assert _close(r.value, sp.sympify(expected)), (r.value, expected)
    assert r.classification.value == "exact_value"
    assert not any(c["status"] == "failed" for c in r.checks)
    assert r.decimal is not None


@pytest.mark.parametrize("f,lo,hi,expected", IMPROPER_CONVERGENT)
def test_convergent_improper_integrals(f, lo, hi, expected):
    r = solve_ok(f, lower=lo, upper=hi)
    assert r.solved, r.message
    assert _close(r.value, sp.sympify(expected)), (r.value, expected)
    assert any("improper" in s.explanation.text() for s in r.steps), "convergence analysis must be shown"


@pytest.mark.parametrize("f,lo,hi", DIVERGENT)
def test_divergent_integrals_never_return_a_value(f, lo, hi):
    r = solve_ok(f, lower=lo, upper=hi)
    assert r.status.value == "divergent", (r.status, r.value, r.message)
    assert r.value is None and r.answer_expr() is None
    assert "DIVERGES" in r.message.text()


def test_interior_singularity_is_not_papered_over():
    # a naive F(b)-F(a) with F = ln|x| would give 0 — the integral diverges instead
    r = solve_ok("1/x", lower="-2", upper="3")
    assert r.status.value == "divergent"


def test_spurious_antiderivative_jump_is_handled():
    # F = arctan(tan(x/2)/√3) jumps at x = π although the integrand is smooth: naive F(2π) - F(0) = 0 is WRONG
    r = solve_ok("1/(2+cos(x))", lower="0", upper="2*pi")
    assert r.solved and _close(r.value, 2 * sp.pi / sp.sqrt(3))
    assert any(s.rule == "antiderivative_jumps" for s in r.steps)


def test_oscillatory_conditionally_convergent_integral():
    r = solve_ok("sin(x)/x", lower="0", upper="oo")
    assert r.solved and _close(r.value, sp.pi / 2)
    assert r.classification.value in ("exact_value", "special_function")


def test_symmetry_shortcuts():
    odd = solve_ok("x^3*cos(x)", lower="-2", upper="2")
    assert odd.solved and odd.value == 0 and odd.technique == "symmetry"
    even = solve_ok("x^2", lower="-1", upper="1")
    assert even.solved and even.value == sp.Rational(2, 3) and even.technique == "symmetry"


def test_king_property():
    r = solve_ok("sin(x)/(sin(x)+cos(x))", lower="0", upper="pi/2")
    assert r.solved and _close(r.value, sp.pi / 4) and r.technique == "king_property"


def test_substitution_shows_transformed_limits():
    r = solve_ok("x*exp(x^2)", lower="0", upper="1")
    tl = [s for s in r.steps if s.rule == "transformed_limits"]
    assert tl, "the substitution with transformed limits should be displayed"
    assert "new limits" in tl[0].explanation.text() or "limits become" in tl[0].explanation.text()


def test_definite_working_is_verified():
    for f, lo, hi in [("x*exp(x^2)", "0", "1"), ("1/(2+cos(x))", "0", "2*pi"), ("ln(x)", "0", "1")]:
        r = solve_ok(f, lower=lo, upper=hi)
        assert r.solved
        assert not check_all_steps(r)


def test_exact_arithmetic_is_kept_symbolic():
    r = solve_ok("1/(1+x^2)", lower="0", upper="sqrt(3)")
    assert r.value == sp.pi / 3 and not r.value.has(sp.Float)


def test_numerical_cross_check_is_independent_and_reported():
    r = solve_ok("exp(-x^2)", lower="0", upper="oo")
    chk = [c for c in r.checks if "quadrature" in c["name"].lower()]
    assert chk and chk[0]["status"] == "numeric"


def test_lower_limit_above_upper_limit_negates():
    a = solve_ok("x^2", lower="0", upper="2")
    b = solve_ok("x^2", lower="2", upper="0")
    assert a.value == -b.value


def test_integrand_not_real_on_interval_is_rejected():
    r = solve_ok("sqrt(x)", lower="-1", upper="1")
    assert not r.solved
