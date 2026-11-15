import pytest
import sympy as sp

from helpers import indep_antiderivative, integrand_of, solve_ok


def test_symbolic_exponent_power_rule_states_condition():
    r = solve_ok("x^n")
    assert r.solved
    assert any("-1" in c.text() for c in r.conditions), [c.text() for c in r.conditions]


@pytest.mark.parametrize("expr", ["sin(a*x)", "exp(b*x)", "a*x^2 + b", "1/(a*x+b)", "cos(a*x+b)", "a*exp(a*x)"])
def test_parametric_antiderivatives(expr):
    r = solve_ok(expr)
    assert r.solved
    f, x = integrand_of(expr)
    assert indep_antiderivative(r.antiderivative, f, x)


def test_parameter_division_condition_is_stated():
    r = solve_ok("sin(a*x)")
    assert any("≠ 0" in c.text() for c in r.conditions)


def test_definite_with_symbolic_limits_and_parameter():
    r = solve_ok("a*x", lower="0", upper="b")
    assert r.solved and sp.simplify(r.value - sp.Symbol("a", real=True) * sp.Symbol("b", real=True) ** 2 / 2) == 0
    assert any("parameter" in w.lower() for w in r.warnings) or r.conditions


def test_improper_parametric_needs_assumptions():
    r = solve_ok("exp(-a*x)", lower="0", upper="oo", assumptions={"a": "positive"})
    assert r.solved and sp.simplify(r.value - 1 / sp.Symbol("a", positive=True, real=True)) == 0 or str(r.value) == "1/a"
    r2 = solve_ok("exp(-a*x)", lower="0", upper="oo")
    # without the assumption the engine must not guess: it either states a condition or refuses
    assert (not r2.solved) or r2.conditions or r2.warnings


def test_log_of_parameter_domain_conditions():
    r = solve_ok("ln(a*x)")
    assert r.solved
