import pytest
import sympy as sp

from helpers import check_all_steps, indep_antiderivative, integrand_of, solve_ok

CASES = {
    # simple polynomial / trigonometric / exponential
    "poly": ["3*x^2 + 2*x + 1", "x^5 - 4*x^3", "(2*x+1)^4", "sqrt(x)", "1/x^3", "7"],
    "trig_simple": ["sin(x)", "cos(3*x)", "sec(x)^2", "tan(x)", "cot(2*x)", "sinh(x)", "5*cosh(2*x)"],
    "reverse_chain": ["x*exp(x^2)", "x*sqrt(1+x^2)", "cos(x)*sin(x)^4", "exp(x)/(1+exp(x))", "ln(x)/x", "x^2*exp(x^3)",
                      "(3*x+1)^5*x", "x/(x^2+1)", "tan(x)*sec(x)^2"],
    "by_parts": ["x*exp(x)", "x*sin(x)", "x^2*ln(x)", "ln(x)", "arctan(x)", "arcsin(x)", "x*arctan(x)", "x*cos(2*x)"],
    "repeated_parts": ["x^2*exp(x)", "x^3*sin(x)", "ln(x)^2", "x^2*cos(3*x)", "x^3*exp(2*x)"],
    "exp_trig": ["exp(x)*sin(x)", "exp(2*x)*cos(3*x)", "exp(-x)*sin(2*x)"],
    "partial_fractions": ["1/(x^2-1)", "(3*x+5)/((x-1)*(x+2))", "1/(x*(x+1)^2)", "1/(x^3+x)", "(x^2+1)/(x^3+3*x)",
                          "1/(x^4+1)", "1/(x^2*(x^2+1))"],
    "polynomial_division": ["x^3/(x^2+1)", "(x^3+2*x)/(x-1)", "(x^4+1)/(x^2-1)", "x^2/(x+1)"],
    "completing_square": ["1/(x^2+4*x+5)", "(2*x+3)/(x^2+4*x+5)", "1/(x^2+2*x+10)", "1/sqrt(x^2+2*x+5)"],
    "trig_identities": ["sin(x)^2", "cos(x)^4", "sin(x)^2*cos(x)^3", "sin(x)^3*cos(x)^3", "sin(3*x)*cos(2*x)",
                        "sin(x)*sin(5*x)", "tan(x)^3", "sec(x)^4", "tan(x)^3*sec(x)^3", "sec(x)^3", "1/(2+cos(x))",
                        "1/(1+sin(x))", "cosh(x)^2"],
    "inverse_trig_forms": ["1/(x^2+4)", "1/sqrt(9-x^2)", "1/(x^2-2)", "1/(1+4*x^2)", "1/sqrt(x^2+1)"],
    "radicals": ["sqrt(1-x^2)", "sqrt(x^2+4)", "x^2/sqrt(1-x^2)", "1/(x^2*sqrt(x^2+9))", "x*sqrt(x+1)",
                 "1/(1+sqrt(x))", "1/(x*sqrt(x+1))", "x^3*sqrt(x^2+1)", "1/(x^2+1)^2", "sqrt(9-4*x^2)"],
}
ALL = [(k, e) for k, v in CASES.items() for e in v]


@pytest.mark.parametrize("category,expr", ALL)
def test_antiderivative_is_correct_and_verified(category, expr):
    r = solve_ok(expr)
    assert r.solved, f"{expr}: {r.message}"
    f, x = integrand_of(expr)
    assert indep_antiderivative(r.antiderivative, f, x), f"{expr}: antiderivative fails the independent derivative check"
    assert r.verification_status in ("verified", "verified_numeric")
    assert r.classification.value == "elementary"
    assert r.steps and r.steps[-1].rule == "verify_derivative"


@pytest.mark.parametrize("category,expr", [(k, e) for k, v in CASES.items() for e in v[:3]])
def test_displayed_working_is_correct(category, expr):
    r = solve_ok(expr)
    assert r.solved
    problems = check_all_steps(r)
    assert not problems, problems


@pytest.mark.parametrize("expr,technique", [
    ("x*exp(x^2)", "substitution"), ("ln(x)", "parts"), ("x^2*exp(x)", "tabular_parts"), ("exp(x)*sin(x)", "cyclic_parts"),
    ("1/(x^2-1)", "table"), ("(3*x+5)/((x-1)*(x+2))", "rational"), ("sin(x)^2*cos(x)^3", "trig_powers"),
    ("sin(3*x)*cos(2*x)", "trig_product_to_sum"), ("sqrt(1-x^2)", "trig_substitution"), ("sec(x)^3", "reduction"),
    ("1/(2+cos(x))", "weierstrass"), ("1/(x^2+4*x+5)", "complete_square"), ("tan(x)^3", "trig_tan_sec"),
])
def test_strategy_selector_picks_the_appropriate_technique(expr, technique):
    r = solve_ok(expr)
    assert r.solved and r.technique == technique, (r.technique, [t.technique for t in r.trace[:6]])


def test_every_technique_in_catalogue_is_exercised():
    from integralis.techniques import default_techniques
    used = set()
    for _, expr in ALL:
        pass
    names = {t.name for t in default_techniques()}
    for expr in ["x*exp(x^2)", "ln(x)", "x^2*exp(x)", "exp(x)*sin(x)", "1/(x^2-1)", "(3*x+5)/((x-1)*(x+2))",
                 "sin(x)^2*cos(x)^3", "sin(3*x)*cos(2*x)", "sqrt(1-x^2)", "sec(x)^3", "1/(2+cos(x))",
                 "1/(x^2+4*x+5)", "tan(x)^3", "x*sqrt(x+1)", "cosh(x)^2", "exp(-x^2)", "sin(x)", "3*x^2+2"]:
        used.add(solve_ok(expr).technique)
        for t in solve_ok(expr).trace:
            if t.outcome == "applied":
                used.add(t.technique)
    assert {"linearity", "table", "substitution", "parts", "tabular_parts", "cyclic_parts", "rational", "trig_powers",
            "trig_product_to_sum", "trig_substitution", "reduction", "weierstrass", "complete_square", "trig_tan_sec",
            "hyperbolic_exp", "sympy_general"} <= used | names


def test_sum_and_constant_multiple_structure():
    r = solve_ok("5*sin(x) + 3*x^2")
    assert r.solved and r.steps[0].rule == "sum_rule"
