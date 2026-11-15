import pytest
import sympy as sp

from helpers import indep_antiderivative, integrand_of, solve_ok

SPECIAL = [("exp(-x^2)", "erf"), ("exp(x)/x", "Ei"), ("sin(x)/x", "Si"), ("cos(x)/x", "Ci"), ("1/ln(x)", "li|Ei"),
           ("exp(x^2)", "erfi")]


@pytest.mark.parametrize("expr,fn", SPECIAL)
def test_special_function_antiderivatives(expr, fn):
    r = solve_ok(expr)
    assert r.solved and r.classification.value == "special_function", (r.classification, r.antiderivative)
    assert any(r.antiderivative.has(getattr(sp, name)) for name in fn.split("|"))  # li(x) == Ei(ln x)
    f, x = integrand_of(expr)
    assert indep_antiderivative(r.antiderivative, f, x)
    assert r.derivation_level in ("result_only", "partial")  # honest: the special-function step is not "derived"


def test_non_elementary_is_only_claimed_when_proved():
    # e^(-x^2) and e^x/x are proven non-elementary by the Risch algorithm
    r = solve_ok("exp(-x^2)")
    assert "no elementary antiderivative" in r.message.text().lower()
    r = solve_ok("x^x")
    assert r.status.value == "failed" and r.classification.value == "non_elementary_proven"


def test_failure_is_not_presented_as_proof_of_non_existence():
    r = solve_ok("sin(sin(x))", timeout=12)
    assert r.status.value == "failed"
    assert r.classification.value in ("no_closed_form_found", "could_not_solve")
    assert r.classification.value != "non_elementary_proven"
    assert "not a proof" in r.message.text().lower() or "time limit" in r.message.text().lower()


def test_failed_results_carry_no_antiderivative_or_value():
    r = solve_ok("sin(sin(x))", timeout=12)
    assert r.antiderivative is None and r.value is None and r.answer_expr() is None


def test_definite_integral_of_special_function_needs_convergence_and_numeric_agreement():
    r = solve_ok("exp(-x^2)", lower="-oo", upper="oo")
    assert r.solved and abs(complex(sp.N(r.value - sp.sqrt(sp.pi), 30))) < 1e-12


def test_no_closed_form_definite_still_refuses_unverified_numbers():
    r = solve_ok("sin(sin(x))", lower="0", upper="1", timeout=15)
    # either a verified value or an honest failure — never an unconfirmed number
    assert r.solved or r.status.value == "failed"
    if r.solved:
        assert any("quadrature" in c["name"].lower() for c in r.checks)
