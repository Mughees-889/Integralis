import pytest
import sympy as sp

from integralis import EngineOptions, solve
from integralis.derivation import make_step
from integralis.engine import Engine
from integralis.errors import ParseError, UnjustifiedStepError
from integralis.models import Claim, Solution
from integralis.parsing import parse_problem
from integralis.render import Msg
from integralis.techniques.base import Technique
from integralis.techniques.linearity import Linearity
from integralis.techniques.table import StandardIntegrals

x = sp.Symbol("x", real=True)


@pytest.mark.parametrize("bad", ["sin(", "x +", "((x)", "2 3", "x**", "∫ dx", "sin(x) dx", "@", "x^^2"])
def test_malformed_input_is_rejected_with_a_parse_error(bad):
    with pytest.raises(ParseError):
        solve(bad)


def test_unjustified_step_is_refused():
    with pytest.raises(UnjustifiedStepError):
        make_step("bogus", "wrong", sp.Integral(x, x), x ** 3, Msg("claimed"),
                  claims=[Claim.derivative(x ** 3, x, x, "d/dx x^3 = x")])
    with pytest.raises(UnjustifiedStepError):
        make_step("bogus", "wrong", sp.Integral(sp.sin(x), x), sp.cos(x), Msg("claimed"),
                  claims=[Claim.derivative(sp.cos(x), sp.sin(x), x, "sign error")])


class LyingTechnique(Technique):
    """Returns a plausible-looking but wrong antiderivative without any verified step."""
    name = "liar"
    title = "liar"

    def suitability(self, feats):
        return 5.0, "always first"

    def attempt(self, f, x, ctx):
        return Solution(2 * sp.sin(x), [], "liar")  # derivative 2cos(x) != cos(x)


def test_engine_rejects_a_wrong_result_from_any_technique():
    eng = Engine(EngineOptions(), techniques=[LyingTechnique(), Linearity(), StandardIntegrals()])
    sol = eng.solve_top(sp.cos(x), x)
    assert sol is not None and sol.technique != "liar"
    assert any(t.technique == "liar" and t.outcome == "unverified" for t in eng.trace)


def test_engine_falls_back_gracefully_and_reports_failure_honestly():
    eng = Engine(EngineOptions(enable_sympy_fallback=False), techniques=[LyingTechnique()])
    assert eng.solve_top(sp.cos(x), x) is None


def test_unsupported_but_valid_input_returns_a_clear_failure():
    r = solve("sin(sin(x))", options=EngineOptions(timeout_seconds=10))
    assert r.status.value == "failed" and r.message is not None


def test_time_limit_is_enforced():
    import time
    t = time.time()
    r = solve("sin(sin(x))*cos(cos(x))*exp(sin(x)^2)", options=EngineOptions(timeout_seconds=3))
    assert time.time() - t < 12
    assert not r.solved or r.verification_status != "not_applicable"


def test_no_hardcoded_answers_distinct_inputs_give_distinct_computed_outputs():
    a = solve("x*exp(x^2)").antiderivative
    b = solve("x*exp(2*x^2)").antiderivative
    assert sp.simplify(a - sp.exp(x ** 2) / 2) == 0 and sp.simplify(b - sp.exp(2 * x ** 2) / 4) == 0


def test_every_step_in_every_demo_derivation_is_verified_or_explanatory():
    for e in ["x*exp(x^2)", "x^2*ln(x)", "1/(x^4+1)", "sqrt(1-x^2)", "exp(x)*sin(x)", "sin(x)^2*cos(x)^3"]:
        r = solve(e)
        assert r.solved
        for st in r.all_steps():
            assert st.verification.value in ("symbolic", "structural", "numeric", "unchecked")
            if st.claims:
                assert st.verification.value != "unchecked"
