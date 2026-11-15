"""Top-level solving: pipeline orchestration for indefinite integrals and the public ``solve`` API.

Pipeline (spec section 7): parse -> normalise -> analyse -> select and attempt strategies -> build a verified
derivation -> verify the result independently -> return an exact result, a special-function form, a
conditionally valid result, or an informative failure.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import sympy as sp

from .closed_form import classify_closed_form, special_functions_in
from .config import EngineOptions
from .engine import Engine
from .errors import EngineTimeout
from .models import (CheckResult, Classification, Solution, SolveResult, Status, Step, Verification)
from .parsing import IntegralProblem, parse_problem
from .render import Msg
from .strategy.features import analyze
from .utils.timeouts import time_limit
from .verify import check_derivative, check_identity, diff_real, set_symbolic_budget

def normalize(f, x):
    """Light normalisation that never destroys useful structure: only cancel a genuine common polynomial factor."""
    f = sp.sympify(f)
    try:
        if f.has(x) and f.is_rational_function(x):
            num, den = sp.fraction(sp.together(f))
            if den.has(x):
                g = sp.gcd(sp.Poly(num, x), sp.Poly(den, x))
                if g.degree() > 0:
                    return sp.cancel(f)
    except Exception:
        pass
    return f


def _func_names(e) -> set:
    return {type(a).__name__ for a in sp.sympify(e).atoms(sp.Function)}


def tidy(F, f, x):
    """Present the antiderivative more compactly. Conservative: keep the original unless the simplified form is
    clearly smaller, introduces no new kind of function, and still differentiates to the integrand. Constant
    additive terms are dropped (they are absorbed in C)."""
    best = F
    try:
        c, rest = F.as_independent(x, as_Add=True)
        if c != 0 and rest != 0 and F.is_Add:
            best = rest
    except Exception:
        pass
    try:
        if sp.count_ops(best) <= 250:
            with time_limit(1.5):
                cands = [sp.simplify(best)]
                if best.has(sp.sin, sp.cos) and best.has(sp.asin, sp.acos, sp.atan):
                    cands.append(sp.simplify(sp.expand_trig(best)))
            for s in cands:
                if (s != best and sp.count_ops(s) < 0.85 * sp.count_ops(best)
                        and _func_names(s) <= _func_names(best)):
                    best = s
                    break
    except EngineTimeout:
        pass
    except Exception:
        pass
    if best == F:
        return F
    chk = check_derivative(best, f, x, symbolic_timeout=2.0)
    return best if chk.ok else F


def verification_step(F, f, x, check: CheckResult) -> Step:
    d = diff_real(F, x)
    ds = None
    try:
        with time_limit(1.0):
            ds = sp.simplify(d.replace(sp.sign, lambda a: a / sp.Abs(a)))
    except Exception:
        ds = None
    if ds is not None and sp.count_ops(ds) > 4 * sp.count_ops(f) + 20:
        ds = None
    if check.status in (Verification.SYMBOLIC, Verification.STRUCTURAL):
        how = "exact symbolic simplification"
    else:
        how = f"numerical comparison at {check.n_points} sample points (symbolic simplification was inconclusive)"
    return Step(
        rule="verify_derivative", title=Msg("Verify by differentiation"), before=sp.Derivative(F, x),
        after=ds if ds is not None else f,
        explanation=Msg("Differentiate the result and compare with the integrand {f}: the derivative matches, so the "
                        "antiderivative is correct (checked by {how}).", f=f, how=how),
        verification=check.status, check_detail=check.detail)


@dataclass
class Antiderivative:
    F: sp.Expr
    steps: List[Step]
    solution: Solution
    check: CheckResult
    classification: Classification
    integrand: sp.Expr


def domain_conditions(f, x) -> List[Msg]:
    out: List[Msg] = []
    ft = analyze(f, x)
    if ft.has_trig or ft.has_hyp:  # sp.singularities is slow/unreliable for periodic functions
        return out
    try:
        with time_limit(2.0):
            pts = sp.singularities(f, x)
        reals = [p for p in pts if p.is_real] if hasattr(pts, "__iter__") else []
        for p in reals[:4]:
            out.append(Msg("The antiderivative holds on each interval where the integrand is continuous; do not use "
                           "it across the singular point {pt}.", pt=sp.Eq(x, p)))
    except Exception:
        pass
    return out


def solve_antiderivative(f, x, engine: Engine) -> Optional[Antiderivative]:
    """Find a verified antiderivative with its derivation (including the final verification step)."""
    f0 = normalize(f, x)
    sol = engine.solve_top(f0, x)
    if sol is None:
        return None
    F = tidy(sol.F, f0, x)
    check = check_derivative(F, f0, x, symbolic_timeout=engine.options.symbolic_check_timeout * 1.5, force=True)
    if not check.ok:
        F = sol.F
        check = check_derivative(F, f0, x, symbolic_timeout=engine.options.symbolic_check_timeout * 1.5, force=True)
        if not check.ok:
            return None
    steps = list(sol.steps)
    if F != sol.F:
        steps.append(Step(rule="simplify", title=Msg("Simplify the result"), before=sol.F, after=F,
                          explanation=Msg("Rewrite the antiderivative in a more compact equivalent form."),
                          verification=check.status, check_detail=check.detail))
    steps.append(verification_step(F, f0, x, check))
    return Antiderivative(F, steps, sol, check, classify_closed_form(F), f0)


# ---------------------------------------------------------------------------------------------------------
def _new_result(problem: IntegralProblem, status: Status, cls: Classification) -> SolveResult:
    return SolveResult(status=status, classification=cls, mode="definite" if problem.definite else "indefinite",
                       integrand=problem.integrand, var=problem.var, lower=problem.lower, upper=problem.upper)


def failure_result(problem: IntegralProblem, cls: Classification, message: Msg) -> SolveResult:
    r = _new_result(problem, Status.FAILED, cls)
    r.message = message
    return r


def failure_classification(engine: Engine, f, x):
    info = engine.failure_info.get(Engine.key(normalize(f, x), x), {})
    if info.get("classification") is Classification.NON_ELEMENTARY:
        return Classification.NON_ELEMENTARY, Msg(
            "No elementary antiderivative exists: the Risch algorithm (as implemented in SymPy) decided this "
            "integral is non-elementary. This engine has no recognised special-function representation for it.")
    if engine.timed_out:
        return Classification.COULD_NOT_SOLVE, Msg(
            "The engine could not solve this integral within the time limit. This does not mean no closed form "
            "exists.")
    risch = info.get("risch", "")
    extra = f" (Risch check: {risch})" if risch else ""
    return Classification.NO_CLOSED_FORM, Msg(
        "No closed form was found by the supported techniques or by SymPy's general algorithms" + extra +
        ". This is NOT a proof that none exists — the integral may have an antiderivative this engine cannot "
        "construct.")


def _verification_summary(steps: List[Step]) -> Dict[str, int]:
    counts = {"symbolic": 0, "numeric": 0, "unchecked": 0}
    for s in steps:
        for t in s.walk():
            if t.verification in (Verification.SYMBOLIC, Verification.STRUCTURAL):
                counts["symbolic"] += 1
            elif t.verification is Verification.NUMERIC:
                counts["numeric"] += 1
            else:
                counts["unchecked"] += 1
    return counts


def solve_indefinite(problem: IntegralProblem, engine: Engine) -> SolveResult:
    x, f = problem.var, problem.integrand
    f0 = normalize(f, x)
    analysis = analyze(f0, x).notes
    anti = solve_antiderivative(f, x, engine)
    if anti is None:
        cls, msg = failure_classification(engine, f, x)
        r = failure_result(problem, cls, msg)
        r.analysis = analysis
        return r
    r = _new_result(problem, Status.SOLVED, anti.classification)
    r.analysis = analysis
    r.antiderivative = anti.F
    r.steps = anti.steps
    r.technique = anti.solution.technique
    r.derivation_level = anti.solution.derivation_level
    r.verification_status = "verified" if anti.check.status in (Verification.SYMBOLIC, Verification.STRUCTURAL) \
        else "verified_numeric"
    counts = _verification_summary(anti.steps)
    r.checks = [
        {"name": "Derivative of the antiderivative equals the integrand", "status": anti.check.status.value,
         "detail": anti.check.detail},
        {"name": "Derivation steps", "status": "verified",
         "detail": f"{counts['symbolic']} proven symbolically, {counts['numeric']} confirmed numerically only, "
                   f"{counts['unchecked']} explanatory"}]
    r.conditions = list(anti.solution.conditions) + domain_conditions(anti.integrand, x)
    specials = special_functions_in(anti.F)
    if specials:
        note = "The antiderivative involves the special function(s) " + ", ".join(sorted(specials)) + "."
        diag = engine.failure_info.get(Engine.key(anti.integrand, x), {})
        if diag.get("classification") is Classification.NON_ELEMENTARY:
            note += (" The Risch algorithm established that no elementary antiderivative exists, so a special-function "
                     "form is the best possible closed form.")
        r.message = Msg(note)
    return r


def solve_problem(problem: IntegralProblem, options: Optional[EngineOptions] = None) -> SolveResult:
    options = options or EngineOptions()
    t0 = time.monotonic()
    engine = Engine(options)
    set_symbolic_budget(0.4 * options.timeout_seconds)
    try:
        with time_limit(options.timeout_seconds):
            if problem.definite:
                from .definite import solve_definite
                result = solve_definite(problem, engine)
            else:
                result = solve_indefinite(problem, engine)
    except EngineTimeout:
        result = failure_result(problem, Classification.COULD_NOT_SOLVE,
                                Msg("The time limit of {t} seconds was reached before a verified result was found.",
                                    t=str(options.timeout_seconds)))
    except Exception as exc:
        if os.environ.get("INTEGRALIS_DEBUG"):
            raise
        result = failure_result(problem, Classification.COULD_NOT_SOLVE,
                                Msg("An internal error occurred ({e}). No result is shown rather than an unverified one.",
                                    e=f"{type(exc).__name__}: {exc}"))
    result.elapsed = time.monotonic() - t0
    result.trace = list(engine.trace)
    result.warnings = list(dict.fromkeys(list(problem.warnings) + list(result.warnings)))
    result.problem_latex = problem.latex()
    result.problem_text = problem.text()
    return result


def solve(expression: str, variable: Optional[str] = None, lower: Optional[str] = None,
          upper: Optional[str] = None, mode: Optional[str] = None,
          assumptions: Optional[Dict[str, str]] = None, options: Optional[EngineOptions] = None) -> SolveResult:
    """Parse and solve. Raises :class:`ParseError` for malformed input; mathematical failures are returned as a
    ``SolveResult`` with ``status == FAILED`` and an explanatory message."""
    options = options or EngineOptions()
    problem = parse_problem(expression, variable, lower, upper, mode, assumptions, options.log_is_natural)
    return solve_problem(problem, options)
