"""Definite and improper integrals.

Route 1 (exact): a verified antiderivative F is evaluated between the limits. The real line is cut at every point
where the integrand OR the antiderivative is singular, and each piece is evaluated with one-sided limits, so
  * an interior singularity of the integrand is detected (and divergence reported, never a finite number);
  * a spurious jump of F where the integrand is continuous (e.g. F = arctan(tan(x/2)/√3)) is corrected for.
Route 2 (special functions): SymPy's definite algorithms, accepted only after a rigorous convergence test and an
independent high-precision quadrature agreement.
Symmetry (even/odd) and the King property are tried first when they apply. Quadrature is only ever a cross-check.
"""
from __future__ import annotations

from typing import List, Optional

import sympy as sp

from .closed_form import special_functions_in
from .definite_support import (trig_poles, convergence_test, digits_of_agreement, discontinuities, is_inf, numeric_integral,
                               numeric_order, numeric_probe_limit, one_sided_limit, value_or_limit)
from .derivation import make_step
from .engine import Engine
from .errors import EngineTimeout
from .models import CheckResult, Classification, Claim, SolveResult, Status, Step, Verification
from .parsing import IntegralProblem
from .render import Msg
from .solver import _new_result, failure_result, normalize, solve_antiderivative
from .strategy.features import analyze
from .utils.timeouts import time_limit
from .verify import check_identity, numeric_compare, _param_value


def _D(f, x, a, b):
    return sp.Integral(f, (x, a, b))


def _same_pt(p, q) -> bool:
    try:
        return p == q or (p.is_number and q.is_number and abs(sp.N(p - q, 30)) < 1e-25)
    except Exception:
        return False


def _sort_points(pts: List[sp.Expr]) -> List[sp.Expr]:
    """Sort break points numerically and drop duplicates. Symbolic (parametric) points keep their given order."""
    def key(p):
        return float(sp.N(p)) if not is_inf(p) else (1e300 if p == sp.oo else -1e300)
    try:
        ordered = sorted(pts, key=key)
    except (TypeError, ValueError):
        ordered = list(pts)
    out: List[sp.Expr] = []
    for p in ordered:
        if not any(_same_pt(p, q) for q in out):
            out.append(p)
    return out


def _finish_numeric(f, x, pts, value, params) -> Optional[CheckResult]:
    """Independent quadrature check of an exact value. None = unavailable, FAILED = disagreement."""
    if params or value.has(sp.I):
        return None
    num = numeric_integral(f, x, pts)
    if num is None:
        return None
    d = digits_of_agreement(value, num)
    if d is None:
        return CheckResult(Verification.FAILED, "quadrature", f"quadrature gives {num} which disagrees with the exact value")
    if d < 10:
        return CheckResult(Verification.FAILED, "quadrature", f"only {d} digits agree with quadrature")
    return CheckResult(Verification.NUMERIC, "quadrature", f"mpmath quadrature (30 digits) agrees with the exact value to "
                       f"≥ {min(d, 28)} digits (independent check, not a proof)", d)


def _sample_check(f, x, a, b, value, params) -> Optional[CheckResult]:
    """Parametric results: compare with quadrature at sample parameter values where the integrand is continuous."""
    good, tried = 0, 0
    syms = sorted(params, key=lambda s: s.name)
    for k in range(6):
        sub = {s: sp.Rational(str(float(_param_value(s, k, j)))) for j, s in enumerate(syms)}
        fs, as_, bs = f.subs(sub), a.subs(sub), b.subs(sub)
        try:
            vs = sp.N(value.subs(sub), 30)
        except Exception:
            continue
        if vs.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or vs.has(sp.I) or numeric_order(as_, bs) is not True:
            continue
        d = discontinuities(fs, x, as_, bs, timeout=1.5)
        if d.points or d.undefined_on_interval or not d.certified:
            continue
        tried += 1
        num = numeric_integral(fs, x, [as_, bs])
        if num is not None and (digits_of_agreement(vs, num) or 0) >= 10:
            good += 1
        elif num is not None:
            return CheckResult(Verification.FAILED, "quadrature", f"disagrees with quadrature at the sample {sub}")
        if good >= 3:
            break
    if good:
        return CheckResult(Verification.NUMERIC, "quadrature", f"matches high-precision quadrature at {good} sample "
                           "parameter values (independent check, not a proof)", good)
    return None


def _result(problem, value, steps, level, technique, conds, checks, extra_warn, cls=None) -> SolveResult:
    r = _new_result(problem, Status.SOLVED, cls or (Classification.SPECIAL_FUNCTION if special_functions_in(value)
                                                    else Classification.EXACT_VALUE))
    r.value, r.steps, r.derivation_level, r.technique, r.conditions = value, steps, level, technique, conds
    r.checks = checks
    r.warnings = extra_warn
    weakest_numeric = any(c["status"] == "numeric" for c in checks[:1])
    r.verification_status = "verified_numeric" if weakest_numeric else "verified"
    try:
        v = sp.N(value, 20)
        if not v.free_symbols and not v.has(sp.I):
            r.decimal = str(v)
    except Exception:
        pass
    return r


def _fail(problem, msg: str, steps=None) -> SolveResult:
    r = failure_result(problem, Classification.COULD_NOT_SOLVE, Msg(msg))
    r.steps = steps or []
    return r


# ------------------------------------------------------------------------------------------------------------
def solve_definite(problem: IntegralProblem, engine: Engine) -> SolveResult:
    x, a, b = problem.var, problem.lower, problem.upper
    f = normalize(problem.integrand, x)
    params = (f.free_symbols | a.free_symbols | b.free_symbols) - {x}
    steps: List[Step] = []
    warnings: List[str] = []
    I0 = _D(f, x, a, b)
    steps.append(make_step("setup", Msg("The definite integral"), I0, I0,
                           Msg("Evaluate the integral of {f} over the interval from {a} to {b}.", f=f, a=a, b=b)))
    analysis = analyze(f, x).notes

    def done(r: SolveResult) -> SolveResult:
        r.analysis = analysis
        return r

    # -- trivial / reversed limits ---------------------------------------------------------------------------
    if a == b:
        steps.append(make_step("zero_width", Msg("Equal limits"), I0, sp.Integer(0),
                               Msg("The two limits coincide, so the integral is 0.")))
        return done(_result(problem, sp.Integer(0), steps, "full", "trivial", [], [], warnings))
    sign = 1
    if numeric_order(a, b) is False:
        steps.append(make_step("reverse_limits", Msg("Reverse the limits"), I0, -_D(f, x, b, a),
                               Msg("Swapping the limits changes the sign: ∫ from {a} to {b} = −∫ from {b} to {a}.", a=a, b=b)))
        a, b, sign = b, a, -1

    # -- continuity analysis ------------------------------------------------------------------------------------
    disc = discontinuities(f, x, a, b)
    if disc.undefined_on_interval:
        return done(_fail(problem, "The integrand is not real-valued on part of the interval of integration, so the "
                                   "definite integral over the reals is not defined.", steps))
    sing = disc.points
    inf_ends = [p for p in (a, b) if is_inf(p)]
    endpoint_sing = [p for p in sing if _same_pt(p, a) or _same_pt(p, b)]
    interior_sing = [p for p in sing if p not in endpoint_sing]
    proper = not inf_ends and not sing
    bits = []
    if inf_ends:
        bits.append("a limit is infinite")
    if endpoint_sing:
        bits.append("the integrand is undefined or unbounded at an endpoint (" + ", ".join(str(p) for p in endpoint_sing) + ")")
    if interior_sing:
        bits.append("the integrand is singular inside the interval at " + ", ".join(str(p) for p in interior_sing) +
                    ", so the integral is split there and each side must converge on its own")
    if proper:
        analysis_msg = Msg("Both limits are finite and the integrand is continuous on [{a}, {b}]" +
                           ("" if disc.certified else " (not rigorously certified: a heuristic singularity search was used)") +
                           ": this is a proper integral.", a=a, b=b)
    else:
        analysis_msg = Msg("This is an improper integral: " + "; ".join(bits) + ". Convergence is decided before any "
                           "value is reported.")
    if params:
        warnings.append("Parameters (" + ", ".join(sorted(s.name for s in params)) + ") are treated as real constants; the "
                        "result is valid where the integrand stays continuous on the interval.")
    steps.append(make_step("analyse_integral", Msg("Check continuity and improper points"), I0, I0, analysis_msg))

    # -- symmetry (even / odd) on [-c, c] -------------------------------------------------------------------------
    if proper and not params and not is_inf(a) and sp.simplify(a + b) == 0:
        fm = f.subs(x, -x)
        odd = check_identity(fm, -f, x, symbolic_timeout=1.5)
        if odd.ok:
            steps.append(make_step("odd_symmetry", Msg("Odd integrand on a symmetric interval"), I0, sp.Integer(0),
                                   Msg("Since f(−{x}) = −f({x}) and the limits are symmetric, the contributions from "
                                       "[{a}, 0] and [0, {b}] cancel.", x=x, a=a, b=b),
                                   claims=[Claim.identity(fm, -f, x, "f(−x) = −f(x)")],
                                   formula=r"\int_{-c}^{c}f(x)\,dx=0\ \text{ for odd }f"))
            chk = _finish_numeric(f, x, [a, b], sp.Integer(0), params)
            if chk is None or chk.ok:
                checks = [{"name": "Symmetry argument", "status": odd.status.value, "detail": odd.detail}]
                if chk:
                    checks.append({"name": "Independent numerical quadrature", "status": chk.status.value, "detail": chk.detail})
                return done(_result(problem, sp.Integer(0) * sign, steps, "full", "symmetry", [], checks, warnings))
        else:
            even = check_identity(fm, f, x, symbolic_timeout=1.5)
            if even.ok:
                half = solve_definite(IntegralProblem(f, x, sp.Integer(0), b), engine)
                if half.solved and half.value is not None:
                    steps.append(make_step("even_symmetry", Msg("Even integrand on a symmetric interval"), I0,
                                           2 * _D(f, x, 0, b),
                                           Msg("Since f(−{x}) = f({x}), the two halves are equal: the integral is twice the "
                                               "integral over [0, {b}].", x=x, b=b),
                                           claims=[Claim.identity(fm, f, x, "f(−x) = f(x)")],
                                           formula=r"\int_{-c}^{c}f(x)\,dx=2\int_0^{c}f(x)\,dx\ \text{ for even }f"))
                    steps.append(make_step("half_interval", Msg("Evaluate over [0, {b}]", b=b), _D(f, x, 0, b), half.value,
                                           Msg("The half-interval integral is evaluated in the sub-steps."),
                                           substeps=half.steps[1:], trust=CheckResult(Verification.SYMBOLIC, "sub-result",
                                                                                      "verified sub-result")))
                    val = sp.simplify(2 * half.value)
                    steps.append(make_step("double", Msg("Double the result"), 2 * half.value, val,
                                           Msg("Multiply by 2."), claims=[Claim.identity(2 * half.value, val, x)]))
                    checks = list(half.checks) + [{"name": "Even-symmetry argument", "status": even.status.value, "detail": even.detail}]
                    return done(_result(problem, val * sign, steps, half.derivation_level, "symmetry", half.conditions,
                                        checks, warnings))

    # -- King property: f(x) + f(a+b-x) constant ---------------------------------------------------------------------
    if proper and not params and not is_inf(a) and not is_inf(b) and not f.is_polynomial(x):
        try:
            g = f.subs(x, a + b - x)
            with time_limit(2.5):
                s = sp.simplify(f + g)
            if not s.has(x) and s.is_number and s != 0 or (not s.has(x) and s.is_number):
                val = sp.simplify(s * (b - a) / 2)
                chk = check_identity(f + g, s, x, symbolic_timeout=1.5)
                if chk.ok:
                    steps.append(make_step(
                        "king_property", Msg("Reflect the variable: x → {a} + {b} − x", a=a, b=b), I0, _D(g, x, a, b),
                        Msg("The substitution u = {a} + {b} − {x} swaps the limits and gives ∫ f(x) dx = ∫ f({a}+{b}−x) dx. "
                            "Adding the two forms: 2I = ∫ [f(x) + f({a}+{b}−x)] dx.", a=a, b=b, x=x),
                        formula=r"\int_a^b f(x)\,dx=\int_a^b f(a+b-x)\,dx"))
                    steps.append(make_step(
                        "king_sum", Msg("The sum is constant"), _D(f + g, x, a, b), _D(s, x, a, b),
                        Msg("f(x) + f({a}+{b}−x) simplifies to the constant {s}.", a=a, b=b, s=s),
                        claims=[Claim.identity(f + g, s, x, "the two reflected integrands add up to a constant")]))
                    steps.append(make_step("king_result", Msg("Solve for the integral"), 2 * sp.Symbol("I"), val * 2,
                                           Msg("2I = {s}·({b} − {a}), so I = {v}.", s=s, a=a, b=b, v=val)))
                    nchk = _finish_numeric(f, x, [a, b], val, params)
                    if nchk is None or nchk.ok:
                        checks = [{"name": "Reflection identity", "status": chk.status.value, "detail": chk.detail}]
                        if nchk:
                            checks.append({"name": "Independent numerical quadrature", "status": nchk.status.value, "detail": nchk.detail})
                        return done(_result(problem, val * sign, steps, "full", "king_property", [], checks, warnings))
        except EngineTimeout:
            pass
        except Exception:
            pass

    # -- Route 1: antiderivative ----------------------------------------------------------------------------------
    anti = solve_antiderivative(f, x, engine)
    if anti is not None:
        out = _evaluate_with_antiderivative(problem, f, x, a, b, anti, sing, params, steps, warnings, sign, engine, disc)
        if out is not None:
            return done(out)
    # -- Route 2: SymPy definite + rigorous convergence tests ---------------------------------------------------------
    return done(_sympy_route(problem, f, x, a, b, sing, params, steps, warnings, sign))


# ------------------------------------------------------------------------------------------------------------
def _evaluate_with_antiderivative(problem, f, x, a, b, anti, sing, params, steps, warnings, sign, engine, disc):
    F = anti.F
    I0 = _D(f, x, a, b)
    steps.append(make_step("find_antiderivative", Msg("Find an antiderivative"), sp.Integral(f, x), F,
                           Msg("An antiderivative F of the integrand is found (indefinite derivation below) and verified "
                               "by differentiation."), substeps=anti.steps, trust=anti.check))
    discF = discontinuities(F, x, a, b) if not params else discontinuities(F, x, a, b, timeout=1.0)
    if not discF.points and not is_inf(a) and not is_inf(b):
        extra = trig_poles(F, x, a, b)
        if extra:
            discF.points = _sort_points(list(extra))
    brk = _sort_points([a, b] + list(sing) + list(discF.points))
    f_special = set(map(sp.srepr, sing))
    spurious = [p for p in discF.points if sp.srepr(p) not in f_special and not _same_pt(p, a) and not _same_pt(p, b)]
    if spurious:
        steps.append(make_step("antiderivative_jumps", Msg("The antiderivative is not continuous everywhere"),
                               F, F, Msg("F is undefined at {pts} although the integrand is continuous there. The interval "
                                         "is therefore split at these points and one-sided limits of F are used, which "
                                         "accounts for the jump.", pts=", ".join(map(str, spurious)))))
    total, parts = sp.Integer(0), []
    divergent = None
    special_pts = list(sing) + list(discF.points)

    def endpoint_info(p, side):
        if is_inf(p) or any(_same_pt(p, s_) for s_ in special_pts):
            return one_sided_limit(F, x, p, side)
        return value_or_limit(F, x, p, side)

    for lo, hi in zip(brk[:-1], brk[1:]):
        lo_info = endpoint_info(lo, "+")
        hi_info = endpoint_info(hi, "-")
        for p, side, info in ((lo, "+", lo_info), (hi, "-", hi_info)):
            if info.kind in ("infinite", "dne"):
                divergent = (p, side, info)
                break
            if info.kind == "unknown":
                return _fail(problem, f"The limit of the antiderivative at {p} could not be determined"
                             + (" (give assumptions on the parameters, e.g. a > 0)" if params else "") +
                             ", so convergence could not be decided and no value is reported.", steps)
        if divergent:
            break
        parts.append((lo, hi, lo_info.value, hi_info.value))
        total += hi_info.value - lo_info.value

    # limit steps (shown even for finite endpoints that needed a limit)
    for lo, hi, lv, hv in parts:
        pass
    shown = set()
    infos = []
    for lo, hi, lv, hv in parts:
        infos.append((hi, "-", hv))
        infos.append((lo, "+", lv))
    def limit_step(p, side, val, label):
        needs = is_inf(p) or any(_same_pt(p, s) for s in list(sing) + list(discF.points))
        if not needs:
            return None
        key = (sp.srepr(p), side)
        if key in shown:
            return None
        shown.add(key)
        if is_inf(p):
            lim = sp.Limit(F, x, p)
        else:
            lim = sp.Limit(F, x, p, "+" if side == "+" else "-")
        ok = numeric_probe_limit(F, x, p, side, val)
        return Step(rule="limit", title=Msg(label), before=lim, after=val,
                    explanation=Msg("Evaluate the one-sided limit of F at the improper/singular point {p}.", p=p),
                    verification=Verification.NUMERIC if ok else Verification.UNCHECKED,
                    check_detail="F evaluated extremely close to the point agrees with the limit" if ok else
                    "limit computed symbolically; numerical probe inconclusive")

    if divergent:
        p, side, info = divergent
        st = limit_step_infinite(F, x, p, side, info)
        steps.append(st)
        verdict, why = convergence_test(f, x, p, side)
        msg = (f"The integral DIVERGES: the antiderivative {'tends to ' + str(info.value) if info.kind == 'infinite' else 'has no limit'} "
               f"as x → {p}{'⁺' if side == '+' else '⁻' if not is_inf(p) else ''}, so the improper integral has no finite value. "
               + (f"Independent comparison test: {why}. " if verdict == 'diverges' else "")
               + ("No finite value is assigned (a Cauchy principal value, if it exists, is a different quantity)."
                  if not is_inf(p) and p in sing and any(_same_pt(p, s) for s in sing) and not (_same_pt(p, a) or _same_pt(p, b)) else ""))
        steps.append(Step(rule="divergence", title=Msg("Conclusion: the integral diverges"), before=I0, after=None,
                          explanation=Msg(msg), verification=Verification.NUMERIC if st.verification is Verification.NUMERIC else Verification.UNCHECKED,
                          check_detail=st.check_detail))
        r = _new_result(problem, Status.DIVERGENT, Classification.DIVERGENT)
        r.steps = steps
        r.message = Msg(msg)
        r.technique = anti.solution.technique
        r.derivation_level = anti.solution.derivation_level
        r.verification_status = "verified_numeric" if st.verification is Verification.NUMERIC else "verified"
        r.checks = [{"name": "Limit of the antiderivative", "status": st.verification.value, "detail": st.check_detail}]
        r.warnings = warnings
        return r

    for lo, hi, lv, hv in parts:
        def lbl(p, side):
            return f"Limit of F as {x} → {p}" + ("" if is_inf(p) else ("⁺" if side == "+" else "⁻"))
        for st in (limit_step(lo, "+", lv, lbl(lo, "+")), limit_step(hi, "-", hv, lbl(hi, "-"))):
            if st is not None:
                steps.append(st)
    raw = sp.Add(*[hv - lv for _, _, lv, hv in parts], evaluate=False) if len(parts) > 1 else (parts[0][3] - parts[0][2])
    try:
        with time_limit(3.0):
            value = sp.simplify(total)
    except EngineTimeout:
        value = total
    if value.has(sp.I):
        try:
            if abs(complex(sp.N(sp.im(value), 30))) < 1e-20:
                value = sp.simplify(sp.re(value))
        except Exception:
            pass
    if value.has(sp.zoo) or value.has(sp.nan):
        return _fail(problem, "The evaluation produced an undefined expression; no result is shown.", steps)
    steps.append(make_step("evaluate_bounds", Msg("Evaluate F at the limits and subtract"), I0, value,
                           Msg("Subtract the value of F at the lower limit from its value at the upper limit"
                               + (" (summing the pieces between the break points)" if len(parts) > 1 else "") + "."),
                           claims=[Claim.identity(sp.sympify(raw), value, x, "simplification keeps the value")],
                           formula=r"\int_a^b f\,dx=F(b)-F(a)"))
    # alternative view with a substitution and transformed limits
    _substitution_view(f, x, a, b, value, anti, steps, engine, params, sign)
    # independent numerical cross-check
    checks = [{"name": "Antiderivative differentiates to the integrand", "status": anti.check.status.value,
               "detail": anti.check.detail}]
    if params:
        nchk = _sample_check(f, x, a, b, value, params)
        free = ", ".join(sorted(s.name for s in params))
    else:
        nchk = _finish_numeric(f, x, brk, value, params)
    if nchk is not None and not nchk.ok:
        return _fail(problem, "The exact evaluation disagrees with independent high-precision quadrature (" + nchk.detail +
                     "). No result is shown rather than an unverified one.", steps)
    if nchk is not None:
        steps.append(Step(rule="numeric_check", title=Msg("Numerical cross-check"), before=value,
                          after=sp.N(value, 15) if not value.free_symbols else value,
                          explanation=Msg("An independent arbitrary-precision quadrature of the original integral agrees. "
                                          "This is a sanity check, not a proof."),
                          verification=nchk.status, check_detail=nchk.detail))
        checks.append({"name": "Independent numerical quadrature", "status": nchk.status.value, "detail": nchk.detail})
    else:
        warnings.append("No independent numerical cross-check was possible for this integral.")
    conds = list(anti.solution.conditions)
    if params:
        conds.append(Msg("Valid for parameter values that keep the integrand continuous on the interval of integration."))
    if not disc.certified:
        warnings.append("Continuity of the integrand could not be rigorously certified; a heuristic search found no singularities.")
    return _result(problem, value * sign, steps, anti.solution.derivation_level, anti.solution.technique, conds, checks, warnings)


def limit_step_infinite(F, x, p, side, info) -> Step:
    lim = sp.Limit(F, x, p) if is_inf(p) else sp.Limit(F, x, p, "+" if side == "+" else "-")
    val = info.value if info.value is not None else sp.nan
    ok = info.kind == "infinite" and numeric_probe_limit(F, x, p, side, sp.Float("1e30") * (1 if val == sp.oo else -1)) is False
    probe = False
    try:
        import mpmath as mp
        f = sp.lambdify(x, F, modules=["mpmath"])
        with mp.workdps(40):
            q = (mp.mpf(10) ** 12 * (1 if p == sp.oo else -1)) if is_inf(p) else (mp.mpmathify(sp.N(p, 40)) + (1 if side == "+" else -1) * mp.mpf(10) ** -12)
            q2 = (mp.mpf(10) ** 20 * (1 if p == sp.oo else -1)) if is_inf(p) else (mp.mpmathify(sp.N(p, 40)) + (1 if side == "+" else -1) * mp.mpf(10) ** -20)
            v1, v2 = abs(mp.mpmathify(f(q))), abs(mp.mpmathify(f(q2)))
            probe = v2 > v1 * 2 and v2 > 1e6 if info.kind == "infinite" else True
    except Exception:
        probe = False
    return Step(rule="limit", title=Msg("Limit of F at the problem point"), before=lim, after=val,
                explanation=Msg("The one-sided limit of the antiderivative at {p} is not finite."), 
                verification=Verification.NUMERIC if probe else Verification.UNCHECKED,
                check_detail="|F| grows without bound as the point is approached" if probe else
                "limit computed symbolically; numerical probe inconclusive")


def _substitution_view(f, x, a, b, value, anti, steps, engine, params, sign):
    """If the antiderivative came from a substitution u = g(x), also show the integral with transformed limits."""
    meta = anti.solution.meta.get("substitution")
    if not meta or params or is_inf(a) or is_inf(b):
        return
    u, g, c, h0 = meta["u"], meta["g"], meta["c"], meta["h0"]
    try:
        if meta["method"] not in ("A", "A2", "B"):
            return
        ga, gb = sp.simplify(g.subs(x, a)), sp.simplify(g.subs(x, b))
        if any(v.has(sp.zoo, sp.nan, sp.oo, -sp.oo) for v in (ga, gb)):
            return
        if discontinuities(g, x, a, b, timeout=1.5).points or discontinuities(h0, u, *sorted([ga, gb], key=lambda v: float(sp.N(v))), timeout=1.5).points:
            return
        gp = sp.diff(g, x)
        zeros = sp.solveset(sp.Eq(gp, 0), x, sp.Interval.open(a, b))
        if zeros != sp.S.EmptySet:
            return
        sub = engine.integrate(h0, u, 1)
        if sub is None:
            return
        v2 = sp.simplify(c * (sub.F.subs(u, gb) - sub.F.subs(u, ga)))
        chk = check_identity(v2, value, x, symbolic_timeout=1.5)
        if not chk.ok:
            return
        steps.append(make_step(
            "transformed_limits", Msg("Alternative: change the limits with the substitution"), _D(f, x, a, b),
            c * sp.Integral(h0, (u, ga, gb)),
            Msg("With {u} = {g}, the limits become {u} = {ga} (at {x} = {a}) and {u} = {gb} (at {x} = {b}). The integral "
                "becomes the one shown, which gives {v}, the same value.", u=u, g=g, ga=ga, gb=gb, x=x, a=a, b=b, v=v2),
            claims=[Claim.identity(v2, value, x, "the transformed-limit evaluation equals the value")],
            conditions=[Msg("{g} is continuous and monotonic on [{a}, {b}]", g=g, a=a, b=b)]))
    except EngineTimeout:
        return
    except Exception:
        return


# ------------------------------------------------------------------------------------------------------------
def _sympy_route(problem, f, x, a, b, sing, params, steps, warnings, sign):
    """No elementary antiderivative: use SymPy's definite algorithms but only after convergence is established."""
    bad = []
    for p in _sort_points([a, b] + list(sing)):
        if is_inf(p):
            bad.append((p, "+"))
        elif any(_same_pt(p, s) for s in sing):
            if _same_pt(p, a):
                bad.append((p, "+"))
            elif _same_pt(p, b):
                bad.append((p, "-"))
            else:
                bad.extend([(p, "-"), (p, "+")])
    tests = []
    for p, side in bad:
        verdict, why = convergence_test(f, x, p, side)
        tests.append((p, side, verdict, why))
        if verdict == "diverges":
            msg = f"The integral DIVERGES: {why}."
            steps.append(Step(rule="divergence", title=Msg("Conclusion: the integral diverges"), before=_D(f, x, a, b), after=None,
                              explanation=Msg(msg), verification=Verification.SYMBOLIC,
                              check_detail="rigorous comparison test"))
            r = _new_result(problem, Status.DIVERGENT, Classification.DIVERGENT)
            r.steps, r.message, r.verification_status = steps, Msg(msg), "verified"
            r.checks = [{"name": "Convergence test", "status": "symbolic", "detail": why}]
            r.derivation_level = "partial"
            return r
    unknown = [t for t in tests if t[2] == "unknown"]
    if unknown:
        return _fail(problem, "No closed form was found for the antiderivative, and convergence at " +
                     ", ".join(f"{t[0]}" for t in unknown) + " could not be established by the comparison tests (" +
                     unknown[0][3] + "). A numerical value is deliberately not reported without convergence.", steps)
    for p, side, verdict, why in tests:
        steps.append(Step(rule="convergence_test", title=Msg("Convergence at {p}", p=p), before=_D(f, x, a, b), after=None,
                          explanation=Msg("Converges: " + why + "."), verification=Verification.SYMBOLIC,
                          check_detail="comparison test"))
    pts = _sort_points([a, b] + list(sing))
    total = sp.Integer(0)
    try:
        with time_limit(12.0):
            for lo, hi in zip(pts[:-1], pts[1:]):
                total += sp.integrate(f, (x, lo, hi))
    except EngineTimeout:
        return _fail(problem, "SymPy's definite-integration algorithms timed out; no value is reported.", steps)
    except Exception:
        return _fail(problem, "No closed form was found for this definite integral.", steps)
    if total.has(sp.Integral) or total.has(sp.Piecewise) or total.has(sp.nan) or total.has(sp.zoo) or total.has(sp.oo):
        return _fail(problem, "No closed form was found for this definite integral (SymPy returned an unevaluated or "
                              "conditional result).", steps)
    value = sp.simplify(total)
    if params:
        return _fail(problem, "A parametric closed form was not found; try giving assumptions on the parameters.", steps)
    chk = _finish_numeric(f, x, pts, value, params)
    if chk is None or not chk.ok:
        return _fail(problem, "SymPy returned a value but it could not be confirmed by independent high-precision "
                              "quadrature, so it is not shown." + (f" ({chk.detail})" if chk else ""), steps)
    steps.append(make_step("sympy_definite", Msg("Closed form from general-purpose algorithms"), _D(f, x, a, b), value,
                           Msg("No elementary antiderivative was available. The value comes from SymPy's definite-integration "
                               "algorithms (Meijer G / special functions); convergence was established above and the value is "
                               "confirmed by independent quadrature. The derivation of this value is not shown."),
                           trust=CheckResult(Verification.NUMERIC, "quadrature", chk.detail)))
    checks = [{"name": "Convergence tests", "status": "symbolic", "detail": "; ".join(t[3] for t in tests) or "proper integral"},
              {"name": "Independent numerical quadrature", "status": chk.status.value, "detail": chk.detail}]
    return _result(problem, value * sign, steps, "result_only", "sympy_definite", [], checks, warnings)
