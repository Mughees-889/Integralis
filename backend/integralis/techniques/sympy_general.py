"""Last-resort technique: SymPy's general algorithms (Risch, heuristic Risch, Meijer G, ...).

Whatever it returns is only accepted after the engine verifies it by differentiation. No derivation is invented:
the step says plainly that the result comes from a general algorithm and is justified by differentiation alone.
When it fails, the Risch decision procedure is consulted so that "no elementary antiderivative exists" is only
claimed when it is actually established.
"""
from __future__ import annotations

from typing import Optional

import sympy as sp

from ..closed_form import special_functions_in
from ..derivation import Int, make_step
from ..errors import EngineTimeout
from ..models import Classification, Claim, Solution
from ..render import Msg
from ..strategy.features import Features
from ..utils.timeouts import time_limit
from .base import Context, Technique


def realify_logs(F: sp.Expr) -> sp.Expr:
    """Replace ``log(g)`` by ``log(|g|)`` for real ``g`` of unknown sign (same derivative, valid on both signs)."""

    def cond(e):
        return isinstance(e, sp.log) and len(e.args) == 1

    def repl(e):
        arg = e.args[0]
        if arg.is_positive or arg.is_real is not True:
            return e
        return sp.log(sp.Abs(arg))

    return F.replace(cond, repl)


def risch_diagnosis(f, x, timeout: float = 6.0) -> dict:
    """Ask the Risch algorithm whether an elementary antiderivative exists (transcendental case only)."""
    info = {"classification": Classification.NO_CLOSED_FORM, "risch": "not applicable", "elementary_result": None}
    try:
        from sympy.integrals.risch import NonElementaryIntegral, risch_integrate
        with time_limit(timeout):
            r = risch_integrate(f, x)
        if r.has(NonElementaryIntegral):
            info["classification"] = Classification.NON_ELEMENTARY
            info["risch"] = "proved non-elementary"
        else:
            info["risch"] = "elementary result"
            info["elementary_result"] = r
    except EngineTimeout:
        raise
    except NotImplementedError as exc:
        info["risch"] = f"not applicable: {str(exc)[:80]}"
    except Exception as exc:  # pragma: no cover - defensive
        info["risch"] = f"failed: {type(exc).__name__}"
    return info


class SympyGeneral(Technique):
    name = "sympy_general"
    title = "SymPy general algorithms (last resort)"
    family = "fallback"
    description = ("Hands the integral to SymPy's general-purpose algorithms when none of the human-style "
                   "techniques works. The result is accepted only if it differentiates back to the integrand; "
                   "special-function results (erf, Ei, Si, li, polylog, ...) come from here.")

    def suitability(self, feats: Features):
        return 0.01, "Last resort: general-purpose algorithms."

    def attempt(self, f: sp.Expr, x: sp.Symbol, ctx: Context) -> Optional[Solution]:
        engine = ctx.engine
        if not (ctx.options.enable_sympy_fallback and engine.allow_fallback):
            return None
        result = None
        try:
            with time_limit(ctx.options.technique_timeout):
                result = sp.integrate(f, x)
        except EngineTimeout:
            raise
        except Exception:
            result = None
        if result is None or result.has(sp.Integral):
            diag = risch_diagnosis(f, x)
            engine.failure_info[engine.key(f, x)] = diag
            er = diag.get("elementary_result")
            if er is not None and not er.has(sp.Integral):
                result = er
            else:
                return None
        F = realify_logs(result)
        if special_functions_in(F) and engine.key(f, x) not in engine.failure_info:
            try:
                engine.failure_info[engine.key(f, x)] = risch_diagnosis(f, x, timeout=3.0)
            except EngineTimeout:
                pass
        conditions = []
        if isinstance(F, sp.Piecewise):
            for expr, cond in F.args:
                if cond is not sp.true and cond != True:  # noqa: E712
                    conditions.append(Msg("the branch {e} applies when {c}", e=expr, c=cond))
        step = make_step(
            "sympy_general", Msg("Closed form from general-purpose algorithms"), Int(f, x), F,
            Msg("None of the human-style techniques completed this integral, so the antiderivative comes from "
                "SymPy's general algorithms (Risch / heuristic Risch / Meijer G). How it was found is not shown; it "
                "is justified only by differentiating it, as checked below."),
            claims=[Claim.derivative(F, f, x, "differentiating the antiderivative gives the integrand")],
            conditions=conditions)
        return Solution(F, [step], self.name, conditions, derivation_level="result_only")
