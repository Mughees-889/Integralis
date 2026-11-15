"""The orchestration layer: strategy selection, recursion, verification and fallback.

Separation of concerns
  * strategy selection   -> ``strategy.features`` / ``strategy.selector`` (structure only, no integration)
  * symbolic computation -> inside each technique, delegated to SymPy
  * derivation building  -> ``derivation.make_step`` (every step verified at construction)
  * verification         -> ``verify`` (applied to every sub-result and to the final result)
  * presentation         -> ``render`` / ``SolveResult.to_dict``
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import sympy as sp

from .config import EngineOptions
from .errors import EngineTimeout, TechniqueFailure, UnjustifiedStepError
from .models import CheckResult, Solution, TraceEntry, Verification
from .strategy.features import analyze
from .strategy.selector import rank
from .techniques.base import Context
from .utils.timeouts import outermost_deadline_passed, time_limit
from .verify import check_derivative

_CANON = sp.Symbol("_v", real=True)
MAX_TRACE = 400


class Engine:
    def __init__(self, options: Optional[EngineOptions] = None, techniques=None):
        from .techniques import default_techniques
        self.options = options or EngineOptions()
        self.techniques = list(techniques) if techniques is not None else default_techniques()
        self.trace: List[TraceEntry] = []
        self.failure_info: Dict[Any, dict] = {}
        self.allow_fallback = False
        self.timed_out = False
        self._cache: Dict[Any, Solution] = {}
        self._failed: set = set()
        self._path: List[str] = []
        self._cycle_hits = 0

    # -- keys -----------------------------------------------------------------------------------------
    @staticmethod
    def key(f: sp.Expr, x: sp.Symbol):
        return (sp.srepr(f), x.name)

    @staticmethod
    def cycle_key(f: sp.Expr, x: sp.Symbol) -> str:
        return sp.srepr(f.subs(x, _CANON))

    def check_deadline(self):
        if outermost_deadline_passed():
            raise EngineTimeout("time limit exceeded")

    # -- public entry ---------------------------------------------------------------------------------
    def solve_top(self, f: sp.Expr, x: sp.Symbol) -> Optional[Solution]:
        """Two passes: first with human-style techniques only (full derivations), then allowing SymPy's general
        algorithms at any level (partial or result-only derivations)."""
        for allow in (False, True):
            self.allow_fallback = allow
            self._failed.clear()
            if not allow and not self.options.enable_sympy_fallback:
                pass
            try:
                if allow:
                    sol = self.integrate(f, x, 0)
                else:
                    with time_limit(self.options.timeout_seconds * 0.5):
                        sol = self.integrate(f, x, 0)
            except EngineTimeout:
                self.timed_out = True
                if outermost_deadline_passed():
                    raise
                sol = None
            if sol is not None:
                return sol
            if not self.options.enable_sympy_fallback:
                break
        return None

    # -- recursion ------------------------------------------------------------------------------------
    def integrate(self, f: sp.Expr, x: sp.Symbol, depth: int = 0) -> Optional[Solution]:
        f = sp.sympify(f)
        if depth > self.options.max_depth:
            return None
        self.check_deadline()
        key, ckey = self.key(f, x), self.cycle_key(f, x)
        if key in self._cache:
            return self._cache[key]
        if ckey in self._path:
            self._cycle_hits += 1
            return None
        if key in self._failed:
            return None
        self._path.append(ckey)
        hits_before = self._cycle_hits
        try:
            sol = self._integrate_uncached(f, x, depth)
        finally:
            self._path.pop()
        if sol is not None:
            self._cache[key] = sol
        elif self._cycle_hits == hits_before:  # failures influenced by a cycle are path-dependent: do not cache
            self._failed.add(key)
        return sol

    def _record(self, entry: TraceEntry):
        if len(self.trace) < MAX_TRACE:
            self.trace.append(entry)

    def _integrate_uncached(self, f: sp.Expr, x: sp.Symbol, depth: int) -> Optional[Solution]:
        feats = analyze(f, x)
        ctx = Context(self, depth)
        for tech, score, reason in rank(self.techniques, feats):
            self.check_deadline()
            entry = TraceEntry(tech.name, score, reason, "inapplicable", "", depth)
            self._record(entry)
            try:
                with time_limit(self.options.technique_timeout):
                    sol = tech.attempt(f, x, ctx)
            except EngineTimeout:
                self.timed_out = True
                if outermost_deadline_passed():
                    raise
                entry.outcome, entry.detail = "timeout", "technique time limit reached"
                continue
            except (TechniqueFailure, UnjustifiedStepError) as exc:
                entry.outcome, entry.detail = "failed", str(exc)[:300]
                continue
            except RecursionError:
                entry.outcome, entry.detail = "failed", "recursion limit"
                continue
            except Exception as exc:  # SymPy internals (NotImplementedError, ValueError, ...)
                entry.outcome, entry.detail = "failed", f"{type(exc).__name__}: {exc}"[:300]
                continue
            if sol is None:
                continue
            tmo = self.options.symbolic_check_timeout if depth == 0 else 1.0
            check = check_derivative(sol.F, f, x, symbolic_timeout=tmo)
            if check.status is Verification.FAILED:
                entry.outcome, entry.detail = "unverified", "result rejected: " + check.detail
                continue
            sol.verification = check
            entry.outcome = "applied"
            return sol
        return None
