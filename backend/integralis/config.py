"""Tunable options for the integration engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineOptions:
    #: wall-clock budget for one complete solve (soft limit, enforced with SIGALRM when possible)
    timeout_seconds: float = 30.0
    #: budget for a single technique attempt (nested attempts share the outer budget)
    technique_timeout: float = 12.0
    #: budget for one symbolic equivalence check
    symbolic_check_timeout: float = 3.0
    #: maximum recursion depth of nested sub-integrals
    max_depth: int = 9
    #: substitution candidates tried per integrand
    max_sub_candidates: int = 6
    #: working precision of numerical cross-checks
    numeric_dps: int = 30
    #: digits shown for decimal approximations
    decimal_places: int = 20
    #: read log(x) as the natural logarithm (university convention). False reads it as log10.
    log_is_natural: bool = True
    #: allow SymPy's general algorithms (Risch / Meijer-G / heuristics) as a last resort
    enable_sympy_fallback: bool = True
