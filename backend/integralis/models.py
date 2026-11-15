"""Core data model.

A derivation is a tree of :class:`Step` objects. Every step that makes a mathematical claim carries
:class:`Claim` objects; a step can only be constructed (see ``derivation.make_step``) after all of its claims
have been verified, so a derivation never contains an unverified transformation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import sympy as sp

from . import render
from .render import Msg, as_msg


class Verification(str, Enum):
    SYMBOLIC = "symbolic"      # proven by exact symbolic simplification
    STRUCTURAL = "structural"  # the two sides are syntactically identical
    NUMERIC = "numeric"        # agreement at many high-precision sample points only (weaker than a proof)
    UNCHECKED = "unchecked"    # explanatory step that asserts no new mathematical claim
    FAILED = "failed"


@dataclass
class CheckResult:
    status: Verification
    method: str = ""
    detail: str = ""
    n_points: int = 0

    @property
    def ok(self) -> bool:
        return self.status is not Verification.FAILED

    def to_dict(self) -> dict:
        return {"status": self.status.value, "method": self.method, "detail": self.detail,
                "n_points": self.n_points}


@dataclass
class Claim:
    """A checkable statement: ``lhs == rhs`` (identity) or ``d(lhs)/dx == rhs`` (derivative)."""
    kind: str
    lhs: sp.Basic
    rhs: sp.Basic
    x: sp.Symbol
    description: str = ""
    domain: Optional[Tuple[Optional[float], Optional[float]]] = None

    @staticmethod
    def identity(lhs, rhs, x, description: str = "", domain=None) -> "Claim":
        return Claim("identity", sp.sympify(lhs), sp.sympify(rhs), x, description, domain)

    @staticmethod
    def derivative(F, f, x, description: str = "", domain=None) -> "Claim":
        return Claim("derivative", sp.sympify(F), sp.sympify(f), x, description, domain)


def _pair(expr: Optional[sp.Basic]) -> Optional[dict]:
    if expr is None:
        return None
    return {"latex": render.latex(expr), "text": render.text(expr)}


def _msg_dict(m: Union[Msg, str]) -> dict:
    m = as_msg(m)
    return {"latex": m.latex(), "text": m.text()}


@dataclass
class Step:
    rule: str
    title: Union[Msg, str]
    before: Optional[sp.Basic]
    after: Optional[sp.Basic]
    explanation: Msg
    conditions: List[Msg] = field(default_factory=list)
    verification: Verification = Verification.UNCHECKED
    check_detail: str = ""
    claims: List[Claim] = field(default_factory=list)
    substeps: List["Step"] = field(default_factory=list)
    table: Optional[dict] = None
    formula: Optional[str] = None  # LaTeX of a general formula displayed with the step
    meta: Dict[str, Any] = field(default_factory=dict)

    def walk(self) -> Iterator["Step"]:
        yield self
        for s in self.substeps:
            yield from s.walk()

    def to_dict(self, number: str = "") -> dict:
        return {
            "number": number,
            "rule": self.rule,
            "title": _msg_dict(self.title),
            "before": _pair(self.before),
            "after": _pair(self.after),
            "explanation": _msg_dict(self.explanation),
            "conditions": [_msg_dict(c) for c in self.conditions],
            "verification": {"status": self.verification.value, "detail": self.check_detail},
            "formula": self.formula,
            "table": self.table,
            "substeps": [s.to_dict(f"{number}.{i + 1}" if number else str(i + 1))
                         for i, s in enumerate(self.substeps)],
        }


def number_steps(steps: List[Step], prefix: str = "") -> List[dict]:
    return [s.to_dict(f"{prefix}{i + 1}") for i, s in enumerate(steps)]


@dataclass
class Solution:
    """An antiderivative together with the verified steps that produced it."""
    F: sp.Expr
    steps: List[Step]
    technique: str
    conditions: List[Msg] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    verification: Optional[CheckResult] = None
    derivation_level: str = "full"  # "full" | "result_only"


@dataclass
class TraceEntry:
    technique: str
    score: float
    reason: str
    outcome: str  # applied | inapplicable | failed | unverified | timeout | skipped
    detail: str = ""
    depth: int = 0

    def to_dict(self) -> dict:
        return {"technique": self.technique, "score": round(self.score, 3), "reason": self.reason,
                "outcome": self.outcome, "detail": self.detail, "depth": self.depth}


class Classification(str, Enum):
    ELEMENTARY = "elementary"
    SPECIAL_FUNCTION = "special_function"
    NON_ELEMENTARY = "non_elementary_proven"
    NO_CLOSED_FORM = "no_closed_form_found"
    COULD_NOT_SOLVE = "could_not_solve"
    DIVERGENT = "divergent"
    EXACT_VALUE = "exact_value"
    NOT_APPLICABLE = "not_applicable"


class Status(str, Enum):
    SOLVED = "solved"
    DIVERGENT = "divergent"
    FAILED = "failed"


@dataclass
class SolveResult:
    status: Status
    classification: Classification
    mode: str  # "indefinite" | "definite"
    integrand: sp.Expr
    var: sp.Symbol
    lower: Optional[sp.Expr] = None
    upper: Optional[sp.Expr] = None
    antiderivative: Optional[sp.Expr] = None
    value: Optional[sp.Expr] = None
    steps: List[Step] = field(default_factory=list)
    verification_status: str = "not_applicable"  # verified | verified_numeric | failed | not_applicable
    checks: List[dict] = field(default_factory=list)
    conditions: List[Msg] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    message: Optional[Msg] = None
    derivation_level: str = "none"  # full | partial | result_only | none
    trace: List[TraceEntry] = field(default_factory=list)
    analysis: List[str] = field(default_factory=list)
    decimal: Optional[str] = None
    elapsed: float = 0.0
    technique: Optional[str] = None
    problem_latex: str = ""
    problem_text: str = ""

    @property
    def solved(self) -> bool:
        return self.status is Status.SOLVED

    def answer_expr(self) -> Optional[sp.Expr]:
        return self.value if self.mode == "definite" else self.antiderivative

    def all_steps(self) -> Iterator[Step]:
        for s in self.steps:
            yield from s.walk()

    def to_dict(self) -> dict:
        ans = self.answer_expr()
        answer = None
        if ans is not None:
            if self.mode == "indefinite":
                answer = {"kind": "antiderivative", "latex": render.latex(ans) + " + C",
                          "text": render.text(ans) + " + C",
                          "latex_no_constant": render.latex(ans), "text_no_constant": render.text(ans)}
            else:
                answer = {"kind": "value", "latex": render.latex(ans), "text": render.text(ans)}
        return {
            "status": self.status.value,
            "classification": self.classification.value,
            "mode": self.mode,
            "problem": {"latex": self.problem_latex, "text": self.problem_text,
                        "variable": self.var.name},
            "answer": answer,
            "decimal": self.decimal,
            "message": _msg_dict(self.message) if self.message else None,
            "verification": {"status": self.verification_status, "checks": self.checks},
            "conditions": [_msg_dict(c) for c in self.conditions],
            "warnings": list(self.warnings),
            "derivation_level": self.derivation_level,
            "technique": self.technique,
            "steps": number_steps(self.steps),
            "trace": [t.to_dict() for t in self.trace],
            "analysis": list(self.analysis),
            "elapsed_ms": int(self.elapsed * 1000),
        }
