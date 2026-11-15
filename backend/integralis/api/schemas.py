"""Pydantic models describing the HTTP API (requests and responses)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    expression: str = Field(..., max_length=2000, description="Integrand or full notation, e.g. '∫ x e^(x²) dx'")
    variable: Optional[str] = Field(None, max_length=20, description="Integration variable (default: inferred)")
    mode: Optional[Literal["indefinite", "definite"]] = None
    lower: Optional[str] = Field(None, max_length=200, description="Lower limit, e.g. '0', '-oo', 'pi/2'")
    upper: Optional[str] = Field(None, max_length=200)
    assumptions: Optional[Dict[str, str]] = Field(
        None, description="Parameter assumptions, e.g. {'a': 'positive'} (positive, negative, nonzero, integer)")


class IntegrateRequest(ParseRequest):
    timeout_seconds: float = Field(30.0, ge=1.0, le=120.0)
    decimal_places: int = Field(20, ge=1, le=100)


class Rendered(BaseModel):
    latex: str
    text: str


class StepOut(BaseModel):
    number: str
    rule: str
    title: Rendered
    before: Optional[Rendered] = None
    after: Optional[Rendered] = None
    explanation: Rendered
    conditions: List[Rendered] = []
    verification: Dict[str, Any]
    formula: Optional[str] = None
    table: Optional[Dict[str, Any]] = None
    substeps: List["StepOut"] = []


StepOut.model_rebuild()


class ErrorDetail(BaseModel):
    message: str
    position: Optional[int] = None
    length: int = 1
    hint: Optional[str] = None
    source: Optional[str] = None


class ParseResponse(BaseModel):
    ok: bool
    latex: Optional[str] = None
    text: Optional[str] = None
    variable: Optional[str] = None
    mode: Optional[str] = None
    warnings: List[str] = []
    error: Optional[ErrorDetail] = None


class IntegrateResponse(BaseModel):
    ok: bool = True
    status: Literal["solved", "divergent", "failed"]
    classification: str
    mode: str
    problem: Dict[str, Any]
    answer: Optional[Dict[str, Any]] = None
    decimal: Optional[str] = None
    message: Optional[Rendered] = None
    verification: Dict[str, Any]
    conditions: List[Rendered] = []
    warnings: List[str] = []
    derivation_level: str
    technique: Optional[str] = None
    steps: List[StepOut] = []
    trace: List[Dict[str, Any]] = []
    analysis: List[str] = []
    elapsed_ms: int = 0
    error: Optional[ErrorDetail] = None


class TechniqueInfo(BaseModel):
    name: str
    title: str
    family: str
    description: str
