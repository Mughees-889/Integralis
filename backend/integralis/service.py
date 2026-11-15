"""Framework-independent request handlers (the FastAPI app and the stdlib dev server are thin wrappers).

Every handler takes and returns plain JSON-serialisable dicts, so the whole API contract can be tested without any
web framework.
"""
from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Optional

import sympy as sp

from . import __version__
from .config import EngineOptions
from .errors import ParseError
from .parsing import parse_problem
from .solver import solve_problem
from .techniques import default_techniques

EXAMPLES = [
    "∫ sin(x) dx", "∫ x e^(x²) dx", "∫ x² ln(x) dx", "∫ x/(x² + 1) dx", "∫ sin²(x) cos³(x) dx",
    "∫ 1/(x⁴ + 1) dx", "∫ ln(x)/x dx", "∫ e^x sin(x) dx", "∫ arctan(x) dx", "∫ sqrt(1 − x²) dx",
    "∫ x³/(x² + 1) dx", "∫ 1/(x² + 4x + 5) dx", "∫ sec(x)^3 dx", "∫ 1/(x sqrt(x + 1)) dx", "∫ e^(−x²) dx",
    "∫ sin(x)/x dx", "∫ x^x dx",
]
DEFINITE_EXAMPLES = [
    {"expression": "x e^(x²)", "lower": "0", "upper": "1"},
    {"expression": "1/(2 + cos(x))", "lower": "0", "upper": "2pi"},
    {"expression": "e^(−x²)", "lower": "0", "upper": "oo"},
    {"expression": "1/x", "lower": "−1", "upper": "1"},
    {"expression": "sin(x)/(sin(x)+cos(x))", "lower": "0", "upper": "pi/2"},
    {"expression": "ln(x)", "lower": "0", "upper": "1"},
]


def health() -> Dict[str, Any]:
    return {"status": "ok", "version": __version__, "sympy": sp.__version__, "python": sys.version.split()[0]}


def techniques_catalogue() -> List[Dict[str, str]]:
    return [{"name": t.name, "title": t.title, "family": t.family, "description": t.description}
            for t in default_techniques()]


def examples() -> Dict[str, Any]:
    return {"indefinite": EXAMPLES, "definite": DEFINITE_EXAMPLES}


def _assumptions(raw: Any) -> Optional[Dict[str, str]]:
    """Accept {"a": "positive"} or the string "a>0, b<0, n integer"."""
    if not raw:
        return None
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    out: Dict[str, str] = {}
    for part in re.split(r"[;,]\s*", str(raw)):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"([A-Za-z]\w*)\s*(>|<|≠|!=|\s)\s*(0|positive|negative|nonzero|integer)?", part)
        if not m:
            raise ParseError(f"Cannot read the assumption {part!r}", None, "Use e.g. 'a>0' or 'n integer'")
        name, op, what = m.groups()
        out[name] = {">": "positive", "<": "negative", "≠": "nonzero", "!=": "nonzero"}.get(op) or what or "real"
        if what in ("positive", "negative", "nonzero", "integer"):
            out[name] = what
    return out


def _problem(payload: Dict[str, Any], options: EngineOptions):
    return parse_problem(payload.get("expression", ""), payload.get("variable") or None, payload.get("lower") or None,
                         payload.get("upper") or None, payload.get("mode") or None,
                         _assumptions(payload.get("assumptions")), options.log_is_natural)


def parse_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input and return the LaTeX preview (no solving)."""
    options = EngineOptions()
    try:
        p = _problem(payload, options)
    except ParseError as exc:
        return {"ok": False, "error": exc.to_dict()}
    return {"ok": True, "latex": p.latex(), "text": p.text(), "variable": p.var.name,
            "mode": "definite" if p.definite else "indefinite", "warnings": p.warnings}


def integrate_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Solve an integration request. Parse problems are reported as ``{"ok": False, "error": ...}``; mathematical
    failures are ordinary results with ``status == "failed"`` and an explanatory message."""
    options = EngineOptions()
    try:
        options.timeout_seconds = max(1.0, min(float(payload.get("timeout_seconds") or 30.0), 120.0))
        options.decimal_places = max(1, min(int(payload.get("decimal_places") or 20), 100))
        problem = _problem(payload, options)
    except ParseError as exc:
        return {"ok": False, "error": exc.to_dict()}
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": {"message": f"Invalid option: {exc}", "position": None, "length": 1,
                                       "hint": None, "source": None}}
    result = solve_problem(problem, options)
    out = result.to_dict()
    out["ok"] = True
    if out.get("decimal") and options.decimal_places != 20:
        try:
            out["decimal"] = str(sp.N(result.value, options.decimal_places))
        except Exception:
            pass
    return out
