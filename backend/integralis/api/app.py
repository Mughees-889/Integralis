"""FastAPI application.

Run:   uvicorn integralis.api.app:app --reload        (from the backend/ directory)
Docs:  http://127.0.0.1:8000/docs

Solves run in a pool of worker *processes* so that a hard wall-clock limit can be enforced (the engine's own soft
limits use signals, which only work in a process's main thread). All mathematics lives in ``integralis.service``;
this module only adds HTTP concerns.
"""
from __future__ import annotations

import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, service
from .schemas import (IntegrateRequest, IntegrateResponse, ParseRequest, ParseResponse, TechniqueInfo)

app = FastAPI(title="INTEGRALIS", version=__version__,
              description="Symbolic integration with verified step-by-step working.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_pool: ProcessPoolExecutor | None = None


def _make_pool() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=max(2, min(4, (os.cpu_count() or 2))),
                               mp_context=multiprocessing.get_context("spawn"))


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = _make_pool()
    return _pool


def _reset_pool() -> None:
    """Drop (and kill) workers after a hard timeout so a runaway computation cannot pile up."""
    global _pool
    if _pool is not None:
        for p in list(getattr(_pool, "_processes", {}).values()):
            try:
                p.terminate()
            except Exception:
                pass
        _pool.shutdown(wait=False, cancel_futures=True)
    _pool = None


def _run_integrate(payload: Dict[str, Any]) -> Dict[str, Any]:  # executes in a worker process
    return service.integrate_request(payload)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return service.health()


@app.get("/api/techniques", response_model=List[TechniqueInfo])
def techniques():
    return service.techniques_catalogue()


@app.get("/api/examples")
def examples() -> Dict[str, Any]:
    return service.examples()


@app.post("/api/parse", response_model=ParseResponse)
def parse(req: ParseRequest):
    out = service.parse_request(req.model_dump())
    if not out["ok"]:
        return JSONResponse(status_code=200, content=out)  # a syntax error is an expected outcome, not a server error
    return out


@app.post("/api/integrate", response_model=IntegrateResponse, response_model_exclude_none=True)
async def integrate(req: IntegrateRequest):
    payload = req.model_dump()
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_get_pool(), _run_integrate, payload)
    try:
        out = await asyncio.wait_for(fut, timeout=req.timeout_seconds + 15.0)
    except asyncio.TimeoutError:
        _reset_pool()
        raise HTTPException(status_code=504, detail="The computation exceeded the hard time limit and was stopped.")
    except Exception as exc:  # worker crashed
        _reset_pool()
        raise HTTPException(status_code=500, detail=f"Internal error: {type(exc).__name__}")
    if not out.get("ok", True):
        raise HTTPException(status_code=400, detail=out["error"])
    return out


_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _dist.is_dir():  # serve the built frontend from the same origin
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
