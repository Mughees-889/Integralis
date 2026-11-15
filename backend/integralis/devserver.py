"""Dependency-free development server (standard library only).

It exposes exactly the same JSON API as the FastAPI app (``integralis.api.app``) and serves ``frontend/dist``. It
exists so the application can be run where FastAPI/uvicorn are not installed; FastAPI is the intended production
backend.

    python -m integralis.devserver --port 8000
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import service

DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=2)
    return _pool


def _reset_pool():
    global _pool
    if _pool is not None:
        for p in list(getattr(_pool, "_processes", {}).values()):
            try:
                p.terminate()
            except Exception:
                pass
        _pool.shutdown(wait=False, cancel_futures=True)
    _pool = None


class Handler(BaseHTTPRequestHandler):
    server_version = "integralis-dev"

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json(200, service.health())
        if path == "/api/techniques":
            return self._json(200, service.techniques_catalogue())
        if path == "/api/examples":
            return self._json(200, service.examples())
        if path.startswith("/api/"):
            return self._json(404, {"detail": "not found"})
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        f = (DIST / rel).resolve()
        if DIST.exists() and str(f).startswith(str(DIST.resolve())) and f.is_file():
            ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            return self._send(200, f.read_bytes(), ctype)
        self._json(404, {"detail": "frontend not built (run `npm run build` in frontend/)"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
        except Exception:
            return self._json(400, {"detail": "invalid JSON body"})
        if path == "/api/parse":
            return self._json(200, service.parse_request(payload))
        if path == "/api/integrate":
            timeout = max(1.0, min(float(payload.get("timeout_seconds") or 30.0), 120.0))
            fut = _get_pool().submit(service.integrate_request, payload)
            try:
                out = fut.result(timeout=timeout + 15.0)
            except FutTimeout:
                _reset_pool()
                return self._json(504, {"detail": "The computation exceeded the hard time limit and was stopped."})
            except Exception as exc:
                _reset_pool()
                return self._json(500, {"detail": f"Internal error: {type(exc).__name__}"})
            if not out.get("ok", True):
                return self._json(400, {"detail": out["error"]})
            return self._json(200, out)
        self._json(404, {"detail": "not found"})

    def log_message(self, fmt, *args):  # quiet
        if os.environ.get("INTEGRALIS_LOG"):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"INTEGRALIS dev server on http://{a.host}:{a.port}  (frontend: {'built' if DIST.exists() else 'not built'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
