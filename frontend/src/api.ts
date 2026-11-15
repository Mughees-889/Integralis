import type { Examples, ParseError, ParseResponse, Request, SolveResponse } from "./types";

const BASE: string = (import.meta as any).env?.VITE_API_BASE ?? "";

export class ApiError extends Error {
  parse?: ParseError;
  constructor(message: string, parse?: ParseError) { super(message); this.parse = parse; }
}

function body(req: Request) {
  const mode = req.mode;
  const assumptions = req.assumptions?.trim() ? req.assumptions.trim() : undefined;
  return JSON.stringify({
    expression: req.expression,
    variable: req.variable || undefined,
    mode,
    lower: mode === "definite" ? req.lower : undefined,
    upper: mode === "definite" ? req.upper : undefined,
    assumptions,
    timeout_seconds: req.timeout_seconds ?? 30,
  });
}

export async function parseExpression(req: Request, signal?: AbortSignal): Promise<ParseResponse> {
  const r = await fetch(`${BASE}/api/parse`, { method: "POST", headers: { "Content-Type": "application/json" }, body: body(req), signal });
  if (!r.ok) throw new ApiError(`Server error (${r.status})`);
  return r.json();
}

export async function integrate(req: Request, signal?: AbortSignal): Promise<SolveResponse> {
  const r = await fetch(`${BASE}/api/integrate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: body(req), signal });
  if (r.ok) return r.json();
  let detail: any = null;
  try { detail = (await r.json()).detail; } catch { /* ignore */ }
  if (r.status === 400 && detail && typeof detail === "object") throw new ApiError(detail.message, detail as ParseError);
  if (r.status === 504) throw new ApiError("The computation exceeded the hard time limit and was stopped.");
  throw new ApiError(typeof detail === "string" ? detail : `Server error (${r.status})`);
}

export async function getExamples(): Promise<Examples> {
  const r = await fetch(`${BASE}/api/examples`);
  if (!r.ok) throw new ApiError("Could not load examples");
  return r.json();
}
