export interface Rendered { latex: string; text: string }
export interface StepVerification { status: string; detail: string }
export interface TableData {
  kind?: string; heading?: string; note?: string;
  rows?: string[][]; columns?: string[]; equations?: string[]; solution?: string[];
}
export interface Step {
  number: string; rule: string; title: Rendered;
  before: Rendered | null; after: Rendered | null; explanation: Rendered;
  conditions: Rendered[]; verification: StepVerification;
  formula: string | null; table: TableData | null; substeps: Step[];
}
export interface TraceEntry { technique: string; score: number; reason: string; outcome: string; detail: string; depth: number }
export interface Check { name: string; status: string; detail: string }
export interface SolveResponse {
  ok: boolean;
  status: "solved" | "divergent" | "failed";
  classification: string;
  mode: "indefinite" | "definite";
  problem: { latex: string; text: string; variable: string };
  answer: { kind: string; latex: string; text: string; latex_no_constant?: string; text_no_constant?: string } | null;
  decimal: string | null;
  message: Rendered | null;
  verification: { status: string; checks: Check[] };
  conditions: Rendered[];
  warnings: string[];
  derivation_level: string;
  technique: string | null;
  steps: Step[];
  trace: TraceEntry[];
  analysis: string[];
  elapsed_ms: number;
}
export interface ParseError { message: string; position: number | null; length: number; hint: string | null; source: string | null }
export interface ParseResponse { ok: boolean; latex?: string; text?: string; variable?: string; mode?: string; warnings?: string[]; error?: ParseError }
export interface Request {
  expression: string; variable?: string; mode: "indefinite" | "definite";
  lower?: string; upper?: string; assumptions?: string; timeout_seconds?: number;
}
export interface Examples { indefinite: string[]; definite: { expression: string; lower: string; upper: string }[] }
