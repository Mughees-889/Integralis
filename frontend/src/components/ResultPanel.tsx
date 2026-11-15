import { useState } from "react";
import type { SolveResponse } from "../types";
import { CopyButton } from "./CopyButton";
import { MathView, RichText } from "./Math";
import { StepView, stepText } from "./StepView";

const CLASS_LABEL: Record<string, string> = {
  elementary: "Elementary antiderivative", special_function: "Special-function form", non_elementary_proven: "Proven non-elementary",
  no_closed_form_found: "No closed form found", could_not_solve: "Could not solve", divergent: "Divergent", exact_value: "Exact value",
};
const VERIF_LABEL: Record<string, string> = {
  verified: "Verified — exact symbolic check", verified_numeric: "Verified — numerical evidence (symbolic proof inconclusive)",
  failed: "Verification failed", not_applicable: "No result to verify",
};
const LEVEL_LABEL: Record<string, string> = {
  full: "Complete derivation", partial: "Partial derivation (one or more steps are result-only)",
  result_only: "Result only — no human-style derivation is claimed", none: "",
};

export function solutionText(r: SolveResponse): string {
  const head = `${r.problem.text}\n= ${r.answer?.text ?? r.message?.text ?? ""}\n`;
  return head + "\n" + r.steps.map((s) => stepText(s)).join("\n");
}

export function ResultPanel({ r }: { r: SolveResponse }) {
  const [showDec, setShowDec] = useState(false);
  const [openAll, setOpenAll] = useState(0);
  const [showTrace, setShowTrace] = useState(false);
  const tone = r.status === "solved" ? "good" : r.status === "divergent" ? "diverge" : "fail";
  return (
    <section className={`result tone-${tone}`} aria-live="polite">
      <div className="answer-card">
        <div className="badges">
          <span className={`pill class-${r.classification}`}>{CLASS_LABEL[r.classification] ?? r.classification}</span>
          <span className={`pill verif-${r.verification.status}`}>{VERIF_LABEL[r.verification.status] ?? r.verification.status}</span>
          {LEVEL_LABEL[r.derivation_level] && r.status === "solved" && <span className="pill level">{LEVEL_LABEL[r.derivation_level]}</span>}
        </div>
        {r.answer ? (
          <>
            <div className="final">
              <MathView latex={`${r.problem.latex} = ${r.answer.latex}`} display />
            </div>
            <div className="answer-tools">
              <CopyButton label="Copy answer (LaTeX)" text={r.answer.latex} />
              <CopyButton label="Copy answer (text)" text={r.answer.text} />
              <CopyButton label="Copy full solution" text={() => solutionText(r)} />
              {r.decimal && <button type="button" className="chip-btn" onClick={() => setShowDec(!showDec)}>{showDec ? "Hide decimal" : "Decimal approximation"}</button>}
            </div>
            {showDec && r.decimal && <p className="decimal">≈ <code>{r.decimal}</code> <span className="muted">(30-digit arithmetic; the exact value above is what counts)</span></p>}
          </>
        ) : (
          <div className="final failure">
            <MathView latex={r.problem.latex} display />
            <h3>{r.status === "divergent" ? "The integral diverges" : "No result"}</h3>
          </div>
        )}
        {r.message && <p className="message"><RichText text={r.message.latex} /></p>}
        {r.conditions.length > 0 && (
          <div className="conds-box"><h5>Conditions of validity</h5><ul>{r.conditions.map((c, i) => <li key={i}><RichText text={c.latex} /></li>)}</ul></div>
        )}
        {r.warnings.length > 0 && <ul className="warnings">{r.warnings.map((w, i) => <li key={i}>⚠ {w}</li>)}</ul>}
        <details className="checks">
          <summary>Verification details ({r.verification.checks.length}) · {r.elapsed_ms} ms</summary>
          <ul>{r.verification.checks.map((c, i) => <li key={i}><b>{c.name}</b> — <span className={`st-${c.status}`}>{c.status}</span>: {c.detail}</li>)}</ul>
          {r.analysis.length > 0 && <><h5>Integrand analysis</h5><ul>{r.analysis.map((a, i) => <li key={i}>{a}</li>)}</ul></>}
        </details>
      </div>

      {r.steps.length > 0 && (
        <div className="derivation">
          <div className="deriv-head">
            <h2>Step-by-step working</h2>
            <div className="deriv-tools">
              <button type="button" className="chip-btn" onClick={() => setOpenAll((n) => (n > 0 ? -n - 1 : n + 1) )}>Expand / collapse all</button>
            </div>
          </div>
          <ol className="steps">{r.steps.map((s) => <StepView key={s.number} step={s} forceOpen={openAll} />)}</ol>
        </div>
      )}

      <div className="trace">
        <button type="button" className="toggle" onClick={() => setShowTrace(!showTrace)}>{showTrace ? "▾" : "▸"} Strategy selector trace</button>
        {showTrace && (
          <table className="trace-table"><thead><tr><th>depth</th><th>technique</th><th>score</th><th>outcome</th><th>why considered / detail</th></tr></thead>
            <tbody>{r.trace.map((t, i) => (
              <tr key={i} className={`out-${t.outcome}`}><td>{t.depth}</td><td>{t.technique}</td><td>{t.score.toFixed(2)}</td><td>{t.outcome}</td><td>{t.detail || t.reason}</td></tr>))}</tbody></table>
        )}
      </div>
    </section>
  );
}
