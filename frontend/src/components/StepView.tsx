import { useState } from "react";
import type { Step, TableData } from "../types";
import { CopyButton } from "./CopyButton";
import { MathView, RichText } from "./Math";

export function stepText(s: Step, indent = ""): string {
  const lines = [`${indent}${s.number}. ${s.title.text}`];
  if (s.before && s.after && s.before.text !== s.after.text) lines.push(`${indent}   ${s.before.text}  =  ${s.after.text}`);
  else if (s.after) lines.push(`${indent}   ${s.after.text}`);
  lines.push(`${indent}   ${s.explanation.text}`);
  s.conditions.forEach((c) => lines.push(`${indent}   Condition: ${c.text}`));
  s.substeps.forEach((c) => lines.push(stepText(c, indent + "   ")));
  return lines.join("\n");
}

export function stepLatex(s: Step): string {
  const eq = s.before && s.after && s.before.latex !== s.after.latex ? `${s.before.latex} \\;=\\; ${s.after.latex}` : s.after?.latex ?? "";
  return `% ${s.number}. ${s.title.text}\n${eq}`;
}

const BADGE: Record<string, { label: string; cls: string; tip: string }> = {
  symbolic: { label: "proved", cls: "ok", tip: "Proven by exact symbolic simplification" },
  structural: { label: "proved", cls: "ok", tip: "Both sides are identical expressions" },
  numeric: { label: "numeric", cls: "warn", tip: "Confirmed at many high-precision sample points; symbolic simplification was inconclusive (evidence, not a proof)" },
  unchecked: { label: "explanation", cls: "muted", tip: "An explanatory step that makes no new mathematical claim" },
  failed: { label: "failed", cls: "bad", tip: "Verification failed" },
};

function TableView({ t }: { t: TableData }) {
  if (t.kind === "key_value" && t.rows)
    return (
      <div className="tbl"><table><tbody>{t.rows.map((r, i) => (
        <tr key={i}><th><MathView latex={r[0]} /></th><td><MathView latex={r[1]} /></td></tr>))}</tbody></table></div>
    );
  if (t.kind === "tabular" && t.rows && t.columns)
    return (
      <div className="tbl">
        <table className="tabular">
          <thead><tr>{t.columns.map((c, i) => <th key={i}>{c}</th>)}</tr></thead>
          <tbody>{t.rows.map((r, i) => (<tr key={i}>{r.map((c, j) => <td key={j}>{j === 0 ? c : <MathView latex={c} />}</td>)}</tr>))}</tbody>
        </table>
        {t.note && <p className="note">{t.note}</p>}
      </div>
    );
  if (t.kind === "equations" && t.equations)
    return (
      <div className="tbl eqs">
        {t.heading && <p className="note">{t.heading}</p>}
        <div className="eqlist">{t.equations.map((e, i) => <MathView key={i} latex={e} />)}</div>
        {t.solution && <><p className="note">Solution</p><div className="eqlist sol">{t.solution.map((e, i) => <MathView key={i} latex={e} />)}</div></>}
      </div>
    );
  return null;
}

export function StepView({ step, depth = 0, forceOpen }: { step: Step; depth?: number; forceOpen: number }) {
  const [open, setOpen] = useState(depth < 1);
  const [seen, setSeen] = useState(forceOpen);
  if (seen !== forceOpen) { setSeen(forceOpen); setOpen(forceOpen > 0 ? true : false); }
  const v = BADGE[step.verification.status] ?? BADGE.unchecked;
  const same = step.before && step.after && step.before.latex === step.after.latex;
  const hasSub = step.substeps.length > 0;
  return (
    <li className={`step depth-${Math.min(depth, 3)} rule-${step.rule}`} id={`step-${step.number}`}>
      <div className="step-head">
        <span className="num">{step.number}</span>
        <h4><RichText text={step.title.latex} /></h4>
        <span className={`badge ${v.cls}`} title={`${v.tip}${step.verification.detail ? " — " + step.verification.detail : ""}`}>{v.label}</span>
        <span className="step-actions">
          <CopyButton label="LaTeX" text={() => stepLatex(step)} title="Copy this step as LaTeX" />
          <CopyButton label="text" text={() => stepText(step)} title="Copy this step as plain text" />
        </span>
      </div>
      <div className="eq">
        {step.before && step.after && !same ? (
          <>
            <MathView latex={step.before.latex} display />
            <div className="arrow" aria-hidden>⟹</div>
            <MathView latex={step.after.latex} display />
          </>
        ) : step.after ? <MathView latex={step.after.latex} display /> : step.before ? <MathView latex={step.before.latex} display /> : null}
      </div>
      <p className="explain"><RichText text={step.explanation.latex} /></p>
      {step.formula && <div className="formula"><span className="tag">formula</span><MathView latex={step.formula} display /></div>}
      {step.table && <TableView t={step.table} />}
      {step.conditions.length > 0 && (
        <ul className="conds">{step.conditions.map((c, i) => <li key={i}><span className="tag">condition</span> <RichText text={c.latex} /></li>)}</ul>
      )}
      {hasSub && (
        <div className="sub">
          <button type="button" className="toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
            {open ? "▾" : "▸"} {step.substeps.length} sub-step{step.substeps.length > 1 ? "s" : ""}
          </button>
          {open && <ol className="steps">{step.substeps.map((s) => <StepView key={s.number} step={s} depth={depth + 1} forceOpen={forceOpen} />)}</ol>}
        </div>
      )}
    </li>
  );
}
