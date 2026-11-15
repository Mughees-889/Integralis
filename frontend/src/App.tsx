import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getExamples, integrate, parseExpression } from "./api";
import { MathView } from "./components/Math";
import { ResultPanel } from "./components/ResultPanel";
import type { Examples, ParseError, ParseResponse, Request, SolveResponse } from "./types";

const VARIABLES = ["x", "y", "z", "t", "u", "v", "w", "s", "theta"];
const SYMBOLS: { label: string; insert: string; title: string }[] = [
  { label: "π", insert: "pi", title: "pi" }, { label: "e", insert: "e", title: "Euler's number" },
  { label: "√", insert: "sqrt(", title: "square root" }, { label: "xⁿ", insert: "^", title: "power" },
  { label: "ln", insert: "ln(", title: "natural logarithm" }, { label: "∞", insert: "oo", title: "infinity (limits only)" },
];

function ErrorBox({ err }: { err: ParseError }) {
  const caret = err.source != null && err.position != null;
  return (
    <div className="error" role="alert">
      <strong>{err.message}</strong>
      {caret && (<pre className="caret">{err.source}{"\n"}{" ".repeat(err.position!)}{"^".repeat(Math.max(1, err.length))}</pre>)}
      {err.hint && <div className="hint">Hint: {err.hint}</div>}
    </div>
  );
}

export default function App() {
  const [expression, setExpression] = useState("∫ x e^(x²) dx");
  const [variable, setVariable] = useState("x");
  const [mode, setMode] = useState<"indefinite" | "definite">("indefinite");
  const [lower, setLower] = useState("0");
  const [upper, setUpper] = useState("1");
  const [assumptions, setAssumptions] = useState("");
  const [preview, setPreview] = useState<ParseResponse | null>(null);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [error, setError] = useState<ParseError | string | null>(null);
  const [loading, setLoading] = useState(false);
  const [examples, setExamples] = useState<Examples | null>(null);
  const exprRef = useRef<HTMLInputElement>(null);
  const solveAbort = useRef<AbortController | null>(null);

  const request = useCallback((): Request => ({ expression, variable, mode, lower, upper, assumptions }), [expression, variable, mode, lower, upper, assumptions]);

  useEffect(() => { getExamples().then(setExamples).catch(() => undefined); }, []);

  // live LaTeX preview (debounced)
  useEffect(() => {
    if (!expression.trim()) { setPreview(null); return; }
    const ctl = new AbortController();
    const id = setTimeout(() => {
      parseExpression(request(), ctl.signal).then(setPreview).catch(() => undefined);
    }, 250);
    return () => { clearTimeout(id); ctl.abort(); };
  }, [request, expression]);

  const solve = async () => {
    solveAbort.current?.abort();
    const ctl = new AbortController();
    solveAbort.current = ctl;
    setLoading(true); setError(null); setResult(null);
    try {
      setResult(await integrate(request(), ctl.signal));
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      if (e instanceof ApiError && e.parse) setError(e.parse); else setError((e as Error).message || "Something went wrong");
    } finally { setLoading(false); }
  };

  const insert = (text: string) => {
    const el = exprRef.current;
    if (!el) { setExpression((s) => s + text); return; }
    const s = el.selectionStart ?? expression.length, e2 = el.selectionEnd ?? expression.length;
    setExpression(expression.slice(0, s) + text + expression.slice(e2));
    requestAnimationFrame(() => { el.focus(); el.setSelectionRange(s + text.length, s + text.length); });
  };

  const useExample = (ex: string) => { setExpression(ex); setMode("indefinite"); setResult(null); setError(null); };
  const useDefinite = (d: { expression: string; lower: string; upper: string }) => {
    setExpression(d.expression); setLower(d.lower); setUpper(d.upper); setMode("definite"); setResult(null); setError(null);
  };

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand"><span className="glyph" aria-hidden>∫</span><h1>INTEGRALIS</h1></div>
        <p className="tagline">Integration with working you can check. Every step is verified before it is shown; when the engine cannot solve something, it says so.</p>
      </header>

      <main>
        <form className="input-card" onSubmit={(ev) => { ev.preventDefault(); solve(); }}>
          <div className="mode-row" role="tablist" aria-label="Integral type">
            <button type="button" role="tab" aria-selected={mode === "indefinite"} className={mode === "indefinite" ? "on" : ""} onClick={() => setMode("indefinite")}>Indefinite ∫ f dx</button>
            <button type="button" role="tab" aria-selected={mode === "definite"} className={mode === "definite" ? "on" : ""} onClick={() => setMode("definite")}>Definite ∫ₐᵇ f dx</button>
          </div>

          <label className="field big">
            <span>Integral</span>
            <input ref={exprRef} value={expression} onChange={(e) => setExpression(e.target.value)} spellCheck={false} autoComplete="off"
              placeholder="e.g.  x^2 * exp(x)    or    ∫ sin(x)/x dx" aria-label="Integrand or integral" />
          </label>
          <div className="symbols" aria-label="Insert symbol">{SYMBOLS.map((s) => (
            <button type="button" key={s.label} title={s.title} onClick={() => insert(s.insert)}>{s.label}</button>))}</div>

          {mode === "definite" && (
            <div className="limits">
              <label className="field"><span>Lower limit</span><input value={lower} onChange={(e) => setLower(e.target.value)} spellCheck={false} placeholder="0, -oo, pi/2 …" /></label>
              <label className="field"><span>Upper limit</span><input value={upper} onChange={(e) => setUpper(e.target.value)} spellCheck={false} placeholder="1, oo, 2pi …" /></label>
            </div>
          )}

          <div className="row">
            <label className="field narrow"><span>Variable</span>
              <select value={variable} onChange={(e) => setVariable(e.target.value)}>{VARIABLES.map((v) => <option key={v} value={v}>{v}</option>)}</select>
            </label>
            <label className="field grow"><span>Parameter assumptions <em>(optional)</em></span>
              <input value={assumptions} onChange={(e) => setAssumptions(e.target.value)} spellCheck={false} placeholder="a>0, n integer" />
            </label>
            <button className="solve" type="submit" disabled={loading || !expression.trim()}>{loading ? "Working…" : "Solve"}</button>
          </div>

          <div className="preview" aria-label="Equation preview">
            {preview?.ok && preview.latex ? <MathView latex={preview.latex} display /> : null}
            {preview && !preview.ok && preview.error && <ErrorBox err={preview.error} />}
            {preview?.ok && preview.warnings && preview.warnings.length > 0 && <ul className="warnings">{preview.warnings.map((w, i) => <li key={i}>⚠ {w}</li>)}</ul>}
          </div>

          {examples && (
            <div className="examples">
              <span className="lbl">Try</span>
              {examples.indefinite.slice(0, 9).map((e) => <button type="button" key={e} onClick={() => useExample(e)}>{e.replace(/^∫ /, "").replace(/ dx$/, "")}</button>)}
              {examples.definite.slice(0, 5).map((d) => <button type="button" className="def" key={d.expression + d.lower + d.upper} onClick={() => useDefinite(d)}>{d.expression} [{d.lower}, {d.upper}]</button>)}
            </div>
          )}
        </form>

        {loading && <div className="working" role="status"><span className="spinner" /> Selecting strategies, building a verified derivation…</div>}
        {error && (typeof error === "string" ? <div className="error" role="alert"><strong>{error}</strong></div> : <ErrorBox err={error} />)}
        {result && <ResultPanel r={result} />}
      </main>

      <footer>
        <p>Exact arithmetic throughout (SymPy). Steps are verified by exact simplification where possible; where only numerical agreement was obtained the step is badged <em>numeric</em>.
          Numerical quadrature is used only as an independent cross-check, never as the answer. <code>log(x)</code> means ln x.</p>
      </footer>
    </div>
  );
}
