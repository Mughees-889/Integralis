import { useMemo } from "react";
import katex from "katex";

function esc(s: string) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Renders LaTeX with KaTeX. The LaTeX is produced by our own backend; KaTeX output is sanitised HTML. */
export function MathView({ latex, display = false, className }: { latex: string; display?: boolean; className?: string }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(latex, { displayMode: display, throwOnError: false, strict: "ignore", trust: false });
    } catch {
      return esc(latex);
    }
  }, [latex, display]);
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Text with inline $math$ segments (the backend marks formulas with dollar signs). */
export function RichText({ text }: { text: string }) {
  const parts = text.split("$");
  return (
    <>
      {parts.map((p, i) => (i % 2 === 1 ? <MathView key={i} latex={p} /> : <span key={i}>{p}</span>))}
    </>
  );
}
