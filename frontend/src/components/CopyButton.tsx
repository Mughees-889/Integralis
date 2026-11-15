import { useState } from "react";

async function copy(text: string) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  }
}

export function CopyButton({ label, text, title }: { label: string; text: string | (() => string); title?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button" className="chip-btn" title={title ?? `Copy ${label}`}
      onClick={async () => {
        await copy(typeof text === "function" ? text() : text);
        setDone(true);
        setTimeout(() => setDone(false), 1200);
      }}
    >
      {done ? "Copied ✓" : label}
    </button>
  );
}
