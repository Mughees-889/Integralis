import sys, time, os
os.environ["INTEGRALIS_DEBUG"] = "1"
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from integralis import solve, EngineOptions, render

def R(e): return "" if e is None else render.text(e)

def show(expr, verbose=True, **kw):
    t = time.time()
    r = solve(expr, **kw)
    print(f"\n=== {expr!r} {kw}  [{time.time()-t:.2f}s] status={r.status.value} class={r.classification.value} verif={r.verification_status} level={r.derivation_level} tech={r.technique}")
    if r.answer_expr() is not None:
        print("   answer:", r.to_dict()["answer"]["text"])
    if r.message: print("   message:", r.message.text())
    if verbose:
        def walk(steps, ind=1, pre=""):
            for i, s in enumerate(steps, 1):
                print("   " * ind + f"{pre}{i}. [{s.verification.value[:4]}] {s.title.text() if hasattr(s.title,'text') else s.title}:  {R(s.before)}  ->  {R(s.after)}")
                if os.environ.get("EXPL"): print("   " * ind + "     · " + s.explanation.text())
                walk(s.substeps, ind + 1, f"{pre}{i}.")
        walk(r.steps)
    return r

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    for e in args:
        show(e, verbose="-q" not in sys.argv)
