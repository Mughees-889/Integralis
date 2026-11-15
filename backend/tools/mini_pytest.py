"""A minimal stand-in for pytest, used ONLY when real pytest is not installed (e.g. an offline sandbox).

It supports exactly what this project's tests use: plain ``test_*`` functions, ``pytest.mark.parametrize``,
``pytest.raises``, ``pytest.importorskip``, ``pytest.skip`` and arbitrary marks. The tests are ordinary pytest tests;
run them with ``pytest`` when it is available:

    python tools/mini_pytest.py [-k SUBSTR] [-x] [-q] [tests/test_file.py ...]
"""
from __future__ import annotations

import importlib.util
import inspect
import itertools
import os
import sys
import time
import traceback
import types
from pathlib import Path


class Skipped(Exception):
    pass


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match, self.value = exc, match, None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc):
            return False
        if self.match is not None:
            import re
            if not re.search(self.match, str(ev)):
                raise AssertionError(f"exception message {str(ev)!r} does not match {self.match!r}")
        self.value = ev
        return True


class _Mark:
    def parametrize(self, argnames, argvalues, ids=None):
        names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

        def deco(fn):
            rows = []
            for i, v in enumerate(argvalues):
                vals = (v,) if len(names) == 1 else tuple(v)
                rows.append((ids[i] if ids else None, dict(zip(names, vals))))
            fn.__dict__.setdefault("_param_sets", []).append(rows)
            return fn
        return deco

    def __getattr__(self, name):  # any other mark (slow, ...) is a no-op decorator
        def deco(*a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]
            return lambda fn: fn
        return deco


def install_shim():
    m = types.ModuleType("pytest")
    m.mark = _Mark()
    m.raises = lambda exc, match=None: _Raises(exc, match)
    m.fail = lambda msg="": (_ for _ in ()).throw(AssertionError(msg))

    def skip(reason=""):
        raise Skipped(reason)

    def importorskip(name):
        try:
            return __import__(name)
        except ImportError:
            raise Skipped(f"could not import {name!r}")
    m.skip, m.importorskip = skip, importorskip
    sys.modules["pytest"] = m


def _expand(fn):
    sets = fn.__dict__.get("_param_sets")
    if not sets:
        return [("", {})]
    out = []
    for combo in itertools.product(*sets):
        kw, ids = {}, []
        for pid, d in combo:
            kw.update(d)
            ids.append(pid if pid else "-".join(str(v)[:30] for v in d.values()))
        out.append(("[" + "-".join(ids) + "]", kw))
    return out


def main(argv):
    try:
        import pytest  # noqa: F401
        print("real pytest is installed — please run `pytest` instead of this stand-in.")
    except ImportError:
        install_shim()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests"))
    sel = None
    stop_first = "-x" in argv
    quiet = "-q" in argv
    if "-k" in argv:
        sel = argv[argv.index("-k") + 1]
    files = [a for a in argv if a.endswith(".py")] or sorted(str(p) for p in (root / "tests").glob("test_*.py"))
    passed = failed = skipped = 0
    failures, slow = [], []
    t_all = time.time()
    for path in files:
        spec = importlib.util.spec_from_file_location(Path(path).stem, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Skipped as e:
            skipped += 1
            print(f"{path}: SKIPPED ({e})")
            continue
        for name, fn in sorted(vars(mod).items(), key=lambda kv: kv[1].__code__.co_firstlineno if hasattr(kv[1], "__code__") else 0):
            if not (name.startswith("test_") and callable(fn) and getattr(fn, "__module__", None) == mod.__name__):
                continue
            for suffix, kw in _expand(fn):
                tid = f"{Path(path).name}::{name}{suffix}"
                if sel and sel not in tid:
                    continue
                t0 = time.time()
                try:
                    fn(**kw)
                    passed += 1
                    status = "PASSED"
                except Skipped as e:
                    skipped += 1
                    status = f"SKIPPED ({e})"
                except BaseException as e:  # noqa: BLE001
                    failed += 1
                    status = "FAILED"
                    failures.append((tid, traceback.format_exc()))
                dt = time.time() - t0
                if dt > 5:
                    slow.append((dt, tid))
                if not quiet or status == "FAILED":
                    print(f"{tid:100.100} {status}  ({dt:.2f}s)")
                if status == "FAILED" and stop_first:
                    break
    print("\n" + "=" * 100)
    for tid, tb in failures:
        print(f"--- FAILURE: {tid}\n{tb}")
    if slow:
        print("slowest:", ", ".join(f"{t} {d:.1f}s" for d, t in sorted(slow, reverse=True)[:5]))
    print(f"{passed} passed, {failed} failed, {skipped} skipped in {time.time() - t_all:.1f}s  "
          f"[runner: {'pytest-compatible stand-in' }]")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
