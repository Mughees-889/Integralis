"""Nested soft time limits.

SymPy is pure Python, so a SIGALRM handler can interrupt it between bytecodes. Signals only work in the main
thread of a process, so in other threads ``time_limit`` degrades to a no-op (the API runs solves in worker
processes whose main thread executes the job, so limits are effective there).
"""
from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from typing import List, Optional

from ..errors import EngineTimeout

_stack: List[float] = []
_REPEAT = 0.05


def _handler(signum, frame):  # pragma: no cover - exercised through time_limit
    raise EngineTimeout("time limit exceeded")


def _can_signal() -> bool:
    return hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()


@contextmanager
def time_limit(seconds: Optional[float]):
    """Raise :class:`EngineTimeout` inside the block once ``seconds`` have elapsed.

    Limits nest: the effective deadline of an inner block never exceeds the deadline of an enclosing one.
    """
    if seconds is None or not _can_signal():
        yield
        return
    now = time.monotonic()
    deadline = now + max(float(seconds), 0.0)
    if _stack:
        deadline = min(deadline, _stack[-1])
    _stack.append(deadline)
    previous = signal.signal(signal.SIGALRM, _handler)
    # Repeating timer: SymPy contains broad ``except`` blocks that can swallow one EngineTimeout, so the signal is
    # re-delivered every 50 ms until the block exits.
    signal.setitimer(signal.ITIMER_REAL, max(deadline - now, 1e-3), _REPEAT)
    try:
        yield
    finally:
        _stack.pop()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if _stack:  # re-arm the enclosing limit
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, max(_stack[-1] - time.monotonic(), 1e-3), _REPEAT)


def outermost_deadline_passed() -> bool:
    """True when the outermost active limit has expired (used to stop trying alternatives)."""
    return bool(_stack) and time.monotonic() >= _stack[0]
