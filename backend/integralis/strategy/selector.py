"""Strategy selection: rank candidate techniques from structural features (no computation of integrals here)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .features import Features


def rank(techniques: Sequence, feats: Features) -> List[Tuple[object, float, str]]:
    """Return ``(technique, score, reason)`` for every applicable technique, best first.

    Scores are heuristics in (0, 1]. They order the attempts but never force one: the engine falls through to the
    next candidate whenever a technique fails or its result does not verify. Ties keep registry order.
    """
    out = []
    for order, t in enumerate(techniques):
        try:
            r = t.suitability(feats)
        except Exception:  # a broken heuristic must never take the engine down
            r = None
        if r:
            out.append((t, float(r[0]), r[1], order))
    out.sort(key=lambda a: (-a[1], a[3]))
    return [(t, s, why) for t, s, why, _ in out]
