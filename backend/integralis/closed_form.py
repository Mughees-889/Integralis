"""Classification helpers: elementary versus special-function closed forms."""
from __future__ import annotations

import sympy as sp

from .models import Classification

SPECIAL = {"erf", "erfc", "erfi", "erf2", "fresnels", "fresnelc", "Ei", "expint", "li", "Si", "Ci", "Shi", "Chi",
           "polylog", "gamma", "loggamma", "uppergamma", "lowergamma", "besselj", "bessely", "besseli",
           "besselk", "LambertW", "elliptic_e", "elliptic_f", "elliptic_k", "elliptic_pi", "zeta", "lerchphi",
           "airyai", "airybi", "hankel1", "hankel2", "hyper", "meijerg"}


def special_functions_in(F) -> set:
    F = sp.sympify(F)
    found = {type(a).__name__ for a in F.atoms(sp.Function)} & SPECIAL
    if F.has(sp.hyper):
        found.add("hyper")
    if F.has(sp.meijerg):
        found.add("meijerg")
    return found


def classify_closed_form(F) -> Classification:
    return Classification.SPECIAL_FUNCTION if special_functions_in(F) else Classification.ELEMENTARY
