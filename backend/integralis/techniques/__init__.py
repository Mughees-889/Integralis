"""Technique registry. Order matters only for ties in the strategy selector."""
from __future__ import annotations

from typing import List

from .base import Context, Technique  # noqa: F401
from .linearity import Linearity
from .parts import CyclicParts, IntegrationByParts, TabularParts
from .rational import ExpandAlgebra, QuadraticCompletion, RationalFunctions
from .reduction import ReductionFormulae
from .substitution import Substitution
from .sympy_general import SympyGeneral
from .table import StandardIntegrals
from .trig import (HyperbolicViaExponentials, ProductToSum, SinCosPowers, TanSecPowers,
                   WeierstrassSubstitution)
from .trigsub import TrigSubstitution


def default_techniques() -> List[Technique]:
    return [Linearity(), StandardIntegrals(), RationalFunctions(), Substitution(), CyclicParts(), TabularParts(),
            IntegrationByParts(), ReductionFormulae(), SinCosPowers(), TanSecPowers(), ProductToSum(),
            HyperbolicViaExponentials(), WeierstrassSubstitution(), TrigSubstitution(), QuadraticCompletion(),
            ExpandAlgebra(), SympyGeneral()]
