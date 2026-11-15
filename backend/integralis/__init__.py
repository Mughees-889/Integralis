"""INTEGRALIS — a symbolic integration engine with verified step-by-step working."""
__version__ = "1.0.0"

from .config import EngineOptions  # noqa: F401,E402
from .errors import ParseError  # noqa: F401,E402
from .models import Classification, SolveResult, Status  # noqa: F401,E402
from .solver import solve, solve_problem  # noqa: F401,E402
