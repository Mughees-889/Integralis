# Adding a new integration strategy

A technique is one class in `backend/integralis/techniques/`. The engine supplies recursion, cycle protection, caching,
time limits and verification; you supply (1) a cheap structural test and (2) the construction of a verified derivation.

```python
from ..derivation import Int, make_step, wrap_solution_steps
from ..errors import TechniqueFailure
from ..models import Claim, Solution
from ..render import Msg
from .base import Context, Technique

class MyTechnique(Technique):
    name = "my_technique"            # unique id shown in the strategy trace
    title = "My technique"
    family = "basic"                 # used to group the catalogue
    description = "One or two sentences for docs/TECHNIQUES.md and GET /api/techniques."

    def suitability(self, feats):    # feats: strategy.features.Features — structure only, no integration here
        if looks_like_my_pattern(feats.expr, feats.x):
            return 0.8, "why this technique is plausible"     # (score in (0, 1], reason)
        return None                  # not applicable

    def attempt(self, f, x, ctx: Context):
        g = rewrite(f, x)                                      # your mathematics (SymPy does the algebra)
        step = make_step(
            "my_rule", Msg("Rewrite using …"), Int(f, x), Int(g, x),
            Msg("Because … we can write {g}.", g=g),
            claims=[Claim.identity(g, f, x, "the rewritten integrand equals the original")],   # VERIFIED before the step exists
        )
        sub = ctx.solve_sub(g, x)                              # recursion into the simpler integral
        if sub is None:
            raise TechniqueFailure("the rewritten integral could not be solved")
        return Solution(sub.F, [step] + wrap_solution_steps(g, x, sub), self.name, sub.conditions,
                        derivation_level=sub.derivation_level)
```

Register it in `techniques/__init__.py` (`default_techniques()`); order only matters for ties in score.

## Rules that keep the system honest
* **Every mathematical claim goes into a `Claim`.** `make_step` verifies all claims and raises `UnjustifiedStepError` if one fails, so
  a wrong transformation can never be displayed. Use `Claim.identity(lhs, rhs, x)` for rewrites and `Claim.derivative(F, f, x)` for
  antiderivatives. A step with no claims is displayed as an *explanation*.
* **The engine re-verifies your final `F`** by differentiation (`engine._integrate_uncached`); a wrong result is discarded and the next
  technique is tried. You cannot make the engine return an unverified antiderivative.
* **Return `None`** when the pattern does not apply after all; **raise `TechniqueFailure`** when it applies but cannot be finished.
* **Use `ctx.solve_sub`** (never `engine.integrate` directly) for sub-integrals; it handles depth limits and cycles. Use
  `ctx.fresh_symbol(f, x)` for new variables.
* Put anything the definite-integral code should reuse (e.g. a substitution `u = g(x)`) in `Solution.meta["substitution"]`.
* Add tests: a parametrised case in `tests/test_indefinite.py`, plus an assertion in `test_strategy_selector_picks_the_appropriate_technique`.
