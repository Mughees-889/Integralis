"""Algebraic expansion, rational functions (polynomial division + partial fractions) and completing the square."""
from __future__ import annotations

from typing import List, Optional, Tuple

import sympy as sp

from ..derivation import Int, make_step, merge_conditions, wrap_solution_steps
from ..errors import EngineTimeout, TechniqueFailure
from ..models import Claim, Solution
from ..render import Msg, latex
from ..strategy.features import Features
from ..utils.timeouts import time_limit
from .base import Context, Technique
from .table import _square_term, linear_form


# ---------------------------------------------------------------------------------------------------------
class ExpandAlgebra(Technique):
    name = "expand"
    title = "Algebraic expansion"
    family = "algebra"
    description = ("Multiply out a product or power of polynomials so the integrand becomes a sum of simple "
                   "powers that can be integrated term by term.")

    def suitability(self, feats: Features):
        f = feats.expr
        if feats.is_polynomial and not f.is_Add and f.has(feats.x):
            try:
                if sp.expand(f) != f:
                    return 0.6, "A product of polynomials: expand it into a sum of powers."
            except Exception:
                return None
        return None

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        e = sp.expand(f)
        if e == f or not e.is_Add:
            return None
        step = make_step("expand", Msg("Expand the integrand"), Int(f, x), Int(e, x),
                         Msg("Multiply out the brackets so the integrand becomes a sum of powers of {x}.", x=x),
                         claims=[Claim.identity(f, e, x, "the expanded form equals the integrand")])
        sub = ctx.solve_sub(e, x)
        if sub is None:
            raise TechniqueFailure("the expanded polynomial could not be integrated")
        return Solution(sub.F, [step] + wrap_solution_steps(e, x, sub), self.name, sub.conditions,
                        derivation_level=sub.derivation_level)


# ---------------------------------------------------------------------------------------------------------
def _fresh_names(avoid: set):
    for ch in "ABCDEFGHJKLMNPQRSTUVWXYZ":
        if ch not in avoid:
            yield sp.Symbol(ch, real=True)
    k = 1
    while True:
        yield sp.Symbol(f"A_{k}", real=True)
        k += 1


def _num_den(f, x) -> Optional[Tuple[sp.Expr, sp.Expr]]:
    num, den = sp.fraction(sp.together(f))
    if not (num.is_polynomial(x) and den.is_polynomial(x)):
        return None
    return num, den


def real_split(fac, x) -> Optional[Tuple[sp.Expr, List[Tuple[sp.Expr, int]]]]:
    """Split a polynomial that is irreducible over Q into real linear / quadratic factors with exact roots."""
    deg = sp.degree(fac, x)
    if deg <= 2:
        return sp.Integer(1), [(fac, 1)]
    try:
        with time_limit(2.5):
            rts = sp.roots(sp.Poly(fac, x))
    except EngineTimeout:
        return None
    except Exception:
        return None
    if sum(rts.values()) != deg:
        return None
    lead = sp.Poly(fac, x).LC()
    out: List[Tuple[sp.Expr, int]] = []
    for r, m in rts.items():
        re_, im_ = sp.simplify(sp.re(r)), sp.simplify(sp.im(r))
        if re_.has(sp.re, sp.im) or im_.has(sp.re, sp.im) or re_.has(sp.I) or im_.has(sp.I):
            return None
        if im_ == 0:
            out.append((x - re_, m))
        else:
            pos = im_.is_positive
            if pos is None:
                pos = float(im_) > 0
            if pos:
                out.append((sp.expand(x ** 2 - 2 * re_ * x + re_ ** 2 + im_ ** 2), m))
    prod = lead * sp.Mul(*[f ** m for f, m in out])
    try:
        with time_limit(2.0):
            ok = sp.simplify(sp.expand(prod) - sp.expand(fac)) == 0
    except Exception:
        ok = False
    return (lead, out) if ok else None


class RationalFunctions(Technique):
    name = "rational"
    title = "Rational functions: polynomial division and partial fractions"
    family = "rational"
    description = ("For P(x)/Q(x): divide when deg P ≥ deg Q, factor the denominator, set up the partial-fraction "
                   "form, solve for the coefficients, and integrate each fraction.")

    def __init__(self):
        self._plan_cache = {}

    def _plan(self, f, x):
        key = (sp.srepr(f), x.name)
        if key in self._plan_cache:
            return self._plan_cache[key]
        plan = None
        nd = _num_den(f, x)
        if nd is not None and nd[1].has(x):
            num, den = nd
            dn, dd = sp.degree(num, x), sp.degree(den, x)
            if dn >= dd:
                plan = ("divide", num, den)
            else:
                try:
                    content, facs = sp.factor_list(den, x)
                except Exception:
                    facs = None
                if facs:
                    simple = (len(facs) == 1 and (facs[0][1] == 1 and sp.degree(facs[0][0], x) <= 2
                                                   or (sp.degree(facs[0][0], x) == 1 and dn == 0)))
                    if not simple:
                        plan = ("partial", num, den, content, facs)
        self._plan_cache[key] = plan
        return plan

    def suitability(self, feats: Features):
        if not feats.is_rational or feats.is_polynomial:
            return None
        try:
            plan = self._plan(feats.expr, feats.x)
        except Exception:
            return None
        if plan is None:
            return None
        if plan[0] == "divide":
            return 0.9, "Improper rational function: divide the polynomials first."
        return 0.88, "A proper rational function: decompose it into partial fractions."

    # ------------------------------------------------------------------------------------------------
    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        plan = self._plan(f, x)
        if plan is None:
            return None
        if plan[0] == "divide":
            return self._divide(f, x, plan, ctx)
        return self._partial(f, x, plan, ctx)

    def _divide(self, f, x, plan, ctx) -> Solution:
        _, num, den = plan
        q, r = sp.div(num, den, x)
        g = q + r / den if r != 0 else q
        step = make_step(
            "polynomial_division", Msg("Polynomial division"), Int(f, x), Int(g, x),
            Msg("The numerator has degree at least that of the denominator, so divide: quotient {q}, remainder {r}, "
                "giving {f} = {q} + ({r})/({d}).", q=q, r=r, f=f, d=den),
            claims=[Claim.identity(f, g, x, "quotient plus remainder over divisor equals the integrand")])
        sub = ctx.solve_sub(g, x)
        if sub is None:
            raise TechniqueFailure("the result of the polynomial division could not be integrated")
        return Solution(sub.F, [step] + wrap_solution_steps(g, x, sub), self.name, sub.conditions,
                        derivation_level=sub.derivation_level)

    def _partial(self, f, x, plan, ctx) -> Solution:
        _, num, den, content, facs = plan
        # real factorisation of irreducible factors of degree >= 3
        factors: List[Tuple[sp.Expr, int]] = []
        lead = sp.Integer(1)
        for fac, m in facs:
            if sp.degree(fac, x) >= 3:
                split = real_split(fac, x)
                if split is None:
                    raise TechniqueFailure("a factor of the denominator cannot be split into real linear/quadratic factors")
                lc, parts = split
                lead *= lc ** m
                factors.extend((p, pm * m) for p, pm in parts)
            else:
                factors.append((fac, m))
        den_factored = content * lead * sp.Mul(*[p ** m for p, m in factors])
        steps = []
        steps.append(make_step(
            "factor_denominator", Msg("Factor the denominator"), f, num / den_factored,
            Msg("Factor the denominator completely over the real numbers: {d} = {df}.", d=den, df=den_factored),
            claims=[Claim.identity(den, den_factored, x, "the factorisation multiplies back to the denominator")]))
        # ansatz
        avoid = {s.name for s in f.free_symbols}
        names = _fresh_names(avoid)
        unknowns, terms = [], []
        for fac, m in factors:
            d = sp.degree(fac, x)
            for k in range(1, m + 1):
                if d == 1:
                    A = next(names)
                    unknowns.append(A)
                    terms.append(A / fac ** k)
                elif d == 2:
                    B, C = next(names), next(names)
                    unknowns += [B, C]
                    terms.append((B * x + C) / fac ** k)
                else:
                    raise TechniqueFailure("unsupported factor degree in the denominator")
        ansatz = sp.Add(*terms)
        steps.append(make_step(
            "partial_fraction_form", Msg("Write the partial-fraction form"), num / den_factored, ansatz,
            Msg("Each linear factor gets a constant numerator, each irreducible quadratic a linear numerator, and "
                "repeated factors one term for every power up to the multiplicity.")))
        # solve for the coefficients
        residual = sp.together(ansatz - num / den_factored)
        numer = sp.fraction(sp.cancel(residual))[0]
        eqs = sp.Poly(sp.expand(numer), x).all_coeffs()
        try:
            with time_limit(4.0):
                sol = sp.solve(eqs, unknowns, dict=True)
        except EngineTimeout:
            raise TechniqueFailure("solving for the partial-fraction coefficients timed out")
        if not sol:
            raise TechniqueFailure("the partial-fraction coefficient equations have no solution")
        sol = sol[0]
        if any(u not in sol for u in unknowns):
            raise TechniqueFailure("the partial-fraction coefficients are not uniquely determined")
        decomp = ansatz.xreplace(sol)
        eq_rows = [{"equation": latex(sp.Eq(e, 0))} for e in eqs]
        table = {"kind": "equations",
                 "heading": "Compare coefficients of powers of " + x.name,
                 "equations": [r["equation"] for r in eq_rows],
                 "solution": [latex(sp.Eq(u, sol[u])) for u in unknowns]}
        steps.append(make_step(
            "solve_coefficients", Msg("Solve for the coefficients"), ansatz, decomp,
            Msg("Multiply both sides by the denominator and compare coefficients of like powers of {x}. Solving this "
                "linear system gives the coefficients below.", x=x),
            claims=[Claim.identity(num / den_factored, decomp, x, "the decomposition equals the original fraction")],
            table=table))
        steps.append(make_step(
            "rewrite_as_partial_fractions", Msg("Rewrite the integral"), Int(f, x), Int(decomp, x),
            Msg("Replace the integrand by its partial fractions.")))
        sub = ctx.solve_sub(decomp, x)
        if sub is None:
            raise TechniqueFailure("the partial fractions could not all be integrated")
        steps.extend(wrap_solution_steps(decomp, x, sub))
        return Solution(sub.F, steps, self.name, sub.conditions, derivation_level=sub.derivation_level)


# ---------------------------------------------------------------------------------------------------------
class QuadraticCompletion(Technique):
    name = "complete_square"
    title = "Completing the square"
    family = "rational"
    description = ("Rewrites N(x)/(ax²+bx+c)^p (or the same with a root) as terms in (x+s)²+d, splitting the linear "
                   "numerator into a part proportional to x+s (a logarithm / power after u = (x+s)²+d) and a constant "
                   "part (an arctangent/arcsine-type integral).")

    @staticmethod
    def _parse(f, x):
        """Return (N0, B, p) with f = N0 * B**p, B quadratic in x, N0 polynomial of degree <= 1 (else None)."""
        quad = None
        rest = []
        for a in sp.Mul.make_args(f):
            b, e = a.as_base_exp()
            if quad is None and e.is_Rational and e != 0 and b.has(x) and b.is_polynomial(x) and sp.degree(b, x) == 2 \
                    and not e.has(x):
                quad = (b, e)
            else:
                rest.append(a)
        if quad is None:
            return None
        N0 = sp.Mul(*rest)
        if not N0.is_polynomial(x) or (N0.has(x) and sp.degree(N0, x) != 1):
            return None
        if not (quad[1].is_negative or quad[1] == sp.Rational(1, 2)):
            return None
        return N0, quad[0], quad[1]

    def suitability(self, feats: Features):
        try:
            parsed = self._parse(feats.expr, feats.x)
        except Exception:
            return None
        if parsed is None:
            return None
        N0, B, p = parsed
        x = feats.x
        if self._already_completed(N0, B, x):
            if not self._needs_split(N0, B, x):
                return None
        return 0.8, "A quadratic expression: complete the square."

    @staticmethod
    def _completed_form(B, x):
        """If B is already ``k + c*L**2`` with L linear in x, return (k, c, L)."""
        if not B.is_Add:
            return None
        k, rest = B.as_independent(x, as_Add=True)
        if not rest.has(x):
            return None
        c, r = rest.as_coeff_Mul()
        if r.is_Pow and r.exp == 2 and linear_form(r.base, x) is not None:
            return k, c, r.base
        return None

    def _already_completed(self, N0, B, x) -> bool:
        return self._completed_form(B, x) is not None

    def _needs_split(self, N0, B, x) -> bool:
        k, c, L = self._completed_form(B, x)
        if not N0.has(x):
            return False
        p1 = sp.Poly(N0, x).coeff_monomial(x)
        q1 = N0 - p1 * x
        a, b = linear_form(L, x)
        # numerator proportional to L: the substitution u = B handles it directly
        return sp.simplify(q1 * a - p1 * b) != 0

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        parsed = self._parse(f, x)
        if parsed is None:
            return None
        N0, B, p = parsed
        cf = self._completed_form(B, x)
        if cf is not None:
            k, c, L = cf
            a1, b1 = linear_form(L, x)
            s = L  # already of the form k + c*L^2
            Bnew = B
            if not self._needs_split(N0, B, x):
                return None
            Lx = L
            shift_form = None
        else:
            poly = sp.Poly(B, x)
            a, b, c0 = poly.all_coeffs()
            sft = sp.simplify(b / (2 * a))
            d = sp.simplify(c0 / a - sft ** 2)
            Lx = x + sft
            Bnew = a * (Lx ** 2 + d)
            shift_form = (a, sft, d)
        # numerator in terms of L: N0 = p1*x + q1 = (p1/a1)*(L) + (q1 - p1*b1/a1)
        if N0.has(x):
            p1 = sp.Poly(N0, x).coeff_monomial(x)
            q1 = sp.simplify(N0 - p1 * x)
        else:
            p1, q1 = sp.Integer(0), N0
        a1, b1 = linear_form(Lx, x)
        part_L = sp.simplify(p1 / a1)
        part_c = sp.simplify(q1 - p1 * b1 / a1)
        pieces = []
        if part_L != 0:
            pieces.append(part_L * Lx * Bnew ** p)
        if part_c != 0:
            pieces.append(part_c * Bnew ** p)
        g = sp.Add(*pieces)
        steps = []
        if shift_form is not None:
            a, sft, d = shift_form
            steps.append(make_step(
                "complete_square", Msg("Complete the square"), B, Bnew,
                Msg("Write the quadratic as a multiple of a square plus a constant: {B} = {Bn}.", B=B, Bn=Bnew),
                claims=[Claim.identity(B, Bnew, x, "completing the square gives the same polynomial")]))
        steps.append(make_step(
            "split_numerator", Msg("Split the numerator"), Int(f, x), Int(g, x),
            Msg("Express the numerator through {L}: one part is proportional to {L} (it will be handled by "
                "u = (…)² + const) and the other is constant.", L=Lx),
            claims=[Claim.identity(f, g, x, "the split form equals the integrand")]))
        if g == f:
            return None
        sub = ctx.solve_sub(g, x)
        if sub is None:
            raise TechniqueFailure("the completed-square form could not be integrated")
        steps.extend(wrap_solution_steps(g, x, sub))
        return Solution(sub.F, steps, self.name, sub.conditions, derivation_level=sub.derivation_level)
