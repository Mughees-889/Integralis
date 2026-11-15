"""Trigonometric integrals: identities chosen by the structure of the integrand.

Each technique here *rewrites* the integrand with a named identity (verified as an identity) and hands the result
back to the engine, which then typically finishes with a substitution or the table. The identity used and the
reason it is appropriate are displayed with the step.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import sympy as sp

from ..derivation import Int, make_step, merge_conditions, split_constant, wrap_solution_steps
from ..errors import TechniqueFailure
from ..models import Claim, Solution
from ..render import Msg
from ..strategy.features import Features
from .base import Context, Technique
from .table import linear_form


def _monomial(f, x, classes, allow_negative=False) -> Optional[Tuple[sp.Expr, sp.Expr, Dict[type, int]]]:
    """f = c * prod(fn(theta)**k) over the given function classes, same linear theta, integer k (>= 0 unless
    ``allow_negative``)."""
    c, rest = split_constant(f, x)
    exps: Dict[type, int] = {}
    theta = None
    for a in sp.Mul.make_args(rest):
        base, e = a.as_base_exp()
        if type(base) not in classes or not e.is_Integer or (e < 0 and not allow_negative):
            return None
        arg = base.args[0]
        if linear_form(arg, x) is None:
            return None
        if theta is None:
            theta = arg
        elif theta != arg:
            return None
        exps[type(base)] = exps.get(type(base), 0) + int(e)
    if theta is None:
        return None
    return c, theta, exps


class _RewriteTechnique(Technique):
    """Shared plumbing: verify the identity, show the step, integrate the rewritten form."""

    def _rewrite_and_solve(self, f, x, g, ctx: Context, title, explanation, formula=None, conditions=()) -> Solution:
        step = make_step("trig_identity", title, Int(f, x), Int(g, x), explanation,
                         claims=[Claim.identity(f, g, x, "the rewritten integrand equals the original")],
                         formula=formula, conditions=conditions)
        sub = ctx.solve_sub(g, x)
        if sub is None:
            raise TechniqueFailure("the rewritten integrand could not be integrated")
        return Solution(sub.F, [step] + wrap_solution_steps(g, x, sub), self.name,
                        merge_conditions(list(conditions), sub.conditions), derivation_level=sub.derivation_level)


class SinCosPowers(_RewriteTechnique):
    name = "trig_powers"
    title = "Powers of sine and cosine"
    family = "trigonometric"
    description = ("∫ sin^m θ cos^n θ: split off one factor when a power is odd and use sin²+cos²=1 so that "
                   "u = cos θ or u = sin θ applies; reduce even powers with the double-angle identities "
                   "sin²θ = (1−cos 2θ)/2 and cos²θ = (1+cos 2θ)/2.")

    @staticmethod
    def _plan(f, x):
        parsed = _monomial(f, x, (sp.sin, sp.cos), allow_negative=True)
        if parsed is None:
            return None
        c, th, ex = parsed
        m, n = ex.get(sp.sin, 0), ex.get(sp.cos, 0)
        S, C = sp.sin(th), sp.cos(th)
        if m < 0 and n < 0:
            return None
        odd_cos = n >= 3 and n % 2 == 1
        odd_sin = m >= 3 and m % 2 == 1
        if odd_cos and (not odd_sin or n <= m):
            g = c * S ** m * (1 - S ** 2) ** ((n - 1) // 2) * C
            return ("cos_odd", g, th, m, n)
        if odd_sin:
            g = c * C ** n * (1 - C ** 2) ** ((m - 1) // 2) * S
            return ("sin_odd", g, th, m, n)
        if m >= 0 and n >= 0 and m % 2 == 0 and n % 2 == 0 and m + n >= 2:
            g = sp.expand(c * ((1 - sp.cos(2 * th)) / 2) ** (m // 2) * ((1 + sp.cos(2 * th)) / 2) ** (n // 2))
            return ("even", g, th, m, n)
        if n < 0 and m >= 2 and m % 2 == 0:
            return ("neg_cos", sp.expand(c * (1 - C ** 2) ** (m // 2) * C ** n), th, m, n)
        if m < 0 and n >= 2 and n % 2 == 0:
            return ("neg_sin", sp.expand(c * (1 - S ** 2) ** (n // 2) * S ** m), th, m, n)
        return None

    def suitability(self, feats: Features):
        p = self._plan(feats.expr, feats.x)
        if p is None:
            return None
        return (0.84 if p[0] == "even" else 0.88), "A product of powers of sin and cos: use a Pythagorean or double-angle identity."

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        p = self._plan(f, x)
        if p is None:
            return None
        kind, g, th, m, n = p
        if kind == "cos_odd":
            title = Msg("Use sin² + cos² = 1")
            expl = Msg("The power of cosine ({n}) is odd: split off one cos {th} and write the remaining even power "
                       "as (1 − sin² {th})^{k}. Then u = sin {th} turns the integral into a polynomial.",
                       n=str(n), th=th, k=str((n - 1) // 2))
            formula = r"\cos^2\theta=1-\sin^2\theta"
        elif kind in ("neg_cos", "neg_sin"):
            nm = "sin" if kind == "neg_cos" else "cos"
            other = "cos" if kind == "neg_cos" else "sin"
            title = Msg("Use sin² + cos² = 1")
            expl = Msg("The power of {nm} is even and {other} appears in a denominator: replace {nm}² by 1 − {other}² "
                       "and expand into powers of {other} (positive ones and reciprocals), each of which is standard.",
                       nm=nm, other=other)
            formula = r"\sin^2\theta=1-\cos^2\theta"
        elif kind == "sin_odd":
            title = Msg("Use sin² + cos² = 1")
            expl = Msg("The power of sine ({m}) is odd: split off one sin {th} and write the remaining even power as "
                       "(1 − cos² {th})^{k}. Then u = cos {th} turns the integral into a polynomial.",
                       m=str(m), th=th, k=str((m - 1) // 2))
            formula = r"\sin^2\theta=1-\cos^2\theta"
        else:
            title = Msg("Use the power-reduction (double-angle) identities")
            expl = Msg("Both powers are even, so no single factor can be split off. Replace sin²{th} by "
                       "(1 − cos 2{th})/2 and cos²{th} by (1 + cos 2{th})/2 and expand.", th=th)
            formula = r"\sin^2\theta=\tfrac{1-\cos2\theta}{2},\quad\cos^2\theta=\tfrac{1+\cos2\theta}{2}"
        return self._rewrite_and_solve(f, x, g, ctx, title, expl, formula)


class TanSecPowers(_RewriteTechnique):
    name = "trig_tan_sec"
    title = "Powers of tangent and secant (and cotangent and cosecant)"
    family = "trigonometric"
    description = ("∫ tan^m sec^n and ∫ cot^m csc^n: with sec² = 1 + tan² an even power of sec leads to u = tan, an "
                   "odd power of tan leads to u = sec; otherwise tan² = sec² − 1 reduces the power.")

    _PAIRS = ((sp.tan, sp.sec), (sp.cot, sp.csc))

    def _plan(self, f, x):
        for T, S in self._PAIRS:
            parsed = _monomial(f, x, (T, S))
            if parsed is None:
                continue
            c, th, ex = parsed
            m, n = ex.get(T, 0), ex.get(S, 0)
            t, s = T(th), S(th)
            if n >= 4 and n % 2 == 0:
                g = c * t ** m * (1 + t ** 2) ** (n // 2)  # sec^n = (1 + tan^2)^(n/2)
                return ("sec_even", g, th, m, n, T, S)
            if m >= 3 and m % 2 == 1 and n >= 1:
                g = c * (s ** 2 - 1) ** ((m - 1) // 2) * s ** (n - 1) * (s * t)
                return ("tan_odd", g, th, m, n, T, S)
            if n == 0 and m >= 2:
                g = c * t ** (m - 2) * (s ** 2 - 1)
                return ("tan_reduce", sp.expand(g), th, m, n, T, S)
            if m % 2 == 0 and m >= 2 and n % 2 == 1:
                g = sp.expand(c * (s ** 2 - 1) ** (m // 2) * s ** n)
                return ("tan_even_sec_odd", g, th, m, n, T, S)
        return None

    def suitability(self, feats: Features):
        try:
            p = self._plan(feats.expr, feats.x)
        except Exception:
            return None
        if p is None:
            return None
        score = 0.95 if p[0] == "tan_reduce" else 0.87
        return score, "Powers of tan/sec (or cot/csc): use sec² = 1 + tan² or tan² = sec² − 1."

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        p = self._plan(f, x)
        if p is None:
            return None
        kind, g, th, m, n, T, S = p
        tn, sn = T.__name__, S.__name__
        if kind == "sec_even":
            title = Msg("Use sec² = 1 + tan²")
            expl = Msg("The power of {sn} is even, so write {sn}^{n} = ({sn}²)^{k} = (1 + {tn}²)^{k} using "
                       "{sn}² = 1 + {tn}². The derivative of {tn} {th} is {sn}² {th} = 1 + {tn}² {th}, so u = {tn} {th} "
                       "applies.", sn=sn, tn=tn, th=th, n=str(n), k=str(n // 2))
        elif kind == "tan_odd":
            title = Msg("Use tan² = sec² − 1")
            expl = Msg("The power of {tn} is odd: keep one {sn} {tn} (the derivative of {sn}) and write the other "
                       "{tn}'s with {tn}² = {sn}² − 1. Then u = {sn} {th}.", sn=sn, tn=tn, th=th)
        elif kind == "tan_reduce":
            title = Msg("Use tan² = sec² − 1 to lower the power")
            expl = Msg("Write {tn}^{m} = {tn}^{m2}({sn}² − 1): the {sn}² part integrates by u = {tn}, the rest has a "
                       "lower power.", tn=tn, sn=sn, m=str(m), m2=str(m - 2))
        else:
            title = Msg("Use tan² = sec² − 1 to reach powers of secant")
            expl = Msg("Even power of {tn} with an odd power of {sn}: write {tn}² = {sn}² − 1 and expand into odd "
                       "powers of {sn}, which have standard reduction formulae.", tn=tn, sn=sn)
        formula = (r"\sec^2\theta=1+\tan^2\theta" if T is sp.tan else r"\csc^2\theta=1+\cot^2\theta")
        return self._rewrite_and_solve(f, x, g, ctx, title, expl, formula)


class ProductToSum(_RewriteTechnique):
    name = "trig_product_to_sum"
    title = "Product-to-sum identities"
    family = "trigonometric"
    description = ("sin A cos B, sin A sin B and cos A cos B with different linear angles are rewritten as sums of "
                   "sines/cosines of A ± B, which integrate term by term.")

    @staticmethod
    def _plan(f, x):
        c, rest = split_constant(f, x)
        fs = [a for a in sp.Mul.make_args(rest) if a.has(x)]
        if len(fs) != 2 or not all(isinstance(a, (sp.sin, sp.cos)) for a in fs):
            return None
        A, B = fs[0].args[0], fs[1].args[0]
        if A == B or linear_form(A, x) is None or linear_form(B, x) is None:
            return None
        f1, f2 = fs
        if isinstance(f1, sp.sin) and isinstance(f2, sp.sin):
            g = (sp.cos(A - B) - sp.cos(A + B)) / 2
            ident = r"\sin A\sin B=\tfrac12[\cos(A-B)-\cos(A+B)]"
        elif isinstance(f1, sp.cos) and isinstance(f2, sp.cos):
            g = (sp.cos(A - B) + sp.cos(A + B)) / 2
            ident = r"\cos A\cos B=\tfrac12[\cos(A-B)+\cos(A+B)]"
        else:
            if isinstance(f1, sp.cos):  # make f1 the sine
                A, B = B, A
            g = (sp.sin(A + B) + sp.sin(A - B)) / 2
            ident = r"\sin A\cos B=\tfrac12[\sin(A+B)+\sin(A-B)]"
        return c * g, ident

    def suitability(self, feats: Features):
        if self._plan(feats.expr, feats.x) is None:
            return None
        return 0.89, "A product of sines/cosines of different angles: use a product-to-sum identity."

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        p = self._plan(f, x)
        if p is None:
            return None
        g, ident = p
        g = sp.expand(g)
        return self._rewrite_and_solve(
            f, x, g, ctx, Msg("Use a product-to-sum identity"),
            Msg("The two factors have different angles, so convert the product into a sum of single sines/cosines."),
            ident)


class WeierstrassSubstitution(Technique):
    name = "weierstrass"
    title = "Weierstrass substitution t = tan(θ/2)"
    family = "trigonometric"
    description = ("Any rational function of sin θ and cos θ becomes a rational function of t = tan(θ/2) with "
                   "sin θ = 2t/(1+t²), cos θ = (1−t²)/(1+t²) and dθ = 2 dt/(1+t²). Last resort among trigonometric "
                   "techniques because the antiderivative can jump where tan(θ/2) is undefined (θ = π + 2kπ).")

    _FUNCS = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)

    def _plan(self, f, x):
        nodes = [n for n in sp.preorder_traversal(f) if isinstance(n, self._FUNCS)]
        if not nodes:
            return None
        args = {n.args[0] for n in nodes}
        if len(args) != 1:
            return None
        th = args.pop()
        lf = linear_form(th, x)
        if lf is None:
            return None
        syms = {n: sp.Dummy("s") for n in set(nodes)}
        fs = f.xreplace(syms)
        if fs.has(x) or not fs.is_rational_function(*syms.values()):
            return None
        return th, lf[0]

    def suitability(self, feats: Features):
        if self._plan(feats.expr, feats.x) is None:
            return None
        return 0.45, "A rational function of sin and cos: the Weierstrass substitution always works."

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        plan = self._plan(f, x)
        if plan is None:
            return None
        th, a = plan
        t = ctx.fresh_symbol(f, x)
        repl = {sp.sin(th): 2 * t / (1 + t ** 2), sp.cos(th): (1 - t ** 2) / (1 + t ** 2),
                sp.tan(th): 2 * t / (1 - t ** 2), sp.cot(th): (1 - t ** 2) / (2 * t),
                sp.sec(th): (1 + t ** 2) / (1 - t ** 2), sp.csc(th): (1 + t ** 2) / (2 * t)}
        h = f.xreplace(repl) * 2 / (a * (1 + t ** 2))
        h = sp.cancel(sp.together(h))
        if h.has(x):
            raise TechniqueFailure("the substitution left the variable in the integrand")
        g = sp.tan(th / 2)
        c, h0 = split_constant(h, t)
        claim = Claim.identity(h.xreplace({t: g}) * sp.diff(g, x), f, x, "h(g(x))·g'(x) equals the integrand")
        sub = ctx.solve_sub(h0, t)
        if sub is None:
            raise TechniqueFailure("the rational function in t could not be integrated")
        F = (c * sub.F).xreplace({t: g})
        cond = Msg("valid on every interval where tan({th}/2) is defined, i.e. {th} ≠ π + 2kπ", th=th)
        steps = [
            make_step("substitution", Msg("Weierstrass substitution"), Int(f, x), sp.Eq(t, g),
                      Msg("Let {t} = tan({th}/2). Then sin {th} = 2{t}/(1+{t}²), cos {th} = (1−{t}²)/(1+{t}²) and "
                          "d{x} = 2 d{t}/({a}(1+{t}²)).", t=t, th=th, x=x, a=a),
                      formula=r"\sin\theta=\frac{2t}{1+t^2},\ \cos\theta=\frac{1-t^2}{1+t^2},\ d\theta=\frac{2\,dt}{1+t^2}",
                      conditions=[cond]),
            make_step("rewrite_in_new_variable", Msg("Rewrite in terms of {t}", t=t), Int(f, x), c * Int(h0, t),
                      Msg("The integrand becomes the rational function {h} of {t}.", h=h0, t=t),
                      claims=[claim], meta={"u": t, "g": g}),
        ]
        steps.extend(wrap_solution_steps(h0, t, sub))
        steps.append(make_step("back_substitute", Msg("Substitute back"), c * sub.F, F,
                               Msg("Replace {t} by tan({th}/2).", t=t, th=th),
                               claims=[Claim.derivative(F, f, x, "the result differentiates to the integrand")]))
        meta = {"substitution": {"u": t, "g": g, "h": h, "c": c, "h0": h0, "method": "W", "x": x}}
        return Solution(F, steps, self.name, merge_conditions([cond], sub.conditions), meta=meta,
                        derivation_level=sub.derivation_level)


class HyperbolicViaExponentials(_RewriteTechnique):
    name = "hyperbolic_exp"
    title = "Hyperbolic functions through exponentials"
    family = "trigonometric"
    description = ("Products and powers of sinh and cosh are rewritten with sinh θ = (e^θ − e^−θ)/2 and "
                   "cosh θ = (e^θ + e^−θ)/2 and expanded into a sum of exponentials.")

    @staticmethod
    def _plan(f, x):
        parsed = _monomial(f, x, (sp.sinh, sp.cosh))
        if parsed is None:
            return None
        c, th, ex = parsed
        if sum(ex.values()) < 2:
            return None
        g = sp.expand(f.rewrite(sp.exp))
        if not g.is_Add:
            return None
        return g

    def suitability(self, feats: Features):
        if self._plan(feats.expr, feats.x) is None:
            return None
        return 0.6, "Powers of sinh/cosh: rewrite with exponentials."

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        g = self._plan(f, x)
        if g is None:
            return None
        return self._rewrite_and_solve(
            f, x, g, ctx, Msg("Write the hyperbolic functions as exponentials"),
            Msg("Use sinh θ = (e^θ − e^−θ)/2 and cosh θ = (e^θ + e^−θ)/2, then expand into a sum of exponentials."),
            r"\sinh\theta=\frac{e^{\theta}-e^{-\theta}}{2},\ \cosh\theta=\frac{e^{\theta}+e^{-\theta}}{2}")
