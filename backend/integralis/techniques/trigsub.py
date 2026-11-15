"""Trigonometric substitution (tan for a²+u², sin for a²−u²) and Euler's substitution for u²−a².

The integrand must have the form  R(x) · B(x)^p  where B = k + c·L² with L linear in x (the completed-square form
produced by the ``complete_square`` technique) and p is a half-integer or an integer ≤ −2. The change of variable is
verified as an identity before it is shown, exactly like an ordinary substitution.

Branch safety: the tan and sin substitutions use θ in (−π/2, π/2) where sec θ and cos θ are positive, so radicals
simplify without absolute values. Euler's substitution t = L + √(L²−γ²) is purely algebraic and valid on both
branches |L| > γ.
"""
from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from ..derivation import Int, make_step, merge_conditions, split_constant, wrap_solution_steps
from ..errors import TechniqueFailure
from ..models import Claim, Solution
from ..render import Msg
from ..strategy.features import Features
from ..verify import _canon_trig
from .base import Context, Technique
from .rational import QuadraticCompletion
from .table import linear_form


class TrigSubstitution(Technique):
    name = "trig_substitution"
    title = "Trigonometric (and Euler) substitution"
    family = "substitution"
    description = ("For R(x)(k + c·L²)^p with L linear: L = γ tan θ for c, k > 0; L = γ sin θ for c < 0 < k; Euler's "
                   "substitution t = L + √(L² − γ²) for c > 0 > k. Half-integer powers (radicals) and integer powers "
                   "≤ −2 are covered.")

    @staticmethod
    def _parse(f, x) -> Optional[Tuple]:
        nodes = []
        for n in sp.preorder_traversal(f):
            if n.is_Pow and n.base.has(x) and n.exp.is_Rational and not n.exp.has(x):
                p = n.exp
                if (p.q == 2) or (p.is_Integer and p <= -2):
                    cf = QuadraticCompletion._completed_form(n.base, x)
                    if cf is not None:
                        nodes.append((n, cf))
        if not nodes:
            return None
        bases = {n.base for n, _ in nodes}
        if len(bases) != 1:
            return None
        node, (k, c, L) = nodes[0]
        if not (k.is_number and c.is_number):
            return None
        if k.is_zero or c.is_zero:
            return None
        base = node.base
        # every occurrence of the base must be a pure power with an admissible exponent
        stripped = f
        for n, _ in nodes:
            stripped = stripped.xreplace({n: sp.Integer(1)})
        if stripped.has(base):
            return None
        R = stripped
        if R.has(x) and not R.is_rational_function(x):
            return None
        if c > 0 and k > 0:
            kind = "tan"
        elif c < 0 and k > 0:
            kind = "sin"
        elif c > 0 and k < 0:
            kind = "euler"
            if all(n.exp.is_Integer for n, _ in nodes):
                return None  # integer powers of u²−a²: partial fractions over the reals handle these
        else:
            return None
        return kind, base, k, c, L, nodes

    def suitability(self, feats: Features):
        try:
            p = self._parse(feats.expr, feats.x)
        except Exception:
            return None
        if p is None:
            return None
        return 0.86, {"tan": "a²+u² under a power: u = a tan θ.", "sin": "a²−u² under a power: u = a sin θ.",
                      "euler": "u²−a² under a root: Euler's substitution."}[p[0]]

    def attempt(self, f, x, ctx: Context) -> Optional[Solution]:
        parsed = self._parse(f, x)
        if parsed is None:
            return None
        kind, base, k, c, L, nodes = parsed
        a, b = linear_form(L, x)
        th = ctx.fresh_symbol(f, x)
        S = sp.Dummy("S", positive=True)
        if kind == "tan":
            gam = sp.sqrt(k / c)
            phi_L = gam * sp.tan(th)                     # L as a function of θ
            ginv = sp.atan(L / gam)                      # θ as a function of x
            mapping = {n: (c * gam ** 2) ** n.exp * S ** (2 * n.exp) for n, _ in nodes}
            back = {S: sp.sec(th)}
            title = Msg("Trigonometric substitution (tangent)")
            why = Msg("The base {B} = {k} + {c}·({L})² has the form a² + u², so put {L} = {gam} tan θ. Then "
                      "{B} = {val} sec² θ, which turns the root/power into a power of sec θ.",
                      B=base, k=k, c=c, L=L, gam=gam, val=c * gam ** 2)
            formula = r"u=a\tan\theta,\quad a^2+u^2=a^2\sec^2\theta,\quad du=a\sec^2\theta\,d\theta"
            cond = []
        elif kind == "sin":
            gam = sp.sqrt(k / -c)
            phi_L = gam * sp.sin(th)
            ginv = sp.asin(L / gam)
            mapping = {n: (-c * gam ** 2) ** n.exp * S ** (2 * n.exp) for n, _ in nodes}
            back = {S: sp.cos(th)}
            title = Msg("Trigonometric substitution (sine)")
            why = Msg("The base {B} = {k} − {m}·({L})² has the form a² − u², so put {L} = {gam} sin θ with "
                      "−π/2 < θ < π/2. Then {B} = {val} cos² θ.", B=base, k=k, m=-c, L=L, gam=gam, val=-c * gam ** 2)
            formula = r"u=a\sin\theta,\quad a^2-u^2=a^2\cos^2\theta,\quad du=a\cos\theta\,d\theta"
            cond = [Msg("|{L}| < {gam}", L=L, gam=gam)]
        else:
            gam = sp.sqrt(-k / c)
            # Euler: t = L + sqrt(L²−γ²); L = (t²+γ²)/(2t); sqrt(L²−γ²) = (t²−γ²)/(2t)
            phi_L = (th ** 2 + gam ** 2) / (2 * th)
            ginv = L + sp.sqrt(L ** 2 - gam ** 2)
            root = (th ** 2 - gam ** 2) / (2 * th)
            mapping = {n: c ** n.exp * root ** (2 * n.exp) for n, _ in nodes}
            back = {}
            title = Msg("Euler substitution")
            why = Msg("The base {B} = {c}·({L}² − {g2}) has the form u² − a², so use t = {L} + √({L}² − {g2}). "
                      "This is algebraic: {L} = (t² + {g2})/(2t) and √({L}² − {g2}) = (t² − {g2})/(2t).",
                      B=base, c=c, L=L, g2=gam ** 2)
            formula = r"t=u+\sqrt{u^2-a^2},\ u=\frac{t^2+a^2}{2t},\ \sqrt{u^2-a^2}=\frac{t^2-a^2}{2t},\ du=\frac{t^2-a^2}{2t^2}dt"
            cond = [Msg("|{L}| > {gam}", L=L, gam=gam)]
        x_of_th = (phi_L - b) / a
        stripped = f
        for n, _ in nodes:
            stripped = stripped.xreplace({n: mapping[n]})
        h = stripped.xreplace({x: x_of_th}) * sp.diff(x_of_th, th)
        h = h.xreplace(back)
        try:
            if kind == "euler":
                h = sp.cancel(sp.together(h))
            else:
                h = sp.simplify(h)
                if kind == "tan":  # tan/sec -> sin/cos so the sin/cos techniques (and reductions) recognise it
                    h = sp.powsimp(sp.simplify(_canon_trig(h)))
        except Exception:
            pass
        if h.has(x) or h.has(S):
            raise TechniqueFailure("the substitution did not remove the original variable")
        cc, h0 = split_constant(h, th)
        claim = Claim.identity(h.xreplace({th: ginv}) * sp.diff(ginv, x), f, x, "h(g(x))·g'(x) equals the integrand")
        sub = ctx.solve_sub(h0, th)
        if sub is None:
            raise TechniqueFailure("the transformed integral could not be solved")
        H = sp.expand_trig(cc * sub.F) if kind != "euler" else cc * sub.F
        F = H.xreplace({th: ginv})
        steps = [
            make_step("substitution", title,
                      Int(f, x), sp.Eq(L, phi_L) if kind != "euler" else sp.Eq(th, ginv), why, formula=formula,
                      conditions=cond),
            make_step("rewrite_in_new_variable", Msg("Rewrite the integral in terms of {t}", t=th), Int(f, x),
                      cc * Int(h0, th), Msg("Substituting gives the integrand {h} in the new variable.", h=h0),
                      claims=[claim], meta={"u": th, "g": ginv}),
        ]
        steps.extend(wrap_solution_steps(h0, th, sub))
        steps.append(make_step("back_substitute", Msg("Substitute back"), cc * sub.F, F,
                               Msg("Return to {x} using {t} = {g}.", x=x, t=th, g=ginv),
                               claims=[Claim.derivative(F, f, x, "the result differentiates to the integrand")]))
        meta = {"substitution": {"u": th, "g": ginv, "h": h, "c": cc, "h0": h0, "method": "T", "x": x}}
        return Solution(F, steps, self.name, merge_conditions(cond, sub.conditions), meta=meta,
                        derivation_level=sub.derivation_level)
