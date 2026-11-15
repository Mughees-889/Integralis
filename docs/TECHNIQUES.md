# Catalogue of integration techniques

Generated from the technique registry (`integralis/techniques/__init__.py`). The strategy selector ranks these from the structure of the
integrand (see `strategy/selector.py`); the engine tries them best-first and falls through when a technique fails or its result does
not verify. Heuristics, not guarantees: a pattern match never forces an inappropriate method.

## Basic rules

### `linearity` — Linearity (sum rule and constant multiples)

Integrate a sum term by term and take constant factors outside the integral. The resulting integrals are solved recursively by the other techniques.

### `table` — Standard integrals and the reverse chain rule

Direct use of standard antiderivatives: power rule, reciprocal, exponentials, trigonometric and hyperbolic functions, and inverse-trigonometric/logarithmic forms — each generalised to a linear inner function ax+b by the reverse chain rule.

## Algebra

### `expand` — Algebraic expansion

Multiply out a product or power of polynomials so the integrand becomes a sum of simple powers that can be integrated term by term.

## Rational functions

### `rational` — Rational functions: polynomial division and partial fractions

For P(x)/Q(x): divide when deg P ≥ deg Q, factor the denominator, set up the partial-fraction form, solve for the coefficients, and integrate each fraction.

### `complete_square` — Completing the square

Rewrites N(x)/(ax²+bx+c)^p (or the same with a root) as terms in (x+s)²+d, splitting the linear numerator into a part proportional to x+s (a logarithm / power after u = (x+s)²+d) and a constant part (an arctangent/arcsine-type integral).

## Substitution

### `substitution` — Substitution (reverse chain rule and change of variable)

Finds an inner function g(x) whose derivative appears in the integrand (or that can be inverted uniquely), rewrites the integral in u = g(x), integrates, and substitutes back. Every rewriting is verified by checking h(g(x))·g'(x) = f(x).

### `trig_substitution` — Trigonometric (and Euler) substitution

For R(x)(k + c·L²)^p with L linear: L = γ tan θ for c, k > 0; L = γ sin θ for c < 0 < k; Euler's substitution t = L + √(L² − γ²) for c > 0 > k. Half-integer powers (radicals) and integer powers ≤ −2 are covered.

## Integration by parts

### `cyclic_parts` — Repeated integration by parts returning to the original integral

For e^{ax+b}·sin(cx+d) and e^{ax+b}·cos(cx+d): two applications of integration by parts bring back a multiple of the original integral I, giving an equation I = A + kI that is solved for I.

### `tabular_parts` — Tabular integration by parts

For P(x)·e^{ax+b}, P(x)·sin(ax+b), P(x)·cos(ax+b) (and hyperbolic analogues) with a polynomial P: differentiate P down to zero in one column, integrate the other factor repeatedly in a second column, and add the diagonal products with alternating signs.

### `parts` — Integration by parts

∫u dv = uv − ∫v du with u chosen by the LIATE guideline (logarithmic, inverse-trigonometric, algebraic, trigonometric, exponential). A lone ln x or arctan x uses dv = dx. Repeated application happens automatically when the new integral again needs parts.

## Trigonometric and hyperbolic

### `trig_powers` — Powers of sine and cosine

∫ sin^m θ cos^n θ: split off one factor when a power is odd and use sin²+cos²=1 so that u = cos θ or u = sin θ applies; reduce even powers with the double-angle identities sin²θ = (1−cos 2θ)/2 and cos²θ = (1+cos 2θ)/2.

### `trig_tan_sec` — Powers of tangent and secant (and cotangent and cosecant)

∫ tan^m sec^n and ∫ cot^m csc^n: with sec² = 1 + tan² an even power of sec leads to u = tan, an odd power of tan leads to u = sec; otherwise tan² = sec² − 1 reduces the power.

### `trig_product_to_sum` — Product-to-sum identities

sin A cos B, sin A sin B and cos A cos B with different linear angles are rewritten as sums of sines/cosines of A ± B, which integrate term by term.

### `hyperbolic_exp` — Hyperbolic functions through exponentials

Products and powers of sinh and cosh are rewritten with sinh θ = (e^θ − e^−θ)/2 and cosh θ = (e^θ + e^−θ)/2 and expanded into a sum of exponentials.

### `weierstrass` — Weierstrass substitution t = tan(θ/2)

Any rational function of sin θ and cos θ becomes a rational function of t = tan(θ/2) with sin θ = 2t/(1+t²), cos θ = (1−t²)/(1+t²) and dθ = 2 dt/(1+t²). Last resort among trigonometric techniques because the antiderivative can jump where tan(θ/2) is undefined (θ = π + 2kπ).

## Reduction formulae

### `reduction` — Reduction formulae

Recurrences for ∫sinⁿθ, ∫cosⁿθ, ∫tanⁿθ, ∫cotⁿθ, ∫secⁿθ and ∫cscⁿθ that lower the power by one or two at each application until a standard integral remains. Each instance of the recurrence is verified by differentiation before it is shown.

## Fallback

### `sympy_general` — SymPy general algorithms (last resort)

Hands the integral to SymPy's general-purpose algorithms when none of the human-style techniques works. The result is accepted only if it differentiates back to the integrand; special-function results (erf, Ei, Si, li, polylog, ...) come from here.

## Definite integrals (`integralis/definite.py`)

- **Antiderivative route** — evaluates a verified antiderivative between the limits. The line is cut at every singularity of the integrand *and* of the antiderivative; one-sided limits are used on each piece.
- **Spurious-jump correction** — antiderivatives such as `arctan(tan(x/2)/√3)` jump where the integrand is smooth; the break points handle this.
- **Divergence and convergence** — infinite limits and singularities (endpoint or interior) are analysed through exact limits; a divergent integral never gets a value. When no antiderivative exists, rigorous p-tests / limit comparison decide convergence first.
- **Substitution with transformed limits** — shown when the antiderivative came from `u = g(x)` and `g` is continuous and monotonic on the interval.
- **Symmetry** — odd integrands on `[−c, c]` give 0, even integrands double the half-interval.
- **King property** — `∫ f(x) dx = ∫ f(a+b−x) dx`; used when `f(x)+f(a+b−x)` is constant.
- **Special-function route** — SymPy's definite algorithms, accepted only after a convergence test and agreement with arbitrary-precision quadrature.
- **Numerical cross-check** — `mpmath` quadrature at 30 digits, reported as evidence, never as proof. For oscillatory improper integrals, where quadrature is unreliable, no cross-check is claimed.
