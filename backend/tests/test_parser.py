import pytest
import sympy as sp

from integralis.errors import ParseError
from integralis.parsing import parse_expression, parse_problem

x = sp.Symbol("x", real=True)

SPEC_EXAMPLES = [
    "∫ sin(x) dx", "∫ x e^(x²) dx", "∫ x² ln(x) dx", "∫ x/(x² + 1) dx", "∫ sin²(x) cos³(x) dx",
    "∫ 1/(x⁴ + 1) dx", "∫ ln(x)/x dx", "∫ e^x sin(x) dx", "∫ arctan(x) dx", "∫ sqrt(1 − x²) dx",
    "∫ x³/(x² + 1) dx", "∫ 1/(x² + 4x + 5) dx",
    "sin(x)", "sin x", "exp(x^2)", "sqrt(1-x^2)", "ln(x)", "log(x)", "arctan(x)", "(x^2+1)/(x^3+3*x)", "x^2 * exp(x)",
]


@pytest.mark.parametrize("text", SPEC_EXAMPLES)
def test_spec_examples_parse(text):
    p = parse_problem(text)
    assert p.var.name == "x" and p.integrand.has(p.var)


@pytest.mark.parametrize("text,expected", [
    ("x e^(x²)", x * sp.exp(x ** 2)), ("2x(x+1)", 2 * x * (x + 1)), ("xsinx", x * sp.sin(x)),
    ("sin 2x", sp.sin(2 * x)), ("sin^2 x", sp.sin(x) ** 2), ("-x^2", -x ** 2), ("2^-x", 2 ** (-x)),
    ("e^-x^2", sp.exp(-x ** 2)), ("x²³".replace("²³", "²"), x ** 2), ("1/2 x", x / 2), ("0.5x", x / 2),
    ("sqrt(x)*sqrt(x)", x), ("|x-1|", sp.Abs(x - 1)), ("ln x / x", sp.log(x) / x), ("3^x", 3 ** x),
])
def test_notation_and_implicit_multiplication(text, expected):
    assert sp.simplify(parse_expression(text).expr - expected) == 0


def test_exact_rationals_no_floats():
    e = parse_expression("0.25*x + 1/3 + 2^(1/2)").expr
    assert not e.has(sp.Float)
    assert e.coeff(x) == sp.Rational(1, 4)


def test_definite_notation_and_limits():
    p = parse_problem("∫_0^pi sin(x) dx")
    assert p.definite and p.lower == 0 and p.upper == sp.pi
    p = parse_problem("x^2", lower="-oo", upper="oo")
    assert p.lower == -sp.oo and p.upper == sp.oo
    p = parse_problem("∫₀¹ x dx")
    assert p.lower == 0 and p.upper == 1


def test_variable_selection():
    assert parse_problem("t^2 + a t", variable="t").var.name == "t"
    assert parse_problem("t^2").var.name == "t"
    assert parse_problem("a*s", variable="s").var.name == "s"
    assert parse_problem("∫ u^2 du").var.name == "u"
    with pytest.raises(ParseError):
        parse_problem("a*b")  # ambiguous variable


@pytest.mark.parametrize("bad,fragment", [
    ("sin(", "unexpectedly"), ("2 3", "Unexpected number"), ("x +", "unexpectedly"), ("(x+1", "Missing ')'"),
    ("x)", "Unmatched"), ("x@y", "Unsupported character"), ("", "Empty"), ("1/0", "Division by zero"),
    ("sin^-1(x)", "ambiguous"), ("foo(x)", "Unknown function"), ("5!", "Factorials"), ("x dx", "differential"),
    ("∫ x", "∫"),
])
def test_malformed_input_gives_precise_errors(bad, fragment):
    with pytest.raises(ParseError) as ei:
        parse_problem(bad)
    assert fragment.lower() in ei.value.message.lower() or fragment.lower() in str(ei.value).lower()


def test_error_reports_position():
    with pytest.raises(ParseError) as ei:
        parse_expression("2 + * 3")
    assert ei.value.position == 4


@pytest.mark.parametrize("attack", [
    "__import__('os').system('echo hacked')", "x.__class__", "().__class__.__bases__", "exec('1')", "lambda: 1",
    "open('/etc/passwd')", "x; y", "x\nimport os", "[i for i in range(3)]", "globals()", "'a'*10",
])
def test_parser_never_executes_input(attack):
    # Either a clean ParseError or a harmless symbolic product — never code execution.
    try:
        r = parse_expression(attack)
        assert isinstance(r.expr, sp.Basic)
    except ParseError:
        pass


def test_limits_cannot_depend_on_variable():
    with pytest.raises(ParseError):
        parse_problem("x", lower="0", upper="x")


def test_log_convention_and_warning():
    p = parse_problem("log(x)")
    assert p.integrand == sp.log(x) and any("natural" in w for w in p.warnings)
    assert sp.simplify(parse_expression("log10(x)").expr - sp.log(x) / sp.log(10)) == 0
