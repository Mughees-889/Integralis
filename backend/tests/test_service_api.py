import json

import pytest

from integralis import service


def test_parse_endpoint_contract():
    ok = service.parse_request({"expression": "∫ x e^(x²) dx"})
    assert ok["ok"] and "\\int" in ok["latex"] and ok["variable"] == "x" and ok["mode"] == "indefinite"
    bad = service.parse_request({"expression": "sin("})
    assert not bad["ok"] and bad["error"]["position"] == 4 and bad["error"]["hint"]


def test_integrate_endpoint_contract_is_json_serialisable():
    out = service.integrate_request({"expression": "∫ x e^(x²) dx"})
    s = json.dumps(out)
    assert out["ok"] and out["status"] == "solved" and out["answer"]["text"].endswith("+ C")
    st = out["steps"][0]
    for key in ("number", "rule", "title", "before", "after", "explanation", "conditions", "verification", "substeps"):
        assert key in st
    assert st["title"]["latex"] and out["verification"]["status"] in ("verified", "verified_numeric")
    assert len(s) > 100


def test_definite_request_and_decimal_places():
    out = service.integrate_request({"expression": "sin(x)", "lower": "0", "upper": "pi", "decimal_places": 8})
    assert out["status"] == "solved" and out["answer"]["text"] == "2" and out["decimal"].startswith("2.0")


def test_failure_is_a_normal_result_not_an_exception():
    out = service.integrate_request({"expression": "x^x"})
    assert out["ok"] and out["status"] == "failed" and out["message"]["text"]


def test_parse_errors_in_integrate_request():
    out = service.integrate_request({"expression": "2 +"})
    assert not out["ok"] and out["error"]["message"]


def test_assumption_strings():
    out = service.integrate_request({"expression": "exp(-a*x)", "lower": "0", "upper": "oo", "assumptions": "a>0"})
    assert out["status"] == "solved" and out["answer"]["text"] == "1/a"


def test_catalogue_and_examples():
    cat = service.techniques_catalogue()
    assert len(cat) >= 15 and all(t["description"] for t in cat)
    assert service.examples()["indefinite"]


def test_fastapi_app_if_available():
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from integralis.api.app import app
    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    r = c.post("/api/parse", json={"expression": "x^2"})
    assert r.status_code == 200 and r.json()["ok"]
    r = c.post("/api/integrate", json={"expression": "x e^(x^2)"})
    assert r.status_code == 200 and r.json()["status"] == "solved"
    r = c.post("/api/integrate", json={"expression": "sin("})
    assert r.status_code == 400
