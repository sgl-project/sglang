import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_enrollment(monkeypatch):
    # Configure env with enrollment_url
    monkeypatch.setenv("DEPLOYMENT_CONFIG_JSON", json.dumps({
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "tp_size": 1,
        "enable_metrics": False,
        "h100_only": False,
        "namespace": "ns-a",
        "deployment_id": "dep-123",
        "customer_id": "cust-xyz",
        "enrollment_url": "http://dev-platform/enroll"
    }))

    import importlib
    app_module = importlib.import_module("serve_hathora")

    # Patch engine to avoid heavy init
    class DummyEngine:
        async def async_generate(self, prompt, sampling_params=None, stream=False):
            return {"text": prompt + " ok"}
        def shutdown(self):
            return None
    app_module.engine = DummyEngine()

    # Patch httpx.Client.post to capture payload
    captured = {}
    class DummyClient:
        def __init__(self, timeout=None):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def post(self, url, json=None):
            captured["url"] = url
            captured["json"] = json
            class R: pass
            return R()

    import serve_hathora as mod
    mod.httpx = type("_", (), {"Client": DummyClient})

    return app_module.app, captured


def test_enrollment_callback(app_with_enrollment):
    app, captured = app_with_enrollment
    client = TestClient(app)
    # trigger startup by first request
    r = client.get("/health")
    assert r.status_code == 200
    # we expect enrollment payload captured
    assert captured["url"].endswith("/enroll")
    p = captured["json"]
    assert p["deployment_id"] == "dep-123"
    assert p["namespace"] == "ns-a"
    assert p["customer_id"] == "cust-xyz"

