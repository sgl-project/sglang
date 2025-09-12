import asyncio
import json
import types

import pytest
from fastapi.testclient import TestClient


def _patch_engine(app_module):
    class DummyEngine:
        async def async_generate(self, prompt, sampling_params=None, stream=False):
            if stream:
                async def gen():
                    # Simulate streaming increments
                    yield {"text": prompt + " hello"}
                    await asyncio.sleep(0)
                    yield {"text": prompt + " hello world"}
                return gen()
            return {"text": prompt + " hello world"}

        def shutdown(self):
            return None

    app_module.engine = DummyEngine()


@pytest.fixture
def app(monkeypatch):
    # Configure minimal env
    monkeypatch.setenv("DEPLOYMENT_CONFIG_JSON", json.dumps({
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "tp_size": 1,
        "enable_metrics": False,
        "h100_only": False,
        "namespace": "ns-a",
        "deployment_id": "dep-123",
        "customer_id": "cust-xyz"
    }))

    import importlib
    app_module = importlib.import_module("serve_hathora")
    _patch_engine(app_module)
    return app_module.app


def test_health(app):
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine_status"] == "ready"
    assert body["namespace"] == "ns-a"
    assert body["deployment_id"] == "dep-123"
    assert body["customer_id"] == "cust-xyz"


def test_chat_non_stream(app):
    client = TestClient(app)
    payload = {
        "model": "sglang",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
        "stream": False
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"].strip().startswith("hello")


def test_chat_stream(app):
    client = TestClient(app)
    payload = {
        "model": "sglang",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
        "stream": True
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        chunks = list(r.iter_lines())
        # Ensure we saw at least one chunk beginning with data:
        def starts_with_data(x):
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="ignore").startswith("data:")
            return str(x).startswith("data:")
        assert any(starts_with_data(line) for line in chunks)


