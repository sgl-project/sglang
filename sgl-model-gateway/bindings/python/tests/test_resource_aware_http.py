"""HTTP-edge tests for resource-aware virtual model routing."""

import time

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import sglang_router.resource_aware_router as resource_router
from sglang_router.multi_model import ModelResolverConfig, RoutingProfile
from sglang_router.resource_aware_router import (
    ResourceAwareModelResolver,
    RuntimeLoadCollector,
    RuntimeEndpoint,
    RuntimeLoadState,
)


def test_http_edge_rewrites_only_virtual_model_requests(monkeypatch):
    config = ModelResolverConfig(
        host="127.0.0.1",
        port=30000,
        refresh_interval_secs=1.0,
        stale_after_secs=3.0,
        request_timeout_secs=30.0,
        profiles=(
            RoutingProfile(
                model_id="general-chat",
                candidates=("qwen-awq", "qwen-fp16"),
                max_kv_utilization=0.9,
                max_waiting_requests=4,
                min_free_tokens=32,
            ),
        ),
    )
    resolver = ResourceAwareModelResolver(config)
    now = time.monotonic()
    resolver.update(
        RuntimeLoadState(
            model_id="qwen-awq",
            url="http://awq",
            observed_at=now,
            healthy=True,
            num_total_tokens=800,
            max_total_num_tokens=1000,
            token_usage=0.8,
        )
    )
    resolver.update(
        RuntimeLoadState(
            model_id="qwen-fp16",
            url="http://fp16",
            observed_at=now,
            healthy=True,
            num_total_tokens=100,
            max_total_num_tokens=1000,
            token_usage=0.1,
        )
    )
    collector = RuntimeLoadCollector(
        resolver,
        (
            RuntimeEndpoint("qwen-awq", "http://awq"),
            RuntimeEndpoint("qwen-fp16", "http://fp16"),
        ),
        refresh_interval_secs=10.0,
        request_timeout_secs=1.0,
    )
    monkeypatch.setattr(collector, "start", lambda: None)
    monkeypatch.setattr(collector, "stop", lambda: None)
    captured = {}

    async def fake_proxy(_request, body, **kwargs):
        captured["body"] = body
        captured.update(kwargs)
        return JSONResponse({"model": body["model"]})

    monkeypatch.setattr(resource_router, "_proxy_resolved_request", fake_proxy)
    app = resource_router.create_resource_aware_app(
        resolver,
        collector,
        backend_url="http://router-backend:30001",
        request_timeout_secs=30.0,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "general-chat", "messages": [], "max_tokens": 16},
        )
        models_response = client.get("/v1/models")
        ready_response = client.get("/ready")
        health_response = client.get("/health/verbose")

    assert response.status_code == 200
    assert response.json() == {"model": "qwen-fp16"}
    assert response.headers["x-sglang-requested-model"] == "general-chat"
    assert response.headers["x-sglang-resolved-model"] == "qwen-fp16"
    assert captured["body"]["model"] == "qwen-fp16"
    assert captured["backend_url"] == "http://router-backend:30001"
    assert models_response.status_code == 200
    assert [model["id"] for model in models_response.json()["data"]] == [
        "general-chat",
        "qwen-awq",
        "qwen-fp16",
    ]
    assert ready_response.status_code == 200
    assert ready_response.json()["profiles"] == [
        {
            "model_id": "general-chat",
            "ready": True,
            "fresh_candidates": ["qwen-awq", "qwen-fp16"],
        }
    ]
    assert health_response.status_code == 200
    assert len(health_response.json()["runtimes"]) == 2
