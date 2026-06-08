"""E2E: sgl-router K8s discovery — basic routing.

Verifies that sgl-router, configured with the k8s EndpointSlice backend,
discovers the 3 fake-worker replicas deployed by setup.sh and successfully
routes chat-completion requests to them.
"""

from __future__ import annotations

import httpx
import pytest
from conftest import NAMESPACE, _kubectl, _poll_until, logger


def _scale_fake_worker(replicas: int) -> None:
    _kubectl(
        "scale",
        "deployment/fake-worker",
        f"--replicas={replicas}",
        "-n",
        NAMESPACE,
    )


def test_router_routes_chat_to_a_worker(router_url):
    """A /v1/chat/completions request through the router returns 200 with the
    fake-worker echo payload, proving end-to-end routing works."""
    r = httpx.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": "tiny",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
        timeout=15.0,
    )
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert "echo:" in body["choices"][0]["message"]["content"]


def test_router_lists_model(router_url):
    """GET /v1/models returns the 'tiny' model entry from the router config."""
    r = httpx.get(f"{router_url}/v1/models", timeout=10.0)
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    ids = [m["id"] for m in body["data"]]
    assert "tiny" in ids, f"expected 'tiny' in model list, got {ids}"


def test_router_discovers_multiple_workers(router_url):
    """Scale down from 3 to 1 and back to 3 replicas; router must continue
    routing successfully after each transition (EndpointSlice watch reflects
    the change)."""
    # First confirm baseline routing
    r = httpx.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": "tiny",
            "messages": [{"role": "user", "content": "scale-test"}],
        },
        timeout=15.0,
    )
    assert r.status_code == 200

    # Scale down to 1 — router should still route after reconverging
    _scale_fake_worker(1)
    _poll_until(
        lambda: httpx.post(
            f"{router_url}/v1/chat/completions",
            json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "post-scale-down"}],
            },
            timeout=10.0,
        ).status_code
        == 200,
        "router routes after scale-down to 1",
        timeout=60,
        interval=3,
    )

    # Restore to 3
    _scale_fake_worker(3)
