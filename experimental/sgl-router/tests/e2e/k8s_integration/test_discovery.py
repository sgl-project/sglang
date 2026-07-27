"""E2E: sgl-router K8s discovery — basic routing.

Verifies that sgl-router, configured with the k8s EndpointSlice backend,
discovers the 3 fake-worker replicas deployed by setup.sh and successfully
routes chat-completion requests to them.
"""

from __future__ import annotations

import httpx
from conftest import NAMESPACE, _kubectl, _poll_until


def _scale_fake_worker(replicas: int) -> None:
    """Scale the fake-worker deployment to the requested replica count."""
    _kubectl(
        "scale",
        "deployment/fake-worker",
        f"--replicas={replicas}",
        "-n",
        NAMESPACE,
    )


def _snapshot_has_exact_fresh_workers(router_url: str, expected: int) -> bool:
    """Return whether the monitor exposes exactly `expected` fresh workers."""
    response = httpx.get(
        f"{router_url}/v1/load_monitor/snapshot",
        timeout=10.0,
    )
    response.raise_for_status()
    snapshot = response.json()
    workers = snapshot["workers"]
    return (
        snapshot["enabled"] is True
        and len(workers) == expected
        and all(worker["freshness"] == "fresh" for worker in workers)
    )


def test_router_routes_chat_to_a_worker(router_url):
    """A /v1/chat/completions request through the router returns 200 with the
    fake-worker echo payload, proving end-to-end routing works."""
    # HTTP readiness deliberately does not wait for engine load. Wait for the
    # reporting loop here before expecting a routable fresh candidate.
    _poll_until(
        lambda: _snapshot_has_exact_fresh_workers(router_url, 3),
        "load monitor exposes fresh workers before routing",
        timeout=30,
        interval=1,
    )
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


def test_load_monitor_snapshot_contains_fresh_workers(router_url):
    """All discovered fake workers eventually publish fresh load snapshots."""
    _poll_until(
        lambda: _snapshot_has_exact_fresh_workers(router_url, 3),
        "load monitor exposes all three fresh fake-worker reports",
        timeout=30,
        interval=1,
    )


def test_router_discovers_multiple_workers(router_url):
    """Scale down from 3 to 1 and back to 3 replicas; router must continue
    routing successfully after each transition (EndpointSlice watch reflects
    the change)."""
    # Do not depend on pytest definition order: each routing test establishes
    # the fresh-load precondition independently.
    _poll_until(
        lambda: _snapshot_has_exact_fresh_workers(router_url, 3),
        "load monitor exposes three fresh workers before scale testing",
        timeout=30,
        interval=1,
    )
    # First confirm baseline routing.
    r = httpx.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": "tiny",
            "messages": [{"role": "user", "content": "scale-test"}],
        },
        timeout=15.0,
    )
    assert r.status_code == 200

    # Scale down to 1 and require the immutable snapshot to remove both old
    # worker entries rather than merely routing around them.
    _scale_fake_worker(1)
    _poll_until(
        lambda: _snapshot_has_exact_fresh_workers(router_url, 1),
        "load monitor removes scaled-down workers and keeps one fresh worker",
        timeout=60,
        interval=1,
    )

    # Restore to 3 and wait for discovery, registration, and reporting to
    # converge before leaving shared cluster state for later tests.
    _scale_fake_worker(3)
    _poll_until(
        lambda: _snapshot_has_exact_fresh_workers(router_url, 3),
        "load monitor restores three fresh workers after scale-up",
        timeout=90,
        interval=1,
    )
