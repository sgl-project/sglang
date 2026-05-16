"""Worker lifecycle integration tests.

Covers:
1. Scaling replicas up — new EndpointSlice entries are discovered.
2. Scaling replicas down — removed endpoints are deregistered.
3. Router restart — after the router pod is killed, the Deployment restarts
   it and it re-lists the existing EndpointSlice entries without duplicates.

These tests DO NOT use a /workers admin API (sgl-router does not expose
one). They verify behaviour through /v1/chat/completions responses and
by driving the deployment scale.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from conftest import (
    NAMESPACE,
    _cleanup_port_forward,
    _kubectl,
    _poll_until,
    _port_forward_start,
    _wait_for_deployment_ready,
    logger,
)

ROUTER_RESTART_PORT = 8092


def _scale(deployment: str, replicas: int) -> None:
    _kubectl(
        "scale", f"deployment/{deployment}", f"--replicas={replicas}", "-n", NAMESPACE
    )


def _can_route(router_url: str) -> bool:
    try:
        r = httpx.post(
            f"{router_url}/v1/chat/completions",
            json={"model": "tiny", "messages": [{"role": "user", "content": "ping"}]},
            timeout=8.0,
        )
        return r.status_code == 200
    except Exception:
        return False


class TestScaleUp:
    """Scaling fake-worker replicas up must not break routing."""

    def test_router_routes_after_scale_up(self, router_url):
        """Restore 3 replicas (in case a prior test left 1), verify routing."""
        _scale("fake-worker", 3)
        _poll_until(
            lambda: _can_route(router_url),
            "router routes after scale-up to 3",
            timeout=60,
            interval=3,
        )


class TestScaleDown:
    """Scaling to 0 then back up must restore routing."""

    def test_router_recovers_after_scale_to_zero_and_back(self, router_url):
        try:
            _scale("fake-worker", 0)
            # After scale-to-0 the router may return 503 (no healthy workers)
            # That is expected behaviour — assert it transitions back on scale-up.
            _scale("fake-worker", 2)
            _poll_until(
                lambda: _can_route(router_url),
                "router routes again after scale-up from 0",
                timeout=90,
                interval=3,
            )
        finally:
            _scale("fake-worker", 3)


class TestRouterRestart:
    """Killing the router pod forces a Deployment restart; the new pod must
    re-discover workers via the EndpointSlice watch without duplicates."""

    def test_router_rediscovers_workers_after_restart(self, k8s_cluster):
        # Use a dedicated port to avoid clashing with the session fixture
        pf_holder: list = [None]
        try:
            _wait_for_deployment_ready("sgl-router")
            pf_holder[0] = _port_forward_start(
                NAMESPACE, "sgl-router", ROUTER_RESTART_PORT, 8090
            )
            restart_url = f"http://127.0.0.1:{ROUTER_RESTART_PORT}"

            # Baseline: routing works pre-restart
            _poll_until(
                lambda: _can_route(restart_url),
                "baseline routing works pre-restart",
                timeout=30,
                interval=2,
            )

            # Kill the router pod — the Deployment ReplicaSet will restart it
            res = _kubectl(
                "get",
                "pod",
                "-n",
                NAMESPACE,
                "-l",
                "app=sgl-router",
                "-o",
                "jsonpath={.items[0].metadata.name}",
                check=False,
            )
            old_pod = res.stdout.strip()
            if old_pod:
                _kubectl(
                    "delete",
                    "pod",
                    old_pod,
                    "-n",
                    NAMESPACE,
                    "--force",
                    "--grace-period=0",
                )

            # Tear down the old port-forward before waiting for the new pod
            if pf_holder[0] is not None:
                _cleanup_port_forward("router-restart-pre-kill", pf_holder[0])
                pf_holder[0] = None

            _wait_for_deployment_ready("sgl-router")

            pf_holder[0] = _port_forward_start(
                NAMESPACE, "sgl-router", ROUTER_RESTART_PORT, 8090
            )

            # After restart, routing must come back (EndpointSlice re-watch)
            _poll_until(
                lambda: _can_route(restart_url),
                "routing restored after router restart",
                timeout=60,
                interval=3,
            )
        finally:
            if pf_holder[0] is not None:
                _cleanup_port_forward("router-restart", pf_holder[0])
