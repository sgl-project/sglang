"""SIGTERM readiness-drain integration tests.

The drain exists to produce a *Kubernetes* behaviour, and the Rust tests can
only argue it: they substitute a channel for the real `Signal` and an
in-process `AppContext` for a real pod. These run the shipped container, so
they cover `main.rs::shutdown_signal` — the signal handler, the SIGTERM/SIGINT
branch, and the `ctx` wiring — which no in-process test reaches.

Why `kill -TERM 1` rather than `kubectl delete pod`: deleting a pod stamps a
`deletionTimestamp`, and the endpoints controller marks the endpoint not-ready
on that alone, without ever consulting `/readyz`. A delete-based test would
therefore pass identically with the drain removed — it would look like
coverage while pinning nothing. Signalling the process directly leaves the pod
undeleted, so a `/readyz` 503 can only have come from the drain calling
`AppContext::mark_not_ready`.

The router image is `debian:bookworm-slim` with an exec-form ENTRYPOINT, so
the binary is PID 1 and `kill -TERM 1` reaches it exactly as kubelet's SIGTERM
would.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import httpx
from conftest import (
    NAMESPACE,
    _cleanup_port_forward,
    _container_restart_count,
    _kubectl,
    _pod_names,
    _pod_ready_condition,
    _poll_until,
    _port_forward_start,
    _wait_for_deployment_ready,
)

logger = logging.getLogger(__name__)

# Distinct from the shared 8090 forward so this test's pod-scoped forward
# cannot collide with a leaked service-scoped one from another test.
DRAIN_PORT = 8094

_ROUTER_MANIFEST = Path(__file__).parent / "manifests" / "router.yaml"


def _manifest_drain_secs() -> int:
    """Read `--shutdown-drain-secs` out of the manifest the pod is started
    from. Read rather than restated, because every assertion below is scaled to
    the drain window: a manifest edit that this file did not track would leave
    the test green while measuring the wrong window."""
    args = _ROUTER_MANIFEST.read_text()
    match = re.search(
        r'"--shutdown-drain-secs"\s*\n\s*-\s*"(\d+)"',
        args,
    )
    assert match, f"--shutdown-drain-secs not found in {_ROUTER_MANIFEST}"
    return int(match.group(1))


CONFIGURED_DRAIN_SECS = _manifest_drain_secs()

# Budget for observing the /readyz flip, deliberately a fraction of the drain:
# the assertions that follow it must still land inside the window, so the poll
# cannot be allowed to consume the whole thing.
FLIP_OBSERVATION_SECS = max(2, CONFIGURED_DRAIN_SECS // 2)


def _router_pod() -> str:
    pods = _pod_names("app=sgl-router")
    assert len(pods) == 1, f"expected exactly one live router pod, got {pods}"
    return pods[0]


class TestReadinessDrain:
    """SIGTERM must flip /readyz to 503 while the pod keeps serving."""

    def test_sigterm_flips_readyz_while_the_pod_keeps_serving(self, k8s_cluster):
        _wait_for_deployment_ready("sgl-router")
        pod = _router_pod()
        restarts_before = _container_restart_count(pod, "router")

        # Bind the pod, not the Service: a draining pod leaves the Service's
        # ready endpoints, and the point of this test is to keep talking to it
        # after that happens.
        pf = _port_forward_start(NAMESPACE, pod, DRAIN_PORT, 8090, resource="pod")
        base = f"http://127.0.0.1:{DRAIN_PORT}"
        try:
            assert httpx.get(f"{base}/readyz", timeout=5.0).status_code == 200, (
                "router must be ready before SIGTERM"
            )

            # Through `sh -c`: the slim image ships no `kill` binary, and
            # `kubectl exec` execs directly rather than through a shell, so the
            # builtin is the only way to signal PID 1 from outside.
            sigterm_at = time.monotonic()
            _kubectl("exec", "-n", NAMESPACE, pod, "--", "sh", "-c", "kill -TERM 1")

            # The flip is observable from outside the pod.
            _poll_until(
                lambda: httpx.get(f"{base}/readyz", timeout=3.0).status_code == 503,
                "/readyz returns 503 after SIGTERM",
                timeout=FLIP_OBSERVATION_SECS,
                interval=0.2,
            )

            # ...and the pod is still serving while it reports not-ready.
            # `/healthz` staying 200 is what stops the liveness probe
            # restarting a pod that is draining on purpose.
            assert httpx.get(f"{base}/healthz", timeout=5.0).status_code == 200, (
                "liveness must stay green while the pod drains"
            )

            # A proxied completion still succeeding does double duty: it is the
            # request k8s may still route during the window, AND it proves the
            # worker registry is non-empty — so the 503 above can only be the
            # readiness flip, not `/readyz`'s other term.
            chat = httpx.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": "tiny",
                    "messages": [{"role": "user", "content": "drain"}],
                },
                timeout=10.0,
            )
            assert chat.status_code == 200, (
                f"a proxied request must still succeed mid-drain, got {chat.status_code}"
            )

            # Everything above claims to have run *inside* the window. Say so,
            # so an overrun reads as "the window closed" and not as whichever
            # transport error the closed listener happened to raise next.
            mid_drain_elapsed = time.monotonic() - sigterm_at
            assert mid_drain_elapsed < CONFIGURED_DRAIN_SECS, (
                f"the mid-drain assertions took {mid_drain_elapsed:.1f}s, past the "
                f"{CONFIGURED_DRAIN_SECS}s window they claim to observe"
            )

            # Recorded, not asserted: k8s needs failureThreshold consecutive
            # failing probes, periodSeconds apart, to mark the pod not-ready —
            # longer than the drain at the values in router.yaml. That is
            # exactly why the default is sized for the deletionTimestamp path
            # instead, and why probe-driven setups must raise it.
            logger.info("pod Ready condition mid-drain: %s", _pod_ready_condition(pod))

        finally:
            _cleanup_port_forward(f"pod/{pod}", pf)

        # The drain must END in an exit, and must have lasted roughly the
        # configured window. Both are read off `restartCount`: the process
        # exits when the drain elapses and kubelet restarts the container in
        # place, same pod. Timing it this way rather than by watching the
        # listener close is deliberate — the restart is fast enough that a
        # port-forward probe can miss the closed window entirely and hang,
        # whereas `restartCount` is monotonic and cannot be missed. Kubelet's
        # own latency only ever makes this longer, so the lower bound is safe.
        _poll_until(
            lambda: _container_restart_count(pod, "router") > restarts_before,
            "router container restarts once the drain elapses",
            timeout=CONFIGURED_DRAIN_SECS + 90,
            interval=0.5,
        )
        held_open_for = time.monotonic() - sigterm_at
        # One second of slack for signal delivery and poll granularity.
        assert held_open_for >= CONFIGURED_DRAIN_SECS - 1, (
            f"the router exited {held_open_for:.1f}s after SIGTERM, short of the "
            f"configured {CONFIGURED_DRAIN_SECS}s drain"
        )
        _wait_for_deployment_ready("sgl-router")
