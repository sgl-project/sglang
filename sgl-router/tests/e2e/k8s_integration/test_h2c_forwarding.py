"""E2E: the router forwards to an h2c-capable worker over cleartext HTTP/2.

The in-process tests (`tests/proxy/h2c_forward.rs`, `inbound_h2c.rs`) already
drive real HTTP/2 sockets, and dropping reqwest's `http2` feature fails the
build outright, so neither the client nor the framing needs covering again
here. What no in-process test can assemble is the *chain*: a worker discovered
through a real EndpointSlice, introspected over the network, resolved to
`WireProtocol::H2c` from its own `/server_info`, and then actually forwarded to
over h2c.

The fleet is deliberately mixed. `setup.sh` leaves three uvicorn workers
(HTTP/1.1 only) behind the `app=sglang` Service; this module adds one Granian
worker reporting `enable_http2: true` to the same Service, so both protocols
must be in use simultaneously. That is the e2e form of the per-worker-protocol
property: a router that resolved one protocol fleet-wide would either fail
against the h2c worker or break the three HTTP/1.1 ones, and either way this
test fails.
"""

from __future__ import annotations

import httpx
import pytest
from conftest import (
    NAMESPACE,
    _apply_from_stdin,
    _kubectl,
    _poll_until,
    _wait_for_deployment_ready,
    logger,
)

H2C_DEPLOYMENT = "fake-worker-h2c"

# Round-robin over a 4-worker pool: 12 requests give every worker ~3 turns, so
# a miss means a routing or resolution failure rather than an unlucky draw.
_PROBE_REQUESTS = 12

# Kept low enough that a whole probe round (12 x 5 s worst case) fits inside the
# 90 s poll budget below. A fake worker answers instantly; a request that needs
# more than 5 s is already a failure, and letting a round outlast its own poll
# would make that timeout non-binding.
_CHAT_TIMEOUT = 5.0
_CONVERGE_TIMEOUT = 90

_H2C_WORKER_MANIFEST = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {H2C_DEPLOYMENT}
  namespace: {NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sglang-h2c
  template:
    metadata:
      labels:
        app: sglang-h2c
    spec:
      containers:
        - name: worker
          image: sgl-router-fake-worker:e2e
          imagePullPolicy: Never
          env:
            - name: FAKE_WORKER_HTTP2
              value: "1"
            - name: MODEL_ID
              value: "tiny"
          ports:
            - containerPort: 30000
          readinessProbe:
            httpGet:
              path: /health
              port: 30000
            initialDelaySeconds: 2
            periodSeconds: 3
---
apiVersion: v1
kind: Service
metadata:
  name: {H2C_DEPLOYMENT}
  namespace: {NAMESPACE}
  # The router watches ENDPOINTSLICES whose labels match `--selector
  # app=sglang`, and Kubernetes mirrors a Service's labels onto the slices it
  # manages -- so this label, not the pods', is what puts these workers in the
  # router's view.
  labels:
    app: sglang
spec:
  # Pods are labelled `app: sglang-h2c`, deliberately NOT `app: sglang`: the
  # fake-worker Deployment's selector is a bare `app=sglang`, so sharing that
  # label would put these pods inside another controller's selector and into
  # the HTTP/1.1 Service as well.
  selector:
    app: sglang-h2c
  ports:
    - port: 30000
      targetPort: 30000
"""


@pytest.fixture(scope="module")
def h2c_worker(k8s_cluster):
    """Add one Granian/h2c worker, behind its own Service, to the router's view.

    Its own Service rather than the existing one: the router selects
    EndpointSlices, so a second Service labelled `app: sglang` is watched just
    the same, while its pods stay out of the `fake-worker` Deployment's bare
    `app=sglang` selector. Torn down afterwards so the suite's other modules
    see the three-worker fleet they expect.
    """
    _apply_from_stdin(_H2C_WORKER_MANIFEST)
    try:
        _wait_for_deployment_ready(H2C_DEPLOYMENT)
        yield
    finally:
        for kind in ("deployment", "service"):
            _kubectl(
                "delete",
                kind,
                H2C_DEPLOYMENT,
                "-n",
                NAMESPACE,
                "--ignore-not-found",
                "--wait=true",
                check=False,
            )


def _chat(router_url: str, content: str) -> httpx.Response:
    return httpx.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": "tiny",
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        },
        timeout=_CHAT_TIMEOUT,
    )


def _observed_protocols(router_url: str, *, strict: bool) -> set[str]:
    """Fan out round-robin and collect the HTTP version each worker saw.

    `x_http_version` is reported by the worker itself, not inferred from the
    client side: the test's own connection to the router is a separate hop, so
    only the worker can say what the forward leg used. Distinct content per
    request keeps any content-derived routing from collapsing onto one worker.

    `strict=False` while converging. The router runs `--cb-threshold 1`
    (manifests/router.yaml), so one refused connection to the still-starting h2c
    pod opens its breaker and round-robin hands back a 502 for that turn. That
    is precisely what the poll is meant to wait out — and `conftest._poll_until`
    retries only transport-level errors, so an `AssertionError` raised here
    would escape the retry budget and fail the test on the first blip. Skip
    non-200s while converging; assert on them once converged.
    """
    seen: set[str] = set()
    for i in range(_PROBE_REQUESTS):
        r = _chat(router_url, f"h2c-probe-{i}")
        if r.status_code != 200:
            if strict:
                raise AssertionError(f"request {i} failed {r.status_code}: {r.text}")
            continue
        version = r.json().get("x_http_version")
        # Fatal either way: this is a stale fake-worker image or a router that
        # stopped returning the upstream body verbatim, neither of which a retry
        # fixes, and without it the test cannot tell h2c from HTTP/1.1 at all.
        assert version is not None, (
            "worker did not report `x_http_version` — the fake-worker image is "
            "stale, or the router stopped returning the upstream body verbatim; "
            "either way this test cannot tell h2c from HTTP/1.1"
        )
        seen.add(version)
    logger.info("protocols observed across %d requests: %s", _PROBE_REQUESTS, seen)
    return seen


def test_router_forwards_over_h2c_to_an_http2_worker(router_url, h2c_worker):
    """A worker advertising `enable_http2` is reached over HTTP/2, and the
    HTTP/1.1 workers alongside it keep their own protocol."""
    # The router must first see the new pod's EndpointSlice entry and
    # introspect it; until then every response comes back "1.1".
    _poll_until(
        lambda: "2" in _observed_protocols(router_url, strict=False),
        "router forwards to the h2c worker over HTTP/2",
        timeout=_CONVERGE_TIMEOUT,
        interval=5,
    )

    seen = _observed_protocols(router_url, strict=True)
    assert "2" in seen, f"expected an HTTP/2 forward, saw {seen}"
    assert "1.1" in seen, (
        f"expected the three uvicorn workers to stay on HTTP/1.1, saw {seen}; "
        "a fleet-wide protocol would have taken them with it"
    )
