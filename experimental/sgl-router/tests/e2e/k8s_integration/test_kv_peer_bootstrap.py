"""Cache-aware peer bootstrap in a real cluster: do new replicas match the old?

Covers the two questions this feature exists to answer:

  1. A replica that joins a warm fleet ends up with the SAME cache-aware view as
     the replicas it bootstrapped from.
  2. After bootstrapping, it keeps up with new engine events — the snapshot is
     spliced under the live stream, not substituted for it.

Plus the rolling-update hazard: new replicas must not bootstrap from each other
and inherit an empty tree.

Anti-flake design
-----------------
Three specific decisions, because the naive version of this test is flaky:

  * **Events are driven, never timed.** The fake worker publishes only when the
    test POSTs ``/control/store``, so there is never an event in flight that the
    test did not ask for. A worker emitting on a timer would make every view
    comparison a race.

  * **Compare only after a proven quiesce.** After injecting, the test polls
    until every replica's reported cursor equals the worker's ``last_seq``. Only
    then are views compared. Comparing on a fixed sleep is the single biggest
    source of flakiness here, because the tree is eventually consistent by
    design.

  * **Never assert on a transient mid-rollout state.** Catching "2 new pods
     alongside 3 old pods" by racing a rollout is inherently timing-dependent.
     Instead the scale-up case is asserted directly (deterministic), and the
     rollout case is asserted after ``rollout status`` reports completion. If a
     new replica had bootstrapped from a cold sibling, its final view would be
     empty or short — which these comparisons catch either way.

Views are compared **canonically**: snapshot node order is unspecified (it
follows per-shard hash-map iteration), so each view is reduced to a set of
``(root-to-node hash path, sorted carrier list)`` before comparing.
"""

from __future__ import annotations

import itertools
import json
import logging

import httpx
import pytest
from conftest import (  # type: ignore[import-not-found]
    NAMESPACE,
    _apply_from_stdin,
    _cleanup_port_forward,
    _kubectl,
    _poll_until,
    _port_forward_start,
    _port_forward_target_start,
    _wait_for_deployment_ready,
)

logger = logging.getLogger(__name__)

ROUTER_DEPLOY = "sgl-router-kv"
WORKER_DEPLOY = "fake-kv-worker"

# Distinct chains so a partial view is obvious in the diff, and enough of them
# to span many tree shards (shard = f(root hash)).
WARM_CHAINS = [[r * 4096 + 11, r * 4096 + 12, r * 4096 + 13] for r in range(24)]
POST_BOOTSTRAP_CHAINS = [[900_000 + r, 900_100 + r] for r in range(8)]
# Disjoint from every asserted chain, so the connectivity probe cannot affect a
# view comparison.
PROBE_CHAIN_BASE = 7_000_000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _ready_router_pods() -> list[str]:
    """Pods whose Ready condition is true — i.e. the ones in the EndpointSlice.

    This is the same set peer discovery offers as bootstrap candidates, so
    filtering here keeps the test's notion of "the fleet" aligned with the
    router's.
    """
    out = _kubectl(
        "get", "pods", "-n", NAMESPACE, "-l", f"app={ROUTER_DEPLOY}", "-o", "json"
    )
    ready = []
    for item in json.loads(out.stdout).get("items", []):
        meta = item.get("metadata", {})
        # Terminating pods keep reporting ready through graceful shutdown, so
        # `kubectl rollout status` can return while old pods still match.
        # Counting them made "did the rollout replace every pod?" flaky, and
        # could port-forward into a dying pod.
        if meta.get("deletionTimestamp"):
            continue
        statuses = item.get("status", {}).get("containerStatuses") or []
        if statuses and all(c.get("ready") for c in statuses):
            ready.append(meta["name"])
    return ready


def _await_ready_router_pods(expected: int, timeout: int = 300) -> list[str]:
    """Poll until exactly `expected` non-Terminating ready pods exist.

    Asserting the count once races graceful shutdown; polling converges.
    """
    pods: list[str] = []

    def settled() -> bool:
        nonlocal pods
        pods = _ready_router_pods()
        return len(pods) == expected

    _poll_until(
        settled,
        f"exactly {expected} ready {ROUTER_DEPLOY} pods",
        timeout=timeout,
        interval=3,
    )
    return pods


class _PodClient:
    """Port-forward to one specific pod.

    Per-pod rather than through the Service on purpose: a Service would
    round-robin across replicas and make "compare replica A to replica B"
    meaningless.
    """

    def __init__(self, pod: str, local_port: int, remote_port: int = 8090) -> None:
        self.pod = pod
        self._pf = _port_forward_target_start(
            NAMESPACE, f"pod/{pod}", local_port, remote_port
        )
        self.base = f"http://127.0.0.1:{local_port}"

    def close(self) -> None:
        _cleanup_port_forward(self.pod, self._pf)

    def snapshot(self) -> dict:
        r = httpx.get(f"{self.base}/internal/kv_snapshot", timeout=30.0)
        r.raise_for_status()
        return r.json()

    def metrics(self) -> str:
        r = httpx.get(f"{self.base}/metrics", timeout=10.0)
        r.raise_for_status()
        return r.text


def _canonical_view(snap: dict) -> set[tuple[tuple[int, ...], tuple[str, ...]]]:
    """Reduce a snapshot to an order-independent set of (path, carriers).

    Node records are parent-linked by index, so a root-to-node path is a walk up
    the parent chain. Carrier-less nodes are dropped: they carry no routing
    meaning, and whether one exists depends on eviction/pruning timing that both
    replicas need not agree on.
    """
    workers = [(w["url"], w["dp_rank"]) for w in snap["workers"]]
    nodes = snap["nodes"]
    paths: list[tuple[int, ...]] = []
    view: set[tuple[tuple[int, ...], tuple[str, ...]]] = set()
    for rec in nodes:
        parent = rec["parent"]
        path = (
            paths[parent] + (rec["block_hash"],)
            if parent is not None
            else (rec["block_hash"],)
        )
        paths.append(path)
        carriers = tuple(
            sorted(f"{workers[i][0]}#{workers[i][1]}" for i in rec["workers"])
        )
        if carriers:
            view.add((path, carriers))
    return view


def _store(worker_base: str, chains: list[list[int]]) -> int:
    r = httpx.post(
        f"{worker_base}/control/store",
        json={"chains": chains, "dp_rank": 0},
        timeout=30.0,
    )
    r.raise_for_status()
    return int(r.json()["last_seq"])


def _replica_cursor(snap: dict) -> int:
    """Highest cursor the replica reports across all ranks, or -1 if none."""
    return max((seq for _, seq in snap.get("cursors", [])), default=-1)


def _await_quiesce(
    clients: list[_PodClient], target_seq: int, timeout: int = 120
) -> None:
    """Block until every replica has applied through ``target_seq``.

    This is the assertion-enabling step: without it a view comparison can race a
    delta that one replica has applied and another has not.
    """

    def all_caught_up() -> bool:
        cursors = {c.pod: _replica_cursor(c.snapshot()) for c in clients}
        behind = {p: s for p, s in cursors.items() if s < target_seq}
        if behind:
            logger.info(
                "waiting for %s to reach seq %d: %s", len(behind), target_seq, behind
            )
        return not behind

    _poll_until(
        all_caught_up,
        f"all {len(clients)} replicas applied through seq {target_seq}",
        timeout=timeout,
        interval=2,
    )


def _await_live_stream(clients: list[_PodClient], worker_base: str) -> int:
    """Prove every replica's ZMQ SUB is actually delivering, not just connected.

    THE flake this test would otherwise have. A bootstrapped replica's cursor is
    seeded entirely from the snapshot, and `/readyz` only requires
    registry-non-empty + tracker-settled — so nothing in "pod is Ready and
    quiesced" implies its SUB socket finished ZMQ's connect handshake. PUB
    silently discards messages for a not-yet-connected subscriber, so the next
    injection could be invisible to a replica and the following quiesce would
    simply time out.

    Publishing a throwaway probe and requiring every replica's cursor to ADVANCE
    converts that into a deterministic gate.
    """
    before = {c.pod: _replica_cursor(c.snapshot()) for c in clients}
    target = _store(worker_base, [[PROBE_CHAIN_BASE, PROBE_CHAIN_BASE + 1]])

    def all_advanced() -> bool:
        lagging = {}
        for c in clients:
            now = _replica_cursor(c.snapshot())
            if now <= before[c.pod]:
                lagging[c.pod] = (before[c.pod], now)
        if lagging:
            logger.info("waiting for live stream on %s: %s", len(lagging), lagging)
        return not lagging

    _poll_until(
        all_advanced,
        f"all {len(clients)} replicas receiving live KV events",
        timeout=120,
        interval=2,
    )
    return target


def _assert_views_agree(clients: list[_PodClient]) -> set:
    views = {c.pod: _canonical_view(c.snapshot()) for c in clients}
    reference_pod, reference = next(iter(views.items()))
    assert reference, f"{reference_pod} has an empty view; nothing was learned"
    for pod, view in views.items():
        missing = reference - view
        extra = view - reference
        assert not missing and not extra, (
            f"{pod} view differs from {reference_pod}: "
            f"{len(missing)} missing, {len(extra)} extra. "
            f"sample missing={sorted(missing)[:3]} sample extra={sorted(extra)[:3]}"
        )
    return reference


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kv_fixture(k8s_cluster):
    """Deploy the KV worker + 3-replica router, and tear both down after."""
    manifest = (
        __import__("pathlib").Path(__file__).parent / "manifests" / "kv-bootstrap.yaml"
    ).read_text()
    _apply_from_stdin(manifest)
    _wait_for_deployment_ready(WORKER_DEPLOY, timeout=180)
    _wait_for_deployment_ready(ROUTER_DEPLOY, timeout=300)
    try:
        yield
    finally:
        for kind_name in (
            f"deployment/{ROUTER_DEPLOY}",
            f"service/{ROUTER_DEPLOY}",
            f"deployment/{WORKER_DEPLOY}",
            f"service/{WORKER_DEPLOY}",
        ):
            _kubectl(
                "delete", kind_name, "-n", NAMESPACE, "--ignore-not-found", check=False
            )


@pytest.fixture(scope="module")
def worker_url(kv_fixture):
    pf = _port_forward_start(NAMESPACE, "fake-kv-worker", 8100, 8000)
    try:
        yield "http://127.0.0.1:8100"
    finally:
        _cleanup_port_forward("fake-kv-worker", pf)


# Ports are never reused within a session. A reused port can be inherited by a
# surviving port-forward from an earlier batch, and `_wait_for_port` only checks
# that *something* accepts a connection — so a stale forward would silently make
# two _PodClients read the same pod, turning "compare A to B" into a false pass.
_next_local_port = itertools.count(8200)


def _clients_for(pods: list[str]) -> list[_PodClient]:
    clients: list[_PodClient] = []
    try:
        for pod in pods:
            clients.append(_PodClient(pod, next(_next_local_port)))
    except Exception:
        # Without this, a failure partway through leaks kubectl port-forwards
        # that hold their ports for the rest of the session.
        for c in clients:
            c.close()
        raise
    return clients


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_new_replicas_match_warm_fleet_and_track_new_events(worker_url):
    """Scale-up case, asserted deterministically (no rollout timing involved).

    A replica added to a warm fleet takes the same code path as a rolling
    update's surge pod: it discovers ready siblings, pulls a snapshot, and
    splices it under its own live stream.
    """
    # --- warm the original fleet -------------------------------------------
    # Explicit, so the test does not depend on file ordering leaving 3 behind.
    _kubectl("scale", f"deployment/{ROUTER_DEPLOY}", "--replicas=3", "-n", NAMESPACE)
    _wait_for_deployment_ready(ROUTER_DEPLOY, timeout=300)
    warm_pods = _await_ready_router_pods(3)
    warm_clients = _clients_for(warm_pods)
    try:
        target = _store(worker_url, WARM_CHAINS)
        _await_quiesce(warm_clients, target)
        warm_view = _assert_views_agree(warm_clients)
        logger.info("warm fleet agrees on %d chains", len(warm_view))
    finally:
        for c in warm_clients:
            c.close()

    # --- add two replicas ---------------------------------------------------
    _kubectl("scale", f"deployment/{ROUTER_DEPLOY}", "--replicas=5", "-n", NAMESPACE)
    _wait_for_deployment_ready(ROUTER_DEPLOY, timeout=300)
    all_pods = _await_ready_router_pods(5)
    new_pods = [p for p in all_pods if p not in warm_pods]
    assert len(new_pods) == 2, f"expected 2 new replicas, got {new_pods}"

    clients = _clients_for(all_pods)
    try:
        # Diagnose first: a failed bootstrap should read as "empty tree", not as
        # an opaque timeout inside the quiesce below.
        for pod in new_pods:
            client = next(c for c in clients if c.pod == pod)
            snap = client.snapshot()
            assert snap["nodes"], f"new replica {pod} bootstrapped an empty tree"
            assert snap[
                "producer_ready"
            ], f"new replica {pod} never became a valid source"

        # The new replicas must already agree with the warm view. No new events
        # were injected, so their cursors are whatever the snapshot seeded.
        _await_quiesce(clients, target)
        after_join = _assert_views_agree(clients)
        assert after_join == warm_view, (
            "view changed when replicas joined: "
            f"{len(warm_view - after_join)} lost, {len(after_join - warm_view)} gained"
        )

        # --- new events after bootstrap must reach everyone ----------------
        # Gate on live delivery first; otherwise a not-yet-connected SUB socket
        # makes the injection below invisible to that replica.
        _await_live_stream(clients, worker_url)
        target2 = _store(worker_url, POST_BOOTSTRAP_CHAINS)
        assert target2 > target
        _await_quiesce(clients, target2)
        final = _assert_views_agree(clients)
        assert final - after_join, "post-bootstrap events did not extend the view"
        for chain in POST_BOOTSTRAP_CHAINS:
            assert any(
                path == tuple(chain) for path, _ in final
            ), f"chain {chain} published after bootstrap is missing from the fleet view"
    finally:
        for c in clients:
            c.close()


@pytest.mark.slow
def test_view_survives_a_full_rolling_update(worker_url):
    """Replace every replica; the fleet view must be preserved end to end.

    Asserted after ``rollout status`` reports completion, so the test never
    depends on catching a transient mix of old and new pods. With
    ``maxSurge=2 / maxUnavailable=0`` the surge pods always have warm siblings to
    copy from, which is the configuration the design assumes.
    """
    pre_pods = _ready_router_pods()
    pre_clients = _clients_for(pre_pods)
    try:
        target = _store(worker_url, WARM_CHAINS)
        _await_quiesce(pre_clients, target)
        pre_view = _assert_views_agree(pre_clients)
    finally:
        for c in pre_clients:
            c.close()

    _kubectl("rollout", "restart", f"deployment/{ROUTER_DEPLOY}", "-n", NAMESPACE)
    _kubectl(
        "rollout",
        "status",
        f"deployment/{ROUTER_DEPLOY}",
        "-n",
        NAMESPACE,
        "--timeout=600s",
    )

    post_pods = _await_ready_router_pods(len(pre_pods))
    assert not (set(post_pods) & set(pre_pods)), "rollout did not replace every pod"
    post_clients = _clients_for(post_pods)
    try:
        _await_quiesce(post_clients, target)
        post_view = _assert_views_agree(post_clients)
        lost = pre_view - post_view
        assert not lost, (
            f"{len(lost)} chains were lost across the rolling update; "
            f"sample={sorted(lost)[:3]}"
        )

        # And the replaced fleet still tracks the engine.
        _await_live_stream(post_clients, worker_url)
        target2 = _store(worker_url, POST_BOOTSTRAP_CHAINS)
        _await_quiesce(post_clients, target2)
        _assert_views_agree(post_clients)
    finally:
        for c in post_clients:
            c.close()


@pytest.mark.slow
def test_bootstrap_metrics_show_grafted_not_merely_settled(worker_url):
    """Assert the metrics prove a real graft, not just that bootstrap finished.

    The earlier version of this test checked only `bootstrap_settled == 1` and
    "no rank is Pending". Both are satisfied by a replica that gave up and ran
    cold — `settled` latches on deadline expiry, and `Failed` renders as 2 — so a
    total regression to "everyone boots cold" passed. The series that actually
    distinguishes grafted from cold is
    `peer_snapshot_total{outcome="accepted"}`, and the state must be 1
    (Recovered), not merely non-zero.

    Asserts on the peer-fetch counter rather than the per-rank one on purpose:
    the fleet is quiesced here, so a grafted rank has nothing to prove its splice
    against yet and `bootstrap_rank_total{outcome="warm"}` legitimately lags
    until a batch or a probe resolves it. `accepted` is recorded synchronously
    when the fetch is taken, so it is deterministic at this point.

    Scoped to freshly added replicas, since a long-running pod's counters say
    nothing about whether bootstrap works now.
    """

    def series_value(body: str, prefix: str) -> float | None:
        for line in body.splitlines():
            if line.startswith(prefix):
                return float(line.split()[-1])
        return None

    warm_pods = _await_ready_router_pods(len(_ready_router_pods()))
    warm_clients = _clients_for(warm_pods)
    try:
        target = _store(worker_url, WARM_CHAINS)
        _await_quiesce(warm_clients, target)
    finally:
        for c in warm_clients:
            c.close()

    before = len(warm_pods)
    _kubectl(
        "scale",
        f"deployment/{ROUTER_DEPLOY}",
        f"--replicas={before + 1}",
        "-n",
        NAMESPACE,
    )
    _wait_for_deployment_ready(ROUTER_DEPLOY, timeout=300)
    all_pods = _await_ready_router_pods(before + 1)
    new_pods = [p for p in all_pods if p not in warm_pods]
    assert len(new_pods) == 1, f"expected exactly one new replica, got {new_pods}"

    clients = _clients_for(new_pods)
    try:
        for c in clients:
            body = c.metrics()
            accepted = series_value(
                body, 'sgl_router_kv_peer_snapshot_total{outcome="accepted"}'
            )
            assert accepted is not None and accepted >= 1, (
                f"{c.pod} never took a peer snapshot "
                f"(outcome=accepted absent or zero) — bootstrap is a no-op"
            )
            nodes = series_value(body, "sgl_router_kv_tree_nodes ")
            assert nodes is not None and nodes > 0, f"{c.pod} has an empty tree"
            assert (
                series_value(body, "sgl_router_kv_bootstrap_settled ") == 1
            ), f"{c.pod} did not settle"
            states = [
                line
                for line in body.splitlines()
                if line.startswith("sgl_router_kv_bootstrap_state{")
            ]
            assert states, f"{c.pod} exposes no per-rank bootstrap state"
            for line in states:
                assert line.endswith(
                    " 1"
                ), f"{c.pod} rank did not reach Recovered (1): {line}"
    finally:
        for c in clients:
            c.close()
