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
Each of these exists because the naive version of this test is flaky:

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

import contextlib
import itertools
import logging
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from conftest import (  # type: ignore[import-not-found]
    NAMESPACE,
    _apply_from_stdin,
    _cleanup_port_forward,
    _is_live,
    _kubectl,
    _pods,
    _poll_until,
    _port_forward_start,
    _wait_for_deployment_ready,
)

logger = logging.getLogger(__name__)

ROUTER_DEPLOY = "sgl-router-kv"
WORKER_DEPLOY = "fake-kv-worker"
MANIFEST = Path(__file__).parent / "manifests" / "kv-bootstrap.yaml"

# Distinct chains so a partial view is obvious in the diff, and enough of them
# to span many tree shards (shard = f(root hash)).
WARM_CHAINS = [[r * 4096 + 11, r * 4096 + 12, r * 4096 + 13] for r in range(24)]
POST_BOOTSTRAP_CHAINS = [[900_000 + r, 900_100 + r] for r in range(8)]
# Published by `_await_live_stream`. Disjoint from every asserted chain, and
# re-publishing it is idempotent, so every replica holds the same probe nodes
# and view equality is unaffected. It does land in the views, so "what did
# these events add" must be asked per chain, never as a whole-view difference.
PROBE_CHAIN = [7_000_000, 7_000_001]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _ready_router_pods() -> list[str]:
    """Pods whose containers are all ready — i.e. the ones in the EndpointSlice.

    This is the same set peer discovery offers as bootstrap candidates, so
    filtering here keeps the test's notion of "the fleet" aligned with the
    router's.
    """
    ready = []
    for pod in _pods(f"app={ROUTER_DEPLOY}"):
        # Terminating pods keep reporting ready through graceful shutdown, and
        # `kubectl rollout status` can return while old pods still match.
        # Excluded so "did the rollout replace every pod?" is well-defined and
        # no client port-forwards into a dying pod.
        if not _is_live(pod):
            continue
        statuses = pod.get("status", {}).get("containerStatuses") or []
        if statuses and all(c.get("ready") for c in statuses):
            ready.append(pod["metadata"]["name"])
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


def _scale_routers(replicas: int) -> None:
    _kubectl(
        "scale",
        f"deployment/{ROUTER_DEPLOY}",
        f"--replicas={replicas}",
        "-n",
        NAMESPACE,
    )


def _settled_fleet() -> list[str]:
    """The Deployment's full replica set, once it has settled.

    Sized from `.spec.replicas`, never from a read of whichever pods happen to
    be ready: mid-scale or mid-rollout that count is transient, and a test that
    sizes itself from it waits for a fleet that never forms or scales from the
    wrong base.
    """
    _wait_for_deployment_ready(ROUTER_DEPLOY, timeout=300)
    out = _kubectl(
        "get",
        f"deployment/{ROUTER_DEPLOY}",
        "-n",
        NAMESPACE,
        "-o",
        "jsonpath={.spec.replicas}",
    )
    return _await_ready_router_pods(int(out.stdout))


def _add_replicas(warm_pods: list[str], added: int) -> list[str]:
    """Scale the settled `warm_pods` fleet up by `added`; return the new pods."""
    _scale_routers(len(warm_pods) + added)
    all_pods = _settled_fleet()
    new_pods = [p for p in all_pods if p not in warm_pods]
    assert len(new_pods) == added, f"expected {added} new replicas, got {new_pods}"
    return new_pods


class _PodClient:
    """Port-forward to one specific pod.

    Per-pod rather than through the Service on purpose: a Service would
    round-robin across replicas and make "compare replica A to replica B"
    meaningless.
    """

    def __init__(self, pod: str, local_port: int, remote_port: int = 8090) -> None:
        self.pod = pod
        self._pf = _port_forward_start(
            NAMESPACE, pod, local_port, remote_port, resource="pod"
        )
        self.base = f"http://127.0.0.1:{local_port}"

    def close(self) -> None:
        _cleanup_port_forward(self.pod, self._pf)

    def snapshot(self) -> dict:
        """A full export sampled no earlier than this request's arrival.

        `max_age_ms=0` matters because cursors are polled live (see
        `cursor`): an export cached before this request can predate the seq a
        poll just proved applied, and would then be compared as if it
        reflected it. httpx advertises gzip and inflates the response, so the
        encoding the route picks is invisible here.
        """
        r = httpx.get(
            f"{self.base}/internal/kv_snapshot",
            params={"max_age_ms": 0},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()

    def _cursor_table(self) -> dict:
        """The live cursor table, with no tree.

        `cursors_only` reads the cursor map directly, so polling it neither
        walks the tree nor touches the export cache bootstrapping peers read.
        """
        r = httpx.get(
            f"{self.base}/internal/kv_snapshot",
            params={"cursors_only": "true"},
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()

    def cursor(self) -> int:
        """Highest live cursor across ranks, or -1 if none."""
        cursors = self._cursor_table().get("cursors", [])
        return max((seq for _, seq in cursors), default=-1)

    def producer_ready(self) -> bool:
        """Whether this replica is a valid bootstrap source: settled, with its
        hash config established and a non-empty tree."""
        return bool(self._cursor_table()["producer_ready"])

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
    workers = [f"{w['url']}#{w['dp_rank']}" for w in snap["workers"]]
    paths: list[tuple[int, ...]] = []
    view: set[tuple[tuple[int, ...], tuple[str, ...]]] = set()
    for rec in snap["nodes"]:
        parent = rec["parent"]
        path = (
            paths[parent] + (rec["block_hash"],)
            if parent is not None
            else (rec["block_hash"],)
        )
        paths.append(path)
        carriers = tuple(sorted(workers[i] for i in rec["workers"]))
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


def _await_quiesce(
    clients: list[_PodClient], target_seq: int, timeout: int = 120
) -> None:
    """Block until every replica has applied through ``target_seq``.

    This is the assertion-enabling step: without it a view comparison can race a
    delta that one replica has applied and another has not.
    """

    def all_caught_up() -> bool:
        cursors = {c.pod: c.cursor() for c in clients}
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


def _await_live_stream(clients: list[_PodClient], worker_base: str) -> None:
    """Prove every replica's ZMQ SUB is actually delivering, not just connected.

    Nothing in "pod is Ready and quiesced" implies a replica's SUB socket
    finished ZMQ's connect handshake: `/readyz` does not wait on it, and a
    bootstrapped replica's cursor is seeded entirely from the snapshot. PUB
    silently discards messages for a not-yet-connected subscriber, so an
    injection could be partly invisible to a replica — the quiesce then passes
    on the last batch while the view is short, or times out outright.

    Requiring every replica's cursor to ADVANCE past a throwaway probe converts
    that into a gate. The probe is re-published each round a replica still
    lags, because a single probe dropped that way is gone for good and the gate
    would time out instead of waiting. A grafted rank that misses one probe and
    sees the next reads it as a gap: it drops the graft and its cursor, holds
    its live batches, and re-sweeps for an export newer than the gap. Its
    cursor reappears once that retry grafts, which this gate and the following
    quiesce wait out.
    """
    before = {c.pod: c.cursor() for c in clients}
    _store(worker_base, [PROBE_CHAIN])

    def all_advanced() -> bool:
        lagging = {}
        for c in clients:
            now = c.cursor()
            if now <= before[c.pod]:
                lagging[c.pod] = (before[c.pod], now)
        if lagging:
            logger.info("waiting for live stream on %s: %s", len(lagging), lagging)
            _store(worker_base, [PROBE_CHAIN])
        return not lagging

    _poll_until(
        all_advanced,
        f"all {len(clients)} replicas receiving live KV events",
        timeout=120,
        interval=2,
    )


def _publish_and_quiesce(
    clients: list[_PodClient], worker_base: str, chains: list[list[int]]
) -> int:
    """Publish `chains` once every replica is provably subscribed, and wait
    until all of them have applied it. Returns the worker's last seq.

    The live-stream gate comes first because a replica that misses part of
    the batch still reaches the final seq, so the quiesce alone cannot catch
    it.
    """
    _await_live_stream(clients, worker_base)
    target = _store(worker_base, chains)
    _await_quiesce(clients, target)
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
    _apply_from_stdin(MANIFEST.read_text())
    # The waits sit inside the try: a rollout that never becomes ready must
    # still be torn down, or it runs alongside every later module.
    try:
        _wait_for_deployment_ready(WORKER_DEPLOY, timeout=180)
        _wait_for_deployment_ready(ROUTER_DEPLOY, timeout=300)
        yield
    finally:
        _kubectl("delete", "-f", str(MANIFEST), "--ignore-not-found", check=False)


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


@contextlib.contextmanager
def _pod_clients(pods: list[str]) -> Iterator[list[_PodClient]]:
    """One `_PodClient` per pod, all closed on exit — including when opening a
    later one fails, which would otherwise leak kubectl port-forwards that hold
    their ports for the rest of the session."""
    clients: list[_PodClient] = []
    try:
        for pod in pods:
            clients.append(_PodClient(pod, next(_next_local_port)))
        yield clients
    finally:
        for c in clients:
            c.close()


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
    _scale_routers(3)
    warm_pods = _settled_fleet()
    with _pod_clients(warm_pods) as warm_clients:
        target = _publish_and_quiesce(warm_clients, worker_url, WARM_CHAINS)
        warm_view = _assert_views_agree(warm_clients)
        logger.info("warm fleet agrees on %d chains", len(warm_view))

    # --- add two replicas ---------------------------------------------------
    new_pods = _add_replicas(warm_pods, 2)

    with _pod_clients(warm_pods + new_pods) as clients:
        # Diagnose first: a failed bootstrap should read as "empty tree", not as
        # an opaque timeout inside the quiesce below. Polled, not asserted once:
        # Ready does not imply the graft has landed unless `/readyz` gates on
        # bootstrap, and this must not depend on that.
        for client in clients:
            if client.pod not in new_pods:
                continue
            _poll_until(
                client.producer_ready,
                f"new replica {client.pod} grafted a non-empty tree and became "
                "a valid source",
                timeout=60,
                interval=2,
            )

        # The new replicas must already agree with the warm view. No new events
        # were injected, so their cursors are whatever the snapshot seeded.
        _await_quiesce(clients, target)
        after_join = _assert_views_agree(clients)
        assert after_join == warm_view, (
            "view changed when replicas joined: "
            f"{len(warm_view - after_join)} lost, {len(after_join - warm_view)} gained"
        )

        # --- new events after bootstrap must reach everyone ----------------
        target2 = _publish_and_quiesce(clients, worker_url, POST_BOOTSTRAP_CHAINS)
        assert target2 > target
        final = _assert_views_agree(clients)
        # Per chain: the probe alone already makes `final` differ from
        # `after_join`, so a whole-view difference proves nothing here.
        for chain in POST_BOOTSTRAP_CHAINS:
            assert any(path == tuple(chain) for path, _ in final), (
                f"chain {chain} published after bootstrap is missing from the fleet view"
            )


@pytest.mark.slow
def test_view_survives_a_full_rolling_update(worker_url):
    """Replace every replica; the fleet view must be preserved end to end.

    Asserted after ``rollout status`` reports completion, so the test never
    depends on catching a transient mix of old and new pods. With
    ``maxSurge=2 / maxUnavailable=0`` the surge pods always have warm siblings to
    copy from, which is the configuration the design assumes.
    """
    pre_pods = _settled_fleet()
    with _pod_clients(pre_pods) as pre_clients:
        target = _publish_and_quiesce(pre_clients, worker_url, WARM_CHAINS)
        pre_view = _assert_views_agree(pre_clients)

    _kubectl("rollout", "restart", f"deployment/{ROUTER_DEPLOY}", "-n", NAMESPACE)
    _wait_for_deployment_ready(ROUTER_DEPLOY, timeout=600)

    post_pods = _await_ready_router_pods(len(pre_pods))
    assert not (set(post_pods) & set(pre_pods)), "rollout did not replace every pod"
    with _pod_clients(post_pods) as post_clients:
        _await_quiesce(post_clients, target)
        post_view = _assert_views_agree(post_clients)
        lost = pre_view - post_view
        assert not lost, (
            f"{len(lost)} chains were lost across the rolling update; "
            f"sample={sorted(lost)[:3]}"
        )

        # And the replaced fleet still tracks the engine.
        _publish_and_quiesce(post_clients, worker_url, POST_BOOTSTRAP_CHAINS)
        _assert_views_agree(post_clients)


@pytest.mark.slow
def test_bootstrap_metrics_show_grafted_not_merely_settled(worker_url):
    """Assert the metrics prove a real graft, not just that bootstrap finished.

    `bootstrap_settled == 1` and "no rank is Pending" are both satisfied by a
    replica that gave up and ran cold — `settled` latches on deadline expiry,
    and `Failed` renders as 2 — so they cannot tell a total regression to
    "everyone boots cold" from a working bootstrap. The series that does is
    `peer_snapshot_total{outcome="accepted"}`, and the state must be 1
    (Recovered), not merely non-zero.

    Asserts on the peer-fetch counter rather than the per-rank one on purpose:
    the fleet is quiesced here, so a grafted rank has nothing to prove its splice
    against yet and `bootstrap_rank_total{outcome="warm"}` legitimately lags
    until a batch or a probe resolves it. `accepted` is recorded synchronously
    when the fetch is taken, so it is deterministic at this point.

    Scoped to freshly added replicas, since a long-running pod's counters say
    nothing about whether bootstrap works now.

    Depends on the readiness-gate change ("gate readiness on peer bootstrap,
    and make it observable"), which adds every series asserted here. Without
    it, `/metrics` carries only `sgl_router_kv_bootstrap_peers` and
    `sgl_router_kv_bootstrap_peers_synced`, and this test fails on the first
    assert.
    """

    def series_value(body: str, prefix: str) -> float | None:
        for line in body.splitlines():
            if line.startswith(prefix):
                return float(line.split()[-1])
        return None

    warm_pods = _settled_fleet()
    with _pod_clients(warm_pods) as warm_clients:
        _publish_and_quiesce(warm_clients, worker_url, WARM_CHAINS)

    new_pods = _add_replicas(warm_pods, 1)

    with _pod_clients(new_pods) as clients:
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
            assert series_value(body, "sgl_router_kv_bootstrap_settled ") == 1, (
                f"{c.pod} did not settle"
            )
            states = [
                line
                for line in body.splitlines()
                if line.startswith("sgl_router_kv_bootstrap_state{")
            ]
            assert states, f"{c.pod} exposes no per-rank bootstrap state"
            for line in states:
                assert line.endswith(" 1"), (
                    f"{c.pod} rank did not reach Recovered (1): {line}"
                )
