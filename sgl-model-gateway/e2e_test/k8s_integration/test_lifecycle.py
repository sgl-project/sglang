"""Worker lifecycle integration tests.

Covers three scenarios that the existing reconciliation/PD tests don't:

1. Gateway pod restart with persistent workers — verifies the K8s watcher
   re-discovers existing pods after the gateway restarts, with no duplicate
   registrations.
2. Pod IP change (same pod name, new IP) — verifies the gateway's worker
   registry tracks the new IP after a pod is force-deleted and recreated,
   not the stale one.
3. Graceful drain — verifies the gateway deregisters a worker as soon as
   K8s sets `metadata.deletionTimestamp` (handled by handle_pod_deletion in
   sgl-model-gateway/src/service_discovery.rs:533), instead of waiting for
   the pod to fully terminate. This is what keeps the registry fresh during
   long terminationGracePeriodSeconds windows / preStop hooks.

Run with:
    cd e2e_test/k8s_integration
    pytest test_lifecycle.py -v -s
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

import pytest
from conftest import (  # pytest's rootdir adds the test dir to sys.path
    KUBECTL_CONTEXT,
    NAMESPACE,
    RECONCILIATION_WAIT_SECS,
    _cleanup_port_forward,
    _get_worker_count,
    _get_workers,
    _kubectl,
    _kubectl_json,
    _poll_until,
    _port_forward_start,
    _wait_for_deployment_ready,
    _wait_for_pod_ready,
)

logger = logging.getLogger(__name__)

MANIFESTS_DIR = Path(__file__).parent / "manifests"

RESTART_GATEWAY_HTTP_PORT = 30005


def _deploy_worker_pod(
    name: str,
    extra_labels: dict[str, str] | None = None,
    grace_period_secs: int | None = None,
    prestop_sleep_secs: int | None = None,
):
    """Deploy a fake-worker pod with optional extra labels and grace period.

    `prestop_sleep_secs` adds an exec preStop hook (`sleep N`) so the pod
    stays in the Terminating state for at least N seconds after a graceful
    delete — long enough for the test to observe the watcher's
    deletionTimestamp event firing before SIGKILL.
    """
    labels = {"app": "fake-worker"}
    if extra_labels:
        labels.update(extra_labels)

    container: dict = {
        "name": "worker",
        "image": "python:3.12-slim",
        "imagePullPolicy": "IfNotPresent",
        "command": ["python3", "/app/fake_worker.py"],
        "ports": [{"containerPort": 8000}],
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "initialDelaySeconds": 2,
            "periodSeconds": 3,
        },
        "volumeMounts": [{"name": "app", "mountPath": "/app"}],
    }
    if prestop_sleep_secs is not None:
        container["lifecycle"] = {
            "preStop": {"exec": {"command": ["sleep", str(prestop_sleep_secs)]}}
        }

    spec: dict = {
        "containers": [container],
        "volumes": [{"name": "app", "configMap": {"name": "fake-worker-script"}}],
    }
    if grace_period_secs is not None:
        spec["terminationGracePeriodSeconds"] = grace_period_secs

    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": labels,
        },
        "spec": spec,
    }
    proc = subprocess.run(
        ["kubectl", "--context", KUBECTL_CONTEXT, "apply", "-f", "-"],
        input=json.dumps(pod_manifest),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # Surface kubectl's actual error (webhook denial, schema, etc.)
        # instead of an opaque "non-zero exit status 1" CalledProcessError.
        raise RuntimeError(
            f"Failed to deploy pod {name} (rc={proc.returncode}): "
            f"stderr={proc.stderr.strip()!r}"
        )
    logger.info(
        "Deployed pod %s (labels=%s, grace=%s, prestop=%s)",
        name,
        labels,
        grace_period_secs,
        prestop_sleep_secs,
    )


def _safe_force_delete(name: str):
    """Force-delete a pod, swallowing errors (used in cleanup)."""
    try:
        _kubectl(
            "delete",
            "pod",
            name,
            "-n",
            NAMESPACE,
            "--ignore-not-found",
            "--force",
            "--grace-period=0",
        )
    except Exception as e:
        logger.warning("Cleanup failed for pod %s: %s", name, e)


def _wait_for_pod_gone(name: str, timeout: int = 60):
    """Wait until a pod no longer exists.

    Uses `kubectl get -o name --ignore-not-found`: empty stdout means the pod
    is gone (no need to substring-match "NotFound" against stderr, which is
    locale- and version-fragile). Any non-zero rc is a real cluster error
    (apiserver unreachable, RBAC drift) and propagates immediately via
    `_poll_until` (it only retries transient network errors, not RuntimeError).
    """

    def _gone() -> bool:
        result = _kubectl(
            "get",
            "pod",
            name,
            "-n",
            NAMESPACE,
            "-o",
            "name",
            "--ignore-not-found",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"kubectl get pod {name} failed unexpectedly "
                f"(rc={result.returncode}): {result.stderr.strip()}"
            )
        return not result.stdout.strip()

    _poll_until(_gone, f"pod {name} gone", timeout=timeout, interval=2)


def _get_pod_ip(name: str) -> str:
    """Return the current podIP for a pod (must be running)."""
    pod = _kubectl_json("get", "pod", name, "-n", NAMESPACE)
    ip = pod.get("status", {}).get("podIP")
    if not ip:
        raise RuntimeError(f"Pod {name} has no podIP yet: {pod.get('status')}")
    return ip


@pytest.fixture
def restart_gateway(deploy_base):
    """Deploy a dedicated gateway for the restart test on its own ports.

    Function-scoped so each invocation starts from a clean Deployment — that
    keeps the per-test pod-restart count predictable. The selector is
    `app=fake-worker,lifecycle=restart`, distinct from the default gateway,
    so the test owns the worker fleet exclusively.

    Yields a single-element list `[pf]` so the test can swap in a fresh
    port-forward after killing the gateway pod (the original `pf` exits with
    the pod). Fixture teardown cleans up whichever handle is current.
    """
    manifest = MANIFESTS_DIR / "gateway-restart.yaml"
    _kubectl("apply", "-f", str(manifest))
    pf_holder: list[subprocess.Popen | None] = [None]
    try:
        _wait_for_deployment_ready("smg-gateway-restart")
        pf_holder[0] = _port_forward_start(
            NAMESPACE,
            "smg-gateway-restart",
            RESTART_GATEWAY_HTTP_PORT,
            RESTART_GATEWAY_HTTP_PORT,
        )
        yield f"http://127.0.0.1:{RESTART_GATEWAY_HTTP_PORT}", pf_holder
    finally:
        if pf_holder[0] is not None:
            _cleanup_port_forward("restart_gateway", pf_holder[0])
        _kubectl("delete", "-f", str(manifest), "--ignore-not-found", check=False)


class TestGatewayRestart:
    """Killing the gateway pod must not lose worker state — the new pod
    re-lists existing workers via the K8s watcher and registers them once."""

    def test_workers_re_discovered_without_duplicates_after_restart(
        self, restart_gateway
    ):
        gateway_url, pf_holder = restart_gateway
        worker_names = ["restart-worker-a", "restart-worker-b", "restart-worker-c"]

        try:
            for name in worker_names:
                _deploy_worker_pod(name, extra_labels={"lifecycle": "restart"})
            for name in worker_names:
                _wait_for_pod_ready(name)

            _poll_until(
                lambda: _get_worker_count(gateway_url) >= len(worker_names),
                f"all {len(worker_names)} workers discovered (pre-restart)",
                timeout=30,
                interval=3,
            )

            urls_before = sorted(
                w["url"] for w in _get_workers(gateway_url).get("workers", [])
            )
            logger.info("Workers before restart: %s", urls_before)
            assert len(urls_before) == len(worker_names)

            # Kill the gateway pod. The Deployment's ReplicaSet restarts it.
            # Guard against an empty list — `_wait_for_deployment_ready` above
            # gates on the rollout, but a controller race could in principle
            # leave the selector momentarily empty, and an IndexError here
            # would mask the real problem.
            res = _kubectl_json(
                "get",
                "pod",
                "-n",
                NAMESPACE,
                "-l",
                "app=smg-gateway-restart",
            )
            assert res.get(
                "items"
            ), "No pods found for selector app=smg-gateway-restart"
            old_pod = res["items"][0]["metadata"]["name"]
            _kubectl(
                "delete",
                "pod",
                old_pod,
                "-n",
                NAMESPACE,
                "--force",
                "--grace-period=0",
            )

            # Tear down the fixture's port-forward NOW, before the new pod is
            # ready. Otherwise the stale kubectl process can briefly hold port
            # 30005 while attempting to reconnect, and our fresh port-forward
            # below races with it for "address already in use". After cleanup
            # we clear the holder so the fixture teardown doesn't double-free.
            if pf_holder[0] is not None:
                _cleanup_port_forward("restart_gateway_pre_kill", pf_holder[0])
                pf_holder[0] = None

            _wait_for_deployment_ready("smg-gateway-restart")

            # Open a fresh port-forward to the new gateway pod and stash it
            # in the holder so the fixture cleans it up even if the assertion
            # block raises before the explicit teardown.
            pf_holder[0] = _port_forward_start(
                NAMESPACE,
                "smg-gateway-restart",
                RESTART_GATEWAY_HTTP_PORT,
                RESTART_GATEWAY_HTTP_PORT,
            )

            _poll_until(
                lambda: _get_worker_count(gateway_url) >= len(worker_names),
                f"all {len(worker_names)} workers re-discovered (post-restart)",
                timeout=60,
                interval=3,
            )

            urls_after = sorted(
                w["url"] for w in _get_workers(gateway_url).get("workers", [])
            )
            logger.info("Workers after restart: %s", urls_after)

            # No duplicates: each pod should appear exactly once.
            assert len(urls_after) == len(
                set(urls_after)
            ), f"Duplicate worker registrations after gateway restart: {urls_after}"
            # Set equality: the same workers come back, neither dropped
            # nor duplicated.
            assert set(urls_after) == set(urls_before), (
                f"Worker set diverged across gateway restart. "
                f"before={urls_before}, after={urls_after}"
            )

            # Liveness probe of the new watcher: deploy one more worker AFTER
            # the relist has converged. The relist can't pick this up — only
            # an active watch loop can. A regression where the gateway does
            # the initial list correctly but exits the watch loop afterward
            # would silently pass the assertions above; this catches it.
            extra_name = "restart-worker-d"
            try:
                _deploy_worker_pod(extra_name, extra_labels={"lifecycle": "restart"})
                _wait_for_pod_ready(extra_name)
                _poll_until(
                    lambda: _get_worker_count(gateway_url) >= len(worker_names) + 1,
                    f"post-restart watcher picked up {extra_name}",
                    timeout=30,
                    interval=2,
                )
            finally:
                _safe_force_delete(extra_name)
        finally:
            for name in worker_names:
                _safe_force_delete(name)


class TestPodIpChange:
    """When a pod is deleted and recreated with the same name, the gateway's
    worker registry must drop the old URL and pick up the new one. The test
    skips on the rare CNI-IP-reuse case (since both URLs would be identical
    and the assertions can't distinguish), so it does not exercise the
    "old IP recycled by a different pod" scenario — covering that would need
    a static-IP harness or controlled IP exhaustion, neither of which is
    cheap on kind."""

    def test_recreated_pod_yields_current_ip_in_worker_registry(
        self, gateway_port_forward
    ):
        gateway_url, _ = gateway_port_forward
        pod_name = "ip-change-worker"

        try:
            _deploy_worker_pod(pod_name)
            _wait_for_pod_ready(pod_name)

            _poll_until(
                lambda: _get_worker_count(gateway_url) >= 1,
                "initial worker discovered",
                timeout=30,
                interval=3,
            )

            ip_before = _get_pod_ip(pod_name)
            urls_before = {w["url"] for w in _get_workers(gateway_url)["workers"]}
            assert any(
                ip_before in url for url in urls_before
            ), f"Expected initial worker URL containing {ip_before}, got {urls_before}"
            logger.info("Pod IP before: %s, urls: %s", ip_before, urls_before)

            # Force-delete and wait until the registry no longer references
            # this pod's IP. Asserting on "URL contains ip_before" rather than
            # "total count == 0" keeps the test robust if a parallel/earlier
            # test left an unrelated worker in the same gateway's registry.
            _kubectl(
                "delete",
                "pod",
                pod_name,
                "-n",
                NAMESPACE,
                "--force",
                "--grace-period=0",
            )
            _wait_for_pod_gone(pod_name)
            _poll_until(
                lambda: not any(
                    ip_before in w["url"]
                    for w in _get_workers(gateway_url).get("workers", [])
                ),
                f"stale worker for IP {ip_before} removed",
                timeout=RECONCILIATION_WAIT_SECS,
                interval=5,
            )

            # Recreate with same pod name. CNI typically assigns a new IP
            # since the previous one isn't immediately recycled, but the
            # assertion below works either way.
            _deploy_worker_pod(pod_name)
            _wait_for_pod_ready(pod_name)

            ip_after = _get_pod_ip(pod_name)
            # If the CNI happened to recycle the same IP, skip — both the
            # "no URL contains ip_before" wait and the "URL contains ip_after"
            # assertion describe the same string, so the test would silently
            # pass without exercising the IP-change path it's meant to cover.
            if ip_after == ip_before:
                pytest.skip(
                    f"CNI reused pod IP {ip_before}; cannot validate IP-change "
                    f"path without two distinct IPs"
                )

            _poll_until(
                lambda: any(
                    ip_after in w["url"]
                    for w in _get_workers(gateway_url).get("workers", [])
                ),
                f"recreated worker (IP {ip_after}) discovered",
                timeout=30,
                interval=3,
            )

            urls_after = {w["url"] for w in _get_workers(gateway_url)["workers"]}
            logger.info("Pod IP after: %s, urls: %s", ip_after, urls_after)

            # The registry must reflect the current pod's IP, and the old IP
            # must not linger. Tested by URL membership rather than count to
            # tolerate unrelated workers from other tests.
            matching_after = [u for u in urls_after if ip_after in u]
            assert len(matching_after) == 1, (
                f"Expected exactly one worker URL containing current IP "
                f"{ip_after}, got {matching_after} (all urls: {urls_after})"
            )
            assert not any(
                ip_before in u for u in urls_after
            ), f"Stale URL with old IP {ip_before} still in registry: {urls_after}"
        finally:
            _safe_force_delete(pod_name)


class TestGracefulDrain:
    """A graceful `kubectl delete pod` sets metadata.deletionTimestamp; the
    watcher's `applied_objects()` stream emits the resulting MODIFIED event,
    which the if-branch at service_discovery.rs:349 routes to
    `handle_pod_deletion` (defined at service_discovery.rs:533). That removes
    the worker immediately rather than waiting out the grace period or the
    eventual SIGKILL → DELETED event (which `applied_objects()` filters out
    anyway). This test pins that behavior so a future regression doesn't keep
    a Terminating pod in the registry while kubelet is already running its
    preStop hook."""

    def test_deregistration_fires_during_grace_period(self, gateway_port_forward):
        gateway_url, _ = gateway_port_forward
        pod_name = "graceful-drain-worker"
        # Grace period and preStop sleep need to be long enough that the
        # watcher reliably observes the deletion *while the pod is still
        # Terminating*. 60s grace + 40s sleep gives ~40s of overlap before
        # SIGKILL — far more than the watcher's typical event latency
        # (sub-second under normal conditions).
        grace_secs = 60
        prestop_sleep = 40

        try:
            _deploy_worker_pod(
                pod_name,
                grace_period_secs=grace_secs,
                prestop_sleep_secs=prestop_sleep,
            )
            _wait_for_pod_ready(pod_name)
            pod_ip = _get_pod_ip(pod_name)
            _poll_until(
                lambda: any(
                    pod_ip in w["url"]
                    for w in _get_workers(gateway_url).get("workers", [])
                ),
                f"worker {pod_name} (ip {pod_ip}) registered before drain",
                timeout=30,
                interval=3,
            )

            # Graceful delete: no --force, no --grace-period=0. K8s sets
            # deletionTimestamp and starts the preStop hook + grace timer.
            delete_started = time.time()
            _kubectl(
                "delete",
                "pod",
                pod_name,
                "-n",
                NAMESPACE,
                f"--grace-period={grace_secs}",
                "--wait=false",
            )

            # Assertion is URL-membership rather than total count: a stale
            # worker leaked from a prior test would skew a count comparison
            # without affecting whether *this* pod's IP got removed. The
            # meaningful timing guarantee (`elapsed < grace_secs`) is below.
            _poll_until(
                lambda: not any(
                    pod_ip in w["url"]
                    for w in _get_workers(gateway_url).get("workers", [])
                ),
                f"worker for ip {pod_ip} deregistered after graceful delete",
                timeout=RECONCILIATION_WAIT_SECS,
                interval=1,
            )
            elapsed = time.time() - delete_started
            logger.info(
                "Worker deregistered %.1fs after delete (grace_period=%ds, "
                "prestop_sleep=%ds)",
                elapsed,
                grace_secs,
                prestop_sleep,
            )
            assert elapsed < grace_secs, (
                f"Deregistration took {elapsed:.1f}s — that's >= grace_period "
                f"({grace_secs}s). The gateway should have acted on the "
                f"watcher's deletionTimestamp event well before the grace "
                f"period elapsed (and well before the post-grace SIGKILL "
                f"would emit a Deleted event that applied_objects() filters)."
            )

            # The pod should still exist (Terminating) at this point: it
            # has roughly grace_secs - elapsed seconds left. This guards
            # against a lucky-pass where the test runs so slowly that the
            # pod is gone by the time we check, which would no longer prove
            # the deletionTimestamp path. Use `--ignore-not-found` so a
            # genuinely-missing pod yields rc=0 with empty stdout; any other
            # non-zero rc is a real cluster error (apiserver blip, RBAC
            # drift) and must not masquerade as "pod is gone".
            get_result = _kubectl(
                "get",
                "pod",
                pod_name,
                "-n",
                NAMESPACE,
                "-o",
                "name",
                "--ignore-not-found",
                check=False,
            )
            if get_result.returncode != 0:
                pytest.fail(
                    f"kubectl get pod {pod_name} failed unexpectedly "
                    f"(rc={get_result.returncode}): {get_result.stderr.strip()}"
                )
            still_exists = bool(get_result.stdout.strip())
            assert still_exists, (
                f"Pod {pod_name} is already gone — the test ran too slowly "
                f"to distinguish the deletionTimestamp event from pod-gone "
                f"reconciliation. Increase grace_period or prestop_sleep."
            )
        finally:
            _safe_force_delete(pod_name)
