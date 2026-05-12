"""Integration tests for K8s service discovery and reconciliation.

Tests verify that:
1. The K8s watcher correctly discovers new pods
2. Stale workers are removed after pod deletion (watcher or reconciliation)
3. All pods are eventually discovered and tracked consistently
4. Prometheus discovery metrics are emitted correctly
5. Reconciliation does not cause instability over multiple cycles

These tests require a kind cluster with the gateway deployed. The
reconciliation interval is 60s (see ServiceDiscoveryConfig.check_interval
in sgl-model-gateway/src/service_discovery.rs), so tests exercising
reconciliation must wait ~90s for a tick to fire.

Run with:
    cd e2e_test/k8s_integration
    source .venv/bin/activate
    pytest test_reconciliation.py -v -s
"""

from __future__ import annotations

import json
import logging
import subprocess
import time

import httpx
import pytest
from conftest import (  # pytest's rootdir adds the test dir to sys.path
    KUBECTL_CONTEXT,
    NAMESPACE,
    RECONCILIATION_WAIT_SECS,
    _get_worker_count,
    _get_workers,
    _kubectl,
    _poll_until,
    _wait_for_pod_ready,
)

logger = logging.getLogger(__name__)


def _get_metrics(metrics_url: str) -> str:
    """GET /metrics from the gateway (Prometheus text format)."""
    resp = httpx.get(f"{metrics_url}/metrics", timeout=10)
    resp.raise_for_status()
    return resp.text


def _get_worker_urls(gateway_url: str) -> set[str]:
    """Return the set of worker URLs currently registered in the gateway."""
    data = _get_workers(gateway_url)
    return {w["url"] for w in data.get("workers", [])}


def _parse_metric_value(
    metrics_text: str, metric_name: str, labels: dict | None = None
) -> float | None:
    """Parse a specific metric value from Prometheus text format.

    Uses exact metric name matching (line must start with the metric name)
    and logs diagnostics when the metric is not found.
    """
    matching_lines = []
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        # Exact metric name match: name must be followed by '{' or ' '
        if not line.startswith(metric_name):
            continue
        rest = line[len(metric_name) :]
        if rest and rest[0] not in ("{", " "):
            continue
        matching_lines.append(line)

    if not matching_lines:
        logger.debug("Metric %s not found in output", metric_name)
        return None

    for line in matching_lines:
        if labels:
            if not all(f'{k}="{v}"' in line for k, v in labels.items()):
                continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                return float(parts[-1])
            except ValueError:
                logger.warning("Could not parse float from metric line: %s", line)
                continue

    logger.debug(
        "Metric %s found but no line matched labels %s. Lines: %s",
        metric_name,
        labels,
        matching_lines,
    )
    return None


def _deploy_worker_pod(name: str, labels: dict[str, str] | None = None):
    """Deploy a single fake worker pod via kubectl apply from stdin."""
    pod_labels = {"app": "fake-worker"}
    if labels:
        pod_labels.update(labels)

    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": pod_labels,
        },
        "spec": {
            "containers": [
                {
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
                    "volumeMounts": [
                        {
                            "name": "app",
                            "mountPath": "/app",
                        }
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "app",
                    "configMap": {"name": "fake-worker-script"},
                }
            ],
        },
    }
    proc = subprocess.run(
        ["kubectl", "--context", KUBECTL_CONTEXT, "apply", "-f", "-"],
        input=json.dumps(pod_manifest),
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info("Deployed pod %s: %s", name, proc.stdout.strip())


def _delete_worker_pod(name: str, force: bool = False):
    """Delete a fake worker pod."""
    args = ["delete", "pod", name, "-n", NAMESPACE, "--ignore-not-found"]
    if force:
        args.extend(["--grace-period=0", "--force"])
    _kubectl(*args)
    logger.info("Deleted pod %s (force=%s)", name, force)


def _safe_delete_worker_pod(name: str):
    """Delete a worker pod in a cleanup context, logging errors instead of raising."""
    try:
        _delete_worker_pod(name, force=True)
    except Exception as e:
        logger.warning("Cleanup failed for pod %s: %s", name, e)


def _wait_for_pod_gone(name: str, timeout: int = 60):
    """Wait until a pod no longer exists in K8s.

    Raises TimeoutError if the pod still exists after timeout, or RuntimeError
    if kubectl returns an unexpected error (e.g., apiserver unreachable, RBAC
    drift). The latter would otherwise surface as a misleading "still exists"
    timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _kubectl(
            "get",
            "pod",
            name,
            "-n",
            NAMESPACE,
            check=False,
        )
        if result.returncode == 0:
            time.sleep(2)
            continue
        stderr = result.stderr.strip()
        if "NotFound" in stderr or "not found" in stderr.lower():
            logger.info("Pod %s is gone", name)
            return
        # Anything else is a real cluster-level error — fail loudly so the
        # caller sees the actual problem instead of a generic timeout.
        raise RuntimeError(
            f"kubectl get pod {name} failed unexpectedly (rc={result.returncode}): {stderr}"
        )
    raise TimeoutError(f"Pod {name} still exists after {timeout}s")


class TestWatcherDiscovery:
    """Tests that the K8s watcher correctly discovers pods on creation."""

    def test_watcher_discovers_new_pod(self, gateway_port_forward):
        """Deploy a new worker pod and verify the watcher picks it up quickly."""
        gateway_url, metrics_url = gateway_port_forward
        pod_name = "test-watcher-discovery"

        try:
            initial_count = _get_worker_count(gateway_url)
            logger.info("Initial worker count: %d", initial_count)

            _deploy_worker_pod(pod_name)
            _wait_for_pod_ready(pod_name)

            # The watcher should pick up the pod within seconds
            _poll_until(
                lambda: _get_worker_count(gateway_url) > initial_count,
                f"worker count > {initial_count}",
                timeout=30,
                interval=3,
            )

            workers = _get_workers(gateway_url)
            logger.info(
                "Workers after pod creation: %s",
                json.dumps(workers, indent=2),
            )
            assert workers["total"] > initial_count

        finally:
            _safe_delete_worker_pod(pod_name)


class TestReconciliationStaleWorkerRemoval:
    """Test that stale workers are removed after pod deletion.

    The watcher DELETE event typically handles this immediately.
    If the watcher misses it (e.g., during restart or backoff),
    reconciliation catches it within ~60s.
    """

    def test_stale_worker_removed_after_pod_deletion(self, gateway_port_forward):
        """Deploy a worker, verify discovery, delete the pod, and verify
        the worker is removed (by either watcher or reconciliation)."""
        gateway_url, metrics_url = gateway_port_forward
        pod_name = "test-stale-removal"

        try:
            _deploy_worker_pod(pod_name)
            _wait_for_pod_ready(pod_name)

            _poll_until(
                lambda: _get_worker_count(gateway_url) >= 1,
                "at least 1 worker discovered",
                timeout=30,
                interval=3,
            )

            count_with_pod = _get_worker_count(gateway_url)
            logger.info("Worker count with test pod: %d", count_with_pod)

            # Force-delete the pod (instant removal from K8s API)
            _delete_worker_pod(pod_name, force=True)
            _wait_for_pod_gone(pod_name)

            # Wait for the gateway to remove the stale worker.
            # The watcher DELETE event may handle this immediately.
            # If it doesn't, reconciliation will catch it within ~60s.
            _poll_until(
                lambda: _get_worker_count(gateway_url) < count_with_pod,
                f"worker count < {count_with_pod} (stale worker removed)",
                timeout=RECONCILIATION_WAIT_SECS,
                interval=5,
            )

            final_count = _get_worker_count(gateway_url)
            logger.info("Worker count after stale removal: %d", final_count)
            assert final_count < count_with_pod

        finally:
            _safe_delete_worker_pod(pod_name)


class TestReconciliationMissedPodDiscovery:
    """Verify reconciliation coexists with watcher discovery without interference.

    Note: this test cannot force the watcher to miss events, so it does NOT
    prove that reconciliation discovers missed pods in isolation. It validates
    that reconciliation maintains consistency when pods are already discovered
    by the watcher, and that no pods are lost.
    """

    def test_all_workers_eventually_discovered(self, gateway_port_forward):
        """Deploy multiple worker pods and verify they are all discovered."""
        gateway_url, metrics_url = gateway_port_forward
        pod_names = ["test-reconcile-a", "test-reconcile-b"]

        try:
            for name in pod_names:
                _deploy_worker_pod(name)

            for name in pod_names:
                _wait_for_pod_ready(name)

            # Wait for watcher (or reconciliation) to discover all pods
            _poll_until(
                lambda: _get_worker_count(gateway_url) >= len(pod_names),
                f"at least {len(pod_names)} workers discovered",
                timeout=RECONCILIATION_WAIT_SECS,
                interval=5,
            )

            workers = _get_workers(gateway_url)
            logger.info(
                "Workers after discovery: %s",
                json.dumps(workers, indent=2),
            )
            assert workers["total"] >= len(pod_names)

        finally:
            for name in pod_names:
                _safe_delete_worker_pod(name)
            for name in pod_names:
                try:
                    _wait_for_pod_gone(name, timeout=30)
                except TimeoutError:
                    logger.warning("Pod %s still present after cleanup", name)


class TestReconciliationMetrics:
    """Test that the gateway emits expected Prometheus discovery metrics."""

    def test_discovery_metrics_populated(self, gateway_port_forward):
        """After pods are discovered, verify registration and gauge metrics."""
        gateway_url, metrics_url = gateway_port_forward
        pod_name = "test-metrics"

        try:
            _deploy_worker_pod(pod_name)
            _wait_for_pod_ready(pod_name)

            # Wait for the watcher to discover the pod
            _poll_until(
                lambda: _get_worker_count(gateway_url) >= 1,
                "at least 1 worker",
                timeout=30,
                interval=3,
            )

            # Poll for the registration metric instead of a fixed sleep
            def _registration_metric_exists():
                text = _get_metrics(metrics_url)
                val = _parse_metric_value(
                    text,
                    "smg_discovery_registrations_total",
                    {"source": "kubernetes", "result": "success"},
                )
                return val is not None and val >= 1

            _poll_until(
                _registration_metric_exists,
                "registration success metric >= 1",
                timeout=30,
                interval=3,
            )

            metrics_text = _get_metrics(metrics_url)

            reg_value = _parse_metric_value(
                metrics_text,
                "smg_discovery_registrations_total",
                {"source": "kubernetes", "result": "success"},
            )
            logger.info("Registration success metric: %s", reg_value)
            assert (
                reg_value is not None and reg_value >= 1
            ), f"Expected at least 1 registration, got {reg_value}"

            gauge_value = _parse_metric_value(
                metrics_text,
                "smg_discovery_workers_discovered",
                {"source": "kubernetes"},
            )
            logger.info("Workers discovered gauge: %s", gauge_value)
            assert (
                gauge_value is not None and gauge_value >= 1
            ), f"Expected workers_discovered >= 1, got {gauge_value}"

        finally:
            _safe_delete_worker_pod(pod_name)

    def test_deregistration_metric_after_pod_deletion(self, gateway_port_forward):
        """Deploy a pod, delete it, and verify a deregistration metric fires.

        Either 'pod_deleted' (from the watcher) or 'reconciled' (from
        periodic reconciliation) should increment. Which one fires depends
        on whether the watcher sees the deletion event first.
        """
        gateway_url, metrics_url = gateway_port_forward
        pod_name = "test-dereg-metric"

        try:
            _deploy_worker_pod(pod_name)
            _wait_for_pod_ready(pod_name)

            _poll_until(
                lambda: _get_worker_count(gateway_url) >= 1,
                "at least 1 worker",
                timeout=30,
                interval=3,
            )

            count_before = _get_worker_count(gateway_url)

            _delete_worker_pod(pod_name, force=True)
            _wait_for_pod_gone(pod_name)

            _poll_until(
                lambda: _get_worker_count(gateway_url) < count_before,
                "worker count decreased after pod deletion",
                timeout=RECONCILIATION_WAIT_SECS,
                interval=5,
            )

            metrics_text = _get_metrics(metrics_url)

            pod_deleted = _parse_metric_value(
                metrics_text,
                "smg_discovery_deregistrations_total",
                {"source": "kubernetes", "reason": "pod_deleted"},
            )
            reconciled = _parse_metric_value(
                metrics_text,
                "smg_discovery_deregistrations_total",
                {"source": "kubernetes", "reason": "reconciled"},
            )

            logger.info(
                "Deregistration metrics — pod_deleted: %s, reconciled: %s",
                pod_deleted,
                reconciled,
            )

            total_dereg = (pod_deleted or 0) + (reconciled or 0)
            assert total_dereg >= 1, (
                f"Expected at least 1 deregistration, got pod_deleted={pod_deleted}, "
                f"reconciled={reconciled}"
            )

        finally:
            _safe_delete_worker_pod(pod_name)


class TestReconciliationConsistency:
    """Test that reconciliation maintains consistency over multiple cycles."""

    @pytest.mark.slow
    def test_repeated_reconciliation_is_stable(self, gateway_port_forward):
        """Deploy pods, wait for 2+ reconciliation cycles, and verify worker
        count stays stable (no duplicate additions or spurious removals)."""
        gateway_url, metrics_url = gateway_port_forward
        pod_names = ["test-stable-a", "test-stable-b"]

        try:
            for name in pod_names:
                _deploy_worker_pod(name)
            for name in pod_names:
                _wait_for_pod_ready(name)

            # Wait for initial discovery
            _poll_until(
                lambda: _get_worker_count(gateway_url) >= len(pod_names),
                f"at least {len(pod_names)} workers",
                timeout=30,
                interval=3,
            )

            stable_count = _get_worker_count(gateway_url)
            logger.info("Stable worker count: %d", stable_count)

            # Wait for 2+ reconciliation cycles: 2*60s interval + 30s margin = 150s total
            wait_time = RECONCILIATION_WAIT_SECS + 60
            logger.info("Waiting %ds for 2+ reconciliation cycles...", wait_time)

            # Sample count periodically to verify stability
            end_time = time.time() + wait_time
            samples = []
            while time.time() < end_time:
                count = _get_worker_count(gateway_url)
                samples.append(count)
                time.sleep(15)

            logger.info("Worker count samples over time: %s", samples)

            assert all(
                s == stable_count for s in samples
            ), f"Worker count fluctuated: {samples} (expected stable at {stable_count})"

        finally:
            for name in pod_names:
                _safe_delete_worker_pod(name)
