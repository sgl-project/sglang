"""Cross-namespace service discovery integration test.

When the gateway is started without --service-discovery-namespace, the K8s
API client falls through to Api::all (sgl-model-gateway/src/service_discovery.rs:279),
watching pods in every namespace. That path requires a ClusterRole rather
than the namespace-scoped Role used by the default gateway.

Validates: a single gateway with cluster-wide RBAC discovers workers running
in two distinct namespaces, with each worker registered exactly once.

Run with:
    cd e2e_test/k8s_integration
    pytest test_cross_namespace.py -v -s
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import pytest
from conftest import (  # pytest's rootdir adds the test dir to sys.path
    FAKE_WORKER_SCRIPT,
    KUBECTL_CONTEXT,
    NAMESPACE,
    _apply_from_stdin,
    _cleanup_port_forward,
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

CLUSTER_GATEWAY_HTTP_PORT = 30002
EXTRA_NAMESPACE = "smg-test-extra"


def _deploy_worker_pod(name: str, namespace: str):
    """Deploy a fake-worker pod in the given namespace.

    Assumes the `fake-worker-script` ConfigMap already exists in that namespace.
    The `cross-ns-test=true` label is paired with the gateway's selector in
    gateway-cluster-scoped.yaml so this test owns its worker fleet exclusively
    and the exact-count assertion isn't racy across files.
    """
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"app": "fake-worker", "cross-ns-test": "true"},
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
                    "volumeMounts": [{"name": "app", "mountPath": "/app"}],
                }
            ],
            "volumes": [{"name": "app", "configMap": {"name": "fake-worker-script"}}],
        },
    }
    proc = subprocess.run(
        ["kubectl", "--context", KUBECTL_CONTEXT, "apply", "-f", "-"],
        input=json.dumps(pod_manifest),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # kubectl's actual error (admission webhook, missing configmap, schema
        # validation) lives in stderr; surface it instead of bubbling up an
        # opaque "non-zero exit status 1".
        raise RuntimeError(
            f"Failed to deploy pod {name} in namespace {namespace} "
            f"(rc={proc.returncode}): stderr={proc.stderr.strip()!r}"
        )
    logger.info("Deployed worker %s in namespace %s", name, namespace)


def _get_pod_ip(name: str, namespace: str) -> str:
    """Return the current podIP for a pod (must be running)."""
    pod = _kubectl_json("get", "pod", name, "-n", namespace)
    ip = pod.get("status", {}).get("podIP")
    if not ip:
        raise RuntimeError(
            f"Pod {namespace}/{name} has no podIP yet: {pod.get('status')}"
        )
    return ip


def _safe_delete_pod(name: str, namespace: str):
    try:
        _kubectl(
            "delete",
            "pod",
            name,
            "-n",
            namespace,
            "--ignore-not-found",
            "--force",
            "--grace-period=0",
        )
    except Exception as e:
        logger.warning("Cleanup failed for pod %s in ns %s: %s", name, namespace, e)


def _ensure_namespace(name: str):
    """Create a namespace if it doesn't already exist (apply is idempotent)."""
    manifest = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": name},
    }
    _apply_from_stdin(json.dumps(manifest))


def _ensure_fake_worker_configmap(namespace: str):
    """Create the fake-worker-script ConfigMap in the given namespace.

    Mirrors the configmap shipped to the default namespace by the deploy_base
    fixture, so workers in other namespaces can boot the same script.
    """
    cm_yaml = _kubectl(
        "create",
        "configmap",
        "fake-worker-script",
        f"--from-file=fake_worker.py={FAKE_WORKER_SCRIPT}",
        "-n",
        namespace,
        "--dry-run=client",
        "-o",
        "yaml",
    )
    _apply_from_stdin(cm_yaml.stdout)


@pytest.fixture(scope="module")
def cluster_scoped_gateway(deploy_base):
    """Deploy the cluster-scoped RBAC + gateway, plus a second namespace.

    Cleanup runs in `finally:` so a port-forward failure does not leak
    deployments / RBAC objects across test runs.
    """
    rbac_manifest = MANIFESTS_DIR / "rbac-cluster-scoped.yaml"
    gateway_manifest = MANIFESTS_DIR / "gateway-cluster-scoped.yaml"

    _kubectl("apply", "-f", str(rbac_manifest))
    _ensure_namespace(EXTRA_NAMESPACE)
    _ensure_fake_worker_configmap(EXTRA_NAMESPACE)
    _kubectl("apply", "-f", str(gateway_manifest))

    pf: subprocess.Popen | None = None
    try:
        _wait_for_deployment_ready("smg-gateway-cluster")
        pf = _port_forward_start(
            NAMESPACE,
            "smg-gateway-cluster",
            CLUSTER_GATEWAY_HTTP_PORT,
            CLUSTER_GATEWAY_HTTP_PORT,
        )
        yield f"http://127.0.0.1:{CLUSTER_GATEWAY_HTTP_PORT}"
    finally:
        if pf is not None:
            _cleanup_port_forward("cluster_gateway", pf)
        _kubectl(
            "delete", "-f", str(gateway_manifest), "--ignore-not-found", check=False
        )
        _kubectl("delete", "-f", str(rbac_manifest), "--ignore-not-found", check=False)
        # Drop the extra namespace last so any worker pods left behind go with
        # it. The bounded --timeout prevents pytest from hanging at session
        # end if a finalizer in the namespace is stuck (kindnet, kube-system
        # controllers); the cleanup error gets logged but pytest continues.
        _kubectl(
            "delete",
            "namespace",
            EXTRA_NAMESPACE,
            "--ignore-not-found",
            "--wait=true",
            "--timeout=60s",
            check=False,
        )


class TestClusterWideDiscovery:
    """Verify that a gateway with ClusterRole + no namespace filter sees pods
    in every namespace."""

    def test_workers_in_two_namespaces_are_both_discovered(
        self, cluster_scoped_gateway
    ):
        gateway_url = cluster_scoped_gateway
        worker_a = "cross-ns-worker-a"
        worker_b = "cross-ns-worker-b"

        try:
            _deploy_worker_pod(worker_a, NAMESPACE)
            _deploy_worker_pod(worker_b, EXTRA_NAMESPACE)
            _wait_for_pod_ready(worker_a, namespace=NAMESPACE)
            _wait_for_pod_ready(worker_b, namespace=EXTRA_NAMESPACE)

            ip_a = _get_pod_ip(worker_a, NAMESPACE)
            ip_b = _get_pod_ip(worker_b, EXTRA_NAMESPACE)
            logger.info(
                "worker_a (%s/%s) ip=%s, worker_b (%s/%s) ip=%s",
                NAMESPACE,
                worker_a,
                ip_a,
                EXTRA_NAMESPACE,
                worker_b,
                ip_b,
            )

            _poll_until(
                lambda: _get_workers(gateway_url)["total"] >= 2,
                "both workers discovered across namespaces",
                timeout=30,
                interval=3,
            )

            workers = _get_workers(gateway_url)
            urls = sorted(w["url"] for w in workers.get("workers", []))
            logger.info("Discovered worker URLs: %s", urls)

            # Each pod should appear once. With two pods deployed and two
            # discovered, the URLs must be distinct (no duplicate registration
            # from cluster-wide watcher firing twice).
            assert len(urls) == len(set(urls)), f"Duplicate worker URLs: {urls}"
            assert workers["total"] == 2, (
                f"Expected exactly 2 workers across namespaces, "
                f"got {workers['total']}: {urls}"
            )

            # Per-pod IP membership: proves the gateway actually listed both
            # namespaces, not "two pods from the same namespace by accident".
            # A regression that quietly hardcoded a namespace filter would
            # still produce total=2 if labels happened to match elsewhere,
            # but only one of these IPs would surface.
            assert any(
                ip_a in u for u in urls
            ), f"worker_a IP {ip_a} (ns {NAMESPACE}) not in {urls}"
            assert any(
                ip_b in u for u in urls
            ), f"worker_b IP {ip_b} (ns {EXTRA_NAMESPACE}) not in {urls}"
        finally:
            _safe_delete_pod(worker_a, NAMESPACE)
            _safe_delete_pod(worker_b, EXTRA_NAMESPACE)
