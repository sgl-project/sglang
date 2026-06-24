"""Cross-namespace service discovery integration test.

Validates that a sgl-router instance with cluster-wide RBAC and no namespace
filter in its k8s discovery config watches EndpointSlices in all namespaces.
Workers deployed in a second namespace (sgl-router-test-extra) must be
discovered alongside those in the primary namespace.

This test deploys a separate router Deployment (sgl-router-cluster) with a
ClusterRole that grants EndpointSlice access across all namespaces.

Run with:
    pytest tests/e2e/k8s_integration/test_cross_namespace.py -v -s
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import httpx
import pytest
from conftest import (
    KUBECTL_CONTEXT,
    NAMESPACE,
    _apply_from_stdin,
    _cleanup_port_forward,
    _kubectl,
    _poll_until,
    _port_forward_start,
    _wait_for_deployment_ready,
    logger,
)

MANIFESTS_DIR = Path(__file__).parent / "manifests"
EXTRA_NAMESPACE = "sgl-router-test-extra"
CLUSTER_ROUTER_PORT = 8093


def _deploy_fake_worker_in_ns(name: str, namespace: str) -> None:
    """Deploy a fake-worker pod with imagePullPolicy=Never in the given namespace."""
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"app": "sglang", "cross-ns-test": "true"},
        },
        "spec": {
            "containers": [
                {
                    "name": "worker",
                    "image": "sgl-router-fake-worker:e2e",
                    "imagePullPolicy": "Never",
                    "ports": [{"containerPort": 30000}],
                    "readinessProbe": {
                        "httpGet": {"path": "/health", "port": 30000},
                        "initialDelaySeconds": 2,
                        "periodSeconds": 3,
                    },
                }
            ]
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
        raise RuntimeError(
            f"Failed to deploy pod {name} in namespace {namespace} "
            f"(rc={proc.returncode}): {proc.stderr.strip()!r}"
        )
    logger.info("Deployed worker %s in namespace %s", name, namespace)


def _safe_delete_pod(name: str, namespace: str) -> None:
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
    except Exception as exc:
        logger.warning("Cleanup failed for pod %s in ns %s: %s", name, namespace, exc)


def _ensure_namespace(name: str) -> None:
    manifest = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": name}}
    _apply_from_stdin(json.dumps(manifest))


def _ensure_service_in_ns(namespace: str, selector: str = "app=sglang") -> None:
    """Create a Service so K8s auto-creates an EndpointSlice for cross-ns workers.

    Service `metadata.labels` propagates to the auto-created EndpointSlice's
    labels — and the cluster-scoped router filters slices server-side by
    `app=sglang,cross-ns-test=true`. Without those labels on the Service,
    its EndpointSlice gets filtered out and the cross-ns worker is invisible.
    """
    svc_manifest = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "fake-worker",
            "namespace": namespace,
            "labels": {"app": "sglang", "cross-ns-test": "true"},
        },
        "spec": {
            "selector": {"app": "sglang", "cross-ns-test": "true"},
            "ports": [{"port": 30000, "targetPort": 30000}],
        },
    }
    _apply_from_stdin(json.dumps(svc_manifest))


def _can_route(router_url: str) -> bool:
    try:
        r = httpx.post(
            f"{router_url}/v1/chat/completions",
            json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "cross-ns"}],
            },
            timeout=8.0,
        )
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def cluster_scoped_router(k8s_cluster):
    """Deploy the cluster-scoped RBAC + router, plus a second namespace."""
    rbac_manifest = MANIFESTS_DIR / "rbac-cluster-scoped.yaml"
    router_manifest = MANIFESTS_DIR / "router-cluster-scoped.yaml"

    _kubectl("apply", "-f", str(rbac_manifest))
    _ensure_namespace(EXTRA_NAMESPACE)
    _ensure_service_in_ns(EXTRA_NAMESPACE)

    # The cluster-scoped router is configured via CLI flags in
    # router-cluster-scoped.yaml: no --service-discovery-namespace (watch
    # all namespaces) and --selector app=sglang,cross-ns-test=true.
    _kubectl("apply", "-f", str(router_manifest))

    # The cluster-scoped router's /readyz blocks on registry-not-empty, so
    # without at least one matching worker the rollout-status check below
    # would hang for 180s. Deploy a "bootstrap" worker in EXTRA_NAMESPACE
    # with the label_selector match (app=sglang,cross-ns-test=true) so the
    # router's k8s discovery picks it up before the readiness probe runs.
    # The test body adds a SECOND worker later to verify dynamic discovery.
    bootstrap_worker = "cross-ns-worker-bootstrap"
    _deploy_fake_worker_in_ns(bootstrap_worker, EXTRA_NAMESPACE)

    pf = None
    try:
        _wait_for_deployment_ready("sgl-router-cluster")
        pf = _port_forward_start(
            NAMESPACE, "sgl-router-cluster", CLUSTER_ROUTER_PORT, 8091
        )
        yield f"http://127.0.0.1:{CLUSTER_ROUTER_PORT}"
    finally:
        if pf is not None:
            _cleanup_port_forward("cluster_router", pf)
        _safe_delete_pod(bootstrap_worker, EXTRA_NAMESPACE)
        _kubectl(
            "delete", "-f", str(router_manifest), "--ignore-not-found", check=False
        )
        _kubectl("delete", "-f", str(rbac_manifest), "--ignore-not-found", check=False)
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
    """Router with ClusterRole and no namespace filter sees workers in every namespace."""

    def test_router_routes_to_worker_in_extra_namespace(self, cluster_scoped_router):
        """Deploy one fake-worker pod in the extra namespace behind a Service;
        the cluster-scoped router must discover it (via its EndpointSlice) and
        successfully route a chat completion to it."""
        router_url = cluster_scoped_router
        worker_name = "cross-ns-worker-extra"

        try:
            _deploy_fake_worker_in_ns(worker_name, EXTRA_NAMESPACE)

            _poll_until(
                lambda: _can_route(router_url),
                "cluster-scoped router routes to worker in extra namespace",
                timeout=60,
                interval=3,
            )

            r = httpx.post(
                f"{router_url}/v1/chat/completions",
                json={
                    "model": "tiny",
                    "messages": [
                        {"role": "user", "content": "cross-namespace routing"}
                    ],
                },
                timeout=15.0,
            )
            assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
            assert "echo:" in r.json()["choices"][0]["message"]["content"]
        finally:
            _safe_delete_pod(worker_name, EXTRA_NAMESPACE)
