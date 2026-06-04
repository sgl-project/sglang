"""Pytest configuration for sgl-router K8s integration tests.

These tests require:
  - A kind cluster named 'sgl-router-kind'
  - The sgl-router:e2e and sgl-router-fake-worker:e2e images loaded into kind
  - kubectl configured to use the kind-sgl-router-kind context

Setup:  ./tests/e2e/k8s_integration/setup.sh
Teardown: ./tests/e2e/k8s_integration/setup.sh teardown
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time

import httpx
import pytest

logger = logging.getLogger(__name__)

NAMESPACE = "sgl-router-test"
CLUSTER_NAME = "sgl-router-kind"
KUBECTL_CONTEXT = f"kind-{CLUSTER_NAME}"

# sgl-router discovery reconciliation: if the watcher misses an event the
# reconciler fires within ~60s.  Tests that exercise removal wait up to 90s.
RECONCILIATION_WAIT_SECS = 90

# Errors safe to retry while polling (transport-level only — HTTP 4xx/5xx
# are intentionally NOT included so real regressions surface immediately).
_TRANSIENT_ERRORS = (
    httpx.TransportError,
    httpx.TimeoutException,
    ConnectionError,
    OSError,
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that wait for multiple reconciliation cycles "
        "(deselect with '-m \"not slow\"')",
    )


def _kubectl(
    *args: str,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    cmd = ["kubectl", "--context", KUBECTL_CONTEXT, *args]
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=capture, text=True, check=check)


def _apply_from_stdin(yaml_content: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "--context", KUBECTL_CONTEXT, "apply", "-f", "-"],
        input=yaml_content,
        capture_output=True,
        text=True,
        check=True,
    )


def _wait_for_deployment_ready(
    name: str,
    namespace: str = NAMESPACE,
    timeout: int = 180,
) -> None:
    _kubectl(
        "rollout",
        "status",
        f"deployment/{name}",
        "-n",
        namespace,
        f"--timeout={timeout}s",
    )


def _wait_for_pod_ready(
    name: str,
    namespace: str = NAMESPACE,
    timeout: int = 120,
) -> None:
    _kubectl(
        "wait",
        "--for=condition=Ready",
        f"pod/{name}",
        "-n",
        namespace,
        f"--timeout={timeout}s",
    )


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: int = 15) -> None:
    """Poll until a TCP connection to localhost:port succeeds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"port-forward process exited early: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Port {port} not ready after {timeout}s")


def _port_forward_start(
    namespace: str,
    service: str,
    local_port: int,
    remote_port: int,
) -> subprocess.Popen:
    """Start kubectl port-forward and wait until the port is reachable."""
    cmd = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "port-forward",
        f"svc/{service}",
        f"{local_port}:{remote_port}",
        "-n",
        namespace,
    ]
    logger.info("Starting port-forward: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _wait_for_port(local_port, proc)
    return proc


def _cleanup_port_forward(name: str, pf: subprocess.Popen) -> None:
    try:
        pf.terminate()
        pf.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Port-forward %s did not exit on SIGTERM after 10s; killing", name
        )
        pf.kill()
        try:
            pf.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Port-forward %s still running after SIGKILL", name)
    except Exception as exc:
        logger.warning("Error cleaning up %s port-forward: %s", name, exc)

    rc = pf.returncode
    stderr = pf.stderr.read().decode() if pf.stderr else ""
    if rc != -15:
        suffix = f": {stderr.strip()}" if stderr.strip() else ""
        logger.warning("Port-forward %s exited rc=%s%s", name, rc, suffix)
    else:
        logger.debug("Port-forward %s exited cleanly (rc=%s)", name, rc)


def _poll_until(
    predicate,
    description: str,
    timeout: int,
    interval: float = 5,
) -> bool:
    """Poll predicate until True, or raise TimeoutError.

    Only transient network errors are retried; HTTP status errors and
    programming errors propagate immediately.
    """
    deadline = time.time() + timeout
    last_error = None
    attempts = 0
    while time.time() < deadline:
        try:
            attempts += 1
            if predicate():
                logger.info(
                    "Condition met: %s (after %d attempts)", description, attempts
                )
                return True
        except _TRANSIENT_ERRORS as exc:
            last_error = exc
            logger.debug("Transient error on attempt %d: %s", attempts, exc)
        time.sleep(interval)
    msg = f"Timeout waiting for: {description} (after {timeout}s, {attempts} attempts)"
    if last_error:
        msg += f" — last error: {last_error}"
    raise TimeoutError(msg)


def _get_router_url(router_base: str) -> str:
    return router_base


def _router_is_healthy(router_base: str) -> bool:
    try:
        r = httpx.get(f"{router_base}/healthz", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def k8s_cluster():
    """Assert the kind cluster exists and kubectl context is reachable."""
    result = subprocess.run(
        ["kind", "get", "clusters"],
        capture_output=True,
        text=True,
        check=True,
    )
    if CLUSTER_NAME not in result.stdout.splitlines():
        pytest.skip(
            f"kind cluster '{CLUSTER_NAME}' not found — run "
            f"./tests/e2e/k8s_integration/setup.sh first"
        )
    _kubectl("cluster-info")
    return True


@pytest.fixture(scope="function")
def router_port_forward(k8s_cluster):
    """Per-test port-forward to sgl-router service.

    Function-scoped because some tests (notably
    test_lifecycle.TestRouterRestart) force-delete the router pod;
    a session-scoped port-forward would be bound to the deleted pod's
    network namespace and stay dead for all subsequent tests in the
    suite. Per-test setup costs ~1-2s.
    """
    _wait_for_deployment_ready("sgl-router")
    pf = _port_forward_start(NAMESPACE, "sgl-router", 8090, 8090)
    try:
        _poll_until(
            lambda: _router_is_healthy("http://127.0.0.1:8090"),
            "sgl-router /healthz returns 200",
            timeout=30,
            interval=1,
        )
        yield "http://127.0.0.1:8090"
    finally:
        _cleanup_port_forward("sgl-router", pf)


@pytest.fixture(scope="function")
def router_url(router_port_forward):
    return router_port_forward
