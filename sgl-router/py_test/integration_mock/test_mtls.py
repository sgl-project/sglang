"""
Integration tests for mTLS (mutual TLS) authentication between router and workers.

Tests verify that:
1. Router can successfully connect to TLS-enabled workers with proper certificates
2. Router fails to connect to mTLS-required workers without client certificates
3. Router with CA certs can connect to TLS-only workers (server auth only)
"""

import subprocess
import time
from pathlib import Path
from typing import Tuple

import pytest
import requests

from ..fixtures.ports import find_free_port


def get_test_certs_dir() -> Path:
    """Get the path to the test certificates directory."""
    return Path(__file__).parent.parent / "fixtures" / "test_certs"


def _spawn_tls_worker(
    port: int,
    worker_id: str,
    ssl_certfile: str,
    ssl_keyfile: str,
    ssl_ca_certs: str = None,
) -> Tuple[subprocess.Popen, str]:
    """Spawn a mock worker with TLS/mTLS enabled."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "py_test" / "fixtures" / "mock_worker.py"

    cmd = [
        "python3",
        str(script),
        "--port",
        str(port),
        "--worker-id",
        worker_id,
        "--ssl-certfile",
        ssl_certfile,
        "--ssl-keyfile",
        ssl_keyfile,
    ]

    if ssl_ca_certs:
        cmd.extend(["--ssl-ca-certs", ssl_ca_certs])

    # Use DEVNULL for stdout to avoid blocking, but keep stderr for debugging
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    url = f"https://127.0.0.1:{port}"

    # Give worker a moment to start or fail
    import time

    time.sleep(3)  # Increased delay to ensure TLS server is fully initialized

    # Check if process died immediately
    if proc.poll() is not None:
        _, stderr = proc.communicate()
        raise RuntimeError(f"Worker failed to start.\nStderr: {stderr}")

    # Wait for worker to be ready (with retries for SSL startup)
    # For mTLS workers (with ssl_ca_certs), provide client cert for health check
    certs_dir = get_test_certs_dir()
    client_cert = certs_dir / "client-cert.pem" if ssl_ca_certs else None
    client_key = certs_dir / "client-key.pem" if ssl_ca_certs else None

    try:
        _wait_tls_health(url, certs_dir / "ca-cert.pem", client_cert, client_key)
    except TimeoutError:
        # If health check times out, capture stderr for debugging
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            raise RuntimeError(f"Worker died during health check.\nStderr: {stderr}")
        raise
    return proc, url


def _wait_tls_health(
    url: str,
    ca_cert_path: Path = None,
    client_cert_path: Path = None,
    client_key_path: Path = None,
    timeout: float = 10.0,
):
    """Wait for TLS-enabled worker to become healthy.

    Args:
        url: HTTPS URL of the worker
        ca_cert_path: Path to CA certificate for verifying server cert
        client_cert_path: Path to client certificate for mTLS
        client_key_path: Path to client private key for mTLS
        timeout: Maximum time to wait in seconds
    """
    start = time.time()
    last_error = None
    with requests.Session() as s:
        while time.time() - start < timeout:
            try:
                # Verify server cert with CA if provided, otherwise skip verification
                verify = str(ca_cert_path) if ca_cert_path else False

                # Provide client cert for mTLS if specified
                cert = None
                if client_cert_path and client_key_path:
                    cert = (str(client_cert_path), str(client_key_path))

                r = s.get(f"{url}/health", timeout=1, verify=verify, cert=cert)
                if r.status_code == 200:
                    return
            except requests.RequestException as e:
                # Save last error for debugging
                last_error = e
            time.sleep(0.2)
    raise TimeoutError(
        f"TLS worker at {url} did not become healthy. Last error: {last_error}"
    )


@pytest.mark.integration
def test_mtls_successful_communication(router_manager, test_certificates):
    """Test that router can successfully communicate with mTLS-enabled worker."""
    certs_dir = test_certificates

    # Start worker with mTLS (requires client certificate)
    port = find_free_port()
    worker_id = f"tls-worker-{port}"
    worker_proc, worker_url = _spawn_tls_worker(
        port=port,
        worker_id=worker_id,
        ssl_certfile=str(certs_dir / "server-cert.pem"),
        ssl_keyfile=str(certs_dir / "server-key.pem"),
        ssl_ca_certs=str(certs_dir / "ca-cert.pem"),  # Require client cert
    )

    try:
        # Start router with mTLS configuration
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                "client_cert_path": str(certs_dir / "client-cert.pem"),
                "client_key_path": str(certs_dir / "client-key.pem"),
                "ca_cert_paths": [str(certs_dir / "ca-cert.pem")],
            },
        )

        # Make request through router - should succeed
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
        )

        assert r.status_code == 200, f"Request failed: {r.status_code} {r.text}"
        data = r.json()
        assert "choices" in data
        assert data.get("worker_id") == worker_id

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()


@pytest.mark.integration
def test_mtls_failure_without_client_cert(router_manager, test_certificates):
    """Test that router fails to connect to mTLS worker without client certificates."""
    certs_dir = test_certificates

    # Start worker with mTLS (requires client certificate)
    port = find_free_port()
    worker_id = f"tls-worker-{port}"
    worker_proc, worker_url = _spawn_tls_worker(
        port=port,
        worker_id=worker_id,
        ssl_certfile=str(certs_dir / "server-cert.pem"),
        ssl_keyfile=str(certs_dir / "server-key.pem"),
        ssl_ca_certs=str(certs_dir / "ca-cert.pem"),  # Require client cert
    )

    try:
        # Start router WITHOUT client certificates (but with CA to verify server)
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                "ca_cert_paths": [str(certs_dir / "ca-cert.pem")],
                # Note: no client_cert_path or client_key_path
            },
        )

        # Make request through router - should fail because worker requires client cert
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
        )

        # Router should return 503 (service unavailable) or 500 because it can't connect to worker
        assert r.status_code in [500, 503], f"Expected 500/503 but got {r.status_code}"

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()


@pytest.mark.integration
def test_tls_server_auth_only(router_manager, test_certificates):
    """Test router can connect to TLS worker that doesn't require client certificates."""
    certs_dir = test_certificates

    # Start worker with TLS but WITHOUT requiring client certificates
    port = find_free_port()
    worker_id = f"tls-worker-{port}"
    worker_proc, worker_url = _spawn_tls_worker(
        port=port,
        worker_id=worker_id,
        ssl_certfile=str(certs_dir / "server-cert.pem"),
        ssl_keyfile=str(certs_dir / "server-key.pem"),
        ssl_ca_certs=None,  # Don't require client cert
    )

    try:
        # Start router with only CA cert (to verify server), no client cert
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                "ca_cert_paths": [str(certs_dir / "ca-cert.pem")],
                # Note: no client_cert_path or client_key_path needed
            },
        )

        # Make request through router - should succeed with server-only TLS
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
        )

        assert r.status_code == 200, f"Request failed: {r.status_code} {r.text}"
        data = r.json()
        assert "choices" in data
        assert data.get("worker_id") == worker_id

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()


@pytest.mark.integration
def test_tls_failure_without_ca_cert(router_manager, test_certificates):
    """Test that router fails to connect to TLS worker without CA certificate."""
    certs_dir = test_certificates

    # Start worker with TLS
    port = find_free_port()
    worker_id = f"tls-worker-{port}"
    worker_proc, worker_url = _spawn_tls_worker(
        port=port,
        worker_id=worker_id,
        ssl_certfile=str(certs_dir / "server-cert.pem"),
        ssl_keyfile=str(certs_dir / "server-key.pem"),
        ssl_ca_certs=None,
    )

    try:
        # Start router WITHOUT CA certificate (can't verify server cert)
        rh = router_manager.start_router(
            worker_urls=[worker_url],
            policy="round_robin",
            extra={
                # Note: no ca_cert_paths - router won't trust self-signed cert
            },
        )

        # Make request through router - should fail because router can't verify server cert
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
        )

        # Router should return 503 (service unavailable) or 500 because it can't verify worker cert
        assert r.status_code in [500, 503], f"Expected 500/503 but got {r.status_code}"

    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
