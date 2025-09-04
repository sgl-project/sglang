import atexit
import socket
import subprocess
import time
from typing import Dict, Optional, Tuple

import pytest
import requests


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RouterProcessManager:
    """Session-scoped manager to launch and reuse router/server processes.

    Caches router processes by a normalized config key so repeated requests with
    the same config reuse the same process instead of launching a new one.
    """

    def __init__(self) -> None:
        self._routers: Dict[Tuple, subprocess.Popen] = {}
        self._router_urls: Dict[Tuple, str] = {}
        self._workers: Dict[str, subprocess.Popen] = {}
        atexit.register(self.cleanup)

    def _router_key(
        self,
        *,
        model: str,
        dp_size: int,
        policy: str,
        max_payload_size: Optional[int],
        api_key: Optional[str],
        service_discovery: bool,
        selector: Optional[Tuple[str, ...]],
        service_discovery_port: int,
        service_discovery_namespace: Optional[str],
        prometheus_port: Optional[int],
        prometheus_host: Optional[str],
        dp_aware: bool,
    ) -> Tuple:
        return (
            model,
            dp_size,
            policy,
            max_payload_size,
            api_key,
            service_discovery,
            selector,
            service_discovery_port,
            service_discovery_namespace,
            prometheus_port,
            prometheus_host,
            dp_aware,
        )

    def ensure_router(
        self,
        *,
        model: str,
        dp_size: int,
        policy: str = "cache_aware",
        max_payload_size: Optional[int] = None,
        api_key: Optional[str] = None,
        log_dir: Optional[str] = None,
        service_discovery: bool = False,
        selector: Optional[Tuple[str, ...]] = None,
        service_discovery_port: int = 80,
        service_discovery_namespace: Optional[str] = None,
        prometheus_port: Optional[int] = None,
        prometheus_host: Optional[str] = None,
        dp_aware: bool = False,
        timeout: float = 300.0,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
    ) -> str:
        """Start or reuse a router with the given config and return its base URL."""
        key = self._router_key(
            model=model,
            dp_size=dp_size,
            policy=policy,
            max_payload_size=max_payload_size,
            api_key=api_key,
            service_discovery=service_discovery,
            selector=tuple(selector) if selector else None,
            service_discovery_port=service_discovery_port,
            service_discovery_namespace=service_discovery_namespace,
            prometheus_port=prometheus_port,
            prometheus_host=prometheus_host,
            dp_aware=dp_aware,
        )

        if key in self._routers:
            return self._router_urls[key]

        if port is None:
            port = _find_available_port()
        base_url = f"http://{host}:{port}"

        command = [
            "python3",
            "-m",
            "sglang_router.launch_server",
            "--model-path",
            model,
            "--host",
            host,
            "--port",
            str(port),
            "--dp",
            str(dp_size),
            "--router-eviction-interval",
            "5",
            "--router-policy",
            policy,
            "--allow-auto-truncate",
        ]

        if api_key is not None:
            command.extend(["--api-key", api_key])
            command.extend(["--router-api-key", api_key])

        if max_payload_size is not None:
            command.extend(["--router-max-payload-size", str(max_payload_size)])

        if service_discovery:
            command.append("--router-service-discovery")

        if selector:
            command.extend(["--router-selector", *selector])

        if service_discovery_port != 80:
            command.extend(["--router-service-discovery-port", str(service_discovery_port)])

        if service_discovery_namespace:
            command.extend([
                "--router-service-discovery-namespace",
                service_discovery_namespace,
            ])

        if prometheus_port is not None:
            command.extend(["--router-prometheus-port", str(prometheus_port)])

        if prometheus_host is not None:
            command.extend(["--router-prometheus-host", prometheus_host])

        if log_dir is not None:
            command.extend(["--log-dir", log_dir])

        if dp_aware:
            command.append("--router-dp-aware")

        process = subprocess.Popen(command, stdout=None, stderr=None)

        # Wait for health
        start_time = time.perf_counter()
        with requests.Session() as session:
            while time.perf_counter() - start_time < timeout:
                try:
                    resp = session.get(f"{base_url}/health", timeout=5)
                    if resp.status_code == 200:
                        self._routers[key] = process
                        self._router_urls[key] = base_url
                        return base_url
                except requests.RequestException:
                    pass
                time.sleep(2)

        # If startup fails, make sure to terminate the process we spawned
        try:
            process.terminate()
        except Exception:
            pass
        raise TimeoutError("Router failed to start within the timeout period.")

    def start_worker(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
    ) -> str:
        """Start a model worker and return the worker URL.

        We intentionally do not wait for worker health; the router will pick it up
        upon /add_worker and its own health probing.
        """
        if port is None:
            port = _find_available_port()
        url = f"http://{host}:{port}"

        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            host,
            "--port",
            str(port),
            "--base-gpu-id",
            "1",
        ]
        if api_key is not None:
            cmd.extend(["--api-key", api_key])

        p = subprocess.Popen(cmd, stdout=None, stderr=None)
        self._workers[url] = p
        return url

    def cleanup(self) -> None:
        # Terminate workers first
        for url, proc in list(self._workers.items()):
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)
            except Exception:
                pass
            finally:
                self._workers.pop(url, None)

        # Then terminate routers
        for key, proc in list(self._routers.items()):
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)
            except Exception:
                pass
            finally:
                self._routers.pop(key, None)
                self._router_urls.pop(key, None)


@pytest.fixture(scope="session")
def router_manager() -> RouterProcessManager:
    mgr = RouterProcessManager()
    yield mgr
    mgr.cleanup()

