import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from .ports import find_free_port


@dataclass
class ProcHandle:
    process: subprocess.Popen
    url: str


class RouterManager:
    """Helper to spawn a router process and interact with admin endpoints."""

    def __init__(self):
        self._children: List[subprocess.Popen] = []

    def start_router(
        self,
        worker_urls: Optional[List[str]] = None,
        policy: str = "round_robin",
        port: Optional[int] = None,
        extra: Optional[Dict] = None,
        # PD options
        pd_disaggregation: bool = False,
        prefill_urls: Optional[List[tuple]] = None,
        decode_urls: Optional[List[str]] = None,
        prefill_policy: Optional[str] = None,
        decode_policy: Optional[str] = None,
    ) -> ProcHandle:
        worker_urls = worker_urls or []
        port = port or find_free_port()
        cmd = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--policy",
            policy,
        ]
        # Avoid Prometheus port collisions by assigning a free port per router
        prom_port = find_free_port()
        cmd.extend(
            ["--prometheus-port", str(prom_port), "--prometheus-host", "127.0.0.1"]
        )
        if worker_urls:
            cmd.extend(["--worker-urls", *worker_urls])

        # PD routing configuration
        if pd_disaggregation:
            cmd.append("--pd-disaggregation")
            if prefill_urls:
                for url, bport in prefill_urls:
                    if bport is None:
                        cmd.extend(["--prefill", url, "none"])
                    else:
                        cmd.extend(["--prefill", url, str(bport)])
            if decode_urls:
                for url in decode_urls:
                    cmd.extend(["--decode", url])
            if prefill_policy:
                cmd.extend(["--prefill-policy", prefill_policy])
            if decode_policy:
                cmd.extend(["--decode-policy", decode_policy])

        # Map supported extras to CLI flags (subset for integration)
        if extra:
            flag_map = {
                "max_payload_size": "--max-payload-size",
                "dp_aware": "--dp-aware",
                "api_key": "--api-key",
                # Health/monitoring
                "worker_startup_check_interval": "--worker-startup-check-interval",
                # Loads/monitoring
                "worker_load_check_interval": "--worker-load-check-interval",
                # Cache-aware tuning
                "cache_threshold": "--cache-threshold",
                "balance_abs_threshold": "--balance-abs-threshold",
                "balance_rel_threshold": "--balance-rel-threshold",
                # Retry
                "retry_max_retries": "--retry-max-retries",
                "retry_initial_backoff_ms": "--retry-initial-backoff-ms",
                "retry_max_backoff_ms": "--retry-max-backoff-ms",
                "retry_backoff_multiplier": "--retry-backoff-multiplier",
                "retry_jitter_factor": "--retry-jitter-factor",
                "disable_retries": "--disable-retries",
                # Circuit breaker
                "cb_failure_threshold": "--cb-failure-threshold",
                "cb_success_threshold": "--cb-success-threshold",
                "cb_timeout_duration_secs": "--cb-timeout-duration-secs",
                "cb_window_duration_secs": "--cb-window-duration-secs",
                "disable_circuit_breaker": "--disable-circuit-breaker",
                # Rate limiting
                "max_concurrent_requests": "--max-concurrent-requests",
                "queue_size": "--queue-size",
                "queue_timeout_secs": "--queue-timeout-secs",
                "rate_limit_tokens_per_second": "--rate-limit-tokens-per-second",
                # mTLS configuration
                "client_cert_path": "--client-cert-path",
                "client_key_path": "--client-key-path",
                "ca_cert_paths": "--ca-cert-paths",
            }
            for k, v in extra.items():
                if v is None:
                    continue
                flag = flag_map.get(k)
                if not flag:
                    continue
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag)
                elif isinstance(v, list):
                    # Handle list arguments (e.g., ca_cert_paths)
                    if v:  # Only add if list is not empty
                        cmd.append(flag)
                        cmd.extend([str(item) for item in v])
                else:
                    cmd.extend([flag, str(v)])

        proc = subprocess.Popen(cmd)
        self._children.append(proc)
        url = f"http://127.0.0.1:{port}"
        self._wait_health(url)
        return ProcHandle(process=proc, url=url)

    def _wait_health(self, base_url: str, timeout: float = 30.0):
        start = time.time()
        with requests.Session() as s:
            while time.time() - start < timeout:
                try:
                    r = s.get(f"{base_url}/health", timeout=2)
                    if r.status_code == 200:
                        return
                except requests.RequestException:
                    pass
                time.sleep(0.2)
        raise TimeoutError(f"Router at {base_url} did not become healthy")

    def add_worker(self, base_url: str, worker_url: str, timeout: float = 30.0) -> None:
        r = requests.post(f"{base_url}/workers", json={"url": worker_url})
        assert (
            r.status_code == 202
        ), f"add_worker failed: {r.status_code} {r.text}"  # ACCEPTED status

        payload = r.json()
        worker_id = payload.get("worker_id")
        assert worker_id, f"add_worker did not return worker_id: {payload}"

        # Poll until worker is actually added and healthy
        start = time.time()
        with requests.Session() as s:
            while time.time() - start < timeout:
                try:
                    r = s.get(f"{base_url}/workers/{worker_id}", timeout=2)
                    if r.status_code == 200:
                        data = r.json()
                        # Check if registration job failed
                        job_status = data.get("job_status")
                        if job_status and job_status.get("state") == "failed":
                            raise RuntimeError(
                                f"Worker registration failed: {job_status.get('message', 'Unknown error')}"
                            )
                        # Check if worker is healthy and registered (not just in job queue)
                        if data.get("is_healthy", False):
                            return
                    # Worker not ready yet, continue polling
                except requests.RequestException:
                    pass
                time.sleep(0.1)
        raise TimeoutError(
            f"Worker {worker_url} was not added and healthy after {timeout}s"
        )

    def remove_worker(
        self, base_url: str, worker_url: str, timeout: float = 30.0
    ) -> None:
        # Resolve worker_id from the current registry snapshot
        r_list = requests.get(f"{base_url}/workers")
        assert (
            r_list.status_code == 200
        ), f"list_workers failed: {r_list.status_code} {r_list.text}"
        workers = r_list.json().get("workers", [])
        worker_id = next(
            (w.get("id") for w in workers if w.get("url") == worker_url), None
        )
        assert (
            worker_id
        ), f"could not find worker_id for url={worker_url}. workers={workers}"

        r = requests.delete(f"{base_url}/workers/{worker_id}")
        assert (
            r.status_code == 202
        ), f"remove_worker failed: {r.status_code} {r.text}"  # ACCEPTED status

        # Poll until worker is actually removed (GET returns 404) or timeout
        start = time.time()
        last_status = None
        with requests.Session() as s:
            while time.time() - start < timeout:
                try:
                    r = s.get(f"{base_url}/workers/{worker_id}", timeout=2)
                    if r.status_code == 404:
                        # Worker successfully removed
                        return
                    elif r.status_code == 200:
                        # Check if removal job failed
                        data = r.json()
                        job_status = data.get("job_status")
                        if job_status:
                            last_status = job_status
                            if job_status.get("state") == "failed":
                                raise RuntimeError(
                                    f"Worker removal failed: {job_status.get('message', 'Unknown error')}"
                                )
                    # Worker still being processed, continue polling
                except requests.RequestException:
                    pass
                time.sleep(0.1)

        # Provide detailed timeout error with last known status
        error_msg = f"Worker {worker_url} was not removed after {timeout}s"
        if last_status:
            error_msg += f". Last job status: {last_status}"
        raise TimeoutError(error_msg)

    def list_workers(self, base_url: str) -> list[str]:
        r = requests.get(f"{base_url}/workers")
        assert r.status_code == 200, f"list_workers failed: {r.status_code} {r.text}"
        data = r.json()
        # Extract URLs from WorkerInfo objects
        workers = data.get("workers", [])
        return [w["url"] for w in workers]

    def stop_all(self):
        for p in self._children:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        self._children.clear()
