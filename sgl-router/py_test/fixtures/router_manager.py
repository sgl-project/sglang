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

    def add_worker(self, base_url: str, worker_url: str) -> None:
        r = requests.post(f"{base_url}/add_worker", params={"url": worker_url})
        assert r.status_code == 200, f"add_worker failed: {r.status_code} {r.text}"

    def remove_worker(self, base_url: str, worker_url: str) -> None:
        r = requests.post(f"{base_url}/remove_worker", params={"url": worker_url})
        assert r.status_code == 200, f"remove_worker failed: {r.status_code} {r.text}"

    def list_workers(self, base_url: str) -> list[str]:
        r = requests.get(f"{base_url}/list_workers")
        assert r.status_code == 200, f"list_workers failed: {r.status_code} {r.text}"
        data = r.json()
        return data.get("urls", [])

    def stop_all(self):
        for p in self._children:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        self._children.clear()
