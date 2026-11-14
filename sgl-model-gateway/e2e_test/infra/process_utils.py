"""Process management utilities for E2E tests."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

import requests

logger = logging.getLogger(__name__)


def kill_process_tree(pid: int, sig: int = signal.SIGTERM) -> None:
    """Kill a process and all its children.

    Args:
        pid: Process ID to kill
        sig: Signal to send (default: SIGTERM)
    """
    try:
        import psutil

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        parent.send_signal(sig)
    except ImportError:
        # Fallback if psutil not available
        os.kill(pid, sig)
    except Exception as e:
        logger.warning("Failed to kill process tree for PID %d: %s", pid, e)


def terminate_process(proc: subprocess.Popen, timeout: float = 30) -> None:
    """Gracefully terminate a process, kill if needed.

    Args:
        proc: Process to terminate
        timeout: Seconds to wait before force-killing
    """
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    start = time.perf_counter()
    while proc.poll() is None:
        if time.perf_counter() - start > timeout:
            proc.kill()
            break
        time.sleep(1)


def wait_for_health(
    url: str,
    timeout: float = 60,
    api_key: str | None = None,
    check_interval: float = 1.0,
) -> None:
    """Wait for a server's /health endpoint to return 200.

    Args:
        url: Base URL of the server
        timeout: Seconds to wait before timing out
        api_key: Optional API key for auth header
        check_interval: Seconds between health checks
    """
    start = time.perf_counter()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                resp = session.get(f"{url}/health", headers=headers, timeout=5)
                if resp.status_code == 200:
                    logger.info("Service healthy at %s", url)
                    return
            except requests.RequestException:
                pass
            time.sleep(check_interval)

    raise TimeoutError(f"Server at {url} did not become healthy within {timeout}s")


def wait_for_workers_ready(
    router_url: str,
    expected_workers: int,
    timeout: float = 300,
    api_key: str | None = None,
) -> None:
    """Wait for router to have all workers connected.

    Args:
        router_url: Base URL of the router
        expected_workers: Number of workers to wait for
        timeout: Seconds to wait before timing out
        api_key: Optional API key for auth header
    """
    start = time.perf_counter()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    while time.perf_counter() - start < timeout:
        try:
            resp = requests.get(f"{router_url}/workers", headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                total = data.get("total", len(data.get("workers", [])))
                if total >= expected_workers:
                    logger.info(
                        "All %d workers connected after %.1fs",
                        expected_workers,
                        time.perf_counter() - start,
                    )
                    return
        except requests.RequestException:
            pass
        time.sleep(2)

    raise TimeoutError(
        f"Router at {router_url} did not get {expected_workers} workers within {timeout}s"
    )


def detect_ib_device() -> str | None:
    """Detect first active InfiniBand device usable for PD KV transfer.

    Enumerates `/sys/class/infiniband/` (the canonical device list that
    sglang's `_validate_ib_devices` checks against) and returns the first
    PORT_ACTIVE port. Prioritizes ``mlx5_ib*`` (native InfiniBand) over
    ``mlx5_eth*`` (Ethernet/RoCE) — on the production CI runners both
    families show up under /sys/class/infiniband, but ``mlx5_eth*`` are
    plain Ethernet ports that don't carry the RDMA traffic mooncake
    needs for prefill/decode KV transfer. Picking an eth port led to
    decode-side 500s on every PD MMLU request (worker comes up healthy
    but KV reads fail at request time).

    Avoids the legacy `mlx5_<N>` alias form: on these runners
    ``ibv_devinfo mlx5_0`` resolves to ``mlx5_ib0`` and reports active,
    but sglang's `_validate_ib_devices` walks /sys/class/infiniband and
    rejects the alias name.

    Returns:
        Device name (e.g., ``mlx5_ib0``), or None if nothing usable.
    """
    ib_dir = "/sys/class/infiniband"
    try:
        all_devs = os.listdir(ib_dir)
    except FileNotFoundError:
        return None

    if not all_devs:
        return None

    # Prefer native IB ports over Ethernet/RoCE. Within each family keep
    # /sys ordering stable.
    ib_first = sorted(d for d in all_devs if "_ib" in d)
    eth_last = sorted(d for d in all_devs if "_ib" not in d)
    candidates = ib_first + eth_last

    try:
        subprocess.run(
            ["ibv_devinfo", "-l"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    for dev in candidates:
        try:
            res = subprocess.run(
                ["ibv_devinfo", dev],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode != 0:
                continue
            for line in res.stdout.splitlines():
                if "state:" in line and "PORT_ACTIVE" in line:
                    logger.info("Detected IB device: %s", dev)
                    return dev
        except Exception:
            pass
    return None
