#!/usr/bin/env python3
"""Kill SGLang and related processes.

In CI mode (SGLANG_IS_IN_CI=true): GPU-scoped kill via nvidia-smi,
scoped to CUDA_VISIBLE_DEVICES (numeric indices only). Aborts with
exit code 1 if GPU memory remains >10% after cleanup.

In local mode: pgrep-based kill of SGLang processes (same as the
original killall_sglang.sh default behavior).

Usage:
    python killall.py

Exit codes:
    0 - Clean (or local mode completed)
    1 - CI mode only: GPU memory still >10% after cleanup
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MEMORY_THRESHOLD_PCT = 10

SGLANG_PROCESS_PATTERN = (
    r"sglang::|sglang\.launch_server|sglang\.bench"
    r"|sglang\.data_parallel|sglang\.srt|sgl_diffusion::"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_smi(query, query_type="gpu"):
    """Run nvidia-smi query and return raw CSV lines."""
    flag = "--query-gpu" if query_type == "gpu" else "--query-compute-apps"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        return [line.strip() for line in out.strip().splitlines() if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def _kill_pids(pids, label=""):
    """Send SIGKILL to PIDs, skipping self and init."""
    my_pid = os.getpid()
    pids = {p for p in pids if p != my_pid and p > 1}
    if not pids:
        return
    if label:
        print(f"  Killing {label}: {sorted(pids)}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# ---------------------------------------------------------------------------
# CI mode helpers
# ---------------------------------------------------------------------------


def _get_target_gpus():
    """Return GPU indices from CUDA_VISIBLE_DEVICES, or all visible GPUs.

    Note: only numeric indices are supported (e.g. "0,1,2").
    UUID-style CUDA_VISIBLE_DEVICES values (e.g. "GPU-d4f1...") are not handled.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return {int(g.strip()) for g in cvd.split(",") if g.strip().isdigit()}
    return {int(line) for line in _run_smi("index") if line.isdigit()}


def _get_gpu_pids(gpu_indices):
    """Return PIDs using the specified GPUs (by index)."""
    target_uuids = set()
    for line in _run_smi("index,uuid"):
        parts = line.split(",", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            if int(parts[0].strip()) in gpu_indices:
                target_uuids.add(parts[1].strip())
    pids = set()
    for line in _run_smi("gpu_uuid,pid", query_type="apps"):
        parts = line.split(",", 1)
        if len(parts) == 2 and parts[0].strip() in target_uuids:
            pid = parts[1].strip()
            if pid.isdigit():
                pids.add(int(pid))
    return pids


def _print_gpu_memory(gpu_indices, label=""):
    """Print memory usage for target GPUs. Returns list of dirty GPU descriptions."""
    if label:
        print(f"\n{label}")
    dirty = []
    for line in _run_smi("index,memory.used,memory.total"):
        parts = line.split(",")
        if len(parts) != 3 or not parts[0].strip().isdigit():
            continue
        idx = int(parts[0].strip())
        if idx not in gpu_indices:
            continue
        try:
            used, total = int(float(parts[1].strip())), int(float(parts[2].strip()))
        except ValueError:
            continue
        pct = used / total * 100 if total > 0 else 0
        print(f"  GPU {idx}: {used} MiB / {total} MiB ({pct:.0f}%)")
        if pct >= MEMORY_THRESHOLD_PCT:
            dirty.append(f"GPU {idx} ({pct:.0f}%)")
    return dirty


def _get_orchestrator_ancestors(pids):
    """Walk process tree upward from PIDs, return ancestors that are test orchestrators.

    Linux-only: reads /proc filesystem. Returns empty set on other platforms.
    """
    patterns = ["run_suite.py", "run_tests.py"]
    ancestors, visited = set(), set()
    for pid in pids:
        current = pid
        while current > 1 and current not in visited:
            visited.add(current)
            try:
                cmdline = Path(f"/proc/{current}/cmdline").read_bytes()
                cmdline = cmdline.decode("utf-8", errors="replace").replace("\x00", " ")
                if any(p in cmdline for p in patterns):
                    ancestors.add(current)
            except (FileNotFoundError, PermissionError):
                break
            try:
                current = int(Path(f"/proc/{current}/stat").read_text().split()[3])
            except (FileNotFoundError, PermissionError, IndexError, ValueError):
                break
    return ancestors


# ---------------------------------------------------------------------------
# CI mode
# ---------------------------------------------------------------------------


def _ci_mode():
    """GPU-scoped kill, abort if GPUs remain dirty."""
    gpu_indices = _get_target_gpus()
    if not gpu_indices:
        print("No GPUs detected, skipping cleanup")
        return 0

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or not cvd.strip():
        print(
            "WARNING: CUDA_VISIBLE_DEVICES is not set in CI mode. "
            "Falling back to all visible GPUs — this may kill processes "
            "from other CI jobs on shared hosts."
        )
    print(f"[CI mode] Target GPUs: {sorted(gpu_indices)}")
    if cvd is not None:
        print(f"CUDA_VISIBLE_DEVICES={cvd}")

    _print_gpu_memory(gpu_indices, "Before cleanup:")

    # Kill orchestrator ancestors first, then GPU processes (retry once)
    gpu_pids = _get_gpu_pids(gpu_indices)
    if not gpu_pids:
        print("  No processes found on target GPUs")
    else:
        _kill_pids(_get_orchestrator_ancestors(gpu_pids), "orchestrator ancestors")
        time.sleep(1)
        for attempt in range(2):
            gpu_pids = _get_gpu_pids(gpu_indices)
            if not gpu_pids:
                break
            label = "GPU processes" if attempt == 0 else "stubborn GPU processes"
            _kill_pids(gpu_pids, label)
            time.sleep(3)

    # Verify
    dirty = _print_gpu_memory(gpu_indices, "After cleanup:")
    if dirty:
        print(
            f"\nERROR: GPU memory >={MEMORY_THRESHOLD_PCT}% after cleanup: "
            f"{', '.join(dirty)}"
        )
        print("Orphaned CUDA contexts — container likely needs restart. Aborting CI.")
        return 1
    print("\nGPUs clean.")
    return 0


# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------


def _local_mode():
    """pgrep-based kill of SGLang processes (original killall_sglang.sh behavior)."""
    # Show current GPU status (if nvidia-smi available)
    subprocess.run(["nvidia-smi"], capture_output=False, check=False)

    # Find and kill SGLang processes
    try:
        result = subprocess.run(
            ["pgrep", "-f", SGLANG_PROCESS_PATTERN],
            capture_output=True,
            text=True,
        )
        pids = {
            int(p.strip())
            for p in result.stdout.strip().splitlines()
            if p.strip().isdigit()
        }
    except FileNotFoundError:
        print("pgrep not found, skipping process cleanup")
        return 0

    if pids:
        _kill_pids(pids, "SGLang processes")
    else:
        print("No SGLang processes found")

    # Show GPU status after cleanup
    subprocess.run(["nvidia-smi"], capture_output=False, check=False)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    is_ci = os.environ.get("SGLANG_IS_IN_CI", "").lower() in ("true", "1")
    if is_ci:
        return _ci_mode()
    else:
        return _local_mode()


if __name__ == "__main__":
    sys.exit(main())
