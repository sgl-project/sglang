#!/usr/bin/env python3
"""Kill SGLang processes on CUDA_VISIBLE_DEVICES GPUs (CI mode only).

Called at the start of every CI job to clean up orphaned processes from
previous (possibly cancelled) runs. Requires SGLANG_IS_IN_CI=true.

For local/non-CI usage, use scripts/killall_sglang.sh instead.

Usage:
    python killall.py

Exit codes:
    0 - Clean: all target GPUs have <10% memory usage after cleanup
    1 - Dirty: GPU memory still >10% after cleanup, indicating stuck processes
        or orphaned CUDA contexts that need a container restart
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MEMORY_THRESHOLD_PCT = 10


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


def _get_pid_cmdline(pid):
    """Get command line for a PID. Linux-only via /proc."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        cmdline = cmdline.decode("utf-8", errors="replace").replace("\x00", " ").strip()
        # Truncate long command lines
        return cmdline[:120] + ("..." if len(cmdline) > 120 else "")
    except (FileNotFoundError, PermissionError):
        return "<unknown>"


def _kill_pids(pids, label=""):
    """Send SIGKILL to PIDs, skipping self and init. Logs to _LOG_LINES."""
    my_pid = os.getpid()
    pids = {p for p in pids if p != my_pid and p > 1}
    if not pids:
        return
    if label:
        _log(f"  Killing {label}:")
    for pid in sorted(pids):
        cmdline = _get_pid_cmdline(pid)
        _log(f"    PID {pid}: {cmdline}")
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            _log(f"    PID {pid}: failed (already dead or no permission)")


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


def _log_gpu_memory(gpu_indices):
    """Log memory usage for target GPUs. Returns list of dirty GPU descriptions."""
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
        _log(f"  GPU {idx}: {used} MiB / {total} MiB ({pct:.0f}%)")
        if pct >= MEMORY_THRESHOLD_PCT:
            dirty.append(f"GPU {idx} ({pct:.0f}%)")
    return dirty


_LOG_LINES = []


def _log(msg=""):
    """Buffer a line for boxed output."""
    _LOG_LINES.append(msg)


def _flush_box(title, status=""):
    """Print all buffered lines inside a box, then clear buffer."""
    lines = _LOG_LINES.copy()
    _LOG_LINES.clear()

    # Build content width from title, status, and all lines
    all_text = [title] + ([status] if status else []) + lines
    width = max((len(line) for line in all_text), default=40) + 4
    width = max(width, 60)

    h_bar = "─" * (width - 2)
    print(f"\n┌{h_bar}┐")
    print(f"│ {title:<{width - 3}}│")
    print(f"├{h_bar}┤")
    for line in lines:
        print(f"│ {line:<{width - 3}}│")
    if status:
        print(f"├{h_bar}┤")
        print(f"│ {status:<{width - 3}}│")
    print(f"└{h_bar}┘")


def _ci_mode():
    """GPU-scoped kill, abort if GPUs remain dirty."""
    gpu_indices = _get_target_gpus()
    if not gpu_indices:
        _log("No GPUs detected, skipping cleanup")
        _flush_box("killall_sglang", status="SKIP")
        return 0

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_list = ", ".join(str(g) for g in sorted(gpu_indices))

    if cvd is None or not cvd.strip():
        _log(
            "WARNING: CUDA_VISIBLE_DEVICES is not set. "
            "Falling back to all visible GPUs."
        )
        _log("This may kill processes from other CI jobs on shared hosts.")
    else:
        _log(f"CUDA_VISIBLE_DEVICES={cvd}")
    _log()

    # Before cleanup
    _log("Before cleanup:")
    _log_gpu_memory(gpu_indices)
    gpu_pids = _get_gpu_pids(gpu_indices)
    if not gpu_pids:
        _log("  No processes on target GPUs")
    else:
        _log(f"  Processes ({len(gpu_pids)}):")
        for pid in sorted(gpu_pids):
            _log(f"    PID {pid}: {_get_pid_cmdline(pid)}")
    _log()

    # Kill orchestrator ancestors first, then GPU processes (retry once)
    if gpu_pids:
        _kill_pids(_get_orchestrator_ancestors(gpu_pids), "orchestrator ancestors")
        time.sleep(1)
        for attempt in range(2):
            gpu_pids = _get_gpu_pids(gpu_indices)
            if not gpu_pids:
                break
            label = "GPU processes" if attempt == 0 else "stubborn GPU processes"
            _kill_pids(gpu_pids, label)
            time.sleep(3)
    _log()

    # Verify
    _log("After cleanup:")
    dirty = _log_gpu_memory(gpu_indices)
    remaining_pids = _get_gpu_pids(gpu_indices)
    if remaining_pids:
        _log(f"  Remaining processes ({len(remaining_pids)}):")
        for pid in sorted(remaining_pids):
            _log(f"    PID {pid}: {_get_pid_cmdline(pid)}")
    else:
        _log("  No processes on target GPUs")

    if dirty:
        _log()
        _log(f"ERROR: memory >={MEMORY_THRESHOLD_PCT}%: {', '.join(dirty)}")
        _log("Orphaned CUDA contexts — container needs restart.")
        _flush_box(f"killall_sglang: GPUs [{gpu_list}]", status="FAIL — Aborting CI")
        return 1

    _flush_box(f"killall_sglang: GPUs [{gpu_list}]", status="PASS — GPUs clean")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    return _ci_mode()


if __name__ == "__main__":
    sys.exit(main())
