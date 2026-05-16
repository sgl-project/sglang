#!/usr/bin/env python3
"""Kill SGLang processes — CI strict mode and local cleanup utility.

Modes (subcommand):
    (no arg) / ci       Strict GPU-scoped cleanup on CUDA_VISIBLE_DEVICES.
                        Verifies GPU memory < 10%, exit 1 on failure.
                        Used at the start of every CI job.
    local               NVIDIA local cleanup: kill sglang processes by name,
                        show nvidia-smi before/after.
    rocm                ROCm cleanup: kill sglang processes by name only.
    all                 NVIDIA: kill sglang processes + ALL GPU users
                        (via lsof on /dev/nvidia*).
    gpus 0,1,2,3        Kill ALL processes on the listed GPU device files.

Usage:
    killall_sglang                  # CI strict mode (default)
    killall_sglang local
    killall_sglang rocm
    killall_sglang all
    killall_sglang gpus 0,1,2,3

Exit codes (ci mode):
    0 - Clean: all target GPUs have <10% memory usage after cleanup
    1 - Dirty: GPU memory still >10% after cleanup, indicating stuck processes
        or orphaned CUDA contexts that need a container restart

Other modes always exit 0 unless a hard prerequisite is missing (e.g.
`gpus` mode without lsof).
"""

import argparse
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Constants
MEMORY_THRESHOLD_PCT = 10

# Patterns matching SGLang process command lines (equivalent to `pgrep -f`).
_SGLANG_PROCESS_PATTERNS = re.compile(
    r"sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::|sglang serve"
)

# Boxed output helpers
_LOG_LINES = []


def _log(msg=""):
    """Buffer a line for boxed output."""
    _LOG_LINES.append(msg)


def _flush_box(title, status=""):
    """Print all buffered lines inside a box, then clear buffer."""
    lines = _LOG_LINES.copy()
    _LOG_LINES.clear()

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


# nvidia-smi helpers
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


def _get_smi_version():
    """Return nvidia-smi driver version and GPU name, or None on failure."""
    # Inline nvidia-smi query — killall.py runs before pip install, so sglang
    # internals may not be importable.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        driver = result.stdout.strip().split("\n")[0].strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        driver = None
    if driver is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        gpu_name = out.strip().splitlines()[0].strip() if out.strip() else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError, IndexError):
        gpu_name = "unknown"
    return f"driver {driver}, {gpu_name}"


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


def _get_gpu_memory(gpu_indices):
    """Query memory usage for target GPUs.

    Returns list of (idx, used_mib, total_mib, pct) tuples.
    """
    result = []
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
        result.append((idx, used, total, pct))
    return result


def _get_dirty_gpus(gpu_indices):
    """Return list of dirty GPU description strings (memory >= threshold)."""
    return [
        f"GPU {idx} ({pct:.0f}%)"
        for idx, _, _, pct in _get_gpu_memory(gpu_indices)
        if pct >= MEMORY_THRESHOLD_PCT
    ]


def _log_gpu_memory(gpu_indices):
    """Log memory usage for all target GPUs and return dirty GPU descriptions."""
    dirty = []
    for idx, used, total, pct in _get_gpu_memory(gpu_indices):
        _log(f"  GPU {idx}: {used} MiB / {total} MiB ({pct:.0f}%)")
        if pct >= MEMORY_THRESHOLD_PCT:
            dirty.append(f"GPU {idx} ({pct:.0f}%)")
    return dirty


# /proc helpers
def _read_proc_cmdline(pid):
    """Read /proc/{pid}/cmdline and return as decoded string, or None on failure."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.decode("utf-8", errors="replace").replace("\x00", " ")
    except (FileNotFoundError, PermissionError):
        return None


def _get_pid_cmdline(pid):
    """Get truncated command line for a PID."""
    cmdline = _read_proc_cmdline(pid)
    if cmdline is None:
        return "<unknown>"
    cmdline = cmdline.strip()
    return cmdline[:120] + ("..." if len(cmdline) > 120 else "")


def _find_sglang_pids_by_name():
    """Find SGLang process PIDs by command-line pattern matching.

    Scans /proc/*/cmdline for patterns matching known SGLang entry points.
    Equivalent to: pgrep -f 'sglang::|sglang.launch_server|...'

    Safe in shared-GPU containers: without --pid=host, /proc only exposes
    processes in our own PID namespace, so this cannot kill other containers.
    """
    my_pid = os.getpid()
    pids = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid <= 1 or pid == my_pid:
            continue
        cmdline = _read_proc_cmdline(pid)
        if cmdline and _SGLANG_PROCESS_PATTERNS.search(cmdline):
            pids.add(pid)
    return pids


def _check_pid_namespace(pid):
    """Check if a PID is in our PID namespace. Linux-only via /proc."""
    try:
        my_ns = os.readlink("/proc/self/ns/pid")
    except OSError:
        return "unknown (can't read self ns)"
    try:
        target_ns = os.readlink(f"/proc/{pid}/ns/pid")
    except FileNotFoundError:
        return f"NOT in our namespace (pid not in /proc, self={my_ns})"
    except PermissionError:
        return "unknown (no permission to read ns)"
    if my_ns == target_ns:
        return f"same namespace ({my_ns})"
    return f"DIFFERENT namespace (self={my_ns}, target={target_ns})"


def _get_orchestrator_ancestors(pids):
    """Walk process tree upward from PIDs, return ancestors that are test orchestrators.

    Linux-only: reads /proc filesystem. Returns empty set on other platforms.
    """
    orchestrator_patterns = ["run_suite.py", "run_tests.py"]
    ancestors, visited = set(), set()
    for pid in pids:
        current = pid
        while current > 1 and current not in visited:
            visited.add(current)
            cmdline = _read_proc_cmdline(current)
            if cmdline is None:
                break
            if any(p in cmdline for p in orchestrator_patterns):
                ancestors.add(current)
            try:
                current = int(Path(f"/proc/{current}/stat").read_text().split()[3])
            except (FileNotFoundError, PermissionError, IndexError, ValueError):
                break
    return ancestors


# Kill & diagnostic helpers
def _kill_pids(pids, label="", quiet=False):
    """Send SIGKILL to PIDs, skipping self and init.

    Returns dict of {pid: exception_name} for PIDs that could not be killed.
    When quiet=True, does not log individual kill results.
    """
    my_pid = os.getpid()
    pids = {p for p in pids if p != my_pid and p > 1}
    if not pids:
        return {}
    if label and not quiet:
        _log(f"  Killing {label}:")
    failed = {}
    for pid in sorted(pids):
        try:
            os.kill(pid, signal.SIGKILL)
            if not quiet:
                _log(f"    PID {pid}: killed ({_get_pid_cmdline(pid)})")
        except (ProcessLookupError, PermissionError) as e:
            failed[pid] = type(e).__name__
            if not quiet:
                _log(f"    PID {pid}: failed ({type(e).__name__})")
    return failed


def _get_ps_diagnostic():
    """Return ps auxf output filtered for GPU/sglang-related processes."""
    try:
        out = subprocess.run(["ps", "auxf"], capture_output=True, text=True, timeout=5)
        return [
            line.strip()[:140]
            for line in out.stdout.splitlines()
            if any(k in line.lower() for k in ["sglang", "python", "cuda", "gpu"])
        ][:20]
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def _print_diagnostics(unkillable_pids):
    """Print detailed diagnostics after the FAIL box (to stdout, outside box)."""
    if unkillable_pids:
        print("\n[killall] Diagnostic — unkillable PIDs:")
        for pid in sorted(unkillable_pids):
            ns_info = _check_pid_namespace(pid)
            print(f"  PID {pid}: ns: {ns_info}")
    ps_lines = _get_ps_diagnostic()
    if ps_lines:
        print("\n[killall] Diagnostic — processes in this container (ps auxf):")
        for line in ps_lines:
            print(f"  {line}")
    else:
        print(
            "\n[killall] Diagnostic — no sglang/python/gpu processes "
            "in this container"
        )


# CI mode
def _kill_all_targets(gpu_indices, gpu_pids):
    """Kill all target processes: name-matched, orchestrator ancestors, GPU processes."""
    # Kill name-matched SGLang processes (catches processes not visible to nvidia-smi)
    name_only = _find_sglang_pids_by_name() - gpu_pids
    if name_only:
        _kill_pids(name_only, "name-matched SGLang processes")
        time.sleep(1)
        _log()

    # Kill orchestrator ancestors first, then GPU processes (retry once)
    if gpu_pids:
        _kill_pids(_get_orchestrator_ancestors(gpu_pids), "orchestrator ancestors")
        time.sleep(1)
        for attempt in range(2):
            current_pids = _get_gpu_pids(gpu_indices)
            if not current_pids:
                break
            label = "GPU processes" if attempt == 0 else "stubborn GPU processes"
            _kill_pids(current_pids, label)
            time.sleep(3)
    _log()


def _verify_gpu_clean(gpu_indices):
    """Retry loop: wait for GPUs to become clean.

    Returns (dirty_list, unkillable_pids, elapsed_seconds).
    """
    max_wait_secs = 100
    retry_interval = 10
    elapsed = 0
    dirty = None
    unkillable_pids = {}

    while True:
        dirty = _get_dirty_gpus(gpu_indices)
        remaining_pids = _get_gpu_pids(gpu_indices)

        if not dirty:
            _log(f"Check at {elapsed}s: GPUs clean")
            break

        dirty_summary = ", ".join(dirty)

        if elapsed >= max_wait_secs:
            remaining_info = (
                f", {len(remaining_pids)} processes remaining" if remaining_pids else ""
            )
            _log(f"Check at {elapsed}s: still dirty [{dirty_summary}]{remaining_info}")
            break

        # Kill remaining processes before waiting (silently for retries)
        if remaining_pids:
            failed = _kill_pids(remaining_pids, quiet=True)
            unkillable_pids.update(failed)

        print(
            f"[killall] GPUs still dirty at {elapsed}s [{dirty_summary}], "
            f"retrying in {retry_interval}s "
            f"({elapsed + retry_interval}/{max_wait_secs}s)..."
        )
        time.sleep(retry_interval)
        elapsed += retry_interval

    if unkillable_pids:
        parts = [f"{p} ({unkillable_pids[p]})" for p in sorted(unkillable_pids)]
        _log(f"  Unkillable PIDs: {', '.join(parts)}")

    return dirty, unkillable_pids, elapsed


def _ci_mode():
    """GPU-scoped kill, abort if GPUs remain dirty."""
    gpu_indices = _get_target_gpus()
    if not gpu_indices:
        _log("No GPUs detected, skipping cleanup")
        _flush_box("killall_sglang", status="SKIP")
        return 0

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_list = ", ".join(str(g) for g in sorted(gpu_indices))

    smi_info = _get_smi_version()
    if smi_info:
        _log(f"nvidia-smi: {smi_info}")
    if cvd is None or not cvd.strip():
        _log(
            "WARNING: CUDA_VISIBLE_DEVICES is not set. "
            "Falling back to all visible GPUs."
        )
        _log("This may kill processes from other CI jobs on shared hosts.")
    else:
        _log(f"CUDA_VISIBLE_DEVICES={cvd}")
    _log()

    # Log pre-cleanup state
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

    # Kill phase
    _kill_all_targets(gpu_indices, gpu_pids)

    # Verify phase
    dirty, unkillable_pids, elapsed = _verify_gpu_clean(gpu_indices)

    if dirty:
        _log()
        _log("Final GPU memory:")
        _log_gpu_memory(gpu_indices)
        _log(f"ERROR: memory >={MEMORY_THRESHOLD_PCT}%: {', '.join(dirty)}")
        _log(f"Orphaned CUDA contexts after {elapsed}s — container needs restart.")
        _flush_box(f"killall_sglang: GPUs [{gpu_list}]", status="FAIL — Aborting CI")
        _print_diagnostics(unkillable_pids)
        return 1

    _flush_box(f"killall_sglang: GPUs [{gpu_list}]", status="PASS — GPUs clean")
    return 0


# Local cleanup modes
def _show_nvidia_smi_table():
    """Print full nvidia-smi output directly to stdout (outside any box)."""
    try:
        subprocess.run(["nvidia-smi"], timeout=10)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("[killall] nvidia-smi unavailable")


def _ensure_lsof():
    """Make sure `lsof` is available; try apt-get install if missing.

    Returns True if lsof can be used after this call, False otherwise.
    """
    if shutil.which("lsof"):
        return True
    print("[killall] lsof not found, attempting apt-get install...")
    cmd_prefix = ["sudo"] if shutil.which("sudo") else []
    try:
        subprocess.run(
            cmd_prefix + ["apt-get", "update"],
            capture_output=True,
            timeout=120,
        )
        subprocess.run(
            cmd_prefix + ["apt-get", "install", "-y", "lsof"],
            capture_output=True,
            timeout=180,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"[killall] apt-get install failed: {type(e).__name__}")
        return False
    return shutil.which("lsof") is not None


def _find_pids_via_lsof_devices(device_paths):
    """Run `lsof <device_paths>` and return PIDs holding these device files.

    Glob patterns (e.g. `/dev/nvidia*`) are expanded locally; missing devices
    are silently skipped.
    """
    expanded = []
    for p in device_paths:
        if any(c in p for c in "*?["):
            expanded.extend(glob.glob(p))
        elif Path(p).exists():
            expanded.append(p)
    if not expanded:
        return set()
    try:
        out = subprocess.run(
            ["lsof", *expanded],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return set()
    pids = set()
    # Skip header; field 2 is PID.
    for line in out.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            pids.add(int(parts[1]))
    return pids


def _local_mode():
    """NVIDIA local cleanup: kill sglang processes by name, show nvidia-smi."""
    print("[killall] Before cleanup:")
    _show_nvidia_smi_table()
    print()

    pids = _find_sglang_pids_by_name()
    if pids:
        _kill_pids(pids, "sglang processes")
    else:
        _log("No sglang processes found")
    _flush_box("killall_sglang: local")

    print("\n[killall] After cleanup:")
    _show_nvidia_smi_table()
    return 0


def _rocm_mode():
    """ROCm cleanup: kill sglang processes by name (no nvidia-smi)."""
    _log("Running in ROCm mode")
    pids = _find_sglang_pids_by_name()
    if pids:
        _kill_pids(pids, "sglang processes")
    else:
        _log("No sglang processes found")
    _flush_box("killall_sglang: rocm")
    return 0


def _all_mode():
    """NVIDIA: kill sglang processes + ALL GPU users via lsof on /dev/nvidia*."""
    print("[killall] Before cleanup:")
    _show_nvidia_smi_table()
    print()

    sglang_pids = _find_sglang_pids_by_name()
    if sglang_pids:
        _kill_pids(sglang_pids, "sglang processes")

    if _ensure_lsof():
        other_pids = _find_pids_via_lsof_devices(["/dev/nvidia*"]) - sglang_pids
        if other_pids:
            _kill_pids(other_pids, "all GPU users")
    else:
        _log("WARNING: lsof unavailable, only sglang processes were killed")

    _flush_box("killall_sglang: all")

    print("\n[killall] After cleanup:")
    _show_nvidia_smi_table()
    return 0


def _gpus_mode(devices_str):
    """Kill all processes on the listed GPU device files (e.g. '0,1,2,3')."""
    indices = [d.strip() for d in devices_str.split(",") if d.strip()]
    if not indices:
        print(f"[killall] ERROR: invalid devices spec '{devices_str}'")
        return 1
    device_paths = [f"/dev/nvidia{d}" for d in indices]

    print(f"[killall] Killing all processes on GPUs: {','.join(indices)}")
    print(f"[killall] Targeting devices: {' '.join(device_paths)}")
    print()
    _show_nvidia_smi_table()
    print()

    if not _ensure_lsof():
        print("[killall] ERROR: lsof unavailable, cannot continue")
        return 1

    pids = _find_pids_via_lsof_devices(device_paths)
    if pids:
        _kill_pids(pids, f"processes on devices {','.join(indices)}")
    else:
        _log("No processes on target devices")
    _flush_box(f"killall_sglang: gpus [{','.join(indices)}]")

    print("\n[killall] After cleanup:")
    _show_nvidia_smi_table()
    return 0


# Entry point
def main():
    parser = argparse.ArgumentParser(
        prog="killall_sglang",
        description="Kill SGLang processes (CI strict mode + local cleanup utility).",
    )
    subparsers = parser.add_subparsers(dest="mode")
    subparsers.add_parser(
        "ci",
        help="(default) CI strict mode: GPU-scoped kill, verify, exit 1 on dirty.",
    )
    subparsers.add_parser(
        "local",
        help="NVIDIA local cleanup: kill sglang processes by name.",
    )
    subparsers.add_parser(
        "rocm",
        help="ROCm cleanup: kill sglang processes by name only.",
    )
    subparsers.add_parser(
        "all",
        help="NVIDIA: kill sglang processes + ALL GPU users via lsof on /dev/nvidia*.",
    )
    sub_gpus = subparsers.add_parser(
        "gpus",
        help="Kill all processes on specific GPU devices.",
    )
    sub_gpus.add_argument(
        "devices",
        help="comma-separated GPU indices, e.g. 0,1,2,3",
    )

    args = parser.parse_args()
    mode = args.mode or "ci"

    if mode == "ci":
        return _ci_mode()
    if mode == "local":
        return _local_mode()
    if mode == "rocm":
        return _rocm_mode()
    if mode == "all":
        return _all_mode()
    if mode == "gpus":
        return _gpus_mode(args.devices)
    return 1


if __name__ == "__main__":
    sys.exit(main())
