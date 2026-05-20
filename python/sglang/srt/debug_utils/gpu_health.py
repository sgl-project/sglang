"""GPU health snapshot helpers.

When SGLANG_GPU_HEALTH_SNAPSHOT=1, the CI runner captures GPU ECC counters,
thermal, clock, and throttle state before and after each test batch via NVML.
Snapshots land in SGLANG_CUDA_COREDUMP_DIR alongside CUDA coredumps so CI
artifact upload picks them up without extra config.

The hook runs in the CI parent process (not in the sglang server subprocess)
so a subprocess crash / SIGABRT does not lose the post-fault snapshot.
"""

import glob
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    return envs.SGLANG_GPU_HEALTH_SNAPSHOT.get()


def get_dump_dir() -> str:
    return envs.SGLANG_CUDA_COREDUMP_DIR.get()


# (location_label, NVML location enum) pairs we attempt to read.
# Not all locations are present on all GPUs; missing ones land as None.
_ECC_LOCATIONS = [
    ("dram", "NVML_MEMORY_LOCATION_DEVICE_MEMORY"),
    ("l2", "NVML_MEMORY_LOCATION_L2_CACHE"),
    ("sm", "NVML_MEMORY_LOCATION_SM"),
    ("texture", "NVML_MEMORY_LOCATION_TEXTURE_MEMORY"),
    ("register", "NVML_MEMORY_LOCATION_REGISTER_FILE"),
    ("cbu", "NVML_MEMORY_LOCATION_CBU"),
    ("sram", "NVML_MEMORY_LOCATION_SRAM"),
]


def _snap_one(handle, pynvml) -> Dict[str, Any]:
    """Capture per-GPU health counters. Returns a flat dict; values that NVML
    refuses on this hardware are stored as None (kept in output so diffs can
    distinguish 'not supported' from '0')."""
    out: Dict[str, Any] = {}

    for volat_label, volat_enum in (
        ("ecc_volatile", pynvml.NVML_VOLATILE_ECC),
        ("ecc_aggregate", pynvml.NVML_AGGREGATE_ECC),
    ):
        section: Dict[str, Optional[int]] = {}
        for loc_label, loc_enum_name in _ECC_LOCATIONS:
            loc_enum = getattr(pynvml, loc_enum_name, None)
            if loc_enum is None:
                continue
            for err_label, err_enum in (
                ("corrected", pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED),
                ("uncorrected", pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED),
            ):
                key = f"{loc_label}_{err_label}"
                try:
                    section[key] = pynvml.nvmlDeviceGetMemoryErrorCounter(
                        handle, err_enum, volat_enum, loc_enum
                    )
                except pynvml.NVMLError:
                    section[key] = None
        out[volat_label] = section

    def _safe_query(fn, *args, default=None):
        try:
            return fn(handle, *args)
        except pynvml.NVMLError:
            return default

    out["temp_c"] = _safe_query(
        pynvml.nvmlDeviceGetTemperature, pynvml.NVML_TEMPERATURE_GPU
    )
    out["sm_clock_mhz"] = _safe_query(
        pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_SM
    )
    out["mem_clock_mhz"] = _safe_query(
        pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_MEM
    )
    out["throttle_mask"] = _safe_query(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons)
    util = _safe_query(pynvml.nvmlDeviceGetUtilizationRates)
    out["util_gpu"] = util.gpu if util is not None else None
    out["util_mem"] = util.memory if util is not None else None

    name = _safe_query(pynvml.nvmlDeviceGetName)
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    out["gpu_name"] = name
    return out


def snap_all() -> Optional[List[Dict[str, Any]]]:
    """Snapshot all visible GPUs. Returns None when NVML is unavailable."""
    try:
        import pynvml
    except ImportError:
        logger.warning("pynvml unavailable; GPU health snapshot skipped.")
        return None

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        n = pynvml.nvmlDeviceGetCount()
        snapshots = []
        for i in range(n):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            snapshots.append(_snap_one(handle, pynvml))
        return snapshots
    except pynvml.NVMLError as e:
        logger.warning(f"GPU health snapshot failed: {e}")
        return None
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def write_snapshot(label: str) -> Optional[str]:
    """Capture + write a snapshot to the dump dir. Returns the file path,
    or None if NVML was unavailable / the hook is disabled."""
    snap = snap_all()
    if snap is None:
        return None
    dump_dir = get_dump_dir()
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(
        dump_dir,
        f"gpu_health_{label}_{int(time.time())}_{os.getpid()}.json",
    )
    payload = {
        "label": label,
        "pid": os.getpid(),
        "timestamp": time.time(),
        "snapshot": snap,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def cleanup_dump_dir() -> None:
    """Remove stale gpu_health_*.json files from the dump dir."""
    dump_dir = get_dump_dir()
    for f in glob.glob(os.path.join(dump_dir, "gpu_health_*.json")):
        try:
            os.remove(f)
        except OSError:
            pass


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"failed to read {path}: {e}")
        return None


def _ecc_delta(baseline: Dict[str, Any], final: Dict[str, Any]) -> Dict[str, int]:
    """Diff volatile-ECC counters. Only keys where both ends are int and the
    delta is > 0 are returned."""
    delta: Dict[str, int] = {}
    base_vol = baseline.get("ecc_volatile", {})
    fin_vol = final.get("ecc_volatile", {})
    for k, v in fin_vol.items():
        b = base_vol.get(k)
        if isinstance(v, int) and isinstance(b, int):
            d = v - b
            if d > 0:
                delta[k] = d
    return delta


def report() -> None:
    """Print a per-PID baseline-to-final ECC delta summary. Called by the CI
    runner after a test batch finishes."""
    dump_dir = get_dump_dir()
    snaps = sorted(glob.glob(os.path.join(dump_dir, "gpu_health_*.json")))
    if not snaps:
        return

    # Group snapshots by PID; baseline is the earliest tagged 'baseline',
    # final is the latest tagged 'final' (falling back to the last seen
    # snapshot for that PID).
    by_pid: Dict[int, List[Dict[str, Any]]] = {}
    for path in snaps:
        data = _load(path)
        if data is None:
            continue
        data["path"] = path
        by_pid.setdefault(data["pid"], []).append(data)

    print(f"\n{'='*60}")
    print(f"GPU health snapshots ({len(snaps)} file(s)):")
    for path in snaps:
        print(f"  {path}")

    flagged: List[str] = []
    for pid, entries in by_pid.items():
        entries.sort(key=lambda e: e["timestamp"])
        baseline = next((e for e in entries if "baseline" in e["label"]), None)
        final = next(
            (e for e in reversed(entries) if "final" in e["label"]),
            entries[-1],
        )
        if baseline is None or baseline is final:
            continue
        for gpu_idx, (b_gpu, f_gpu) in enumerate(
            zip(baseline["snapshot"], final["snapshot"])
        ):
            delta = _ecc_delta(b_gpu, f_gpu)
            if delta:
                gpu_name = f_gpu.get("gpu_name") or "?"
                flagged.append(f"  pid={pid} GPU{gpu_idx} ({gpu_name}): {delta}")

    if flagged:
        print(f"\n[ECC ACTIVITY DETECTED]")
        for line in flagged:
            print(line)
    else:
        print("\nNo volatile-ECC delta across snapshots.")
    print(f"{'='*60}\n")
