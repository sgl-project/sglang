"""E2E KV size floor (accuracy-style threshold).

Metric: ``GET /server_info`` → ``memory_usage.kv_size_mb``
  = allocated token KV (incl. SWA / DSA / unified) + Mamba/GDN state.
  Weights and CUDA graphs are not included.

Declare on the test class (preferred) or module::

    class TestFoo(CustomTestCase):
        kv_size_thres = 12000
        # multi-runner:
        kv_size_thres = {"h200": 12000, "b200": 18000}

    # module fallback (used by the seeder):
    KV_SIZE_THRES = 12000

After ``popen_launch_server`` / PD worker health: assert ``kv_size_mb >= thres``.
No declaration → no check. Missing GPU key → skip.

Enabled in CI (not AMD). Opt out: ``SGLANG_CHECK_MEMORY_THRESHOLDS=0``.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import sys
import threading
import types
import unittest
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)

CLASS_ATTR = "kv_size_thres"
MODULE_ATTR = "KV_SIZE_THRES"

_CHECKED_PIDS: set[int] = set()
_lock = threading.Lock()

GPU_FAMILY_TOKENS = (
    "gb300",
    "gb200",
    "b200",
    "h200",
    "h100",
    "h20",
    "a100",
    "5090",
    "4090",
    "l40s",
    "l40",
)

FloorOwner = Union[type, types.ModuleType]


def gpu_family_from_text(text: str) -> Optional[str]:
    s = text.lower().replace("_", "-")
    for key in GPU_FAMILY_TOKENS:
        if key in s:
            return key
    if "1-gpu-small" in s:
        return "5090"
    if "1-gpu-large" in s or "2-gpu-large" in s:
        return "h100"
    if re.search(r"4-gpu-h100|deepep-4-gpu-h100", s):
        return "h100"
    if re.search(r"4-gpu-b200|deepep-4-gpu-b200", s):
        return "b200"
    if re.search(r"8-gpu-h200|deepep-8-gpu-h200", s):
        return "h200"
    if re.search(r"8-gpu-b200", s):
        return "b200"
    if re.search(r"8-gpu-h20", s):
        return "h20"
    return None


def detect_gpu_family() -> Optional[str]:
    env = os.environ.get("SGLANG_MEMORY_FLOOR_GPU", "").strip().lower()
    if env:
        return env
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return gpu_family_from_text(torch.cuda.get_device_properties(0).name)
    except Exception:
        return None


def kv_size_mb_from_server_info(info: Dict[str, Any]) -> Optional[float]:
    mem = None
    internal = info.get("internal_states")
    if isinstance(internal, list) and internal and isinstance(internal[0], dict):
        mem = internal[0].get("memory_usage")
    if not isinstance(mem, dict):
        mem = info.get("memory_usage")
    if not isinstance(mem, dict):
        return None
    if mem.get("kv_size_mb") is not None:
        return float(mem["kv_size_mb"])
    # Older servers: sum kvcache + mamba (GB → MB); ignore weight/graph.
    parts = []
    if mem.get("kvcache") is not None:
        parts.append(float(mem["kvcache"]))
    if mem.get("mamba") is not None:
        parts.append(float(mem["mamba"]))
    if not parts:
        return None
    return round(sum(parts) * 1024.0, 1)


def fetch_kv_size_mb(
    base_url: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> Optional[float]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(
        f"{base_url.rstrip('/')}/server_info",
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return kv_size_mb_from_server_info(resp.json())


def _raw_threshold(owner: FloorOwner) -> Any:
    if isinstance(owner, type):
        v = getattr(owner, CLASS_ATTR, None)
        if v is not None:
            return v
        mod = sys.modules.get(owner.__module__)
        return getattr(mod, MODULE_ATTR, None) if mod is not None else None
    return getattr(owner, MODULE_ATTR, None)


def resolve_kv_size_thres(
    spec: Any, *, gpu_family: Optional[str] = None
) -> Optional[float]:
    """Return a single MB floor, or None to skip."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, dict):
        family = gpu_family if gpu_family is not None else detect_gpu_family()
        if family is None:
            logger.warning(
                "kv_size_thres is per-GPU %s but GPU family unknown; skip",
                list(spec.keys()),
            )
            return None
        if family not in spec:
            logger.warning(
                "kv_size_thres keys %s have no entry for %s; skip",
                list(spec.keys()),
                family,
            )
            return None
        return resolve_kv_size_thres(spec[family], gpu_family=family)
    logger.warning("kv_size_thres has unsupported type %s", type(spec).__name__)
    return None


def _owner_label(owner: FloorOwner) -> str:
    if isinstance(owner, type):
        return f"{owner.__module__}.{owner.__qualname__}"
    return getattr(owner, "__name__", repr(owner))


def find_active_test_owner() -> Optional[FloorOwner]:
    for frame_info in inspect.stack(context=0):
        cls = frame_info.frame.f_locals.get("cls")
        if not isinstance(cls, type):
            continue
        try:
            if not issubclass(cls, unittest.TestCase):
                continue
        except TypeError:
            continue
        if _raw_threshold(cls) is not None:
            return cls
    main = sys.modules.get("__main__")
    if main is not None and _raw_threshold(main) is not None:
        return main
    return None


def memory_threshold_check_enabled() -> bool:
    flag = os.environ.get("SGLANG_CHECK_MEMORY_THRESHOLDS", "").lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("SGLANG_IS_IN_CI_AMD", "").lower() in ("1", "true", "yes"):
        return False
    return os.environ.get("SGLANG_IS_IN_CI", "").lower() in ("1", "true", "yes")


def maybe_check_server_memory(
    base_url: str,
    *,
    api_key: Optional[str] = None,
    process: Any = None,
    owner: Optional[FloorOwner] = None,
    kill_processes: Optional[List[Any]] = None,
) -> None:
    """Assert ``kv_size_mb >= kv_size_thres`` when a threshold is set.

    On failure, kill ``kill_processes`` if provided, else ``process``.
    """
    if not memory_threshold_check_enabled():
        return

    pid = getattr(process, "pid", None) if process is not None else None
    if pid is not None:
        with _lock:
            if int(pid) in _CHECKED_PIDS:
                return

    owner = owner or find_active_test_owner()
    if owner is None:
        return
    threshold = resolve_kv_size_thres(_raw_threshold(owner))
    if threshold is None:
        return

    try:
        observed = fetch_kv_size_mb(base_url, api_key=api_key)
    except Exception as e:
        logger.warning("KV size check skipped: /server_info failed (%s)", e)
        return
    if observed is None:
        logger.warning("KV size check skipped: no kv_size_mb in server_info")
        return

    label = _owner_label(owner)
    logger.info(
        "KV size check %s: observed=%.1f threshold=%.1f",
        label,
        observed,
        threshold,
    )
    if observed < threshold:
        victims = kill_processes if kill_processes is not None else [process]
        for proc in victims or []:
            if proc is None:
                continue
            try:
                from sglang.srt.utils import kill_process_tree

                kill_process_tree(proc.pid)
            except Exception as e:
                print(
                    f"Error killing process {getattr(proc, 'pid', None)} "
                    f"after KV size threshold failure: {e}"
                )
        raise AssertionError(
            f"KV size capacity regression ({label}): "
            f"kv_size_mb={observed:g} < kv_size_thres={threshold:g}"
        )
    if pid is not None:
        with _lock:
            _CHECKED_PIDS.add(int(pid))
