"""Where a worker's host memory sits, by component and by kind.

Debug aid behind ``SGLANG_DIFFUSION_DEBUG_HOST_MEMORY``: on a shared
host/device pool every anonymous byte the runtime keeps is a byte the page
cache cannot hold, so the breakdown says what to cut.
"""

from __future__ import annotations

import gc
import logging
from bisect import bisect_right
from collections.abc import Mapping

import torch

logger = logging.getLogger(__name__)

GIB = 1024**3


def _file_backed_ranges() -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    try:
        with open("/proc/self/maps") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) < 6 or fields[5].startswith("["):
                    continue
                start, end = fields[0].split("-")
                ranges.append((int(start, 16), int(end, 16)))
    except OSError:
        return []
    ranges.sort()
    return ranges


def _kind(
    tensor: torch.Tensor, starts: list[int], ranges: list[tuple[int, int]]
) -> str:
    if tensor.is_pinned():
        return "pinned"
    ptr = tensor.data_ptr()
    index = bisect_right(starts, ptr) - 1
    if index >= 0 and ranges[index][0] <= ptr < ranges[index][1]:
        return "mapped"
    return "anonymous"


def _smaps_rollup() -> dict[str, float]:
    totals: dict[str, float] = {}
    try:
        with open("/proc/self/smaps_rollup") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in (
                    "Rss",
                    "Anonymous",
                    "Rss_File",
                    "Rss_Shmem",
                    "Private_Dirty",
                ):
                    totals[key] = int(rest.split()[0]) * 1024 / GIB
    except OSError:
        pass
    return totals


def host_memory_breakdown(modules: Mapping[str, object]) -> dict[str, dict[str, float]]:
    """GiB of CPU tensor storage per component and kind; ``other`` covers
    tensors no module owns (staging buffers, caches, activations kept alive)."""
    ranges = _file_backed_ranges()
    starts = [start for start, _ in ranges]
    owners: dict[int, str] = {}
    for name, module in modules.items():
        if not isinstance(module, torch.nn.Module):
            continue
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.device.type == "cpu":
                owners[tensor.untyped_storage().data_ptr()] = name
    seen: set[int] = set()
    table: dict[str, dict[str, float]] = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor) or obj.device.type != "cpu":
            continue
        try:
            storage = obj.untyped_storage()
            key = storage.data_ptr()
            nbytes = storage.nbytes()
        except Exception:
            continue
        if key in seen or nbytes == 0:
            continue
        seen.add(key)
        owner = owners.get(key, "other")
        kind = _kind(obj, starts, ranges)
        table.setdefault(owner, {})
        table[owner][kind] = table[owner].get(kind, 0.0) + nbytes / GIB
    return table


def log_host_memory_breakdown(modules: Mapping[str, object], *, label: str) -> None:
    table = host_memory_breakdown(modules)
    rollup = _smaps_rollup()
    lines = [f"Host memory breakdown ({label}):"]
    if rollup:
        lines.append(
            "  process: "
            + " ".join(f"{key}={value:.2f}GiB" for key, value in sorted(rollup.items()))
        )
    for owner in sorted(table, key=lambda name: -sum(table[name].values())):
        kinds = " ".join(
            f"{kind}={value:.2f}GiB" for kind, value in sorted(table[owner].items())
        )
        lines.append(f"  {owner}: {kinds}")
    device = torch.get_device_module()
    if hasattr(device, "memory_allocated"):
        lines.append(
            f"  device: allocated={device.memory_allocated() / GIB:.2f}GiB "
            f"reserved={device.memory_reserved() / GIB:.2f}GiB"
        )
    snapshot = getattr(device, "memory_snapshot", None)
    if callable(snapshot):
        try:
            segments = snapshot()
        except Exception:
            segments = []
        big = sorted(
            (
                (int(segment.get("address", 0)), int(segment.get("total_size", 0)))
                for segment in segments
                if int(segment.get("total_size", 0)) >= 256 * 1024**2
            ),
            key=lambda item: -item[1],
        )[:12]
        lines.append(
            "  device segments >= 256 MiB: "
            + ", ".join(f"{address:#x}:{size / GIB:.2f}GiB" for address, size in big)
        )
    logger.info("\n".join(lines))
