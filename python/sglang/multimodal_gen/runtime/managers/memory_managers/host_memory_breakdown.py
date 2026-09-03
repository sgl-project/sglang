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


def _anon_vmas(min_bytes: int = 128 * 1024**2) -> list[tuple[int, int, int, str]]:
    """(start, end, anonymous_bytes, vmflags) of anonymous mappings holding at least min_bytes."""
    out: list[tuple[int, int, int, str]] = []
    try:
        start = end = 0
        path = ""
        anon = 0
        flags = ""
        with open("/proc/self/smaps") as handle:
            for line in handle:
                if line[0] in "0123456789abcdef" and "-" in line.split()[0]:
                    if path in ("", "[anon]") and anon >= min_bytes:
                        out.append((start, end, anon, flags))
                    fields = line.split()
                    start, end = (int(x, 16) for x in fields[0].split("-"))
                    path = fields[5] if len(fields) >= 6 else ""
                    anon = 0
                    flags = ""
                elif line.startswith("Anonymous:"):
                    anon = int(line.split()[1]) * 1024
                elif line.startswith("VmFlags:"):
                    flags = line.split(":", 1)[1].strip()
        if path in ("", "[anon]") and anon >= min_bytes:
            out.append((start, end, anon, flags))
    except OSError:
        pass
    return sorted(out, key=lambda item: -item[2])


def _mallinfo() -> dict[str, float]:
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"))

        class MallInfo2(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_size_t)
                for name in (
                    "arena",
                    "ordblks",
                    "smblks",
                    "hblks",
                    "hblkhd",
                    "usmblks",
                    "fsmblks",
                    "uordblks",
                    "fordblks",
                    "keepcost",
                )
            ]

        libc.mallinfo2.restype = MallInfo2
        info = libc.mallinfo2()
        return {
            "glibc_arena": info.arena / GIB,
            "glibc_mmapped": info.hblkhd / GIB,
            "glibc_in_use": info.uordblks / GIB,
            "glibc_free": info.fordblks / GIB,
        }
    except Exception:
        return {}


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
        ranges = [
            (
                int(s.get("address", 0)),
                int(s.get("address", 0)) + int(s.get("total_size", 0)),
            )
            for s in segments
        ]
        malloc = _mallinfo()
        if malloc:
            lines.append(
                "  glibc: " + " ".join(f"{k}={v:.2f}GiB" for k, v in malloc.items())
            )
        host_stats = getattr(device, "host_memory_stats", None)
        if callable(host_stats):
            try:
                hs = host_stats()
                big = {
                    k: v
                    for k, v in hs.items()
                    if isinstance(v, (int, float)) and v >= 64 * 1024**2
                }
                lines.append(
                    "  pinned host allocator: "
                    + (
                        " ".join(
                            f"{k}={v / GIB:.2f}GiB" for k, v in sorted(big.items())
                        )
                        or "no counter >= 64 MiB"
                    )
                )
            except Exception as exc:
                lines.append(f"  pinned host allocator: unavailable ({exc})")
        for start, end, anon, flags in _anon_vmas()[:16]:
            inside = any(a <= start < b for a, b in ranges)
            lines.append(
                f"  anon vma {start:#x}-{end:#x}: {anon / GIB:.2f}GiB "
                f"{'INSIDE cuda segment' if inside else 'outside cuda segments'} [{flags}]"
            )
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
