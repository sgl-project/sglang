"""Host-memory headroom bounded by the process's visible cgroup hierarchy."""

from __future__ import annotations

import logging
import re
from pathlib import Path, PurePosixPath

import psutil

logger = logging.getLogger(__name__)


def _unescape_mount_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def _cgroup_headroom(
    controller: str,
    limits_v2: tuple[str, ...],
    usage_v2: str,
    limits_v1: tuple[str, ...],
    usage_v1: str,
    proc_root: Path = Path("/proc"),
) -> int | None:
    memberships = {}
    try:
        cgroups = (proc_root / "self/cgroup").read_text()
        mounts = (proc_root / "self/mountinfo").read_text()
    except FileNotFoundError:
        # Non-Linux systems need not expose procfs.
        return None
    for line in cgroups.splitlines():
        _, controllers, path = line.split(":", 2)
        if not controllers:
            memberships["cgroup2"] = PurePosixPath(path)
        elif controller in controllers.split(","):
            memberships["cgroup"] = PurePosixPath(path)

    headroom = None
    resolved = False
    for line in mounts.splitlines():
        before, after = line.split(" - ", 1)
        filesystem, _, options = after.split()[:3]
        if filesystem not in memberships:
            continue
        if filesystem == "cgroup" and controller not in options.split(","):
            continue
        fields = before.split()
        root = PurePosixPath(_unescape_mount_path(fields[3]))
        mount = Path(_unescape_mount_path(fields[4]))
        membership = memberships[filesystem]
        if membership.is_relative_to(root):
            relative = membership.relative_to(root)
        elif root != PurePosixPath("/"):
            # A cgroup namespace can expose membership relative to its root,
            # while mountinfo still identifies the host-side subtree.
            relative = membership.relative_to("/")
        else:
            continue
        if ".." in relative.parts:
            raise ValueError(f"Cannot resolve cgroup {controller} path: {membership}")
        directory = mount / relative
        if not directory.is_dir():
            continue
        resolved = True
        limits = limits_v2 if filesystem == "cgroup2" else limits_v1
        usage_name = usage_v2 if filesystem == "cgroup2" else usage_v1
        while True:
            for name in limits:
                try:
                    value = (directory / name).read_text().strip()
                except FileNotFoundError:
                    # The hierarchy root may not have controller files.
                    continue
                if value == "max":
                    continue
                limit = int(value)
                # Do not silently ignore an unreadable usage file for a known
                # limit: falling back to host RAM could overrun the container.
                usage = int((directory / usage_name).read_text())
                remaining = max(0, limit - usage)
                headroom = remaining if headroom is None else min(headroom, remaining)
            if directory == mount:
                break
            directory = directory.parent
    if memberships and not resolved:
        raise RuntimeError(
            f"Cannot locate the process {controller} cgroup in mounted cgroup "
            "filesystems"
        )
    return headroom


def _cgroup_memory_headroom(proc_root: Path = Path("/proc")) -> int | None:
    return _cgroup_headroom(
        "memory",
        ("memory.max", "memory.high"),
        "memory.current",
        ("memory.limit_in_bytes",),
        "memory.usage_in_bytes",
        proc_root,
    )


def cgroup_hugetlb_headroom_bytes(
    page_size: int, proc_root: Path = Path("/proc")
) -> int | None:
    """Bytes the hugetlb allowed by the cgroup controller.

    None when no limit applies (controller unmounted, or every ancestor unlimited).
    """
    if page_size % 1024**3 == 0:
        label = f"{page_size // 1024**3}GB"
    else:
        label = f"{page_size // 1024**2}MB"
    prefix = f"hugetlb.{label}"
    return _cgroup_headroom(
        "hugetlb",
        (f"{prefix}.max",),
        f"{prefix}.current",
        (f"{prefix}.limit_in_bytes",),
        f"{prefix}.usage_in_bytes",
        proc_root,
    )


def available_host_memory_bytes() -> int:
    """Conservative allocatable RAM; charged file cache is not assumed reclaimable."""
    available = psutil.virtual_memory().available
    cgroup_headroom = _cgroup_memory_headroom()
    if cgroup_headroom is not None:
        logger.info(
            "HiCache memory headroom: host %.1f GiB, cgroup %.1f GiB",
            available / 1024**3,
            cgroup_headroom / 1024**3,
        )
        available = min(available, cgroup_headroom)
    return available
