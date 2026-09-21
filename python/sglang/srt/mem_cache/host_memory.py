"""Host-memory headroom bounded by the process's visible cgroup hierarchy."""

from __future__ import annotations

import logging
import re
from pathlib import Path, PurePosixPath

import psutil

logger = logging.getLogger(__name__)


def _unescape_mount_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def _cgroup_memory_headroom(proc_root: Path = Path("/proc")) -> int | None:
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
        elif "memory" in controllers.split(","):
            memberships["cgroup"] = PurePosixPath(path)

    headroom = None
    resolved = False
    for line in mounts.splitlines():
        before, after = line.split(" - ", 1)
        filesystem, _, options = after.split()[:3]
        if filesystem not in memberships:
            continue
        if filesystem == "cgroup" and "memory" not in options.split(","):
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
            raise ValueError(f"Cannot resolve cgroup memory path: {membership}")
        directory = mount / relative
        if not directory.is_dir():
            continue
        resolved = True
        limits = (
            ("memory.max", "memory.high")
            if filesystem == "cgroup2"
            else ("memory.limit_in_bytes",)
        )
        usage_name = (
            "memory.current" if filesystem == "cgroup2" else "memory.usage_in_bytes"
        )
        while True:
            for name in limits:
                try:
                    value = (directory / name).read_text().strip()
                except FileNotFoundError:
                    # The hierarchy root may not have memory controller files.
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
            "Cannot locate the process memory cgroup in mounted cgroup filesystems"
        )
    return headroom


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
