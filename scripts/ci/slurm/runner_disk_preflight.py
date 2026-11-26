"""Bound diagnostic retention, reclaim this runner's leftover launcher scratch,
and check write headroom on this runner only."""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# Written into each scratch dir by launch_mi355x.sh; keep the two in sync.
SCRATCH_OWNER_FILE = ".runner_name"


def prune_diagnostics(diag: Path, now: float) -> int:
    """Keep seven days and the newest log of each kind; never follow links."""
    if diag.is_symlink() or not diag.is_dir():
        return 0
    removed = 0
    for pattern in ("Runner_*.log", "Worker_*.log"):
        logs = sorted(
            (p for p in diag.glob(pattern) if not p.is_symlink() and p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in logs[1:]:
            if path.stat().st_mtime < now - 7 * 86400:
                size = path.stat().st_size
                path.unlink()
                removed += size
    return removed


def prune_scratch(scratch: Path, runner_name: str, now: float) -> list[str]:
    """Remove launcher scratch dirs left by this runner's earlier legs.

    A runner runs one leg at a time, so before a leg starts nothing it owns is
    in use. Dirs owned by another runner sharing this home are never touched;
    unowned dirs (from launchers that predate the owner file) go after a day.
    """
    if scratch.is_symlink() or not scratch.is_dir():
        return []
    removed = []
    for entry in scratch.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        owner_file = entry / SCRATCH_OWNER_FILE
        if owner_file.is_file():
            stale = owner_file.read_text().strip() == runner_name
        else:
            stale = entry.stat().st_mtime < now - 86400
        if stale:
            # NFS keeps files a dead process still holds as .nfsXXXX; skip them.
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
    return removed


def main() -> None:
    temp = Path(os.environ["RUNNER_TEMP"])
    # Derive the installation from the standard runner layout, never from HOME:
    # two runner installations may share the same user and filesystem quota.
    if temp.is_symlink() or temp.parent.is_symlink():
        raise SystemExit("Refusing a symlinked runner temp/work directory")
    temp = temp.resolve(strict=True)
    root = temp.parent.parent
    if (
        temp.name != "_temp"
        or temp.parent.name != "_work"
        or not (root / ".runner").is_file()
    ):
        raise SystemExit(f"Unrecognized runner layout: {temp}")
    reclaimed = prune_diagnostics(root / "_diag", time.time())
    print(f"Removed {reclaimed} bytes of runner diagnostics older than seven days")
    stale = prune_scratch(
        scratch=Path.home() / ".mi355x_ci",
        runner_name=os.environ["RUNNER_NAME"],
        now=time.time(),
    )
    print(f"Removed leftover launcher scratch: {', '.join(stale) or 'none'}")

    # df/free space alone cannot detect per-user quota exhaustion. Report quota
    # where available, then actually allocate and flush a small probe file.
    if shutil.which("quota"):
        subprocess.run(["quota", "-s"], check=False, timeout=15)
    paths = [root / "_diag", temp, Path(os.environ["GITHUB_WORKSPACE"]), Path.home()]
    for path in paths:
        free = shutil.disk_usage(path).free
        print(f"{path}: {free // (1024**2)} MiB available", flush=True)
        if free < 5 * 1024**3:
            raise SystemExit(f"::error::Less than 5 GiB free at {path}")
        try:
            with tempfile.TemporaryFile(dir=path) as probe:
                for _ in range(8):
                    probe.write(b"\0" * 1024**2)
                probe.flush()
                os.fsync(probe.fileno())
        except OSError as exc:
            raise SystemExit(
                f"::error::Runner storage is not writable at {path}: {exc}"
            )


if __name__ == "__main__":
    main()
