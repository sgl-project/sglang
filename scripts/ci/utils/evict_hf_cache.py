#!/usr/bin/env python3
"""Evict least-recently-used Hugging Face hub repos until the cache has room.

Only for runners whose HF cache is local to the host; see
scripts/ci/cuda/ensure_disk_headroom.sh. Stdlib only, because it runs before
the CI venv exists.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

_GIB = 1 << 30
_REPO_PREFIXES = ("models--", "datasets--", "spaces--")
# A job on another runner of the same host may still be reading the repo;
# two hours matches the stale-download age in cleanup_hf_cache.py.
_DEFAULT_MIN_IDLE_HOURS = 2.0


class CachedRepo(NamedTuple):
    path: Path
    size_bytes: int
    # Newest atime/mtime of any file. Under relatime, atime moves at most once
    # a day, so this orders repos only roughly by last use.
    last_used: float


def _describe_repo(repo_dir: Path) -> CachedRepo:
    size_bytes = 0
    last_used = repo_dir.stat().st_mtime
    for root, _, files in os.walk(repo_dir):
        for name in files:
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue  # vanished mid-scan
            size_bytes += st.st_blocks * 512
            last_used = max(last_used, st.st_atime, st.st_mtime)
    return CachedRepo(path=repo_dir, size_bytes=size_bytes, last_used=last_used)


def scan_repos(cache_dir: Path) -> list[CachedRepo]:
    return [
        _describe_repo(entry)
        for entry in cache_dir.iterdir()
        if entry.is_dir() and entry.name.startswith(_REPO_PREFIXES)
    ]


def evict_least_recently_used(
    *,
    cache_dir: Path,
    target_free_bytes: int,
    min_idle_seconds: float,
    now: float,
    free_bytes_fn: Callable[[], int],
    remove_fn: Callable[[Path], None],
) -> list[CachedRepo]:
    evicted = []
    for repo in sorted(scan_repos(cache_dir), key=lambda r: r.last_used):
        if free_bytes_fn() >= target_free_bytes:
            break
        if now - repo.last_used < min_idle_seconds:
            break  # sorted by last use, so every remaining repo is newer
        try:
            remove_fn(repo.path)
        except OSError as exc:
            print(f"  Warning: could not evict {repo.path.name}: {exc}")
            continue
        idle_hours = (now - repo.last_used) / 3600
        print(
            f"  Evicted {repo.path.name} ({repo.size_bytes / _GIB:.1f} GiB, "
            f"idle {idle_hours:.1f} h)"
        )
        evicted.append(repo)
    return evicted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--target-free-gib", type=float, required=True)
    parser.add_argument("--min-idle-hours", type=float, default=_DEFAULT_MIN_IDLE_HOURS)
    args = parser.parse_args()

    if not args.cache_dir.is_dir():
        print(f"HF cache {args.cache_dir} does not exist; nothing to evict")
        return 0

    print(
        f"Evicting least-recently-used repos from {args.cache_dir} "
        f"until {args.target_free_gib:.0f} GiB is free"
    )
    evicted = evict_least_recently_used(
        cache_dir=args.cache_dir,
        target_free_bytes=int(args.target_free_gib * _GIB),
        min_idle_seconds=args.min_idle_hours * 3600,
        now=time.time(),
        free_bytes_fn=lambda: shutil.disk_usage(args.cache_dir).free,
        remove_fn=shutil.rmtree,
    )
    free_gib = shutil.disk_usage(args.cache_dir).free / _GIB
    print(f"Evicted {len(evicted)} repo(s); {free_gib:.1f} GiB free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
