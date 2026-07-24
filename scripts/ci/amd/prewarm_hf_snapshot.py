"""Synchronously read one cached HF snapshot into the node page cache."""

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path


def _read_files(paths: list[str]) -> int:
    total = 0
    for path in paths:
        with open(path, "rb", buffering=0) as file:
            while data := file.read(32 * 1024 * 1024):
                total += len(data)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--link-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    snapshots = []
    for snapshot in args.model_cache.glob("snapshots/*"):
        weight_files = sorted(snapshot.glob("*.safetensors"))
        if weight_files:
            snapshots.append((len(weight_files), snapshot, weight_files))
    if not snapshots:
        raise RuntimeError(f"No safetensors snapshot found under {args.model_cache}")

    _, snapshot, weight_files = max(snapshots, key=lambda item: item[0])
    groups = [
        [str(path) for path in weight_files[rank :: args.workers]]
        for rank in range(args.workers)
    ]
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        bytes_by_worker = list(executor.map(_read_files, groups))
    elapsed = time.perf_counter() - start

    args.link_path.unlink(missing_ok=True)
    os.symlink(snapshot, args.link_path, target_is_directory=True)
    payload = {
        "snapshot": str(snapshot),
        "link_path": str(args.link_path),
        "files": len(weight_files),
        "bytes": sum(bytes_by_worker),
        "workers": args.workers,
        "elapsed_s": elapsed,
        "gib_s": sum(bytes_by_worker) / (1024**3) / elapsed,
    }
    print(f"DSV4_PREWARM_RESULT {json.dumps(payload, sort_keys=True)}")


if __name__ == "__main__":
    main()
