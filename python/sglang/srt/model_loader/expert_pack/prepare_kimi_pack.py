#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate or build the Kimi K3 expert pack derived from GGUF shards."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
from pathlib import Path

try:
    from .kimi_ggml import (
        SHARD_RE,
        KimiK3Spec,
        discover_gguf_shards,
        estimate_ggml_moe_pack_size,
        scan_gguf_shards,
        validate_expert_tensors,
        validate_ggml_moe_pack,
        write_ggml_moe_pack,
    )
except ImportError:
    from kimi_ggml import (  # type: ignore[no-redef]
        SHARD_RE,
        KimiK3Spec,
        discover_gguf_shards,
        estimate_ggml_moe_pack_size,
        scan_gguf_shards,
        validate_expert_tensors,
        validate_ggml_moe_pack,
        write_ggml_moe_pack,
    )


def expert_pack_path(gguf: Path) -> Path:
    match = SHARD_RE.search(gguf.name)
    if match is None:
        raise ValueError(f"Kimi GGUF name is not a numbered shard: {gguf}")
    return gguf.parent / f"{gguf.name[: match.start()]}.expert-major.pack"


def validate_pack(pack: Path, expert_tensors: dict, spec: KimiK3Spec) -> str:
    result = validate_ggml_moe_pack(
        pack, expert_tensors, spec, payload_samples=6, full_pack_hash=False
    )
    return (
        f"entries={result['index_count']} size={result['size']} "
        f"samples={result['payload_samples_verified']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--safety-margin-gib", type=float, default=2.0)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gguf = args.gguf.expanduser().resolve(strict=True)
    model_config = args.model_config.expanduser().resolve(strict=True)
    if args.safety_margin_gib < 0:
        raise ValueError("--safety-margin-gib must be non-negative")

    shards = discover_gguf_shards(gguf.parent)
    if gguf not in shards:
        raise ValueError(f"--gguf is not part of the discovered shard set: {gguf}")
    config = json.loads(model_config.read_text(encoding="utf-8"))
    spec = KimiK3Spec.from_config(config)
    _, tensors, _ = scan_gguf_shards(shards)
    expert_tensors = validate_expert_tensors(tensors, spec)
    pack = expert_pack_path(gguf)
    partial = pack.with_name(pack.name + ".partial")
    lock_path = pack.with_name(pack.name + ".lock")

    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            detail = validate_pack(pack, expert_tensors, spec)
        except (OSError, ValueError) as exc:
            print(f"EXPERT_PACK_INVALID path={pack} detail={exc}", flush=True)
            if args.check_only:
                return 1
        else:
            print(f"EXPERT_PACK_VALID path={pack} {detail}", flush=True)
            return 0

        for path in (pack, partial):
            if path.exists():
                print(f"EXPERT_PACK_REMOVE_INVALID path={path}", flush=True)
                path.unlink()

        estimated_size = estimate_ggml_moe_pack_size(expert_tensors, spec)
        safety_margin = int(args.safety_margin_gib * 1024**3)
        available = shutil.disk_usage(pack.parent).free
        if available < estimated_size + safety_margin:
            raise OSError(
                f"insufficient space for Kimi Expert Pack: available={available}, "
                f"required={estimated_size + safety_margin}"
            )

        print(
            f"EXPERT_PACK_BUILD_START output={pack} size={estimated_size}",
            flush=True,
        )

        def report_progress(completed: int, total: int) -> None:
            print(
                f"EXPERT_PACK_BUILD_PROGRESS completed={completed} total={total}",
                flush=True,
            )

        write_ggml_moe_pack(pack, expert_tensors, spec, progress=report_progress)
        detail = validate_pack(pack, expert_tensors, spec)
        print(f"EXPERT_PACK_READY path={pack} {detail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
