#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a small, zero-copy Kimi K3 GGUF/expert-pack adapter manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .kimi_ggml import create_manifest, write_json_atomic
except ImportError:
    from kimi_ggml import create_manifest, write_json_atomic  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-dir", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--payload-samples",
        type=int,
        default=6,
        help="Evenly spaced pack entries compared byte-for-byte with GGUF (default: 6).",
    )
    parser.add_argument("--full-source-hashes", action="store_true")
    parser.add_argument("--full-pack-hash", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.payload_samples < 0:
        raise ValueError("--payload-samples must be non-negative")
    repo = Path(__file__).resolve().parent
    manifest = create_manifest(
        gguf_dir=args.gguf_dir,
        expert_pack=args.expert_pack,
        model_config=args.model_config.resolve(strict=True),
        tokenizer_dir=args.tokenizer_dir,
        payload_samples=args.payload_samples,
        full_source_hashes=args.full_source_hashes,
        full_pack_hash=args.full_pack_hash,
        repo=repo,
    )
    write_json_atomic(args.output.resolve(), manifest)
    summary = {
        "manifest": str(args.output.resolve()),
        "format": manifest["format"],
        "source_shards": manifest["source"]["summary"]["shard_count"],
        "source_tensors": manifest["source"]["summary"]["tensor_count"],
        "pack_entries": manifest["expert_pack"]["index_count"],
        "pack_index_sha256": manifest["expert_pack"]["index_sha256"],
        "top_k": manifest["hard_constraints"]["top_k"],
        "payload_samples_verified": manifest["expert_pack"]["payload_samples_verified"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
