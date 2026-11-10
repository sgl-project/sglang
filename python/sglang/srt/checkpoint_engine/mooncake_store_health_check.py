#!/usr/bin/env python3
"""
Mooncake Store checkpoint health check: manifest + bucket key existence.

Note this verifies that every bucket KEY exists in the store; it does not
checksum bucket contents.

  python -m sglang.srt.checkpoint_engine.mooncake_store_health_check \
      --checkpoint-name <name> --master-addr <mooncake-master-host>:50051

Exit codes:
    0 - Checkpoint exists and all buckets are complete
    1 - Checkpoint not found (manifest doesn't exist) -> need to dump
    2 - Error or partial (manifest exists but one or more bucket keys are missing)

Caveat: if the store/master is unreachable the underlying client may crash
hard (segfault, exit code 139) instead of returning an error; callers must
treat any non-zero exit as unhealthy, not just 1/2.

Adapted from the kvcache-ai/checkpoint_engine health-check tool.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from mooncake.store import MooncakeDistributedStore

ALIGN_SIZE = 256


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_hostname = os.getenv("MOONCAKE_HOSTNAME", socket.gethostname())
    default_metadata = os.getenv("MOONCAKE_TE_META_DATA_SERVER",
                                 os.getenv("MOONCAKE_METADATA_SERVER", "P2PHANDSHAKE"))
    default_master = os.getenv("MOONCAKE_MASTER", os.getenv("MOONCAKE_MASTER_ADDR", "localhost:50051"))

    parser = argparse.ArgumentParser(
        description="Check Mooncake Store checkpoint integrity (manifest + all buckets)"
    )
    parser.add_argument("--checkpoint-name", required=True, help="Checkpoint name")
    parser.add_argument("--hostname", default=default_hostname)
    parser.add_argument("--metadata-server", default=default_metadata)
    parser.add_argument("--master-addr", default=default_master)
    parser.add_argument("--segment-size", type=int, default=0)
    # Must hold the manifest in one get(); large-model manifests run to
    # tens of MB (same sizing rationale as the loader).
    parser.add_argument("--buffer-size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--protocol", default="rdma", choices=["tcp", "rdma"])
    parser.add_argument("--rdma-devices", default=os.getenv("MOONCAKE_DEVICE", ""))
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="Only check manifest existence (skip bucket verification)",
    )
    return parser.parse_args()


def verify_checkpoint_integrity(
    store: MooncakeDistributedStore,
    manifest: dict,
) -> tuple[bool, int, int, list[str]]:
    """Verify all buckets in the manifest exist in the Mooncake Store.

    Args:
        store: Mooncake store instance
        manifest: Parsed manifest dict containing "buckets" list

    Returns:
        (all_complete, total_buckets, missing_count, missing_keys)
    """
    buckets = manifest.get("buckets", [])
    if not buckets:
        return False, 0, 0, ["no buckets in manifest"]

    bucket_keys = [b["bucket_key"] for b in buckets]
    total = len(bucket_keys)

    t0 = time.perf_counter()
    missing = []
    for key in bucket_keys:
        if store.is_exist(key) != 1:
            missing.append(key)
    elapsed = time.perf_counter() - t0

    logger.info(
        f"Bucket integrity check: {total - len(missing)}/{total} buckets present "
        f"({elapsed:.2f}s)"
    )

    if missing:
        # Log first few missing keys for debugging
        preview = missing[:5]
        logger.warning(
            f"{len(missing)} buckets missing. First few: {preview}"
        )

    return len(missing) == 0, total, len(missing), missing


def main() -> int:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    store: MooncakeDistributedStore | None = None

    try:
        # Setup store
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "mooncake-transfer-engine is required for Mooncake store access. "
                "Install it with: pip install mooncake-transfer-engine"
            ) from e

        t0 = time.perf_counter()
        store = MooncakeDistributedStore()
        ret = store.setup(
            args.hostname,
            args.metadata_server,
            args.segment_size,
            args.buffer_size,
            args.protocol,
            args.rdma_devices,
            args.master_addr,
        )

        if ret != 0:
            logger.error(
                f"Failed to setup MooncakeDistributedStore with error code {ret}. "
                f"Parameters: hostname={args.hostname}, metadata_server={args.metadata_server}, "
                f"protocol={args.protocol}, rdma_devices={args.rdma_devices}"
            )
            return 2

        logger.info(f"Store setup completed in {time.perf_counter() - t0:.2f}s")

        test_hostname = store.get_hostname()
        logger.info(f"Store running on: {test_hostname}")

        # Step 1: Check manifest existence
        prefix = args.prefix or f"ckpt:{args.checkpoint_name}"
        manifest_key = f"{prefix}:manifest"

        exists = store.is_exist(manifest_key)
        if exists == 0:
            logger.info(f"Manifest not found: {manifest_key}")
            return 1
        elif exists != 1:
            logger.error(f"Error checking manifest existence: is_exist returned {exists}")
            return 2

        logger.info(f"Manifest found: {manifest_key}")

        # Step 2: Load and parse manifest
        if args.manifest_only:
            logger.info("Manifest-only mode: skipping bucket verification")
            return 0

        payload = store.get(manifest_key)
        if payload is None:
            logger.error(f"Failed to read manifest: {manifest_key}")
            return 2

        try:
            manifest = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to parse manifest: {e}")
            return 2

        align_size = manifest.get("align_size")
        if align_size is not None and align_size != ALIGN_SIZE:
            logger.error(f"Manifest align_size mismatch: {align_size} != {ALIGN_SIZE}")
            return 2

        buckets = manifest.get("buckets", [])
        logger.info(f"Manifest contains {len(buckets)} buckets")

        if not buckets:
            logger.error("Manifest contains no buckets")
            return 2

        # Step 3: Verify all buckets exist
        all_complete, total, missing_count, missing_keys = verify_checkpoint_integrity(
            store, manifest
        )

        if all_complete:
            logger.info(
                f"Checkpoint '{args.checkpoint_name}' is complete: "
                f"all {total} buckets present"
            )
            return 0
        else:
            logger.error(
                f"Checkpoint '{args.checkpoint_name}' is INCOMPLETE: "
                f"{missing_count}/{total} buckets missing"
            )
            return 2

    except Exception as e:
        logger.error(f"Health check failed with exception: {e}")
        return 2
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
