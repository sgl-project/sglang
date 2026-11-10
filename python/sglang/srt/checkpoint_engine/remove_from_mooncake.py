#!/usr/bin/env python3
"""
remove_from_mooncake.py - Delete checkpoints (or arbitrary prefixes) from
Mooncake Store. Use before re-uploading a checkpoint with the same name.

  python -m sglang.srt.checkpoint_engine.remove_from_mooncake \
      --checkpoint-name <name> --master-addr <mooncake-master-host>:50051

Adapted from the kvcache-ai/checkpoint_engine remove tool.
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from mooncake.store import MooncakeDistributedStore


def remove_by_prefix(store: MooncakeDistributedStore, prefix: str, force: bool) -> int:
    """Remove all keys matching ``^<prefix>:.*`` from the store."""
    pattern = f"^{re.escape(prefix)}:.*"
    logger.info(f"Removing keys matching regex: {pattern} (force={force})")
    count = store.remove_by_regex(pattern, force)
    if count < 0:
        raise RuntimeError(
            f"remove_by_regex failed with error code {count}. "
            f"If error is -706, the keys have an active lease; retry with --force."
        )
    return count


def remove_all_objects(store: MooncakeDistributedStore, force: bool) -> int:
    """Remove every object in the store."""
    logger.info(f"Removing ALL objects from store (force={force})")
    count = store.remove_all(force)
    if count < 0:
        raise RuntimeError(
            f"remove_all failed with error code {count}. "
            f"If error is -706, some keys have active leases; retry with --force."
        )
    return count


def parse_args() -> argparse.Namespace:
    default_hostname = os.getenv("MOONCAKE_HOSTNAME", socket.gethostname())
    default_metadata = os.getenv("MOONCAKE_TE_META_DATA_SERVER",
                                 os.getenv("MOONCAKE_METADATA_SERVER", "P2PHANDSHAKE"))
    default_master = os.getenv("MOONCAKE_MASTER", os.getenv("MOONCAKE_MASTER_ADDR", "localhost:50051"))
    default_segment = int(os.getenv("MOONCAKE_SEGMENT_SIZE", "0"))
    default_buffer = int(os.getenv("MOONCAKE_BUFFER_SIZE", "0"))
    default_protocol = os.getenv("MOONCAKE_PROTOCOL", "rdma")
    default_rdma = os.getenv("MOONCAKE_DEVICE", os.getenv("MOONCAKE_RDMA_DEVICES", ""))

    parser = argparse.ArgumentParser(
        description="Remove checkpoints or objects from Mooncake Store."
    )

    # --- deletion target (mutually exclusive) ---
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--checkpoint-name",
        default=None,
        help="Remove the checkpoint with this name (deletes all keys under ckpt:<name>).",
    )
    target.add_argument(
        "--store-prefix",
        default=None,
        help="Remove all keys matching <prefix>:* in the store.",
    )
    target.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Remove ALL objects from the store.",
    )

    # --- force ---
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Skip lease checks (use when keys were recently accessed and have active leases).",
    )

    # --- store connection (same as dump_to_store.py) ---
    parser.add_argument(
        "--hostname",
        default=default_hostname,
        help="Local hostname for Mooncake registration (default: %(default)s).",
    )
    parser.add_argument(
        "--metadata-server",
        default=default_metadata,
        help="Mooncake metadata server URL (default: %(default)s).",
    )
    parser.add_argument(
        "--master-addr",
        default=default_master,
        help="Mooncake master address host:port (default: %(default)s).",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=default_segment,
        help="Global segment size in bytes (default from $MOONCAKE_SEGMENT_SIZE or 0).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=default_buffer,
        help="Local buffer size in bytes (default from $MOONCAKE_BUFFER_SIZE or 0).",
    )
    parser.add_argument(
        "--protocol",
        default=default_protocol,
        choices=["tcp", "rdma"],
        help="Mooncake transport protocol.",
    )
    parser.add_argument(
        "--rdma-devices",
        default=default_rdma,
        help="Comma-separated RDMA device list (default from $MOONCAKE_RDMA_DEVICES).",
    )

    # --- logging ---
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for loguru (default INFO).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as e:
        raise ImportError(
            "mooncake-transfer-engine is required for Mooncake store access. "
            "Install it with: pip install mooncake-transfer-engine"
        ) from e

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
        raise RuntimeError(
            f"Failed to setup MooncakeDistributedStore with error code {ret}. "
            f"Parameters: hostname={args.hostname}, metadata_server={args.metadata_server}, "
            f"protocol={args.protocol}, rdma_devices={args.rdma_devices}, "
            f"master_addr={args.master_addr}"
        )

    try:
        if args.all:
            count = remove_all_objects(store, args.force)
            logger.info(f"Removed {count} object(s) from store.")
        elif args.checkpoint_name:
            prefix = f"ckpt:{args.checkpoint_name}"
            count = remove_by_prefix(store, prefix, args.force)
            logger.info(
                f"Removed {count} key(s) for checkpoint '{args.checkpoint_name}'."
            )
        elif args.store_prefix:
            count = remove_by_prefix(store, args.store_prefix, args.force)
            logger.info(
                f"Removed {count} key(s) with prefix '{args.store_prefix}'."
            )
    finally:
        store.close()


if __name__ == "__main__":
    main()
