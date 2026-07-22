"""Exercise repeated HiCache reads/writes across multiple prompt namespaces."""

import argparse
import json
import time
import uuid
from typing import Any

import requests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30001")
    p.add_argument("--length", type=int, default=8000)
    p.add_argument("--keys", type=int, default=4)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--filler-count", type=int, default=8)
    p.add_argument(
        "--min-eviction-tokens",
        type=int,
        default=0,
        help="Require filler_count * length to reach this eviction budget.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument(
        "--min-reread-cached-tokens",
        type=int,
        default=0,
        help="Fail if an eviction reread returns fewer cached tokens.",
    )
    p.add_argument(
        "--expected-reread-cached-tokens",
        type=int,
        help="Require an exact reread hit length and pure L3 accounting.",
    )
    p.add_argument(
        "--expected-reread-backend",
        help="If set, require this storage_backend on every eviction reread.",
    )
    p.add_argument(
        "--min-reread-storage-tokens",
        type=int,
        default=0,
        help="Fail unless this many reread tokens came from the L3 storage tier.",
    )
    p.add_argument(
        "--flush-before-reread",
        action="store_true",
        help="Flush L1/L2 before each reread to require a pure storage-tier hit.",
    )
    p.add_argument(
        "--storage-settle-seconds",
        type=float,
        default=0.0,
        help="Wait after writes before flushing/rereading the storage tier.",
    )
    p.add_argument("--output-file")
    args = p.parse_args()
    effective_length = max(0, (args.length - 1) // 64 * 64)
    filler_tokens = args.filler_count * effective_length
    if filler_tokens < args.min_eviction_tokens:
        raise ValueError(
            "filler workload is too small to exercise eviction: "
            f"{filler_tokens} < required {args.min_eviction_tokens} tokens"
        )
    base = args.base_url.rstrip("/")
    run_id = uuid.uuid4().hex
    signature = [50000 + int(run_id[i : i + 2], 16) for i in range(0, 16, 2)]
    rows: list[dict[str, Any]] = []

    def ids(namespace: int) -> list[int]:
        prefix = [1000 + namespace] + signature
        return (prefix + [2000 + (i % 997) for i in range(args.length)])[: args.length]

    def flush() -> None:
        r = requests.post(
            f"{base}/flush_cache?timeout={args.timeout}", timeout=args.timeout + 10
        )
        r.raise_for_status()

    def request(phase: str, namespace: int) -> dict[str, Any]:
        started = time.perf_counter()
        # Kimi-K3's hybrid radix cache materializes a complete KDA checkpoint
        # when the seed request has no decode tokens. Decode only on the
        # read/reread phases so the aligned sidecar checkpoint is published.
        max_new_tokens = (
            0 if phase in ("write", "filler") else args.max_new_tokens
        )
        r = requests.post(
            f"{base}/generate",
            json={
                "input_ids": ids(namespace),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=args.timeout,
        )
        r.raise_for_status()
        meta = r.json().get("meta_info") or {}
        details = meta.get("cached_tokens_details") or {}
        row = {
            "type": "request",
            "run_id": run_id,
            "cycle": cycle,
            "phase": phase,
            "namespace": namespace,
            "cached_tokens": int(meta.get("cached_tokens", 0)),
            "device": int(details.get("device", 0)),
            "host": int(details.get("host", 0)),
            "storage": int(details.get("storage", 0)),
            "storage_backend": details.get("storage_backend"),
            "latency_s": time.perf_counter() - started,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if int(meta.get("completion_tokens", 0)) != max_new_tokens:
            raise RuntimeError(f"short decode: {row}")
        if phase == "reread_after_eviction":
            if row["cached_tokens"] < args.min_reread_cached_tokens:
                raise RuntimeError(
                    "insufficient cache hit after eviction: "
                    f"expected >= {args.min_reread_cached_tokens}, got {row}"
                )
            if row["storage"] < args.min_reread_storage_tokens:
                raise RuntimeError(
                    "insufficient L3 storage hit after eviction: "
                    f"expected >= {args.min_reread_storage_tokens}, got {row}"
                )
            if args.expected_reread_cached_tokens is not None:
                expected = args.expected_reread_cached_tokens
                if (
                    row["cached_tokens"] != expected
                    or row["device"] != 0
                    or row["host"] != 0
                    or row["storage"] != expected
                    or row["cached_tokens"]
                    != row["device"] + row["host"] + row["storage"]
                ):
                    raise RuntimeError(
                        "reread was not a pure exact L3 hit: "
                        f"expected cached/storage={expected}, device=host=0, got {row}"
                    )
            if (
                args.expected_reread_backend is not None
                and row["storage_backend"] != args.expected_reread_backend
            ):
                raise RuntimeError(
                    "unexpected storage backend after eviction: "
                    f"expected {args.expected_reread_backend!r}, got {row}"
                )
        return row

    flush()
    for cycle in range(args.cycles):
        # Write/read every key, then evict device/host copies with independent keys.
        cycle_base = cycle * (args.keys + args.filler_count)
        for key in range(args.keys):
            namespace = cycle_base + key
            request("write", namespace)
            request("read", namespace)
        for filler in range(args.filler_count):
            request(
                "filler",
                cycle_base + args.keys + filler,
            )
        if args.storage_settle_seconds > 0:
            time.sleep(args.storage_settle_seconds)
        if args.flush_before_reread:
            flush()
        for key in range(args.keys):
            request("reread_after_eviction", cycle_base + key)
    summary = {
        "type": "summary",
        "run_id": run_id,
        "length": args.length,
        "keys": args.keys,
        "cycles": args.cycles,
        "filler_count": args.filler_count,
        "samples": len(rows),
        "status": "PASS",
    }
    print(json.dumps(summary), flush=True)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for row in rows + [summary]:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
