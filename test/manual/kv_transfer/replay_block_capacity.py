"""Fixed-pool geometry with frozen token IDs and simplified node LRU.

This is NOT a serving, byte-copy, nvCOMP or actual-tree test. Lengths default to
synthetic 147900..148220 bytes. --lengths supplies measured lengths indexed by
prefix SHA256; every new object must have a sample (no synthetic substitution).
Only allocator metadata is touched; arena payload correctness has separate tests.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def replay(workload, order, node_pages, lengths=None):
    root = Path(__file__).resolve().parents[3]
    base = root / "python/sglang/srt/kv_compression"
    load("sglang.srt.kv_compression.types", base / "types.py")
    load("sglang.srt.kv_compression.faults", base / "faults.py")
    cls = load("block_capacity_store", base / "store.py").CompressedHostKVCache
    pool = cls(
        147456, 8_000_000_000, reservation_bytes=197632, verify=True, pin_memory=False
    )
    resident = {}
    nodes = OrderedDict()
    ref = 0
    node_id = 0
    targets = {}
    rows = []
    evictions = []
    peaks = []
    for row in order:
        g, c, cycle = row["group"], row["case"], row["cycle"]
        if g == "warmup":
            tokens = workload["warmup"]
        elif g == "lengths":
            tokens = next(x for x in workload["lengths"] if len(x) == int(c))
        elif g == "shared":
            tokens = workload["shared"][0 if c == "prime-1" else 1]
        elif g == "concurrency":
            tokens = workload["concurrency"][int(c)]
        else:
            tokens = workload["l2"][cycle - 1][
                int(c.split("-")[-1]) + 1 if c.startswith("pressure-") else 0
            ]
        assert (
            hashlib.sha256(json.dumps(tokens, sort_keys=True).encode()).hexdigest()
            == row["input_sha256"]
        )
        assert len(tokens) == row["input_tokens"]
        h = hashlib.sha256()
        keys = []
        for token in tokens:
            h.update(int(token).to_bytes(4, "little"))
            keys.append(h.digest())
        keyset = set(keys)
        # Real radix matching splits a partially matched node. Keeping its
        # entire old suffix pinned would exaggerate the 8192-node stress case.
        split = OrderedDict()
        for old_node, old_keys in nodes.items():
            cut = 0
            while cut < len(old_keys) and old_keys[cut] in keyset:
                cut += 1
            if 0 < cut < len(old_keys):
                prefix, suffix = old_keys[:cut], old_keys[cut:]
                split[old_node] = prefix
                split[node_id] = suffix
                for k in suffix:
                    h, r, _ = resident[k]
                    resident[k] = (h, r, node_id)
                node_id += 1
            else:
                split[old_node] = old_keys
        nodes = split
        for key in keys:
            if key in resident:
                nodes.move_to_end(resident[key][2])
        retained = None
        if g == "l2" and c == "restore":
            retained = sum(
                k in resident and resident[k][1] == r for k, r in targets[cycle].items()
            )
        missing = [k for k in keys if k not in resident]
        for begin in range(0, len(missing), node_pages):
            chunk = missing[begin : begin + node_pages]
            while not pool.can_reserve(len(chunk)):
                victim = next(
                    (n for n, ks in nodes.items() if not keyset.intersection(ks)), None
                )
                assert victim is not None, "Pinned request exceeds capacity"
                removed = nodes.pop(victim)
                before = pool.free_block_count
                pool.free([resident[k][0] for k in removed])
                for k in removed:
                    del resident[k]
                evictions.append(
                    {
                        "rid": row["rid"],
                        "pages": len(removed),
                        "released_blocks": pool.free_block_count - before,
                    }
                )
            handles = pool.alloc(len(chunk))
            assert handles is not None
            snap = pool.snapshot()
            peaks.append(snap["payload_bytes"] + snap["reserved_bytes"])
            for start in range(0, len(chunk), 64):
                ck = chunk[start : start + 64]
                hh = handles[start : start + 64]
                sizes = [
                    lengths[k.hex()]
                    if lengths is not None
                    else 147900 + int.from_bytes(k[:2], "little") % 321
                    for k in ck
                ]
                write = pool.prepare_write(hh, sizes)
                # Geometry-only completion stand-in, never used by live serving.
                # Does not validate or claim any payload correctness.
                write.written = True
                refs = list(range(ref + 1, ref + 1 + len(ck)))
                write.publish(
                    refs,
                    [SimpleNamespace(encoding="lz4")] * len(ck),
                    [bytes(32)] * len(ck),
                )
                write.close()
                for handle, key, r in zip(hh.tolist(), ck, refs):
                    resident[key] = (handle, r, node_id)
                ref += len(ck)
            nodes[node_id] = chunk
            node_id += 1
        if g == "l2" and c == "cold":
            targets[cycle] = {k: resident[k][1] for k in keys}
        s = pool.snapshot()
        assert pool.allocated_size() == len(resident) and not pool.has_readers()
        assert (
            s["free_blocks"] * 4096
            + s["payload_bytes"]
            + s["reserved_bytes"]
            + s["retired_bytes"]
            == s["arena_bytes"]
        )
        rows.append(dict(group=g, case=c, cycle=cycle, retained_original=retained, **s))
    checks = [
        r["retained_original"] for r in rows if r["retained_original"] is not None
    ]
    result = {
        "node_pages": node_pages,
        "restore_retention": checks,
        "rows": rows,
        "peak_payload_plus_reservations": max(peaks),
        "max_pages": max(r["used_pages"] for r in rows),
        "evictions": evictions,
        "measured_lengths": lengths is not None,
        "source_sha256": hashlib.sha256((base / "store.py").read_bytes()).hexdigest(),
    }
    pool.free([x[0] for x in resident.values()])
    assert pool.free_block_count == pool.block_count
    del pool
    gc.collect()
    assert checks == [8192] * 3, checks
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--order", type=Path, required=True)
    p.add_argument("--lengths", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--node-pages", type=int, nargs="+", default=[1024, 8192])
    args = p.parse_args()
    workload = json.loads(args.workload.read_text())
    order = [json.loads(x) for x in args.order.read_text().splitlines()]
    lengths = json.loads(args.lengths.read_text()) if args.lengths else None
    results = []
    for n in args.node_pages:
        result = replay(workload, order, n, lengths)
        results.append(result)
        print(n, result["restore_retention"], result["max_pages"], flush=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": __doc__,
                "budget_bytes": 8_000_000_000,
                "workload_sha256": hashlib.sha256(
                    args.workload.read_bytes()
                ).hexdigest(),
                "order_sha256": hashlib.sha256(args.order.read_bytes()).hexdigest(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
