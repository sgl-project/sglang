"""Read random KV pages from a peer process's GPU through KvPeerReader.

The owner allocates one FABRIC VMM allocation laid out like a paged MLA pool with
a second per-layer index buffer (defaults: GLM-5.2, 78 layers of 36864 B +
8448 B pages) and exports it. The reader imports it and reads
``--pages`` random pages into its own pools, in groups of ``--layers-per-group``
layers, for each ``--num-blocks``. Reported per point: the CPU time of
``read()`` (validation, metadata staging, launches) and the wall time until
``wait()`` returns SUCCESS. Every sample checks the first and last page it read
of every buffer.
Pages are drawn afresh per sample from the whole pool, so the source L2 is cold.

    python benchmark/kv_transfer/bench_kv_peer_read.py --pages 16 1024 7187

By default both run on this node (GPU 0 -> GPU 1). Across nodes of one NVLink
domain, start one role per node with a directory both can see; this is also the
cross-node correctness check:

    node A: python bench_kv_peer_read.py --role owner  --rendezvous /shared/dir
    node B: python bench_kv_peer_read.py --role reader --rendezvous /shared/dir
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import statistics
import time
from pathlib import Path

import numpy as np
import torch


class _FileConn:
    """Pipe-like messages through a shared directory, for roles on different nodes."""

    def __init__(self, directory, me: str, peer: str, timeout: float = 600.0):
        self._dir, self._me, self._peer = Path(directory), me, peer
        self._sent = self._received = 0
        self._timeout = timeout
        self._dir.mkdir(parents=True, exist_ok=True)

    def send(self, obj) -> None:
        path = self._dir / f"{self._me}-{self._sent}.msg"
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(obj))
        os.replace(tmp, path)
        self._sent += 1

    def recv(self):
        path = self._dir / f"{self._peer}-{self._received}.msg"
        deadline = time.monotonic() + self._timeout
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"no message {path}")
            time.sleep(0.05)
        self._received += 1
        return pickle.loads(path.read_bytes())


def _geometry(args):
    page_bytes = tuple(args.page_bytes)
    region_bytes = args.layers * args.capacity * sum(page_bytes)
    return page_bytes, region_bytes


def _src_offset(args, page_bytes, layer, cls):
    # Class-major, then layer, each buffer holding `capacity` pages.
    return args.layers * args.capacity * sum(page_bytes[:cls]) + (
        layer * args.capacity * page_bytes[cls]
    )


def _owner(args, conn):
    from cuda.bindings import driver as drv

    from sglang.srt.mem_cache.kv_peer_read import KvPeerExporter
    from sglang.srt.utils.cuda_vmm_utils import (
        align_up,
        check_drv,
        get_allocation_granularity,
        make_device_allocation_prop,
        make_rw_access_desc,
        tensor_from_pointer,
    )

    dev = args.owner_device
    torch.cuda.set_device(dev)
    page_bytes, region_bytes = _geometry(args)
    fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    prop = make_device_allocation_prop(dev, handle_types=int(fabric))
    size = align_up(region_bytes, get_allocation_granularity(prop))
    handle = check_drv(drv.cuMemCreate(size, prop, 0), "cuMemCreate")
    va = int(check_drv(drv.cuMemAddressReserve(size, 0, 0, 0), "reserve"))
    check_drv(drv.cuMemMap(va, size, 0, handle, 0), "map")
    check_drv(drv.cuMemSetAccess(va, size, [make_rw_access_desc(dev)], 1), "access")
    words = tensor_from_pointer(
        va, size, dtype=torch.int32, shape=(size // 4,), device_id=dev
    )
    chunk = 1 << 28
    for start in range(0, words.numel(), chunk):  # word i holds i, mod 2**31
        n = min(chunk, words.numel() - start)
        torch.arange(start, start + n, out=words[start : start + n])
    torch.cuda.synchronize()

    exporter = KvPeerExporter(dev)
    leases = []
    while conn.recv() == "export":
        desc = exporter.export(handle, size, region_id="bench")
        leases.append(desc.lease_id)
        conn.send(desc.to_bytes())
    for lease in leases:  # every reader unmapped before sending "done"
        exporter.release_export(lease)
    check_drv(drv.cuMemUnmap(va, size), "unmap")
    check_drv(drv.cuMemAddressFree(va, size), "free")
    check_drv(drv.cuMemRelease(handle), "release")


def _verify(args, page_bytes, dst, src_pages, dst_pages):
    bad = 0
    for i in (0, len(src_pages) - 1):
        for layer in range(args.layers):
            for cls, pb in enumerate(page_bytes):
                first = (
                    _src_offset(args, page_bytes, layer, cls) + int(src_pages[i]) * pb
                ) // 4
                want = torch.arange(
                    first, first + pb // 4, dtype=torch.int32, device=dst[cls].device
                )
                got = dst[cls][layer, int(dst_pages[i])].view(torch.int32)
                bad += not torch.equal(got, want)
    return bad


def _run(args, conn):
    from sglang.srt.mem_cache.kv_peer_read import (
        KvPeerReader,
        PeerBuffer,
        PeerRegionDescriptor,
        ReadGroup,
        ReadState,
    )

    dev = args.reader_device
    torch.cuda.set_device(dev)
    page_bytes, _ = _geometry(args)
    dst = [
        torch.zeros(
            (args.layers, args.capacity, pb), dtype=torch.uint8, device=f"cuda:{dev}"
        )
        for pb in page_bytes
    ]
    rng = np.random.default_rng(args.seed)
    results = []
    for num_blocks in args.num_blocks:
        reader = KvPeerReader(dev, num_blocks=num_blocks)
        conn.send("export")
        region = reader.import_region(PeerRegionDescriptor.from_bytes(conn.recv()))
        # Built once per region, as a pool owner would: a reused tuple is validated once.
        buffer_sets = [
            tuple(
                PeerBuffer(
                    src_region=region,
                    src_offset=_src_offset(args, page_bytes, layer, cls),
                    dst_base=dst[cls][layer].data_ptr(),
                    page_bytes=pb,
                    dst_num_pages=args.capacity,
                )
                for layer in range(
                    first, min(first + args.layers_per_group, args.layers)
                )
                for cls, pb in enumerate(page_bytes)
            )
            for first in range(0, args.layers, args.layers_per_group)
        ]
        for in_flight, pages in ((n, p) for n in args.in_flight for p in args.pages):
            if (in_flight + 1) * pages > args.capacity:
                continue  # not enough disjoint destination pages
            cpu, wall, bad = [], [], 0
            for it in range(args.warmup + args.iters):
                src_pages = rng.choice(args.capacity, pages, replace=False)
                # Disjoint destination pages for the measured read and the others.
                dst_all = rng.choice(
                    args.capacity, (in_flight + 1) * pages, replace=False
                )
                dst_pages = dst_all[:pages]
                groups = [
                    ReadGroup(buffers=buffers, src_pages=src_pages, dst_pages=dst_pages)
                    for buffers in buffer_sets
                ]
                others = (
                    [
                        reader.read(
                            [
                                ReadGroup(buffers=b, src_pages=src_pages, dst_pages=d)
                                for b in buffer_sets
                            ]
                        )
                        for d in np.split(dst_all[pages:], in_flight)
                    ]
                    if in_flight
                    else []
                )
                if not others:
                    torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                batch = reader.read(groups)
                t1 = time.perf_counter()
                status = reader.wait(batch)
                t2 = time.perf_counter()
                assert status.state == ReadState.SUCCESS, status
                reader.release(batch)
                for other in others:
                    assert reader.wait(other).state == ReadState.SUCCESS
                    reader.release(other)
                if it >= args.warmup:
                    cpu.append((t1 - t0) * 1e3)
                    wall.append((t2 - t0) * 1e3)
                    bad += _verify(args, page_bytes, dst, src_pages, dst_pages)
            nbytes = pages * args.layers * sum(page_bytes)
            point = dict(
                num_blocks=num_blocks,
                pages=pages,
                in_flight=in_flight,
                groups=len(groups),
                bytes=nbytes,
                read_cpu_ms=statistics.median(cpu),
                wall_ms=statistics.median(wall),
                gbps=nbytes / statistics.median(wall) / 1e6,
                verify_bad=bad,
            )
            results.append(point)
            print(
                f"blocks={num_blocks:4d} pages={pages:6d} in_flight={in_flight:3d} "
                f"groups={len(groups):3d} "
                f"read()={point['read_cpu_ms']:7.2f} ms wall={point['wall_ms']:8.2f} ms "
                f"{point['gbps']:6.1f} GB/s bad={bad}",
                flush=True,
            )
        reader.unmap(region)
        reader.close()
    conn.send("done")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, default=78)
    parser.add_argument("--page-bytes", type=int, nargs="+", default=[36864, 8448])
    parser.add_argument("--capacity", type=int, default=8200)
    parser.add_argument("--pages", type=int, nargs="+", default=[16, 128, 1024, 7187])
    parser.add_argument("--layers-per-group", type=int, default=13)
    parser.add_argument("--num-blocks", type=int, nargs="+", default=[32, 64])
    parser.add_argument(
        "--in-flight",
        type=int,
        nargs="+",
        default=[0],
        help="other reads left outstanding while one is timed (its wall time then "
        "includes theirs)",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="write the results here")
    parser.add_argument("--role", choices=("owner", "reader"), help="cross-node mode")
    parser.add_argument("--rendezvous", help="directory shared by both roles")
    parser.add_argument("--owner-device", type=int)
    parser.add_argument("--reader-device", type=int)
    args = parser.parse_args()
    if args.role is not None and args.rendezvous is None:
        parser.error("--role needs --rendezvous")
    # One node: GPU 0 -> GPU 1. Across nodes: each role on its own GPU 0.
    if args.owner_device is None:
        args.owner_device = 0
    if args.reader_device is None:
        args.reader_device = 1 if args.role is None else 0

    if args.role == "owner":
        _owner(args, _FileConn(args.rendezvous, "owner", "reader"))
        return
    if args.role == "reader":
        results = _run(args, _FileConn(args.rendezvous, "reader", "owner"))
        _write_json(args, results)
        return

    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    owner = ctx.Process(target=_owner, args=(args, child))
    owner.start()
    try:
        results = _run(args, conn)
    finally:
        owner.join(timeout=120)
        if owner.is_alive():
            owner.kill()
    _write_json(args, results)


def _write_json(args, results) -> None:
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(args=vars(args), results=results), f, indent=1)


if __name__ == "__main__":
    main()
