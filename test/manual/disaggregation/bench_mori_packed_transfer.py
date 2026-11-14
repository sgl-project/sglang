"""Two-process MoRI PD transfer correctness/latency harness, not model serving.

Same-node RDMA (disable automatic XGMI):
  MORI_DISABLE_AUTO_XGMI=1 torchrun --standalone --nproc-per-node=2 \
    test/manual/disaggregation/bench_mori_packed_transfer.py --mode baseline

Two-node: torchrun --nnodes=2 --nproc-per-node=1 --node-rank={0,1}
  --master-addr=<rank0-eno0-IP> --master-port=29741 <script> ...
Set MORI_BENCH_BIND_HOST separately on each node to its routable management IP.
Use an allocation and a container with GPU/RDMA devices and matching host verbs.
The receiver's unrelated pages must remain byte-exact sentinel values.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


class CountingEngine:
    """Count descriptors actually submitted by either implementation."""

    def __init__(self, engine):
        self.engine = engine
        self.total_register_calls = 0
        self.total_register_us = 0.0
        self.reset()

    def reset(self):
        self.calls = self.sessions = self.descriptors = self.bytes = 0
        self.submit_us = self.deregister_us = 0.0
        self.deregister_calls = 0

    def snapshot(self):
        return dict(
            batch_write_calls=self.calls,
            sessions=self.sessions,
            descriptors=self.descriptors,
            bytes=self.bytes,
            batch_write_submit_us=self.submit_us,
            deregister_calls=self.deregister_calls,
            mr_deregister_us=self.deregister_us,
        )

    def batch_write(self, *args, **kwargs):
        sizes = args[4] if len(args) > 4 else kwargs["sizes"]
        self.calls += 1
        self.sessions += len(sizes)
        self.descriptors += sum(len(entries) for entries in sizes)
        self.bytes += sum(sum(entries) for entries in sizes)
        started = time.perf_counter()
        try:
            return self.engine.batch_write(*args, **kwargs)
        finally:
            self.submit_us += (time.perf_counter() - started) * 1e6

    def deregister_memory(self, *args, **kwargs):
        self.deregister_calls += 1
        started = time.perf_counter()
        try:
            return self.engine.deregister_memory(*args, **kwargs)
        finally:
            self.deregister_us += (time.perf_counter() - started) * 1e6

    def register_memory(self, *args, **kwargs):
        self.total_register_calls += 1
        started = time.perf_counter()
        try:
            return self.engine.register_memory(*args, **kwargs)
        finally:
            self.total_register_us += (time.perf_counter() - started) * 1e6

    def __getattr__(self, name):
        return getattr(self.engine, name)


def wait_statuses(statuses, timeout):
    """Drive nonblocking MoRI progress with a deadline; keep buffers alive."""
    deadline = time.monotonic() + timeout
    for status in statuses:
        while status.InProgress():
            if time.monotonic() > deadline:
                raise TimeoutError("MoRI transfer did not drain before the deadline")
        if not status.Succeeded():
            raise RuntimeError(f"MoRI transfer failed: {status.Message()}")


def exchange(value):
    result = [None, None]
    dist.all_gather_object(result, value)
    return result[1 - dist.get_rank()]


def bind_host():
    configured = os.environ.get("MORI_BENCH_BIND_HOST")
    if configured:
        return configured
    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((master, 9))
        return sock.getsockname()[0]


def make_case(args):
    """Deterministic byte layout with real TP head intersections and page maps.

    Each full/SWA K/V layer has a separate memory registration. SWA is a finite
    ring pool and transfers only its live suffix. This tests the byte contract;
    it does not claim any particular model uses this exact pool representation.
    """
    if args.heads % args.prefill_tp or args.heads % args.decode_tp:
        raise ValueError("heads must divide both attention TP sizes")
    sh, dh = args.heads // args.prefill_tp, args.heads // args.decode_tp
    ss, ds = args.prefill_rank * sh, args.decode_rank * dh
    lo, hi = max(ss, ds), min(ss + sh, ds + dh)
    if hi <= lo:
        raise ValueError("selected source/decode TP ranks have no shared heads")
    src_row_bytes, dst_row_bytes = sh * args.head_dim * 2, dh * args.head_dim * 2
    src_head_off, dst_head_off = (
        (lo - ss) * args.head_dim * 2,
        (lo - ds) * args.head_dim * 2,
    )
    slice_bytes = (hi - lo) * args.head_dim * 2
    rng = np.random.default_rng(args.seed)
    components = []
    for layout in args.layouts.split(","):
        live = args.tokens if layout == "full" else min(args.tokens, args.swa_window)
        pages = (live + args.page_size - 1) // args.page_size
        # Extra pages preserve guard rows and make source/destination unrelated.
        pool_pages = pages + 7
        # Leave the reserved zero page untouched as an additional guard.
        src_pages = rng.permutation(np.arange(1, pool_pages))[:pages]
        dst_pages = rng.permutation(np.arange(1, pool_pages))[:pages]
        for layer in range(args.layers):
            for kv in ("k", "v"):
                tag = len(components) * 37 + 1
                src_size = pool_pages * args.page_size * src_row_bytes
                dst_size = pool_pages * args.page_size * dst_row_bytes
                source = (
                    (np.arange(src_size, dtype=np.uint64) * 13 + tag) % 251 + 1
                ).astype(np.uint8)
                expected = np.full(dst_size, 0xA5, dtype=np.uint8)
                src_offsets, dst_offsets = [], []
                for t in range(live):
                    sr = (
                        int(src_pages[t // args.page_size]) * args.page_size
                        + t % args.page_size
                    )
                    dr = (
                        int(dst_pages[t // args.page_size]) * args.page_size
                        + t % args.page_size
                    )
                    so, do = (
                        sr * src_row_bytes + src_head_off,
                        dr * dst_row_bytes + dst_head_off,
                    )
                    src_offsets.append(so)
                    dst_offsets.append(do)
                    expected[do : do + slice_bytes] = source[so : so + slice_bytes]
                components.append(
                    dict(
                        name=f"{layout}.layer{layer}.{kv}",
                        source=source,
                        expected=expected,
                        src_offsets=src_offsets,
                        dst_offsets=dst_offsets,
                        sizes=[slice_bytes] * live,
                        layout=layout,
                        layer=layer,
                        kv=kv,
                        src_pages=src_pages.tolist(),
                        dst_pages=dst_pages.tolist(),
                        src_page_bytes=args.page_size * src_row_bytes,
                        dst_page_bytes=args.page_size * dst_row_bytes,
                        head_offset_bytes=src_head_off,
                        dst_head_offset_bytes=dst_head_off,
                    )
                )
    return components


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("baseline", "packed", "all"), default="baseline"
    )
    parser.add_argument(
        "--order", default=None, help="explicit measurement order, e.g. packed,baseline"
    )
    parser.add_argument("--transport", choices=("rdma", "xgmi"), default="rdma")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--swa-window", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--layouts", default="full,swa")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--prefill-tp", type=int, default=2)
    parser.add_argument("--decode-tp", type=int, default=4)
    parser.add_argument("--prefill-rank", type=int, default=0)
    parser.add_argument("--decode-rank", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--qp", type=int, default=4)
    parser.add_argument("--gpu-offset", type=int, default=0)
    parser.add_argument("--packed-capacity-mib", type=int, default=64)
    parser.add_argument(
        "--packed-capacity-bytes",
        type=int,
        default=None,
        help="override capacity for small multi-batch correctness tests",
    )
    args = parser.parse_args()
    modes = (
        args.order.split(",")
        if args.order
        else (["baseline", "packed"] if args.mode == "all" else [args.mode])
    )
    if (
        not modes
        or len(set(modes)) != len(modes)
        or set(modes) - {"baseline", "packed"}
    ):
        parser.error("order must list baseline and/or packed once each")
    needs_packed = "packed" in modes
    if set(args.layouts.split(",")) - {"full", "swa"}:
        parser.error("layouts must contain only full,swa")
    if (
        min(args.tokens, args.page_size, args.swa_window, args.layers, args.iterations)
        <= 0
    ):
        parser.error("sizes and iterations must be positive")
    if args.transport == "rdma":
        os.environ["MORI_DISABLE_AUTO_XGMI"] = "1"
    from mori.io import (
        BackendType,
        EngineDesc,
        IOEngine,
        IOEngineConfig,
        MemoryDesc,
        RdmaBackendConfig,
        XgmiBackendConfig,
    )

    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout))
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise ValueError("exactly two processes are required")
    device = int(os.environ.get("LOCAL_RANK", "0")) + args.gpu_offset
    torch.cuda.set_device(device)
    case = make_case(args)
    if needs_packed:
        if args.tokens % args.page_size or args.swa_window % args.page_size:
            raise ValueError(
                "packed page API requires page-aligned live lengths in this harness"
            )
        if any(c["dst_head_offset_bytes"] for c in case):
            raise ValueError(
                "packed API writes complete destination rows; destination head offset must be zero"
            )
    engine = IOEngine(
        key=f"pd_bench_{rank}_{os.environ.get('MASTER_PORT', '0')}",
        config=IOEngineConfig(host=bind_host(), port=0),
    )
    if args.transport == "rdma":
        config = RdmaBackendConfig(
            qp_per_transfer=args.qp,
            post_batch_size=-1,
            num_worker_threads=1,
            enable_notification=False,
        )
        engine.create_backend(BackendType.RDMA, config)
    else:
        engine.create_backend(BackendType.XGMI, XgmiBackendConfig())
    engine = CountingEngine(engine)
    engine.register_remote_engine(
        EngineDesc.unpack(exchange(engine.get_engine_desc().pack()))
    )
    tensors = [
        torch.from_numpy(c["source"]).to(device)
        if rank == 0
        else torch.full((len(c["expected"]),), 0xA5, dtype=torch.uint8, device=device)
        for c in case
    ]
    torch.cuda.synchronize()
    local_descs = [engine.register_torch_tensor(t) for t in tensors]
    peer_descs = [
        MemoryDesc.unpack(p) for p in exchange([d.pack() for d in local_descs])
    ]
    dist.barrier()

    def baseline():
        # Match conn.py's heterogeneous-TP path: one API call per K/V tensor.
        statuses = []
        for c, local_desc, remote_desc in zip(case, local_descs, peer_descs):
            statuses.extend(
                engine.batch_write(
                    [local_desc],
                    [c["src_offsets"]],
                    [remote_desc],
                    [c["dst_offsets"]],
                    [c["sizes"]],
                    [engine.allocate_transfer_uid()],
                )
            )
        wait_statuses(statuses, args.timeout)

    sender = None
    recorded = []
    if needs_packed and rank == 0:
        from sglang.srt.disaggregation.mori.packed import MoriPackedKVSender

        def completion(statuses):
            try:
                wait_statuses(statuses, args.timeout)
            except (TimeoutError, RuntimeError) as exc:
                return str(exc)
            return None

        sender = MoriPackedKVSender(
            engine=engine,
            gpu_id=device,
            size_bytes=(
                args.packed_capacity_bytes
                if args.packed_capacity_bytes is not None
                else args.packed_capacity_mib * 1024 * 1024
            ),
            wait_completion=completion,
            record_statuses=recorded.extend,
        )

    def packed():
        for layout in args.layouts.split(","):
            indices = [i for i, c in enumerate(case) if c["layout"] == layout]
            first = case[indices[0]]
            sender.send(
                src_ptrs=[tensors[i].data_ptr() for i in indices],
                dst_descs=[peer_descs[i] for i in indices],
                src_pages=first["src_pages"],
                dst_pages=first["dst_pages"],
                src_page_bytes=first["src_page_bytes"],
                dst_page_bytes=first["dst_page_bytes"],
                page_size=args.page_size,
                head_offset_bytes=first["head_offset_bytes"],
            )
        if any(status.InProgress() for status in recorded):
            raise AssertionError(
                "packed sender returned before its submissions drained"
            )
        recorded.clear()

    try:
        for mode in modes:
            timings, sender_timings, submissions = [], [], []
            for iteration in range(args.warmup + args.iterations):
                if rank == 1:
                    # Independent clear before EVERY candidate/iteration.
                    for tensor in tensors:
                        tensor.fill_(0xA5)
                torch.cuda.synchronize()
                dist.barrier()
                engine.reset()
                started = time.perf_counter()
                if rank == 0:
                    (baseline if mode == "baseline" else packed)()
                sender_elapsed = (time.perf_counter() - started) * 1e6
                dist.barrier()
                elapsed = (time.perf_counter() - started) * 1e6
                torch.cuda.synchronize()
                if rank == 1:
                    for c, tensor in zip(case, tensors):
                        if not np.array_equal(tensor.cpu().numpy(), c["expected"]):
                            raise AssertionError(
                                f"byte mismatch or overwritten guard: {c['name']}"
                            )
                dist.barrier()
                if iteration >= args.warmup:
                    timings.append(elapsed)
                    sender_timings.append(sender_elapsed)
                    submissions.append(engine.snapshot())
            if rank == 0:
                print(
                    json.dumps(
                        dict(
                            mode=mode,
                            transport=args.transport,
                            correctness="PASS",
                            payload_bytes=sum(sum(c["sizes"]) for c in case),
                            baseline_logical_segments=sum(
                                len(c["sizes"]) for c in case
                            ),
                            baseline_api_pattern="one batch_write per K/V tensor",
                            actual_submission_samples=submissions,
                            order=modes,
                            sender_submit_drain_us_samples=sender_timings,
                            roundtrip_us_samples=timings,
                            iterations=args.iterations,
                            p50_roundtrip_us=statistics.median(timings),
                            p99_roundtrip_us=percentile(timings, 99),
                            p50_sender_submit_drain_us=statistics.median(
                                sender_timings
                            ),
                            p99_sender_submit_drain_us=percentile(sender_timings, 99),
                            note="submit+drain and Gloo barrier wall time; API descriptors are not WQEs; not serving",
                        )
                    ),
                    flush=True,
                )
    finally:
        # Timeout/failure requires terminating the isolated test process group;
        # do not reuse destination pages. Sender retains unsettled registration.
        if sender is not None:
            sender.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
