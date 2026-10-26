"""Benchmark the DSA replay metadata kernels, unbounded vs live-length bounded.

Shapes: a 1M-token page table (the DSA models' default --context-length), live KV
1k / 8k / 64k per request, decode rows = bs, target verify and draft extend with
next_n = 5 (EAGLE 4/1/5). Times are CUDA-graph replays with a cold L2: every call
follows an L2 flush, and the flush-only graph's time is subtracted.

Usage::

    python test/registered/kernels/benchmark/attention/bench_dsa_metadata_bounded.py
"""

import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import get_benchmark_range
from sglang.kernels.ops.attention import dsa_metadata_bounded as bounded
from sglang.kernels.ops.attention.dsa import metadata as unbounded
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.dsa_bounded_metadata_kit import (
    WIDTH_1M,
    MetadataWorld,
    decode_bufs,
    run_decode,
    run_draft_extend,
    run_verify,
    verify_bufs,
)

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NEXT_N = 5
KV_LENS = {"1k": 1024, "8k": 8192, "64k": 65536}
BATCH_SIZES = get_benchmark_range(full_range=[1, 2, 4, 8, 16], ci_range=[1, 16])
KV_NAMES = get_benchmark_range(full_range=list(KV_LENS), ci_range=["8k"])
OPS = ["decode", "target_verify", "draft_extend"]

_WORLD = None
_FLUSH = None


def world() -> MetadataWorld:
    global _WORLD
    if _WORLD is None:
        _WORLD = MetadataWorld(WIDTH_1M, pool=40)
    return _WORLD


def cold_l2_graph_us(fn, reps: int = 10, replays: int = 30) -> float:
    """Median us per call of ``fn`` replayed in a CUDA graph, each call after an
    L2 flush; the time of a flush-only graph is subtracted."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(64 << 20, dtype=torch.int32, device="cuda")  # 256 MB

    def capture(with_fn):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                _FLUSH.zero_()
                if with_fn:
                    fn()
        return graph

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()  # compile and allocate outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs = {True: capture(True), False: capture(False)}
    times = {True: [], False: []}
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for _ in range(replays):
        for with_fn, graph in graphs.items():
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times[with_fn].append(start.elapsed_time(end) * 1000 / reps)
    median = lambda xs: sorted(xs)[len(xs) // 2]
    return max(0.0, median(times[True]) - median(times[False]))


def make_op(op: str, provider: str, bs: int, kv_len: int):
    w = world()
    g = torch.Generator().manual_seed(bs * 7919 + kv_len)
    kv = torch.randint(kv_len, kv_len + kv_len // 8 + 1, (bs,), generator=g).tolist()
    seq, req = w.batch(kv)
    fns = {
        "unbounded": (
            unbounded.fused_dsa_decode_metadata,
            unbounded.fused_dsa_target_verify_metadata,
            unbounded.fused_dsa_draft_extend_metadata,
        ),
        "bounded": (
            bounded.fused_dsa_decode_metadata_bounded,
            bounded.fused_dsa_target_verify_metadata_bounded,
            bounded.fused_dsa_draft_extend_metadata_bounded,
        ),
    }[provider]
    if op == "decode":
        bufs = decode_bufs(w, bs)
        return lambda: run_decode(fns[0], w, seq, req, bs, bufs)
    if op == "target_verify":
        bufs = verify_bufs(w, bs, NEXT_N, with_ctx=True)
        return lambda: run_verify(fns[1], w, seq, req, bs, NEXT_N, bufs)
    bufs = verify_bufs(w, bs, NEXT_N)
    # The unbounded replay also fills the static extend lengths on device.
    if provider == "unbounded":

        def extend():
            ext = torch.full((bs,), NEXT_N, dtype=torch.int32, device="cuda")
            run_draft_extend(fns[2], w, seq, req, bs, NEXT_N, bufs, ext)

        return extend
    return lambda: run_draft_extend(fns[2], w, seq, req, bs, NEXT_N, bufs, None)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["op", "kv", "bs"],
        x_vals=[(op, kv, bs) for op in OPS for kv in KV_NAMES for bs in BATCH_SIZES],
        line_arg="provider",
        line_vals=["unbounded", "bounded"],
        line_names=["Unbounded grid", "Bounded grid"],
        styles=[("orange", "--"), ("blue", "-")],
        ylabel="us",
        plot_name="dsa-replay-metadata-1m-context",
        args={},
    )
)
def benchmark_replay_metadata(op: str, kv: str, bs: int, provider: str) -> float:
    return cold_l2_graph_us(make_op(op, provider, bs, KV_LENS[kv]))


if __name__ == "__main__":
    benchmark_replay_metadata.run(print_data=True)
