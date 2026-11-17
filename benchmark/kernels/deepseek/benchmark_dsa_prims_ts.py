#!/usr/bin/env python3
"""Microbenchmark: the DSA sparse-MLA kernels behind prims_ts, per kernel.

One layer's work for a DSA model (D576 MLA: kv_lora_rank 512 + RoPE 64, top-k
2048) at a given number of query rows (decode batch per rank, or prefill rows):

    attn  prims_ts-nvfp4   FlashInfer Prims-TS over the 352 B/token NVFP4 cache
    attn  prims_ts-fp8     FlashInfer Prims-TS over the raw 576 B/token FP8 cache
    attn  trtllm-fp8       FlashInfer trtllm-gen sparse MLA decode, same FP8 cache
    kv    write-nvfp4      BF16 K -> 352 B rows (prims_k_cache writer)
    kv    write-fp8        BF16 K -> 576 B rows (E4M3 cast + scatter, the FP8 route)
    q     cast-fp8         BF16 q_nope/q_rope -> E4M3 [rows, H, 576] (both routes)

Each row attends to top-k slots drawn at random from a pool of
--pool-slots tokens, so the KV reads are scattered as in serving. Timings are
CUDA-graph replays with L2 flushed before every call (--warm-l2 skips the
flush), median over --iters, in microseconds; LOWER = FASTER. GB/s counts the
KV row bytes read (attention) or written (KV writers) plus Q/O traffic.

The attention outputs are cross-checked: prims_ts-fp8 vs trtllm-fp8 must agree
to FP8 rounding, and prims_ts-nvfp4 vs prims_ts-fp8 shows the NVFP4 error.

Usage (single SM100/SM103 GPU, FlashInfer with Prims-TS nvfp4.fp8rope):
    python benchmark/kernels/deepseek/benchmark_dsa_prims_ts.py
    python benchmark/kernels/deepseek/benchmark_dsa_prims_ts.py --rows 1,8,32 --heads 128
    python benchmark/kernels/deepseek/benchmark_dsa_prims_ts.py --json out.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
from typing import Callable, Optional

import torch

FP8 = torch.float8_e4m3fn
KV_LORA, ROPE, TOPK = 512, 64, 2048
DIM = KV_LORA + ROPE


def time_us(
    fn: Callable[[], object], iters: int, flush: Optional[torch.Tensor]
) -> float:
    """Median microseconds of one CUDA-graph replay of fn."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):  # plans compile and allocate on eager calls
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


class Case:
    def __init__(self, args: argparse.Namespace, rows: int, gen: torch.Generator):
        dev = torch.device("cuda")
        self.rows, self.heads = rows, args.heads
        self.sm_scale = (args.qk_nope_head_dim + ROPE) ** -0.5
        slots = args.pool_slots
        # One BF16 K source for every format, so the outputs are comparable.
        k_nope = torch.randn(slots, KV_LORA, generator=gen, device=dev).bfloat16()
        k_rope = torch.randn(slots, ROPE, generator=gen, device=dev).bfloat16()
        self.k_nope, self.k_rope = k_nope, k_rope
        self.loc = torch.arange(slots, device=dev)
        self.kv_fp8 = torch.cat([k_nope, k_rope], -1).to(FP8).view(slots, 1, DIM)
        self.kv_nvfp4 = torch.zeros((slots, 1, 352), dtype=torch.uint8, device=dev)
        from sglang.srt.layers.attention.dsa.prims_k_cache import (
            quantize_prims_k_cache_into,
        )

        quantize_prims_k_cache_into(k_nope, k_rope, self.kv_nvfp4, self.loc)
        self.q_nope = torch.randn(rows, args.heads, KV_LORA, generator=gen, device=dev)
        self.q_rope = torch.randn(rows, args.heads, ROPE, generator=gen, device=dev)
        self.q_nope, self.q_rope = self.q_nope.bfloat16(), self.q_rope.bfloat16()
        self.q8 = torch.cat([self.q_nope, self.q_rope], -1).to(FP8)
        # Random slots per row (repeats are rare at 2048 of ~2M and harmless).
        self.indices = torch.randint(
            0, slots, (rows, TOPK), generator=gen, device=dev, dtype=torch.int32
        )
        self.lengths = torch.full((rows,), TOPK, dtype=torch.int32, device=dev)


def bench_prims_ts(case: Case, fmt: str, iters, flush):
    from sglang.srt.layers.attention.dsa.prims_ts import PrimsTSSparseMLA

    runner = PrimsTSSparseMLA(torch.device("cuda"), fmt)
    kv = case.kv_nvfp4 if fmt == "nvfp4.fp8rope" else case.kv_fp8
    result = {}

    def run():
        result["out"], _ = runner.forward(
            q=case.q8,
            kv_cache=kv,
            indices=case.indices,
            lengths=case.lengths,
            sm_scale=case.sm_scale,
            kv_scale=1.0,
        )

    return time_us(run, iters, flush), result


def bench_trtllm(case: Case, args, iters, flush):
    import flashinfer.decode

    fn = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
    params = inspect.signature(fn).parameters
    workspace = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    kwargs = dict(
        query=case.q8.view(case.rows, 1, case.heads, DIM),
        kv_cache=case.kv_fp8.view(-1, 1, args.page_size, DIM),
        workspace_buffer=workspace,
        qk_nope_head_dim=args.qk_nope_head_dim,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        block_tables=case.indices.unsqueeze(1),
        seq_lens=torch.full(
            (case.rows,), args.pool_slots, dtype=torch.int32, device="cuda"
        ),
        max_seq_len=args.pool_slots,
        sparse_mla_top_k=TOPK,
        bmm1_scale=case.sm_scale,
        backend="trtllm-gen",
    )
    if "multi_ctas_kv_counter_buffer" in params:
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )

        kwargs["multi_ctas_kv_counter_buffer"] = (
            make_persistent_multi_ctas_kv_counter_buffer(
                torch.device("cuda"), case.heads, case.rows
            )
        )
    result = {}

    def run():
        result["out"] = fn(**kwargs).view(case.rows, case.heads, KV_LORA)

    return time_us(run, iters, flush), result


def bench_write_nvfp4(case: Case, tokens: int, iters, flush):
    from sglang.srt.layers.attention.dsa.prims_k_cache import (
        quantize_prims_k_cache_into,
    )

    loc = case.loc[:tokens]
    return time_us(
        lambda: quantize_prims_k_cache_into(
            case.k_nope[:tokens], case.k_rope[:tokens], case.kv_nvfp4, loc
        ),
        iters,
        flush,
    )


def bench_write_fp8(case: Case, tokens: int, iters, flush):
    from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton

    loc = case.loc[:tokens]
    dst = case.kv_fp8.view(torch.uint8)

    def run():
        nope = case.k_nope[:tokens].unsqueeze(1).to(FP8).view(torch.uint8)
        rope = case.k_rope[:tokens].unsqueeze(1).to(FP8).view(torch.uint8)
        set_mla_kv_buffer_triton(dst, loc, nope, rope)

    return time_us(run, iters, flush)


def bench_q_cast(case: Case, iters, flush):
    from sglang.kernels.ops.attention.mla_qk_prep import concat_and_cast_q_fp8_heads

    out = torch.empty_like(case.q8)
    return time_us(
        lambda: concat_and_cast_q_fp8_heads(out, case.q_nope, case.q_rope),
        iters,
        flush,
    )


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def run_rows(args, rows: int, flush, results: list) -> None:
    gen = torch.Generator(device="cuda").manual_seed(args.seed + rows)
    case = Case(args, rows, gen)
    q_bytes = rows * args.heads * DIM  # E4M3
    o_bytes = rows * args.heads * KV_LORA * 2  # BF16
    entries = []

    t, nv = bench_prims_ts(case, "nvfp4.fp8rope", args.iters, flush)
    entries.append(("attn", "prims_ts-nvfp4", t, rows * TOPK * 352 + q_bytes + o_bytes))
    t, fp = bench_prims_ts(case, "native", args.iters, flush)
    entries.append(("attn", "prims_ts-fp8", t, rows * TOPK * DIM + q_bytes + o_bytes))
    checks: dict[str, object] = {"nvfp4_vs_fp8_rel_l2": rel(nv["out"], fp["out"])}
    if rows <= args.trtllm_max_rows:
        try:
            t, tr = bench_trtllm(case, args, args.iters, flush)
            entries.append(
                ("attn", "trtllm-fp8", t, rows * TOPK * DIM + q_bytes + o_bytes)
            )
            checks["prims_fp8_vs_trtllm_rel_l2"] = rel(fp["out"], tr["out"])
        except Exception as exc:  # report and keep going: optional baseline
            checks["trtllm_error"] = repr(exc)[:200]
    tokens = rows  # one new token per row (decode); prefill writes `rows` tokens
    t = bench_write_nvfp4(case, tokens, args.iters, flush)
    entries.append(("kv", "write-nvfp4", t, tokens * (352 + DIM * 2)))
    t = bench_write_fp8(case, tokens, args.iters, flush)
    entries.append(("kv", "write-fp8", t, tokens * (DIM + DIM * 2)))
    t = bench_q_cast(case, args.iters, flush)
    entries.append(("q", "cast-fp8", t, q_bytes * 3))

    for kind, name, us, nbytes in entries:
        results.append(
            dict(rows=rows, kind=kind, kernel=name, us=us, gbps=nbytes / us / 1e3)
        )
        print(
            f"{rows:>6} {kind:>4} {name:<16} {us:>10.2f} us {nbytes / us / 1e3:>9.1f} GB/s"
        )
    print(
        f"{rows:>6} check {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in checks.items()})}"
    )
    results.append(dict(rows=rows, kind="check", **checks))
    del case
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(
        description="Per-kernel DSA sparse-MLA benchmark for prims_ts."
    )
    p.add_argument("--rows", default="1,2,4,8,16,32,64,128,1024,4096")
    p.add_argument("--heads", type=int, default=64, help="query heads per rank")
    p.add_argument("--qk-nope-head-dim", type=int, default=192, help="sets sm_scale")
    p.add_argument("--pool-slots", type=int, default=1 << 21)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--trtllm-max-rows", type=int, default=128, help="decode shapes")
    p.add_argument("--warm-l2", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    major, minor = torch.cuda.get_device_capability()
    print(
        f"{torch.cuda.get_device_name()} SM{major}{minor}, heads {args.heads}, "
        f"pool {args.pool_slots} slots, L2 {'warm' if args.warm_l2 else 'flushed'}"
    )
    flush = None
    if not args.warm_l2:
        flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    print(f"{'rows':>6} {'kind':>4} {'kernel':<16} {'time':>13} {'bandwidth':>14}")
    results: list = []
    for rows in (int(r) for r in args.rows.split(",")):
        run_rows(args, rows, flush, results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "args": vars(args),
                    "device": torch.cuda.get_device_name(),
                    "results": results,
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
