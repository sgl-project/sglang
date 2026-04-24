"""Benchmark breakdown for CUTLASS W4A8 MoE decode (TP=8 dimensions).

Profiles each stage of cutlass_w4a8_moe and compares with Marlin W4A16:
  1. pre_reorder + FP8 quant
  2. get_cutlass_w4a8_moe_mm_data
  3. GEMM 1 (gate+up)
  4. SiLU + FP8 quant
  5. GEMM 2 (down)
  6. post_reorder

Usage:
    python benchmarks/bench_w4a8_moe_decode.py [--batch 1 2 4 8 16 32]
    python benchmarks/bench_w4a8_moe_decode.py --batch 1 4 8 16 32 --no-marlin
    python benchmarks/bench_w4a8_moe_decode.py --batch 1 4 8 16 32 --no-breakdown
    python benchmarks/bench_w4a8_moe_decode.py --tune-gemm2 --batch 1 4 8 16 32
"""

import argparse
import socket
from unittest.mock import patch

import torch
import torch.distributed

from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)


def init_dist():
    if not torch.distributed.is_initialized():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl",
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
    try:
        initialize_model_parallel(tensor_model_parallel_size=1)
    except AssertionError:
        pass


def pack_int4_to_int8(vals: torch.Tensor) -> torch.Tensor:
    low = vals[..., 0::2].to(torch.int8)
    high = vals[..., 1::2].to(torch.int8)
    return ((high << 4) | (low & 0x0F)).to(torch.int8)


def pack_interleave(num_experts, ref_weight, ref_scale, alignment=None):
    """Matches sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py::pack_interleave."""
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    w_q = pack_int4_to_int8(ref_weight.cpu()).cuda()
    w_q = w_q.view(num_experts, n, k // 2).contiguous()

    if alignment is None:
        alignment = 4 if k % 512 == 0 else 1

    s = ref_scale
    s = s.reshape(s.shape[0], s.shape[1], s.shape[2] // alignment, alignment)
    s = s.permute(0, 2, 1, 3)
    s = s.reshape(
        ref_scale.shape[0],
        ref_scale.shape[2] // alignment,
        ref_scale.shape[1] * alignment,
    )
    w_scale = s.contiguous()
    return w_q, w_scale


class CUDATimer:
    def __init__(self):
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.end_ev = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_ev.record()

    def stop(self) -> float:
        self.end_ev.record()
        torch.cuda.synchronize()
        return self.start_ev.elapsed_time(self.end_ev)


def _stats(vals):
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    return mean, std


# ---------------------------------------------------------------------------
# Prepare CUTLASS weights (shared across benchmarks)
# ---------------------------------------------------------------------------
def prepare_cutlass_weights(E, K, N, group_size):
    device = "cuda"
    ref_w1 = torch.randint(-8, 8, (E, 2 * N, K), dtype=torch.int8, device=device)
    scale_1 = (
        torch.randn(E, 2 * N, K // group_size, dtype=torch.bfloat16, device=device)
        * 0.005
    )
    w1_q, w1_scale = pack_interleave(E, ref_w1, scale_1)

    ref_w2 = torch.randint(-8, 8, (E, K, N), dtype=torch.int8, device=device)
    scale_2 = (
        torch.randn(E, K, N // group_size, dtype=torch.bfloat16, device=device) * 0.005
    )
    w2_q, w2_scale = pack_interleave(E, ref_w2, scale_2, alignment=1)

    del ref_w1, ref_w2, scale_1, scale_2
    torch.cuda.empty_cache()
    return w1_q, w1_scale, w2_q, w2_scale


# ---------------------------------------------------------------------------
# CUTLASS real E2E — calls cutlass_w4a8_moe directly, no per-stage sync
# ---------------------------------------------------------------------------
def benchmark_cutlass_real_e2e(
    M: int,
    w1_q,
    w1_scale,
    w2_q,
    w2_scale,
    E: int = 384,
    hidden_size: int = 7168,
    intermediate_tp: int = 256,
    topk: int = 6,
    warmup: int = 10,
    repeat: int = 50,
):
    from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    device = "cuda"
    K = hidden_size
    N = intermediate_tp

    a_strides1 = torch.full((E, 3), K, device=device, dtype=torch.int64)
    c_strides1 = torch.full((E, 3), 2 * N, device=device, dtype=torch.int64)
    a_strides2 = torch.full((E, 3), N, device=device, dtype=torch.int64)
    c_strides2 = torch.full((E, 3), K, device=device, dtype=torch.int64)
    b_strides1 = a_strides1
    s_strides13 = c_strides1
    b_strides2 = a_strides2
    s_strides2 = c_strides2
    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((E, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((E, 3), dtype=torch.int32, device=device)

    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    score = torch.randn(M, E, dtype=torch.bfloat16, device=device)
    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    topk_weights, topk_ids, _ = topk_output

    times = []
    for i in range(warmup + repeat):
        t = CUDATimer()
        t.start()
        cutlass_w4a8_moe(
            a,
            w1_q,
            w2_q,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            a_strides1,
            b_strides1,
            c_strides1,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides13,
            s_strides2,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
        )
        ms = t.stop()
        if i >= warmup:
            times.append(ms)

    mean, std = _stats(times)
    return mean, std


# ---------------------------------------------------------------------------
# CUTLASS per-stage breakdown (with per-stage sync for profiling)
# ---------------------------------------------------------------------------
def benchmark_cutlass_breakdown(
    M: int,
    w1_q,
    w1_scale,
    w2_q,
    w2_scale,
    E: int = 384,
    hidden_size: int = 7168,
    intermediate_tp: int = 256,
    topk: int = 6,
    group_size: int = 128,
    warmup: int = 10,
    repeat: int = 50,
):
    from sgl_kernel import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data

    from sglang.jit_kernel.per_tensor_absmax_fp8 import per_tensor_absmax_fp8
    from sglang.srt.layers.moe.ep_moe.kernels import (
        cutlass_w4_run_moe_ep_preproess,
        post_reorder_for_cutlass_moe,
        pre_reorder_for_cutlass_moe,
        silu_mul_static_tensorwise_quant_for_cutlass_moe,
    )
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    device = "cuda"
    K = hidden_size
    N = intermediate_tp

    a_strides1 = torch.full((E, 3), K, device=device, dtype=torch.int64)
    c_strides1 = torch.full((E, 3), 2 * N, device=device, dtype=torch.int64)
    a_strides2 = torch.full((E, 3), N, device=device, dtype=torch.int64)
    c_strides2 = torch.full((E, 3), K, device=device, dtype=torch.int64)
    b_strides1 = a_strides1
    s_strides13 = c_strides1
    b_strides2 = a_strides2
    s_strides2 = c_strides2
    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((E, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((E, 3), dtype=torch.int32, device=device)

    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    score = torch.randn(M, E, dtype=torch.bfloat16, device=device)
    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    topk_weights, topk_ids, _ = topk_output

    a1_scale = torch.empty(1, dtype=torch.float32, device=device)
    per_tensor_absmax_fp8(a, a1_scale)

    num_local_experts = E
    timings = {
        "1_pre_reorder_quant": [],
        "2_get_mm_data": [],
        "3_gemm1": [],
        "4_a2scale+silu_quant": [],
        "5_gemm2": [],
        "6_post_reorder": [],
        "total_e2e": [],
    }

    for i in range(warmup + repeat):
        src2dst = cutlass_w4_run_moe_ep_preproess(topk_ids)
        gateup_input = torch.empty(
            (M * topk, K), device=device, dtype=torch.float8_e4m3fn
        )

        te = CUDATimer()
        te.start()

        t1 = CUDATimer()
        t1.start()
        pre_reorder_for_cutlass_moe(
            a,
            gateup_input,
            src2dst,
            topk_ids,
            a1_scale,
            num_local_experts,
            topk,
            M,
            K,
        )
        t1_ms = t1.stop()

        t2 = CUDATimer()
        t2.start()
        a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
        c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
        get_cutlass_w4a8_moe_mm_data(
            topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
            num_local_experts,
            N,
            K,
        )
        t2_ms = t2.stop()

        c1 = torch.empty((M * topk, N * 2), device=device, dtype=torch.bfloat16)
        t3 = CUDATimer()
        t3.start()
        cutlass_w4a8_moe_mm(
            c1,
            gateup_input,
            w1_q,
            a1_scale.float(),
            w1_scale,
            expert_offsets[:-1],
            problem_sizes1,
            a_strides1,
            b_strides1,
            c_strides1,
            s_strides13,
            128,
            topk,
        )
        t3_ms = t3.stop()

        intermediate_q = torch.empty(
            (M * topk, N), dtype=torch.float8_e4m3fn, device=device
        )
        a2_scale = torch.empty(1, dtype=torch.float32, device=device)
        t4 = CUDATimer()
        t4.start()
        per_tensor_absmax_fp8(c1, a2_scale)
        silu_mul_static_tensorwise_quant_for_cutlass_moe(
            c1,
            intermediate_q,
            a2_scale.float(),
            expert_offsets[-1:],
            M * topk,
            N,
        )
        t4_ms = t4.stop()

        c2 = torch.empty((M * topk, K), device=device, dtype=torch.bfloat16)
        t5 = CUDATimer()
        t5.start()
        cutlass_w4a8_moe_mm(
            c2,
            intermediate_q,
            w2_q,
            a2_scale.float(),
            w2_scale,
            expert_offsets[:-1],
            problem_sizes2,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides2,
            128,
            topk,
        )
        t5_ms = t5.stop()

        output = torch.empty_like(a)
        t6 = CUDATimer()
        t6.start()
        post_reorder_for_cutlass_moe(
            c2,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            num_local_experts,
            topk,
            M,
            K,
            1.0,
        )
        t6_ms = t6.stop()

        te_ms = te.stop()

        if i >= warmup:
            timings["1_pre_reorder_quant"].append(t1_ms)
            timings["2_get_mm_data"].append(t2_ms)
            timings["3_gemm1"].append(t3_ms)
            timings["4_a2scale+silu_quant"].append(t4_ms)
            timings["5_gemm2"].append(t5_ms)
            timings["6_post_reorder"].append(t6_ms)
            timings["total_e2e"].append(te_ms)

    e2e_mean = sum(timings["total_e2e"]) / repeat

    print(f"\n{'='*70}")
    print(f"  [CUTLASS W4A8 Breakdown] M={M}, E={E}, N_tp={N}, K={K}, topk={topk}")
    print(f"  GEMM1: N={2*N}, K={K}   GEMM2: N={K}, K={N}")
    print(f"{'='*70}")
    print(f"  {'Stage':<25s}  {'Mean(ms)':>10s}  {'Std(ms)':>10s}  {'Pct':>8s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*8}")

    stage_sum = 0.0
    for name, vals in timings.items():
        mean, std = _stats(vals)
        pct = mean / e2e_mean * 100 if e2e_mean > 0 else 0
        print(f"  {name:<25s}  {mean:10.3f}  {std:10.3f}  {pct:7.1f}%")
        if name != "total_e2e":
            stage_sum += mean

    overhead = e2e_mean - stage_sum
    print(
        f"  {'overhead(sync+alloc+py)':<25s}  {overhead:10.3f}  {'':>10s}  {overhead/e2e_mean*100:7.1f}%"
    )

    return e2e_mean


# ---------------------------------------------------------------------------
# Marlin W4A16 E2E benchmark
# ---------------------------------------------------------------------------
def prepare_marlin_weights(E, K, N, group_size):
    from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
    from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales

    device = "cuda"
    pack_factor = 8

    print("  Preparing Marlin W4A16 weights (one-time, ~30s for E=384)...")

    w1_gptq = torch.randint(
        0,
        2**16,
        (E, K // pack_factor, 2 * N),
        dtype=torch.int32,
        device=device,
    )
    w2_gptq = torch.randint(
        0,
        2**16,
        (E, N // pack_factor, K),
        dtype=torch.int32,
        device=device,
    )

    w1_scale = (
        torch.randn(E, K // group_size, 2 * N, dtype=torch.bfloat16, device=device)
        * 0.005
    )
    w2_scale = (
        torch.randn(E, N // group_size, K, dtype=torch.bfloat16, device=device) * 0.005
    )

    sort_indices = torch.empty((E, 0), dtype=torch.int32, device=device)

    w1 = gptq_marlin_moe_repack(w1_gptq, sort_indices, K, 2 * N, 4)
    w2 = gptq_marlin_moe_repack(w2_gptq, sort_indices, N, K, 4)
    del w1_gptq, w2_gptq

    w1_scale = marlin_moe_permute_scales(w1_scale, K, 2 * N, group_size)
    w2_scale = marlin_moe_permute_scales(w2_scale, N, K, group_size)

    torch.cuda.empty_cache()
    print("  Marlin weights ready.")
    return w1, w2, w1_scale, w2_scale


def benchmark_marlin_e2e(
    M: int,
    marlin_w1,
    marlin_w2,
    marlin_w1_scale,
    marlin_w2_scale,
    E: int = 384,
    hidden_size: int = 7168,
    intermediate_tp: int = 256,
    topk: int = 6,
    warmup: int = 10,
    repeat: int = 50,
):
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        fused_marlin_moe,
    )
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    device = "cuda"
    K = hidden_size

    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    score = torch.randn(M, E, dtype=torch.bfloat16, device=device)
    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    topk_weights, topk_ids, _ = topk_output

    times = []
    for i in range(warmup + repeat):
        t = CUDATimer()
        t.start()
        fused_marlin_moe(
            hidden_states=a,
            w1=marlin_w1,
            w2=marlin_w2,
            w1_scale=marlin_w1_scale,
            w2_scale=marlin_w2_scale,
            gating_output=score,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_bits=4,
        )
        ms = t.stop()
        if i >= warmup:
            times.append(ms)

    mean, std = _stats(times)
    return mean, std


# ---------------------------------------------------------------------------
# GEMM2-only isolated benchmark (for tile config tuning)
# ---------------------------------------------------------------------------
GEMM2_TUNE_CONFIGS = {
    0: "PP< 64, 16,128, 1,1,1>  (baseline)",
    1: "PP< 64, 32,128, 1,1,1>  (2x TileN)",
    2: "PP< 64, 64,128, 1,1,1>  (4x TileN)",
    3: "PP< 64, 16,128, 2,1,1>  (2x ClusterM)",
    4: "CO<128, 16,128, 1,1,1>  (cooperative)",
    5: "CO<128, 32,128, 1,1,1>  (CO + 2x TileN)",
}

GEMM2_TUNE_CONFIGS_MED = {
    0: "PP<128, 32,128, 1,1,1>  (baseline)",
    1: "PP<128, 64,128, 1,1,1>  (2x TileN)",
    2: "PP<128, 32,128, 2,1,1>  (2x ClusterM)",
    3: "CO<128, 16,128, 1,1,1>  (cooperative)",
    4: "CO<128, 32,128, 1,1,1>  (CO + TileN=32)",
    5: "CO<128, 32,128, 2,1,1>  (CO + cluster)",
}


def benchmark_gemm2_only(
    M: int,
    w2_q,
    w2_scale,
    E: int = 384,
    hidden_size: int = 7168,
    intermediate_tp: int = 256,
    topk: int = 6,
    group_size: int = 128,
    warmup: int = 20,
    repeat: int = 200,
):
    """Benchmark only the GEMM2 kernel (cutlass_w4a8_moe_mm for down-proj)."""
    from sgl_kernel import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data

    from sglang.jit_kernel.per_tensor_absmax_fp8 import per_tensor_absmax_fp8
    from sglang.srt.layers.moe.ep_moe.kernels import (
        cutlass_w4_run_moe_ep_preproess,
    )
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    device = "cuda"
    K = hidden_size
    N = intermediate_tp

    a_strides2 = torch.full((E, 3), N, device=device, dtype=torch.int64)
    c_strides2 = torch.full((E, 3), K, device=device, dtype=torch.int64)
    b_strides2 = a_strides2
    s_strides2 = c_strides2
    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((E, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((E, 3), dtype=torch.int32, device=device)

    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    score = torch.randn(M, E, dtype=torch.bfloat16, device=device)
    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    _, topk_ids, _ = topk_output

    src2dst = cutlass_w4_run_moe_ep_preproess(topk_ids)

    a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
    c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
    get_cutlass_w4a8_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        E,
        N,
        K,
    )

    intermediate_q = torch.randn(M * topk, N, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    a2_scale = torch.empty(1, dtype=torch.float32, device=device)
    per_tensor_absmax_fp8(
        torch.randn(M * topk, 2 * N, dtype=torch.bfloat16, device=device), a2_scale
    )

    c2 = torch.empty((M * topk, K), device=device, dtype=torch.bfloat16)

    times = []
    for i in range(warmup + repeat):
        t = CUDATimer()
        t.start()
        cutlass_w4a8_moe_mm(
            c2,
            intermediate_q,
            w2_q,
            a2_scale.float(),
            w2_scale,
            expert_offsets[:-1],
            problem_sizes2,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides2,
            128,
            topk,
        )
        ms = t.stop()
        if i >= warmup:
            times.append(ms)

    mean, std = _stats(times)
    return mean, std


def run_gemm2_tuning_sweep(args):
    """Run GEMM2 tile config sweep: test each config for all batch sizes."""
    import os

    E, K, N = args.E, args.hidden, args.intermediate_tp

    print("  Preparing CUTLASS W4A8 weights (w2 only)...")
    _, _, w2_q, w2_scale = prepare_cutlass_weights(E, K, N, 128)
    print("  Weights ready.\n")

    num_configs = max(len(GEMM2_TUNE_CONFIGS), len(GEMM2_TUNE_CONFIGS_MED))
    results = {}

    for cfg_id in range(num_configs):
        os.environ["SGL_GEMM2_TUNE"] = str(cfg_id)
        cfg_desc_small = GEMM2_TUNE_CONFIGS.get(cfg_id, f"config {cfg_id}")
        cfg_desc_med = GEMM2_TUNE_CONFIGS_MED.get(cfg_id, f"config {cfg_id}")

        print(f"{'='*74}")
        print(f"  Config {cfg_id}")
        print(f"    m<=8:  {cfg_desc_small}")
        print(f"    m<=32: {cfg_desc_med}")
        print(f"{'='*74}")

        results[cfg_id] = {}
        for batch in args.batch:
            try:
                mean, std = benchmark_gemm2_only(
                    M=batch,
                    w2_q=w2_q,
                    w2_scale=w2_scale,
                    E=E,
                    hidden_size=K,
                    intermediate_tp=N,
                    topk=args.topk,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
                results[cfg_id][batch] = (mean, std)
                print(f"    M={batch:4d}  GEMM2={mean:.4f}ms (std={std:.4f})")
            except Exception as e:
                results[cfg_id][batch] = None
                print(f"    M={batch:4d}  FAILED: {e}")
        print()

    os.environ.pop("SGL_GEMM2_TUNE", None)

    print(f"\n{'='*74}")
    print(f"  GEMM2 Tile Config Comparison (N={K}, K={N})")
    print(f"{'='*74}")

    hdr = f"  {'Config':>6s}  {'Description':<36s}"
    for b in args.batch:
        hdr += f"  {'M='+str(b):>8s}"
    print(hdr)
    print(f"  {'-'*6}  {'-'*36}" + "".join(f"  {'-'*8}" for _ in args.batch))

    baseline = results.get(0, {})
    for cfg_id in range(num_configs):
        desc = GEMM2_TUNE_CONFIGS.get(cfg_id, f"config {cfg_id}")
        line = f"  {cfg_id:6d}  {desc:<36s}"
        for b in args.batch:
            r = results.get(cfg_id, {}).get(b)
            bl = baseline.get(b)
            if r is None:
                line += f"  {'FAIL':>8s}"
            elif bl is not None and bl[0] > 0:
                speedup = bl[0] / r[0]
                line += (
                    f"  {r[0]:.3f}ms"
                    if speedup < 1.01 and speedup > 0.99
                    else f"  {speedup:+7.1%}"
                )
            else:
                line += f"  {r[0]:.4f}"
        print(line)

    best_for_batch = {}
    for b in args.batch:
        best_cfg = 0
        best_time = float("inf")
        for cfg_id in range(num_configs):
            r = results.get(cfg_id, {}).get(b)
            if r is not None and r[0] < best_time:
                best_time = r[0]
                best_cfg = cfg_id
        best_for_batch[b] = (best_cfg, best_time)

    print(f"\n  Best config per batch:")
    for b in args.batch:
        cfg_id, t = best_for_batch[b]
        bl = baseline.get(b)
        speedup = bl[0] / t if bl and bl[0] > 0 else 1.0
        desc = GEMM2_TUNE_CONFIGS.get(cfg_id, "")
        print(
            f"    M={b:4d}: config {cfg_id} ({desc.strip()})  {t:.4f}ms  ({speedup:.1%} vs baseline)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--E", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate-tp", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument(
        "--no-marlin", action="store_true", help="Skip Marlin comparison"
    )
    parser.add_argument(
        "--no-breakdown", action="store_true", help="Skip per-stage breakdown"
    )
    parser.add_argument(
        "--tune-gemm2", action="store_true", help="Run GEMM2 tile config tuning sweep"
    )
    args = parser.parse_args()

    init_dist()

    if args.tune_gemm2:
        run_gemm2_tuning_sweep(args)
        return

    K = args.hidden
    N = args.intermediate_tp
    E = args.E

    print("  Preparing CUTLASS W4A8 weights...")
    cutlass_w1_q, cutlass_w1_s, cutlass_w2_q, cutlass_w2_s = prepare_cutlass_weights(
        E, K, N, 128
    )
    print("  CUTLASS weights ready.")

    cutlass_real_e2e = {}
    cutlass_profiled_e2e = {}

    # ---- CUTLASS real E2E (no per-stage sync) ----
    print(f"\n{'='*70}")
    print(f"  CUTLASS W4A8 Real E2E (no per-stage sync)")
    print(f"{'='*70}")
    for batch in args.batch:
        with patch(
            "sglang.srt.layers.moe.cutlass_w4a8_moe.get_moe_expert_parallel_world_size",
            return_value=1,
        ):
            mean, std = benchmark_cutlass_real_e2e(
                M=batch,
                w1_q=cutlass_w1_q,
                w1_scale=cutlass_w1_s,
                w2_q=cutlass_w2_q,
                w2_scale=cutlass_w2_s,
                E=E,
                hidden_size=K,
                intermediate_tp=N,
                topk=args.topk,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            cutlass_real_e2e[batch] = mean
            print(f"  [CUTLASS Real] M={batch:4d}  E2E={mean:.3f}ms (std={std:.3f})")

    # ---- CUTLASS per-stage breakdown (with sync) ----
    if not args.no_breakdown:
        for batch in args.batch:
            with patch(
                "sglang.srt.layers.moe.cutlass_w4a8_moe.get_moe_expert_parallel_world_size",
                return_value=1,
            ):
                profiled = benchmark_cutlass_breakdown(
                    M=batch,
                    w1_q=cutlass_w1_q,
                    w1_scale=cutlass_w1_s,
                    w2_q=cutlass_w2_q,
                    w2_scale=cutlass_w2_s,
                    E=E,
                    hidden_size=K,
                    intermediate_tp=N,
                    topk=args.topk,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
                cutlass_profiled_e2e[batch] = profiled

    del cutlass_w1_q, cutlass_w1_s, cutlass_w2_q, cutlass_w2_s
    torch.cuda.empty_cache()

    # ---- Marlin E2E ----
    marlin_e2e = {}
    if not args.no_marlin:
        print(f"\n{'='*70}")
        print(f"  Marlin W4A16 Benchmark")
        print(f"{'='*70}")
        marlin_w1, marlin_w2, marlin_w1s, marlin_w2s = prepare_marlin_weights(
            E, K, N, 128
        )
        for batch in args.batch:
            mean, std = benchmark_marlin_e2e(
                M=batch,
                marlin_w1=marlin_w1,
                marlin_w2=marlin_w2,
                marlin_w1_scale=marlin_w1s,
                marlin_w2_scale=marlin_w2s,
                E=E,
                hidden_size=K,
                intermediate_tp=N,
                topk=args.topk,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            marlin_e2e[batch] = (mean, std)
            print(f"  [Marlin W4A16]  M={batch:4d}  E2E={mean:.3f}ms (std={std:.3f})")
        del marlin_w1, marlin_w2, marlin_w1s, marlin_w2s
        torch.cuda.empty_cache()

    # ---- Final comparison table ----
    print(f"\n{'='*70}")
    print(f"  Final Comparison — per-MoE-layer E2E")
    print(f"  E={E}, N_tp={N}, K={K}, topk={args.topk}")
    print(f"{'='*70}")

    has_marlin = bool(marlin_e2e)
    has_profiled = bool(cutlass_profiled_e2e)

    hdr = f"  {'M':>4s}  {'Real(ms)':>10s}"
    sep = f"  {'-'*4}  {'-'*10}"
    if has_profiled:
        hdr += f"  {'Profiled(ms)':>12s}  {'SyncOH(ms)':>10s}"
        sep += f"  {'-'*12}  {'-'*10}"
    if has_marlin:
        hdr += f"  {'Marlin(ms)':>10s}  {'Real/M':>8s}  {'RealGap':>10s}"
        sep += f"  {'-'*10}  {'-'*8}  {'-'*10}"
    print(hdr)
    print(sep)

    for batch in args.batch:
        r = cutlass_real_e2e.get(batch, 0)
        line = f"  {batch:4d}  {r:10.3f}"
        if has_profiled:
            p = cutlass_profiled_e2e.get(batch, 0)
            sync_oh = p - r
            line += f"  {p:12.3f}  {sync_oh:+9.3f}"
        if has_marlin:
            m_mean, _ = marlin_e2e.get(batch, (0, 0))
            ratio = r / m_mean if m_mean > 0 else float("inf")
            gap = r - m_mean
            line += f"  {m_mean:10.3f}  {ratio:7.2f}x  {gap:+9.3f}ms"
        print(line)


if __name__ == "__main__":
    main()
