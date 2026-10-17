"""Exact V4.1 TP8 down projection occupancy experiment, independent of API.

PYTHONPATH=python MAX_JOBS=1 python test/manual/layers/moe/bench_mxfp4_sm80_down_blocks.py
Uses original BF16 GEMM rounding followed by BF16 route-weight multiplication.
Rotates >=192 experts (>150MiB packed weights), keeping FP32 reduction and no
atomic add. Requires the shape-scoped blocks override in moe_wna16_marlin.cuh.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace, marlin_permute_scales
from sglang.srt.layers.quantization.marlin_utils_fp4 import mxfp4_marlin_process_scales

FLAG = "SGLANG_DSV41_SM80_MOE_DOWN_BLOCKS"
E, N, K, LOGICAL_K, TOPK = 384, 5120, 320, 288, 6
VARIANTS = ("auto", "1", "2", "4")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--routes", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()
    assert 32 <= args.routes <= 64 and args.rounds >= 2
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)
    assert torch.version.hip is None
    torch.set_num_threads(1)
    os.environ["SGLANG_DSV41_SM80_EXPERIMENTAL"] = "1"
    torch.manual_seed(42080)
    raw = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
    raw[:, :, LOGICAL_K // 2:] = 0
    scale_bytes = torch.randint(120, 125, (E, N, K // 32), dtype=torch.uint8, device="cuda")
    scale_bytes[:, :, LOGICAL_K // 32:] = 127
    perm = torch.empty(0, dtype=torch.int32, device="cuda")
    packed = torch.empty((E, K // 16, N * 2), dtype=torch.int32, device="cuda")
    scale_list = []
    for expert in range(E):
        packed[expert].copy_(gptq_marlin_repack(
            b_q_weight=raw[expert].view(torch.int32).T.contiguous(),
            perm=perm, size_k=K, size_n=N, num_bits=4))
        scales = torch.exp2(scale_bytes[expert].float() - 127).bfloat16().T.contiguous()
        scales = marlin_permute_scales(s=scales, size_k=K, size_n=N, group_size=32)
        scale_list.append(mxfp4_marlin_process_scales(scales, input_dtype=torch.bfloat16))
    scales = torch.stack(scale_list)
    del scale_list
    x = (torch.randn(TOPK, K, device="cuda", dtype=torch.bfloat16) * .1).contiguous()
    x[:, LOGICAL_K:] = 0
    weights = torch.tensor([[.013, .09, .18, .27, .31, .137]], device="cuda", dtype=torch.float32)
    workspace = marlin_make_workspace(x.device, 4)
    permutation = torch.randperm(E).tolist()
    routes = []
    for r in range(args.routes):
        ids = torch.tensor([[permutation[r * TOPK + j] for j in range(TOPK)]],
                           device="cuda", dtype=torch.int32)
        aligned, experts, count = moe_align_single_token(ids, 8)
        out = torch.empty((TOPK, N), device="cuda", dtype=torch.bfloat16)
        routes.append((ids, aligned, experts, count, out))

    def call(route):
        _, aligned, experts, count, out = route
        return moe_wna16_marlin_gemm(
            a=x, c_or_none=out, b_q_weight=packed, b_bias_or_none=None,
            b_scales=scales, global_scale_or_none=None, b_zeros_or_none=None,
            g_idx_or_none=None, perm_or_none=None, workspace=workspace,
            sorted_token_ids=aligned, expert_ids=experts, num_tokens_post_padded=count,
            topk_weights=weights, moe_block_size=8, top_k=1, mul_topk_weights=True,
            is_ep=False, b_q_type=scalar_types.float4_e2m1f, size_m=TOPK,
            size_n=N, size_k=K, is_k_full=True, use_atomic_add=False,
            use_fp32_reduce=True, is_zp_float=False)

    # Independent original-byte CPU FP64 dequant/GEMV oracle, with the TWO
    # BF16 boundaries used by Marlin's output write (not FP32-weighted GEMV).
    selected = routes[0][0].cpu().flatten().tolist()
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
    original_x, original_weights = x.clone(), weights.clone()
    original_raw = [raw[e].cpu() for e in selected]
    original_scale = [scale_bytes[e].cpu() for e in selected]
    del raw, scale_bytes

    def oracle():
        x_cpu, weights_cpu = x.cpu().double(), weights.cpu().bfloat16().flatten()
        result = []
        for j, (b, sb) in enumerate(zip(original_raw, original_scale)):
            b = b.to(torch.int64)
            code = torch.stack((b & 15, b >> 4), -1).reshape(N, K)
            w = lut[code & 7] * torch.where(code & 8 != 0, -1., 1.)
            sf = torch.exp2(sb.double() - 127).repeat_interleave(32, 1)
            gemm_bf16 = (w * sf).matmul(x_cpu[j]).bfloat16()
            result.append(gemm_bf16 * weights_cpu[j])
        return torch.stack(result)

    reference = oracle()
    graphs, snapshots = {}, {}
    for variant in VARIANTS:
        os.environ[FLAG] = variant
        for route in routes:
            call(route)
        torch.cuda.synchronize()
        snapshot = routes[0][-1].cpu().clone()
        torch.testing.assert_close(snapshot, reference, rtol=.01, atol=.0001)
        snapshots[variant] = snapshot
        graphs[variant] = []
        for route in routes:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call(route)
            graphs[variant].append(graph)
        print(json.dumps({"variant": variant, "oracle_max_abs":
                          (snapshot.float() - reference.float()).abs().max().item(),
                          "bitwise_baseline": torch.equal(snapshot, snapshots["auto"])}), flush=True)
    total_out = torch.empty((1, N), device="cuda", dtype=torch.bfloat16)
    for i, route in enumerate(routes):
        graphs["auto"][i].replay()
        base = route[-1].clone()
        for variant in VARIANTS[1:]:
            route[-1].fill_(float("nan"))
            graphs[variant][i].replay()
            assert torch.isfinite(route[-1]).all(), (variant, i)
            torch.testing.assert_close(route[-1], base, rtol=.01, atol=.0001)
        moe_topk_sum(route[-1].view(1, TOPK, N), total_out)
        cpu_rows = route[-1].cpu().float()
        ordered_sum = torch.zeros((1, N), dtype=torch.float32)
        for row in cpu_rows:
            ordered_sum.add_(row)
        torch.testing.assert_close(total_out.cpu(), ordered_sum.bfloat16(), rtol=0, atol=0)
    for mutation in range(4):
        x.copy_(original_x * [1., -.25, 2., .03125][mutation])
        weights.copy_(original_weights.roll(mutation, dims=1))
        reference = oracle()
        for variant in VARIANTS:
            routes[0][-1].fill_(float("nan"))
            graphs[variant][0].replay()
            torch.testing.assert_close(routes[0][-1].cpu(), reference, rtol=.01, atol=.0001)
    x.copy_(original_x)
    weights.copy_(original_weights)
    for _ in range(3):
        for variant in VARIANTS:
            for graph in graphs[variant]:
                graph.replay()
    torch.cuda.synchronize()
    times = {v: [] for v in VARIANTS}
    for round_id in range(args.rounds):
        for variant in VARIANTS if round_id % 2 == 0 else VARIANTS[::-1]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for graph in graphs[variant]:
                graph.replay()
            end.record(); end.synchronize()
            times[variant].append(start.elapsed_time(end) * 1000 / args.routes)
    baseline_us = statistics.median(times["auto"])
    print(json.dumps({"numeric": "ALL_ROUTES_GRAPH_MUTATIONS_AND_ORACLE_PASS",
                      "MNK": [TOPK, N, K], "api_top_k": 1, "route_topk": TOPK,
                      "logical_k": LOGICAL_K, "routes": args.routes,
                      "selected_weight_MiB_per_rotation": args.routes * TOPK * N * K / 2 / 2**20,
                      "median_us": {v: statistics.median(t) for v, t in times.items()},
                      "saved_ms_per_40_layers": {v: (baseline_us - statistics.median(t)) * .04 for v, t in times.items()},
                      "samples_us": times}), flush=True)
    os.environ.pop(FLAG, None)


if __name__ == "__main__":
    main()
