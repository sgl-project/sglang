"""Default-off exact V4.1 gate/up tile experiment; does not launch a model.

PYTHONPATH=python MAX_JOBS=1 python test/manual/layers/moe/bench_mxfp4_sm80_n64.py
Requires patched moe_wna16_marlin.cuh and an A100 with ~2GiB free.
Both variants retain BF16 output, original MXFP4 bytes/scales, FP32 reduction,
and atomic=False. 32 captured routes rotate >L2 weights instead of benchmarking
one permanently cache-hot six-expert set. No checkpoint or precision changes.
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
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import mxfp4_marlin_process_scales

FLAG = "SGLANG_DSV41_SM80_MARLIN_N64"
E, N, K, TOPK = 384, 640, 5120, 6


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--routes", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()
    assert args.routes >= 16, "Use >=16 routes so the selected weights exceed A100 L2"
    assert args.rounds >= 2
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)
    assert torch.version.hip is None
    os.environ["SGLANG_DSV41_SM80_EXPERIMENTAL"] = "1"
    torch.manual_seed(41080)
    raw = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
    scale_bytes = torch.randint(
        120, 125, (E, N, K // 32), dtype=torch.uint8, device="cuda"
    )
    perm = torch.empty(0, dtype=torch.int32, device="cuda")
    packed = torch.empty((E, K // 16, N * 2), dtype=torch.int32, device="cuda")
    scale_list = []
    # Build only one expert's numerical scales at a time; avoid a huge temporary.
    for expert in range(E):
        packed[expert].copy_(
            gptq_marlin_repack(
                b_q_weight=raw[expert].view(torch.int32).T.contiguous(),
                perm=perm,
                size_k=K,
                size_n=N,
                num_bits=4,
            )
        )
        s = (
            torch.exp2(scale_bytes[expert].float() - 127)
            .to(torch.bfloat16)
            .T.contiguous()
        )
        s = marlin_permute_scales(s=s, size_k=K, size_n=N, group_size=32)
        scale_list.append(mxfp4_marlin_process_scales(s, input_dtype=torch.bfloat16))
    scales = torch.stack(scale_list)
    del scale_list
    x = (torch.randn(1, K, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()
    workspace = marlin_make_workspace(x.device, 4)
    routes = []
    # Disjoint until all experts have been exercised; cycle only after E/TOPK.
    permutation = torch.randperm(E).tolist()
    for route in range(args.routes):
        ids = torch.tensor(
            [[permutation[(route * TOPK + j) % E] for j in range(TOPK)]],
            dtype=torch.int32,
            device="cuda",
        )
        sorted_ids, expert_ids, padded_count = moe_align_single_token(ids, 8)
        output = torch.empty((TOPK, N), dtype=torch.bfloat16, device="cuda")
        routes.append((ids, sorted_ids, expert_ids, padded_count, output))
    topk_weights = torch.full((1, TOPK), 1 / TOPK, dtype=torch.float32, device="cuda")

    def call(route):
        _, sorted_ids, expert_ids, padded_count, out = route
        return moe_wna16_marlin_gemm(
            a=x,
            c_or_none=out,
            b_q_weight=packed,
            b_bias_or_none=None,
            b_scales=scales,
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=workspace,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=padded_count,
            topk_weights=topk_weights,
            moe_block_size=8,
            top_k=TOPK,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=1,
            size_n=N,
            size_k=K,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    # Independent CPU FP32 GEMV on selected original bytes/scales, then BF16.
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    x_cpu = x.cpu().float().squeeze(0)
    selected = routes[0][0].cpu().flatten().tolist()
    reference = []
    for expert in selected:
        b = raw[expert].cpu().to(torch.int64)
        codes = torch.stack((b & 15, b >> 4), dim=-1).reshape(N, K)
        w = lut[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
        sf = torch.exp2(scale_bytes[expert].cpu().float() - 127).repeat_interleave(
            32, 1
        )
        reference.append((w * sf).matmul(x_cpu).to(torch.bfloat16))
    reference = torch.stack(reference)
    del raw, scale_bytes, w, sf, codes
    variants = {}
    snapshots = {}
    for enabled in ("0", "1"):
        os.environ[FLAG] = enabled
        for route in routes:
            call(route)
        torch.cuda.synchronize()
        snapshot = routes[0][-1].cpu().clone()
        torch.testing.assert_close(snapshot, reference, rtol=0.01, atol=0.02)
        snapshots[enabled] = snapshot
        graphs = []
        for route in routes:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call(route)
            graphs.append(graph)
        variants[enabled] = graphs
        print(
            json.dumps(
                {
                    "variant": enabled,
                    "reference_max_abs": (snapshot.float() - reference.float())
                    .abs()
                    .max()
                    .item(),
                }
            ),
            flush=True,
        )
    torch.testing.assert_close(snapshots["1"], snapshots["0"], rtol=0.01, atol=0.02)
    print(
        json.dumps(
            {
                "baseline_vs_n64_max_abs": (
                    snapshots["1"].float() - snapshots["0"].float()
                )
                .abs()
                .max()
                .item()
            }
        ),
        flush=True,
    )
    # Validate all routes against baseline, not only the CPU oracle's first six.
    for i in range(len(routes)):
        variants["0"][i].replay()
        baseline = routes[i][-1].clone()
        variants["1"][i].replay()
        torch.testing.assert_close(routes[i][-1], baseline, rtol=0.01, atol=0.02)
    for graphs in variants.values():
        for _ in range(3):
            for graph in graphs:
                graph.replay()
    torch.cuda.synchronize()
    times = {"0": [], "1": []}
    for round_id in range(args.rounds):
        for enabled in ("0", "1") if round_id % 2 == 0 else ("1", "0"):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for graph in variants[enabled]:
                graph.replay()
            end.record()
            end.synchronize()
            times[enabled].append(start.elapsed_time(end) * 1000 / args.routes)
    base_us, n64_us = statistics.median(times["0"]), statistics.median(times["1"])
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "MNK": [1, N, K],
                "experts": E,
                "topk": TOPK,
                "routes": args.routes,
                "rounds": args.rounds,
                "selected_weight_MiB_per_rotation": min(E, args.routes * TOPK)
                * N
                * K
                / 2
                / 2**20,
                "default_median_us": base_us,
                "n64_median_us": n64_us,
                "speedup": base_us / n64_us,
                "saved_ms_per_40_layers": (base_us - n64_us) * 40 / 1000,
                "default_samples_us": times["0"],
                "n64_samples_us": times["1"],
            }
        ),
        flush=True,
    )
    os.environ.pop(FLAG, None)


if __name__ == "__main__":
    main()
