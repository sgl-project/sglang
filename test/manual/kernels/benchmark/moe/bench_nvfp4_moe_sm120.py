#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

import torch
import torch.nn.functional as F
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from safetensors import safe_open

from sglang.kernels.ops.moe.nvfp4_moe_sm120 import (
    NVFP4_MOE_SM120_MAX_TOKENS,
    Nvfp4MoeWorkspace,
    nvfp4_moe_sm120,
)
from sglang.srt.layers.moe.topk import fused_topk
from sglang.srt.utils import is_sm120_supported

HIDDEN = 2560
GLOBAL_INTERMEDIATE = 640
TP = 2
INTERMEDIATE = GLOBAL_INTERMEDIATE // TP
EXPERTS = 512
TOP_K = 10
LAYER = 0
TOKEN_COUNTS = (1, 4, NVFP4_MOE_SM120_MAX_TOKENS)
FP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def swizzle_scale(scale: torch.Tensor) -> torch.Tensor:
    batches, rows, cols = scale.shape
    padded_rows = math.ceil(rows / 128) * 128
    padded_cols = math.ceil(cols / 4) * 4
    if padded_rows != rows or padded_cols != cols:
        padded = torch.zeros(
            (batches, padded_rows, padded_cols),
            dtype=scale.dtype,
            device=scale.device,
        )
        padded[:, :rows, :cols].copy_(scale)
        scale = padded
    return (
        scale.reshape(batches, padded_rows // 128, 4, 32, padded_cols // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(batches, padded_rows, padded_cols)
    )


def load_checkpoint_rank(
    device: torch.device, snapshot: pathlib.Path
) -> dict[str, torch.Tensor]:
    w13 = torch.empty(
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.empty(
        (EXPERTS, HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8, device=device
    )
    s13_linear = torch.empty(
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    s2_linear = torch.empty(
        (EXPERTS, HIDDEN, INTERMEDIATE // 16),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    w13_s2 = torch.empty(EXPERTS, dtype=torch.float32, device=device)
    w2_s2 = torch.empty(EXPERTS, dtype=torch.float32, device=device)
    w13_input = torch.empty(EXPERTS, dtype=torch.float32, device=device)
    w2_input = torch.empty(EXPERTS, dtype=torch.float32, device=device)

    seen = set()
    for shard in sorted(snapshot.glob(f"layer-{LAYER:05d}-experts-*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as source:
            keys = set(source.keys())
            for expert in range(EXPERTS):
                prefix = f"model.language_model.layers.{LAYER}.mlp.experts.{expert}"
                gate_key = f"{prefix}.gate_proj.weight"
                if gate_key not in keys:
                    continue
                seen.add(expert)
                up = source.get_tensor(f"{prefix}.up_proj.weight")[:INTERMEDIATE]
                gate = source.get_tensor(gate_key)[:INTERMEDIATE]
                down = source.get_tensor(f"{prefix}.down_proj.weight")[
                    :, : INTERMEDIATE // 2
                ]
                w13[expert, :INTERMEDIATE].copy_(up)
                w13[expert, INTERMEDIATE:].copy_(gate)
                w2[expert].copy_(down)

                up_sf = source.get_tensor(f"{prefix}.up_proj.weight_scale")[
                    :INTERMEDIATE
                ]
                gate_sf = source.get_tensor(f"{prefix}.gate_proj.weight_scale")[
                    :INTERMEDIATE
                ]
                down_sf = source.get_tensor(f"{prefix}.down_proj.weight_scale")[
                    :, : INTERMEDIATE // 16
                ]
                s13_linear[expert, :INTERMEDIATE].copy_(up_sf)
                s13_linear[expert, INTERMEDIATE:].copy_(gate_sf)
                s2_linear[expert].copy_(down_sf)

                up_s2 = source.get_tensor(f"{prefix}.up_proj.weight_scale_2").float()
                gate_s2 = source.get_tensor(
                    f"{prefix}.gate_proj.weight_scale_2"
                ).float()
                if not torch.equal(up_s2, gate_s2):
                    raise RuntimeError(f"expert {expert} gate/up weight_scale_2 differ")
                w13_s2[expert] = gate_s2
                w2_s2[expert] = source.get_tensor(
                    f"{prefix}.down_proj.weight_scale_2"
                ).float()
                up_input = source.get_tensor(f"{prefix}.up_proj.input_scale").float()
                gate_input = source.get_tensor(
                    f"{prefix}.gate_proj.input_scale"
                ).float()
                if not torch.equal(up_input, gate_input):
                    raise RuntimeError(f"expert {expert} gate/up input_scale differ")
                w13_input[expert] = gate_input
                w2_input[expert] = source.get_tensor(
                    f"{prefix}.down_proj.input_scale"
                ).float()
    if seen != set(range(EXPERTS)):
        missing = sorted(set(range(EXPERTS)) - seen)
        raise RuntimeError(f"missing experts: {missing[:16]}")

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    router_name = f"model.language_model.layers.{LAYER}.mlp.gate.weight"
    router_file = snapshot / index["weight_map"][router_name]
    with safe_open(router_file, framework="pt", device="cpu") as source:
        router = source.get_tensor(router_name).to(device)

    return {
        "w13": w13,
        "w2": w2,
        "s13_linear": s13_linear,
        "s2_linear": s2_linear,
        "s13_swizzled": swizzle_scale(s13_linear),
        "s2_swizzled": swizzle_scale(s2_linear),
        "w13_s2": w13_s2,
        "w2_s2": w2_s2,
        "w13_input": w13_input,
        "w2_input": w2_input,
        "router": router,
    }


def dequant_weight(
    packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    lut = FP4_LUT.to(packed.device)
    output = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.float32,
        device=packed.device,
    )
    output[..., 0::2] = lut[(packed & 0xF).long()]
    output[..., 1::2] = lut[(packed >> 4).long()]
    global_scale = global_scale.float()
    while global_scale.ndim < output.ndim:
        global_scale = global_scale.unsqueeze(-1)
    return output * scale.float().repeat_interleave(16, dim=-1) * global_scale


def reference_moe(
    x: torch.Tensor,
    ids: torch.Tensor,
    routing: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    output = torch.zeros((x.shape[0], HIDDEN), dtype=torch.float32, device=x.device)
    for expert in torch.unique(ids).tolist():
        locations = torch.nonzero(ids == expert, as_tuple=False)
        token_indices = locations[:, 0]
        slot_indices = locations[:, 1]
        w13 = dequant_weight(
            weights["w13"][expert],
            weights["s13_linear"][expert],
            weights["w13_s2"][expert],
        )
        w2 = dequant_weight(
            weights["w2"][expert],
            weights["s2_linear"][expert],
            weights["w2_s2"][expert],
        )
        fc1 = x[token_indices].float() @ w13.T
        linear, gate = fc1.split(INTERMEDIATE, dim=-1)
        expert_output = (F.silu(gate) * linear) @ w2.T
        output.index_add_(
            0,
            token_indices,
            expert_output * routing[token_indices, slot_indices, None],
        )
    return output


def event_samples(fn, flush: torch.Tensor, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        flush.add_(1)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def graph_samples(
    fn, flush: torch.Tensor, repeats: int
) -> tuple[list[float], torch.cuda.CUDAGraph]:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return event_samples(graph.replay, flush, repeats), graph


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "p10_us": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_us": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
    }


def max_errors(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.float()
    difference = (actual - reference).abs()
    denominator = reference.abs().clamp_min(1e-3)
    return {
        "max_abs": difference.max().item(),
        "max_rel_ref_floor_1e-3": (difference / denominator).max().item(),
        "relative_l2": (difference.norm() / reference.norm().clamp_min(1e-12)).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=pathlib.Path,
        required=True,
        help="Qwen3.8 NVFP4 snapshot directory containing model.safetensors.index.json",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    snapshot = args.snapshot
    if not (snapshot / "model.safetensors.index.json").is_file():
        parser.error(f"not a Qwen3.8 NVFP4 snapshot directory: {snapshot}")

    torch.manual_seed(20260827)
    weights = load_checkpoint_rank(torch.device("cuda"), snapshot)
    input_scale_1 = weights["w13_input"].max()
    input_scale_2 = weights["w2_input"].max()
    input_quant_1 = (1.0 / input_scale_1).float()
    input_quant_2 = (1.0 / input_scale_2).float()
    g1_alpha = (input_scale_1 * weights["w13_s2"]).float()
    g2_alpha = (input_scale_2 * weights["w2_s2"]).float()
    workspace = Nvfp4MoeWorkspace.allocate(
        max_tokens=NVFP4_MOE_SM120_MAX_TOKENS,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        device=torch.device("cuda"),
    )
    x_all = (
        torch.randn(
            (NVFP4_MOE_SM120_MAX_TOKENS, HIDDEN),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    ).contiguous()
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    rows = []
    outputs = {}

    for tokens in TOKEN_COUNTS:
        x = x_all[:tokens]
        logits = F.linear(x, weights["router"])
        topk_weights, topk_ids = fused_topk(
            x, logits, TOP_K, True, scoring_func="softmax"
        )
        topk_weights = topk_weights.contiguous()
        topk_ids = topk_ids.contiguous()
        current_out = torch.empty_like(x)
        candidate_out = torch.empty_like(x)

        def current():
            return cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_ids,
                token_final_scales=topk_weights,
                fc1_expert_weights=weights["w13"].view(torch.long),
                fc2_expert_weights=weights["w2"].view(torch.long),
                output_dtype=torch.bfloat16,
                quant_scales=[
                    input_quant_1,
                    weights["s13_swizzled"].view(torch.int32),
                    g1_alpha,
                    input_quant_2,
                    weights["s2_swizzled"].view(torch.int32),
                    g2_alpha,
                ],
                output=current_out,
                tp_size=TP,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                activation_type=ActivationType.Swiglu,
                tune_max_num_tokens=1 << math.ceil(math.log2(tokens)),
                use_fused_finalize=True,
            )[0]

        def candidate():
            launched = nvfp4_moe_sm120(
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                w13_weight=weights["w13"],
                w2_weight=weights["w2"],
                w13_scale=weights["s13_swizzled"],
                w2_scale=weights["s2_swizzled"],
                input_scale_1=input_quant_1,
                input_scale_2=input_quant_2,
                g1_alpha=g1_alpha,
                g1_alpha_up=g1_alpha,
                g2_alpha=g2_alpha,
                global_routed_experts=EXPERTS,
                local_routed_experts=EXPERTS,
                local_expert_start=0,
                output=candidate_out,
                workspace=workspace,
            )
            if not launched:
                raise RuntimeError("cooperative launch is unavailable")
            return candidate_out

        current_samples, current_graph = graph_samples(current, flush, args.repeats)
        candidate_samples, candidate_graph = graph_samples(
            candidate, flush, args.repeats
        )
        current()
        outputs[("current", tokens)] = current_out.clone()
        outputs[("candidate", tokens)] = candidate().clone()
        torch.cuda.synchronize()
        row = {
            "tokens": tokens,
            "unique_experts": int(torch.unique(topk_ids).numel()),
            "current": summarize(current_samples),
            "candidate": summarize(candidate_samples),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del current_graph, candidate_graph

    correctness = {}
    for tokens in (1, NVFP4_MOE_SM120_MAX_TOKENS):
        x = x_all[:tokens]
        logits = F.linear(x, weights["router"])
        routing, ids = fused_topk(x, logits, TOP_K, True, scoring_func="softmax")
        reference = reference_moe(x, ids, routing, weights)
        correctness[str(tokens)] = {
            "cutlass_vs_fp32_dequant": max_errors(
                outputs[("current", tokens)], reference
            ),
            "b12x_vs_fp32_dequant": max_errors(
                outputs[("candidate", tokens)], reference
            ),
            "b12x_vs_cutlass": max_errors(
                outputs[("candidate", tokens)],
                outputs[("current", tokens)].float(),
            ),
        }

    correctness_acceptance = {
        "criterion": (
            "candidate max_abs and relative_l2 are each no worse than current "
            "plus 1e-3"
        ),
        "passed": all(
            correctness[str(tokens)]["b12x_vs_fp32_dequant"][metric]
            <= correctness[str(tokens)]["cutlass_vs_fp32_dequant"][metric] + 1e-3
            for tokens in (1, NVFP4_MOE_SM120_MAX_TOKENS)
            for metric in ("max_abs", "relative_l2")
        ),
    }
    for row in rows:
        row["routed_pairs"] = row["tokens"] * TOP_K
        row["cuda_graph"] = {
            "cutlass_fused": row["current"],
            "b12x_sm12x": row["candidate"],
        }

    result = {
        "schema": "nvfp4_moe_small_m_v1",
        "model": {
            "checkpoint": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
            "layer": LAYER,
            "global_experts": EXPERTS,
            "local_experts": EXPERTS,
            "top_k": TOP_K,
            "hidden": HIDDEN,
            "global_intermediate": GLOBAL_INTERMEDIATE,
            "rank_intermediate": INTERMEDIATE,
            "tp": TP,
            "tp_rank": 0,
            "ep": 1,
            "quant": ("NVFP4 W4A4 group 16, E4M3 block scales, FP32 global scales"),
            "w13_input_scale_max": input_scale_1.item(),
            "w2_input_scale_max": input_scale_2.item(),
        },
        "timing": {
            "method": (
                "CUDA events, 256 MiB L2 flush before each sample, compilation "
                "and two warmups excluded"
            ),
            "repeats": args.repeats,
            "cuda_graph": (
                "one fused MoE call per captured graph; L2 flush outside graph"
            ),
        },
        "rows": rows,
        "correctness": correctness,
        "correctness_acceptance": correctness_acceptance,
        "graph_capture": {
            "cutlass": {str(tokens): {"ok": True} for tokens in TOKEN_COUNTS},
            "b12x_sm12x": {str(tokens): {"ok": True} for tokens in TOKEN_COUNTS},
        },
    }
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    if not is_sm120_supported():
        print("[skip] NVFP4 fused MoE benchmark requires SM120 CUDA.")
        sys.exit(0)
    main()
