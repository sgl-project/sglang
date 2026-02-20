#!/usr/bin/env python3
"""
Minimal layer-level parity harness for FlashInfer MoE backends.

Compares:
1) CUTLASS fused MoE  (per-expert scales, ground truth)
2) CuteDSL MoE wrapper (scalar fc2_input_scale)
3) TRTLLM routed FP4 MoE (per-expert output1_scale_scalar)

Modes (--scale-contract):
  well-behaved: scalar w2_input_scale_quant — CuteDSL exact, all backends match
  realistic:    per-expert w2_input_scale_quant with spread — tests the
                min() (fixed) vs max() (old buggy) scalar reduction

With --scale-contract=realistic, the output includes parity for both CuteDSL
variants so you can see the impact of the fc2_input_scale convention fix.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F


def next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def interleave_linear_and_gate(
    tensor: torch.Tensor, group_size: int = 64, dim: int = 1
) -> torch.Tensor:
    if tensor.shape[dim] % 2 != 0:
        raise ValueError("Expected even size on interleave dimension.")
    split = tensor.shape[dim] // 2
    if split % group_size != 0:
        raise ValueError(f"Expected split divisible by group_size={group_size}.")
    gate = tensor.narrow(dim, 0, split)
    up = tensor.narrow(dim, split, split)
    gate_groups = gate.split(group_size, dim=dim)
    up_groups = up.split(group_size, dim=dim)
    interleaved = [item for pair in zip(gate_groups, up_groups) for item in pair]
    return torch.cat(interleaved, dim=dim)


def reorder_w1_halves(tensor: torch.Tensor, order: str, dim: int = 1) -> torch.Tensor:
    if order not in {"gate_up", "up_gate"}:
        raise ValueError(f"Unsupported w1 order: {order}")
    half = tensor.shape[dim] // 2
    first = tensor.narrow(dim, 0, half)
    second = tensor.narrow(dim, half, half)
    if order == "gate_up":
        return torch.cat([first, second], dim=dim)
    return torch.cat([second, first], dim=dim)


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype != torch.float8_e4m3fn:
        scale = scale.to(torch.float8_e4m3fn)
    if scale.ndim != 3:
        raise ValueError(f"Expected 3D scale tensor, got {scale.ndim}D")

    bsz, m, k = scale.shape
    round_up = lambda x, n: (x + n - 1) // n * n
    m_padded = round_up(m, 128)
    k_padded = round_up(k, 4)
    padded = torch.zeros(
        (bsz, m_padded, k_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:, :m, :k] = scale
    padded = padded.reshape(bsz, m_padded // 128, 4, 32, k_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous()
    return swizzled.reshape(bsz, m_padded, k_padded)


def unswizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype != torch.float8_e4m3fn:
        scale = scale.to(torch.float8_e4m3fn)
    if scale.ndim != 3:
        raise ValueError(f"Expected 3D scale tensor, got {scale.ndim}D")

    bsz, m_padded, k_padded = scale.shape
    if m_padded % 128 != 0 or k_padded % 4 != 0:
        raise ValueError(
            f"Expected padded swizzled shape aligned to (128,4), got {(bsz, m_padded, k_padded)}"
        )
    unswizzled = scale.reshape(bsz, m_padded // 128, k_padded // 4, 32, 4, 4)
    unswizzled = unswizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    return unswizzled.reshape(bsz, m_padded, k_padded)


def build_well_behaved_fp4_scale_contract(
    num_experts: int, device: str, seed: int
) -> dict:
    """Scalar w2_input_scale_quant — CuteDSL scalar approximation is exact."""
    g = torch.Generator(device=device)
    g.manual_seed(seed + 202)
    w13_input_scale_quant = torch.tensor(
        0.8 + 0.4 * torch.rand(1, device=device, generator=g).item(),
        device=device,
        dtype=torch.float32,
    )
    w2_input_scale_quant = torch.tensor(
        0.8 + 0.4 * torch.rand(1, device=device, generator=g).item(),
        device=device,
        dtype=torch.float32,
    )
    w13_weight_scale_2 = (
        0.8 + 0.4 * torch.rand(num_experts, device=device, generator=g)
    ).to(torch.float32)
    w2_weight_scale_2 = (
        0.8 + 0.4 * torch.rand(num_experts, device=device, generator=g)
    ).to(torch.float32)

    w13_input_scale = 1.0 / w13_input_scale_quant
    w2_input_scale = 1.0 / w2_input_scale_quant
    g1_alphas = (w13_input_scale * w13_weight_scale_2).to(torch.float32)
    g2_alphas = (w2_input_scale * w2_weight_scale_2).to(torch.float32)
    g1_scale_c = (w2_input_scale_quant * g1_alphas).to(torch.float32)
    return {
        "w13_input_scale_quant": w13_input_scale_quant,
        "w2_input_scale_quant": w2_input_scale_quant,
        "g1_alphas": g1_alphas,
        "g2_alphas": g2_alphas,
        "g1_scale_c": g1_scale_c,
    }


def build_realistic_fp4_scale_contract(
    num_experts: int, device: str, seed: int, w2_spread: float = 7.5
) -> dict:
    """Per-expert w2_input_scale_quant with realistic spread.

    Real models (e.g. Qwen3-30B-A3B-FP4) have ~7.5x spread in w2_input_scale
    across experts.  CuteDSL must reduce this to a scalar; the TRTLLM convention
    is fc2_input_scale = min(w2_isq) = 1/max(w2_input_scale).  The old buggy
    sglang code used max(w2_isq) instead.

    This contract tests both conventions so the parity harness can show the
    impact of the fix.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed + 303)
    w13_input_scale_quant = torch.tensor(
        0.8 + 0.4 * torch.rand(1, device=device, generator=g).item(),
        device=device,
        dtype=torch.float32,
    )
    w2_isq_min = 350.0
    w2_isq_max = w2_isq_min * w2_spread
    w2_input_scale_quant_per_expert = torch.linspace(
        w2_isq_min, w2_isq_max, num_experts, device=device, dtype=torch.float32
    )

    w13_weight_scale_2 = (
        0.8 + 0.4 * torch.rand(num_experts, device=device, generator=g)
    ).to(torch.float32)
    w2_weight_scale_2 = (
        0.8 + 0.4 * torch.rand(num_experts, device=device, generator=g)
    ).to(torch.float32)

    w13_input_scale = 1.0 / w13_input_scale_quant
    w2_input_scale = 1.0 / w2_input_scale_quant_per_expert
    g1_alphas = (w13_input_scale * w13_weight_scale_2).to(torch.float32)
    g2_alphas = (w2_input_scale * w2_weight_scale_2).to(torch.float32)

    # FIXED convention: fc2 = min(w2_isq) = 1/max(w2_input_scale)
    fc2_fixed = w2_input_scale_quant_per_expert.min().reshape(1)
    w2_alpha_fixed = g2_alphas * w2_input_scale_quant_per_expert / fc2_fixed

    # OLD buggy convention: fc2 = max(w2_isq)
    fc2_buggy = w2_input_scale_quant_per_expert.max().reshape(1)
    w2_alpha_buggy = g2_alphas * w2_input_scale_quant_per_expert / fc2_buggy

    g1_scale_c = (w2_input_scale_quant_per_expert * g1_alphas).to(torch.float32)

    return {
        "w13_input_scale_quant": w13_input_scale_quant,
        "w2_input_scale_quant": fc2_fixed,
        "w2_input_scale_quant_buggy": fc2_buggy,
        "w2_input_scale_quant_per_expert": w2_input_scale_quant_per_expert,
        "g1_alphas": g1_alphas,
        "g2_alphas": w2_alpha_fixed,
        "g2_alphas_buggy": w2_alpha_buggy,
        "g2_alphas_raw": g2_alphas,
        "g1_scale_c": g1_scale_c,
        "w2_spread": w2_spread,
    }


def normalize_cutlass_fc1_act_global(x: torch.Tensor, num_experts: int) -> torch.Tensor:
    if x.ndim == 0:
        return x
    if x.numel() == 1:
        return x.reshape(())
    if x.ndim == 1 and x.shape[0] >= num_experts:
        return x[:num_experts].contiguous()
    raise ValueError(f"Unsupported fc1_act_global shape {tuple(x.shape)}")


def normalize_cutlass_expert_scale(x: torch.Tensor, num_experts: int) -> torch.Tensor:
    if x.ndim == 0 or x.numel() == 1:
        return x.reshape(1).repeat(num_experts).contiguous()
    if x.ndim == 1 and x.shape[0] >= num_experts:
        return x[:num_experts].contiguous()
    raise ValueError(f"Unsupported expert scale shape {tuple(x.shape)}")


def create_moe_tensors(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    seed: int,
    fixed_expert_id: int | None,
    uniform_topk_weights: bool,
    w1_order: str,
    w1_layout: str,
    cutedsl_w1_order: str,
    cutedsl_w1_layout: str,
    cutedsl_preprocess: str,
    scale_contract: str = "well-behaved",
    w2_spread: float = 7.5,
) -> dict:
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize, nvfp4_block_scale_interleave

    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    if w1_layout not in {"plain", "interleaved"}:
        raise ValueError(f"Unsupported w1 layout: {w1_layout}")
    if cutedsl_w1_layout not in {"plain", "interleaved"}:
        raise ValueError(f"Unsupported cutedsl_w1_layout: {cutedsl_w1_layout}")
    if cutedsl_preprocess not in {"legacy", "trtllm_post_quant"}:
        raise ValueError(f"Unsupported cutedsl_preprocess: {cutedsl_preprocess}")

    device = "cuda"
    sf_vec_size = 16
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if scale_contract == "realistic":
        contract = build_realistic_fp4_scale_contract(
            num_experts, device, seed, w2_spread
        )
    else:
        contract = build_well_behaved_fp4_scale_contract(num_experts, device, seed)

    x_bf16 = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) * 0.1
    )
    x_q, x_sf_raw = fp4_quantize(
        x_bf16,
        global_scale=contract["w13_input_scale_quant"],
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )
    # CUTLASS FP4 path expects activation scales in swizzled/interleaved layout.
    x_sf_cutlass_fp4 = nvfp4_block_scale_interleave(
        x_sf_raw.view(num_tokens, hidden_size // sf_vec_size).contiguous()
    )
    x_sf_linear = x_sf_raw.view(torch.float8_e4m3fn)

    if fixed_expert_id is not None:
        if fixed_expert_id < 0 or fixed_expert_id >= num_experts:
            raise ValueError(f"fixed_expert_id={fixed_expert_id} out of range")
        topk_ids = torch.full(
            (num_tokens, top_k), fixed_expert_id, dtype=torch.int32, device=device
        )
        topk_weights = torch.full(
            (num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device
        )
    else:
        router_logits = torch.randn(num_tokens, num_experts, device=device)
        probs = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
        if uniform_topk_weights:
            topk_weights.fill_(1.0 / top_k)
        else:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)

    w1_base = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )

    def quantize_w1(
        base_w1: torch.Tensor, order: str, layout: str, preprocess: str
    ) -> dict:
        w1_kernel = reorder_w1_halves(base_w1, order=order, dim=1)
        if preprocess == "legacy" and layout == "interleaved":
            w1_kernel = interleave_linear_and_gate(w1_kernel, group_size=64, dim=1)

        w1_flat = w1_kernel.view(num_experts * 2 * intermediate_size, hidden_size)
        w1_q_flat, w1_sf_flat = fp4_quantize(
            w1_flat,
            global_scale=torch.tensor([1.0], device=device, dtype=torch.float32),
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=False,
        )
        w1_q = w1_q_flat.view(num_experts, 2 * intermediate_size, hidden_size // 2)
        w1_sf_linear = w1_sf_flat.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 16
        )
        w1_sf_swizzled = swizzle_blockscale(w1_sf_linear)

        if preprocess == "trtllm_post_quant" and layout == "interleaved":
            w1_q = interleave_linear_and_gate(
                w1_q.view(torch.uint8), group_size=64, dim=1
            ).contiguous()
            w1_sf_unswizzled = unswizzle_blockscale(w1_sf_swizzled).view(
                num_experts, 2 * intermediate_size, hidden_size // 16
            )
            w1_sf_unswizzled = interleave_linear_and_gate(
                w1_sf_unswizzled, group_size=64, dim=1
            ).contiguous()
            w1_sf_swizzled = swizzle_blockscale(w1_sf_unswizzled)
            w1_sf_flat = w1_sf_swizzled.contiguous().view(torch.uint8).reshape(-1)

        w1_sf_mma = convert_sf_to_mma_layout(
            w1_sf_swizzled.contiguous().view(torch.uint8).reshape(-1),
            m=2 * intermediate_size,
            k=hidden_size,
            num_groups=num_experts,
            sf_vec_size=sf_vec_size,
        )
        return {
            "w1_q": w1_q,
            "w1_sf_flat": w1_sf_flat,
            "w1_sf_by_layout": {"swizzled": w1_sf_swizzled, "mma": w1_sf_mma},
        }

    w1_cutlass = quantize_w1(
        w1_base, order=w1_order, layout=w1_layout, preprocess="legacy"
    )
    w1_cutedsl = quantize_w1(
        w1_base,
        order=cutedsl_w1_order,
        layout=cutedsl_w1_layout,
        preprocess=cutedsl_preprocess,
    )

    w2_bf16 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    w2_flat = w2_bf16.view(num_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat,
        global_scale=torch.tensor([1.0], device=device, dtype=torch.float32),
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )
    w2_q = w2_q_flat.view(num_experts, hidden_size, intermediate_size // 2)
    w2_sf_linear = w2_sf_flat.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 16
    )
    w2_sf_swizzled = swizzle_blockscale(w2_sf_linear)
    w2_sf_mma = convert_sf_to_mma_layout(
        w2_sf_swizzled.contiguous().view(torch.uint8).reshape(-1),
        m=hidden_size,
        k=intermediate_size,
        num_groups=num_experts,
        sf_vec_size=sf_vec_size,
    )

    (
        trtllm_gemm1_weights,
        trtllm_gemm1_scales,
        trtllm_gemm2_weights,
        trtllm_gemm2_scales,
    ) = prepare_static_weights_for_trtllm_fp4_moe(
        w1_cutlass["w1_q"],
        w2_q,
        w1_cutlass["w1_sf_flat"],
        w2_sf_flat,
        hidden_size,
        intermediate_size,
        num_experts,
    )

    result = {
        "x_bf16": x_bf16,
        "x_q": x_q,
        "x_sf": x_sf_raw,
        "x_sf_cutlass_fp4": x_sf_cutlass_fp4,
        "x_sf_linear": x_sf_linear,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "w1_q": w1_cutlass["w1_q"],
        "w1_sf_by_layout": w1_cutlass["w1_sf_by_layout"],
        "cutedsl_w1_q": w1_cutedsl["w1_q"],
        "cutedsl_w1_sf_by_layout": w1_cutedsl["w1_sf_by_layout"],
        "w2_q": w2_q,
        "w2_sf_by_layout": {"swizzled": w2_sf_swizzled, "mma": w2_sf_mma},
        "fc1_input_scale": contract["w13_input_scale_quant"],
        "w1_alpha": contract["g1_alphas"],
        "fc2_input_scale": contract["w2_input_scale_quant"],
        "w2_alpha": contract["g2_alphas"],
        "trtllm_output1_scale_scalar": contract["g1_scale_c"],
        "trtllm_gemm1_weights": trtllm_gemm1_weights,
        "trtllm_gemm1_scales": trtllm_gemm1_scales,
        "trtllm_gemm2_weights": trtllm_gemm2_weights,
        "trtllm_gemm2_scales": trtllm_gemm2_scales,
    }
    if "w2_input_scale_quant_buggy" in contract:
        result["w2_input_scale_quant_buggy"] = contract["w2_input_scale_quant_buggy"]
    if "g2_alphas_buggy" in contract:
        result["g2_alphas_buggy"] = contract["g2_alphas_buggy"]
    if "g2_alphas_raw" in contract:
        result["w2_alpha_trtllm"] = contract["g2_alphas_raw"]
    if "w2_input_scale_quant_per_expert" in contract:
        result["fc2_input_scale_per_expert"] = contract[
            "w2_input_scale_quant_per_expert"
        ]
        result["w2_alpha_raw"] = contract["g2_alphas_raw"]
    return result


def run_cutlass(
    tensors: dict, hidden_size: int, weight_sf_layout: str, input_mode: str
) -> torch.Tensor:
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    output = torch.empty(
        tensors["topk_ids"].shape[0],
        hidden_size,
        dtype=torch.bfloat16,
        device=tensors["x_q"].device,
    )
    num_experts = tensors["w1_q"].shape[0]
    fc1_act_global = normalize_cutlass_fc1_act_global(
        tensors["fc1_input_scale"].to(torch.float32), num_experts
    )
    w1_alpha = normalize_cutlass_expert_scale(
        tensors["w1_alpha"].to(torch.float32), num_experts
    )
    cutlass_fc2 = tensors.get("fc2_input_scale_per_expert", tensors["fc2_input_scale"])
    fc2_input_scale = normalize_cutlass_expert_scale(
        cutlass_fc2.to(torch.float32), num_experts
    )
    cutlass_w2a = tensors.get("w2_alpha_raw", tensors["w2_alpha"])
    w2_alpha = normalize_cutlass_expert_scale(
        cutlass_w2a.to(torch.float32), num_experts
    )

    if input_mode == "bf16":
        cutlass_input = tensors["x_bf16"]
        cutlass_input_sf = None
    elif input_mode == "fp4":
        cutlass_input = tensors["x_q"]
        cutlass_input_sf = tensors["x_sf_cutlass_fp4"]
    else:
        raise ValueError(f"Unsupported CUTLASS input mode: {input_mode}")

    out = cutlass_fused_moe(
        output=output,
        input=cutlass_input,
        token_selected_experts=tensors["topk_ids"].to(torch.int),
        token_final_scales=tensors["topk_weights"],
        fc1_expert_weights=tensors["w1_q"].view(torch.long),
        fc2_expert_weights=tensors["w2_q"].view(torch.long),
        output_dtype=torch.bfloat16,
        input_sf=cutlass_input_sf,
        quant_scales=[
            fc1_act_global,
            tensors["w1_sf_by_layout"][weight_sf_layout].contiguous().view(torch.int32),
            w1_alpha,
            fc2_input_scale,
            tensors["w2_sf_by_layout"][weight_sf_layout].contiguous().view(torch.int32),
            w2_alpha,
        ],
        ep_size=1,
        ep_rank=0,
        tp_size=1,
        tp_rank=0,
        tune_max_num_tokens=next_pow2(tensors["topk_ids"].shape[0]),
        activation_type=ActivationType.Swiglu,
        enable_alltoall=False,
    )[0]
    return out


def run_cutedsl(
    tensors: dict,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    weight_sf_layout: str,
) -> torch.Tensor:
    from flashinfer import CuteDslMoEWrapper

    moe = CuteDslMoEWrapper(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        use_cuda_graph=False,
        num_local_experts=num_experts,
        local_expert_offset=0,
        output_dtype=torch.bfloat16,
        device=str(tensors["x_q"].device),
    )
    x_sf = tensors["x_sf"].contiguous().view(tensors["x_q"].shape[0], -1)
    if x_sf.dtype != torch.uint8:
        x_sf = x_sf.view(torch.uint8)

    out = moe.run(
        x=tensors["x_q"],
        x_sf=x_sf,
        token_selected_experts=tensors["topk_ids"],
        token_final_scales=tensors["topk_weights"],
        w1_weight=tensors["cutedsl_w1_q"],
        w1_weight_sf=tensors["cutedsl_w1_sf_by_layout"][weight_sf_layout],
        w1_alpha=tensors["w1_alpha"],
        fc2_input_scale=tensors["fc2_input_scale"],
        w2_weight=tensors["w2_q"],
        w2_weight_sf=tensors["w2_sf_by_layout"][weight_sf_layout],
        w2_alpha=tensors["w2_alpha"],
    )
    return out


def run_trtllm_routed(
    tensors: dict,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    routing_method_type: int,
) -> torch.Tensor:
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import ActivationType

    packed_scores = (
        tensors["topk_weights"].to(torch.bfloat16).view(torch.int16).to(torch.int32)
    ) & 0xFFFF
    packed_topk = (tensors["topk_ids"].to(torch.int32) << 16) | packed_scores

    common_kwargs = dict(
        topk_ids=packed_topk,
        routing_bias=None,
        hidden_states=tensors["x_q"],
        hidden_states_scale=tensors["x_sf_linear"].flatten().contiguous(),
        gemm1_weights=tensors["trtllm_gemm1_weights"],
        gemm1_weights_scale=tensors["trtllm_gemm1_scales"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=tensors["trtllm_gemm2_weights"],
        gemm2_weights_scale=tensors["trtllm_gemm2_scales"],
        gemm2_bias=None,
        output1_scale_scalar=tensors["trtllm_output1_scale_scalar"],
        output1_scale_gate_scalar=tensors["w1_alpha"],
        output2_scale_scalar=tensors.get("w2_alpha_trtllm", tensors["w2_alpha"]),
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type,
        do_finalize=True,
    )
    try:
        out = trtllm_fp4_block_scale_routed_moe(
            activation_type=ActivationType.Swiglu.value, **common_kwargs
        )[0]
    except RuntimeError as e:
        if "gated_act_type" not in str(e):
            raise
        out = trtllm_fp4_block_scale_routed_moe(activation_type=0, **common_kwargs)[0]
    return out


def summarize(
    a_out: torch.Tensor, b_out: torch.Tensor, min_norm_for_relative: float = 1e-6
) -> dict:
    a = a_out.float()
    b = b_out.float()
    diff = a - b
    diff_norm = float(torch.norm(diff).item())
    a_norm = float(torch.norm(a).item())
    b_norm = float(torch.norm(b).item())
    rel_l2 = float(diff_norm / b_norm) if b_norm >= min_norm_for_relative else None
    cosine = (
        float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())
        if (a_norm >= min_norm_for_relative and b_norm >= min_norm_for_relative)
        else None
    )
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "rel_l2": rel_l2,
        "cosine": cosine,
        "diff_l2": diff_norm,
        "a_l2": a_norm,
        "b_l2": b_norm,
    }


def timed(fn, repeats: int) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / repeats
    return out, ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal parity harness for Cutlass/CuteDSL/TRTLLM MoE."
    )
    core_group = parser.add_argument_group("core settings")
    core_group.add_argument("--num-tokens", type=int, default=256)
    core_group.add_argument("--hidden-size", type=int, default=2048)
    core_group.add_argument("--intermediate-size", type=int, default=4096)
    core_group.add_argument("--num-experts", type=int, default=16)
    core_group.add_argument("--top-k", type=int, default=1)
    core_group.add_argument("--seed", type=int, default=42)
    core_group.add_argument("--repeats", type=int, default=8)
    core_group.add_argument("--fixed-expert-id", type=int, default=0)
    core_group.add_argument("--uniform-topk-weights", action="store_true")

    contract_group = parser.add_argument_group("backend contract controls")
    contract_group.add_argument(
        "--w1-order", type=str, default="up_gate", choices=["gate_up", "up_gate"]
    )
    contract_group.add_argument(
        "--w1-layout", type=str, default="plain", choices=["plain", "interleaved"]
    )
    contract_group.add_argument(
        "--cutedsl-w1-order",
        type=str,
        default="up_gate",
        choices=["gate_up", "up_gate"],
    )
    contract_group.add_argument(
        "--cutedsl-w1-layout",
        type=str,
        default="interleaved",
        choices=["plain", "interleaved"],
    )
    contract_group.add_argument(
        "--cutedsl-preprocess",
        type=str,
        default="legacy",
        choices=["legacy", "trtllm_post_quant"],
    )
    contract_group.add_argument(
        "--cutlass-weight-sf-layout",
        type=str,
        default="swizzled",
        choices=["swizzled", "mma"],
    )
    contract_group.add_argument(
        "--cutlass-input-mode",
        type=str,
        default="bf16",
        choices=["bf16", "fp4"],
    )
    contract_group.add_argument(
        "--cutedsl-weight-sf-layout",
        type=str,
        default="mma",
        choices=["swizzled", "mma"],
    )

    contract_scale_group = parser.add_argument_group("scale contract selection")
    contract_scale_group.add_argument(
        "--scale-contract",
        type=str,
        default="well-behaved",
        choices=["well-behaved", "realistic"],
        help=(
            "'well-behaved': scalar w2_input_scale_quant (CuteDSL exact). "
            "'realistic': per-expert w2_input_scale_quant with spread, tests "
            "min() (fixed) vs max() (buggy) scalar reduction."
        ),
    )
    contract_scale_group.add_argument(
        "--w2-spread",
        type=float,
        default=7.5,
        help="w2_input_scale spread factor for --scale-contract=realistic (default: 7.5x).",
    )

    advanced_group = parser.add_argument_group("advanced overrides")
    advanced_group.add_argument(
        "--trtllm-routing-method-type",
        type=int,
        default=5,
        help="TRTLLM routed MoE routing_method_type (default: 5=TopK).",
    )
    advanced_group.add_argument(
        "--min-norm-for-relative",
        type=float,
        default=1e-6,
        help="Lower bound on output L2 norm before reporting relative metrics.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    use_realistic = args.scale_contract == "realistic"
    fixed_expert_id = None if args.fixed_expert_id < 0 else args.fixed_expert_id
    tensors = create_moe_tensors(
        num_tokens=args.num_tokens,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        seed=args.seed,
        fixed_expert_id=fixed_expert_id,
        uniform_topk_weights=args.uniform_topk_weights,
        w1_order=args.w1_order,
        w1_layout=args.w1_layout,
        cutedsl_w1_order=args.cutedsl_w1_order,
        cutedsl_w1_layout=args.cutedsl_w1_layout,
        cutedsl_preprocess=args.cutedsl_preprocess,
        scale_contract=args.scale_contract,
        w2_spread=args.w2_spread,
    )

    _ = run_cutlass(
        tensors=tensors,
        hidden_size=args.hidden_size,
        weight_sf_layout=args.cutlass_weight_sf_layout,
        input_mode=args.cutlass_input_mode,
    )
    _ = run_cutedsl(
        tensors=tensors,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        weight_sf_layout=args.cutedsl_weight_sf_layout,
    )

    trtllm_error = None
    try:
        _ = run_trtllm_routed(
            tensors=tensors,
            intermediate_size=args.intermediate_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            routing_method_type=args.trtllm_routing_method_type,
        )
    except Exception as e:
        trtllm_error = str(e)

    cutlass_out, cutlass_ms = timed(
        lambda: run_cutlass(
            tensors=tensors,
            hidden_size=args.hidden_size,
            weight_sf_layout=args.cutlass_weight_sf_layout,
            input_mode=args.cutlass_input_mode,
        ),
        args.repeats,
    )
    cutedsl_out, cutedsl_ms = timed(
        lambda: run_cutedsl(
            tensors=tensors,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            weight_sf_layout=args.cutedsl_weight_sf_layout,
        ),
        args.repeats,
    )

    # With realistic contract, also run CuteDSL with the OLD buggy scales for comparison
    cutedsl_buggy_out = None
    if use_realistic and "g2_alphas_buggy" in tensors:
        buggy_tensors = dict(tensors)
        buggy_tensors["fc2_input_scale"] = tensors["w2_input_scale_quant_buggy"]
        buggy_tensors["w2_alpha"] = tensors["g2_alphas_buggy"]
        _ = run_cutedsl(
            tensors=buggy_tensors,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            weight_sf_layout=args.cutedsl_weight_sf_layout,
        )
        cutedsl_buggy_out, _ = timed(
            lambda: run_cutedsl(
                tensors=buggy_tensors,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                num_experts=args.num_experts,
                top_k=args.top_k,
                weight_sf_layout=args.cutedsl_weight_sf_layout,
            ),
            args.repeats,
        )

    result = {
        "config": {
            "num_tokens": args.num_tokens,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "seed": args.seed,
            "repeats": args.repeats,
            "fixed_expert_id": fixed_expert_id,
            "uniform_topk_weights": args.uniform_topk_weights,
            "scale_contract": args.scale_contract,
            "w2_spread": args.w2_spread if use_realistic else None,
            "w1_order": args.w1_order,
            "w1_layout": args.w1_layout,
            "cutedsl_w1_order": args.cutedsl_w1_order,
            "cutedsl_w1_layout": args.cutedsl_w1_layout,
            "cutedsl_preprocess": args.cutedsl_preprocess,
            "cutlass_weight_sf_layout": args.cutlass_weight_sf_layout,
            "cutlass_input_mode": args.cutlass_input_mode,
            "cutedsl_weight_sf_layout": args.cutedsl_weight_sf_layout,
            "trtllm_routing_method_type": args.trtllm_routing_method_type,
        },
        "latency_ms": {
            "cutlass": cutlass_ms,
            "cutedsl": cutedsl_ms,
            "cutedsl_vs_cutlass_ratio": cutedsl_ms / max(cutlass_ms, 1e-8),
        },
        "parity": {
            "cutedsl_vs_cutlass": summarize(
                cutedsl_out,
                cutlass_out,
                min_norm_for_relative=args.min_norm_for_relative,
            )
        },
    }

    if cutedsl_buggy_out is not None:
        result["parity"]["cutedsl_buggy_vs_cutlass"] = summarize(
            cutedsl_buggy_out,
            cutlass_out,
            min_norm_for_relative=args.min_norm_for_relative,
        )
        result["parity"]["cutedsl_fixed_vs_buggy"] = summarize(
            cutedsl_out,
            cutedsl_buggy_out,
            min_norm_for_relative=args.min_norm_for_relative,
        )

    if trtllm_error is not None:
        result["trtllm_error"] = trtllm_error
    else:
        trtllm_out, trtllm_ms = timed(
            lambda: run_trtllm_routed(
                tensors=tensors,
                intermediate_size=args.intermediate_size,
                num_experts=args.num_experts,
                top_k=args.top_k,
                routing_method_type=args.trtllm_routing_method_type,
            ),
            args.repeats,
        )
        result["latency_ms"]["trtllm"] = trtllm_ms
        result["latency_ms"]["cutedsl_vs_trtllm_ratio"] = cutedsl_ms / max(
            trtllm_ms, 1e-8
        )
        result["latency_ms"]["cutlass_vs_trtllm_ratio"] = cutlass_ms / max(
            trtllm_ms, 1e-8
        )
        result["parity"]["cutedsl_vs_trtllm"] = summarize(
            cutedsl_out, trtllm_out, min_norm_for_relative=args.min_norm_for_relative
        )
        result["parity"]["cutlass_vs_trtllm"] = summarize(
            cutlass_out, trtllm_out, min_norm_for_relative=args.min_norm_for_relative
        )
        if cutedsl_buggy_out is not None:
            result["parity"]["cutedsl_buggy_vs_trtllm"] = summarize(
                cutedsl_buggy_out,
                trtllm_out,
                min_norm_for_relative=args.min_norm_for_relative,
            )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
