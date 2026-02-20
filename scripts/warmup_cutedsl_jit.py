#!/usr/bin/env python3
"""
Pre-warm FlashInfer JIT cache for FP4 MoE backends.

Triggers JIT compilation and autotuning for the specified backends by running
minimal forward passes with dummy tensors sized to the model's architecture.
After this script completes, subsequent sglang server launches will skip
compilation and start in seconds instead of minutes.

Usage:
    # Warm all backends for a model:
    python3 scripts/warmup_cutedsl_jit.py --model nvidia/Qwen3-30B-A3B-FP4

    # Warm only CuteDSL:
    python3 scripts/warmup_cutedsl_jit.py --model nvidia/Qwen3-30B-A3B-FP4 --backends cutedsl

    # Warm with explicit dimensions:
    python3 scripts/warmup_cutedsl_jit.py \\
        --hidden-size 2048 --intermediate-size 1024 \\
        --num-experts 128 --top-k 8

The compiled .so files are cached in ~/.cache/flashinfer/ and persist across runs.
Set FLASHINFER_NVCC_THREADS=4 to speed up parallel compilation.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

SUPPORTED_BACKENDS = ["cutedsl", "cutlass", "trtllm"]


def get_moe_dims_from_model(model_name_or_path: str) -> dict:
    """Read MoE architecture dimensions from a HuggingFace model config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    hidden_size = config.hidden_size

    def _first(*attrs):
        for a in attrs:
            v = getattr(config, a, None)
            if v is not None:
                return v
        return None

    num_experts = _first("num_local_experts", "n_routed_experts", "num_experts")
    top_k = _first("num_experts_per_tok", "top_k")
    intermediate_size = _first("moe_intermediate_size", "intermediate_size")

    if num_experts is None:
        raise ValueError(
            f"Cannot determine num_experts from {model_name_or_path} config. "
            "Use --num-experts explicitly."
        )
    if top_k is None:
        raise ValueError(
            f"Cannot determine top_k from {model_name_or_path} config. "
            "Use --top-k explicitly."
        )
    if intermediate_size is None:
        raise ValueError(
            f"Cannot determine intermediate_size from {model_name_or_path} config. "
            "Use --intermediate-size explicitly."
        )

    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "top_k": top_k,
    }


# ── layout helpers ────────────────────────────────────────────────────────────


def _swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype != torch.float8_e4m3fn:
        scale = scale.to(torch.float8_e4m3fn)
    bsz, m, k = scale.shape
    mp = (m + 127) // 128 * 128
    kp = (k + 3) // 4 * 4
    padded = torch.zeros(bsz, mp, kp, dtype=scale.dtype, device=scale.device)
    padded[:, :m, :k] = scale
    return (
        padded.reshape(bsz, mp // 128, 4, 32, kp // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(bsz, mp, kp)
    )


def _interleave_gate_up(t: torch.Tensor, gs: int = 64, dim: int = 1) -> torch.Tensor:
    half = t.shape[dim] // 2
    gate = t.narrow(dim, 0, half).split(gs, dim=dim)
    up = t.narrow(dim, half, half).split(gs, dim=dim)
    return torch.cat([x for pair in zip(gate, up) for x in pair], dim=dim)


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


# ── shared tensor factory ─────────────────────────────────────────────────────


def _make_shared_tensors(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    num_tokens: int,
):
    """Create shared dummy tensors used by all backends."""
    from flashinfer.fp4_quantization import fp4_quantize

    device = "cuda"
    sf_vec_size = 16

    torch.manual_seed(0)
    global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    x_bf16 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    x_fp4, x_sf_raw = fp4_quantize(
        x_bf16,
        global_scale=global_scale,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )

    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    topk_weights = (
        torch.ones(num_tokens, top_k, dtype=torch.float32, device=device) / top_k
    )

    # Quantize w1 [E, 2*I, H] and w2 [E, H, I]
    def quantize_weight(rows, cols, num_groups):
        w_bf16 = (
            torch.randn(num_groups, rows, cols, dtype=torch.bfloat16, device=device)
            * 0.01
        )
        wq_flat, wsf_flat = fp4_quantize(
            w_bf16.view(num_groups * rows, cols),
            global_scale=global_scale,
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=False,
        )
        wq = wq_flat.view(num_groups, rows, cols // 2)
        wsf_lin = wsf_flat.view(torch.float8_e4m3fn).reshape(
            num_groups, rows, cols // sf_vec_size
        )
        wsf_swiz = _swizzle_blockscale(wsf_lin)
        return wq, wsf_flat, wsf_swiz

    w1q, w1sf_flat, w1sf_swiz = quantize_weight(
        2 * intermediate_size, hidden_size, num_experts
    )
    w2q, w2sf_flat, w2sf_swiz = quantize_weight(
        hidden_size, intermediate_size, num_experts
    )

    alphas = torch.ones(num_experts, dtype=torch.float32, device=device)
    scalar_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    return {
        "device": device,
        "sf_vec_size": sf_vec_size,
        "x_bf16": x_bf16,
        "x_fp4": x_fp4,
        "x_sf_raw": x_sf_raw,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "w1q": w1q,
        "w1sf_flat": w1sf_flat,
        "w1sf_swiz": w1sf_swiz,
        "w2q": w2q,
        "w2sf_flat": w2sf_flat,
        "w2sf_swiz": w2sf_swiz,
        "alphas": alphas,
        "scalar_scale": scalar_scale,
    }


# ── backend warmup functions ──────────────────────────────────────────────────


def warmup_cutedsl(shared: dict, H: int, I: int, E: int, K: int, T: int) -> float:
    from flashinfer import CuteDslMoEWrapper
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    sf = shared["sf_vec_size"]

    w1sf_mma = convert_sf_to_mma_layout(
        shared["w1sf_swiz"].contiguous().view(torch.uint8).reshape(-1),
        m=2 * I,
        k=H,
        num_groups=E,
        sf_vec_size=sf,
    )
    w2sf_mma = convert_sf_to_mma_layout(
        shared["w2sf_swiz"].contiguous().view(torch.uint8).reshape(-1),
        m=H,
        k=I,
        num_groups=E,
        sf_vec_size=sf,
    )
    w1q = _interleave_gate_up(
        shared["w1q"].view(torch.uint8), gs=64, dim=1
    ).contiguous()

    x_sf = shared["x_sf_raw"].contiguous().view(T, -1)
    if x_sf.dtype != torch.uint8:
        x_sf = x_sf.view(torch.uint8)

    moe = CuteDslMoEWrapper(
        num_experts=E,
        top_k=K,
        hidden_size=H,
        intermediate_size=I,
        use_cuda_graph=False,
        num_local_experts=E,
        local_expert_offset=0,
        output_dtype=torch.bfloat16,
        device=shared["device"],
    )

    torch.cuda.synchronize()
    t0 = time.time()
    moe.run(
        x=shared["x_fp4"],
        x_sf=x_sf,
        token_selected_experts=shared["topk_ids"],
        token_final_scales=shared["topk_weights"],
        w1_weight=w1q,
        w1_weight_sf=w1sf_mma,
        w1_alpha=shared["alphas"],
        fc2_input_scale=shared["scalar_scale"],
        w2_weight=shared["w2q"],
        w2_weight_sf=w2sf_mma,
        w2_alpha=shared["alphas"],
    )
    torch.cuda.synchronize()
    return time.time() - t0


def warmup_cutlass(shared: dict, H: int, I: int, E: int, K: int, T: int) -> float:
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    x_sf_interleaved = nvfp4_block_scale_interleave(
        shared["x_sf_raw"].view(T, H // 16).contiguous()
    )

    output = torch.empty(T, H, dtype=torch.bfloat16, device=shared["device"])
    per_expert_scale = shared["scalar_scale"].expand(E).contiguous()

    torch.cuda.synchronize()
    t0 = time.time()
    cutlass_fused_moe(
        output=output,
        input=shared["x_fp4"],
        token_selected_experts=shared["topk_ids"].to(torch.int),
        token_final_scales=shared["topk_weights"],
        fc1_expert_weights=shared["w1q"].view(torch.long),
        fc2_expert_weights=shared["w2q"].view(torch.long),
        output_dtype=torch.bfloat16,
        input_sf=x_sf_interleaved,
        quant_scales=[
            per_expert_scale,
            shared["w1sf_swiz"].contiguous().view(torch.int32),
            shared["alphas"],
            per_expert_scale,
            shared["w2sf_swiz"].contiguous().view(torch.int32),
            shared["alphas"],
        ],
        ep_size=1,
        ep_rank=0,
        tp_size=1,
        tp_rank=0,
        tune_max_num_tokens=_next_pow2(T),
        activation_type=ActivationType.Swiglu,
        enable_alltoall=False,
    )
    torch.cuda.synchronize()
    return time.time() - t0


def warmup_trtllm(shared: dict, H: int, I: int, E: int, K: int, T: int) -> float:
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import (
        ActivationType,
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    cache: dict = {}
    tile = 128
    device = shared["device"]

    w1f = shared["w1q"].view(torch.float8_e4m3fn).reshape(E, 2 * I, H // 2)
    w1s = shared["w1sf_flat"].view(torch.float8_e4m3fn).reshape(E, 2 * I, H // 16)
    w2f = shared["w2q"].view(torch.float8_e4m3fn).reshape(E, H, I // 2)
    w2s = shared["w2sf_flat"].view(torch.float8_e4m3fn).reshape(E, H, I // 16)

    g1w, g1s, g2w, g2s = [], [], [], []
    for i in range(E):
        pi = _maybe_get_cached_w3_w1_permute_indices(
            cache, w1f[i].view(torch.uint8), tile
        )
        g1w.append(w1f[i].view(torch.uint8)[pi.to(device)].contiguous())
        psi = _maybe_get_cached_w3_w1_permute_indices(
            cache, w1s[i].view(torch.uint8), tile, num_elts_per_sf=16
        )
        g1s.append(
            nvfp4_block_scale_interleave(
                w1s[i].view(torch.uint8)[psi.to(device)].contiguous()
            )
        )
        pi2 = get_w2_permute_indices_with_cache(cache, w2f[i].view(torch.uint8), tile)
        g2w.append(w2f[i].view(torch.uint8)[pi2.to(device)].contiguous())
        psi2 = get_w2_permute_indices_with_cache(
            cache, w2s[i].view(torch.uint8), tile, num_elts_per_sf=16
        )
        g2s.append(
            nvfp4_block_scale_interleave(
                w2s[i].view(torch.uint8)[psi2.to(device)].contiguous()
            )
        )

    trt_g1w = torch.stack(g1w)
    trt_g1s = torch.stack(g1s).view(torch.float8_e4m3fn).reshape(E, 2 * I, H // 16)
    trt_g2w = torch.stack(g2w)
    trt_g2s = torch.stack(g2s).view(torch.float8_e4m3fn).reshape(E, H, I // 16)

    packed_scores = (
        shared["topk_weights"].to(torch.bfloat16).view(torch.int16).to(torch.int32)
    ) & 0xFFFF
    packed_topk = (shared["topk_ids"].to(torch.int32) << 16) | packed_scores

    x_sf_linear = shared["x_sf_raw"].view(torch.float8_e4m3fn)

    torch.cuda.synchronize()
    t0 = time.time()
    try:
        trtllm_fp4_block_scale_routed_moe(
            activation_type=ActivationType.Swiglu.value,
            topk_ids=packed_topk,
            routing_bias=None,
            hidden_states=shared["x_fp4"],
            hidden_states_scale=x_sf_linear.flatten().contiguous(),
            gemm1_weights=trt_g1w,
            gemm1_weights_scale=trt_g1s,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=trt_g2w,
            gemm2_weights_scale=trt_g2s,
            gemm2_bias=None,
            output1_scale_scalar=shared["alphas"],
            output1_scale_gate_scalar=shared["alphas"],
            output2_scale_scalar=shared["alphas"],
            num_experts=E,
            top_k=K,
            n_group=None,
            topk_group=None,
            intermediate_size=I,
            local_expert_offset=0,
            local_num_experts=E,
            routed_scaling_factor=None,
            routing_method_type=5,
            do_finalize=True,
        )
    except RuntimeError as e:
        if "gated_act_type" not in str(e):
            raise
        trtllm_fp4_block_scale_routed_moe(
            activation_type=0,
            topk_ids=packed_topk,
            routing_bias=None,
            hidden_states=shared["x_fp4"],
            hidden_states_scale=x_sf_linear.flatten().contiguous(),
            gemm1_weights=trt_g1w,
            gemm1_weights_scale=trt_g1s,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=trt_g2w,
            gemm2_weights_scale=trt_g2s,
            gemm2_bias=None,
            output1_scale_scalar=shared["alphas"],
            output1_scale_gate_scalar=shared["alphas"],
            output2_scale_scalar=shared["alphas"],
            num_experts=E,
            top_k=K,
            n_group=None,
            topk_group=None,
            intermediate_size=I,
            local_expert_offset=0,
            local_num_experts=E,
            routed_scaling_factor=None,
            routing_method_type=5,
            do_finalize=True,
        )
    torch.cuda.synchronize()
    return time.time() - t0


def warmup_fp4_gemm(H: int) -> float:
    """Trigger FlashInfer FP4 GEMM autotuner for dense-layer dimensions."""
    from flashinfer.fp4_quantization import fp4_quantize

    device = "cuda"
    sf_vec_size = 16
    global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    x = torch.randn(16, H, dtype=torch.bfloat16, device=device)
    w = torch.randn(H, H, dtype=torch.bfloat16, device=device)
    xq, xsf = fp4_quantize(
        x,
        global_scale=global_scale,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )
    wq, wsf = fp4_quantize(
        w,
        global_scale=global_scale,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )

    torch.cuda.synchronize()
    t0 = time.time()
    try:
        from flashinfer.gemm import fp4_gemm

        fp4_gemm(xq, xsf, wq, wsf, global_scale, global_scale)
    except (ImportError, Exception) as e:
        print(f"    FP4 GEMM warmup skipped: {e}")
    torch.cuda.synchronize()
    return time.time() - t0


# ── main ──────────────────────────────────────────────────────────────────────


WARMUP_FNS = {
    "cutedsl": warmup_cutedsl,
    "cutlass": warmup_cutlass,
    "trtllm": warmup_trtllm,
}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-warm FlashInfer JIT cache for FP4 MoE backends."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name/path to read MoE dimensions from config.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["all"],
        choices=SUPPORTED_BACKENDS + ["all"],
        help="Backends to warm up (default: all).",
    )
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=16,
        help="Dummy token count (small is fine, just triggers compilation).",
    )
    parser.add_argument(
        "--skip-fp4-gemm",
        action="store_true",
        help="Skip the general FP4 GEMM autotuner warmup.",
    )
    args = parser.parse_args()

    backends = SUPPORTED_BACKENDS if "all" in args.backends else args.backends

    dims = {}
    if args.model:
        print(f"Reading model config from {args.model}...")
        dims = get_moe_dims_from_model(args.model)
        print(f"  Detected: {json.dumps(dims)}")

    H = args.hidden_size or dims.get("hidden_size")
    I = args.intermediate_size or dims.get("intermediate_size")
    E = args.num_experts or dims.get("num_experts")
    K = args.top_k or dims.get("top_k")
    T = args.num_tokens

    missing = []
    if not H:
        missing.append("--hidden-size")
    if not I:
        missing.append("--intermediate-size")
    if not E:
        missing.append("--num-experts")
    if not K:
        missing.append("--top-k")
    if missing:
        parser.error(
            f"Missing required dimensions: {', '.join(missing)}. "
            "Provide --model or specify them explicitly."
        )

    print(f"\nDimensions: H={H} I={I} E={E} top_k={K} tokens={T}")
    print(f"Backends: {', '.join(backends)}")

    if not args.skip_fp4_gemm:
        print(f"\n--- FP4 GEMM autotuner warmup ---")
        elapsed = warmup_fp4_gemm(H)
        print(f"  Completed in {elapsed:.1f}s")

    print(f"\nCreating shared dummy tensors...")
    shared = _make_shared_tensors(H, I, E, K, T)

    total = 0.0
    for backend in backends:
        print(f"\n--- {backend} ---")
        fn = WARMUP_FNS[backend]
        try:
            elapsed = fn(shared, H, I, E, K, T)
            print(f"  Completed in {elapsed:.1f}s")
            total += elapsed
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nTotal warmup time: {total:.1f}s")
    print("Cache populated. Subsequent server launches will skip compilation.")


if __name__ == "__main__":
    main()
