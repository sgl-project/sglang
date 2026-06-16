from typing import Optional, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_cuda_alike

from .utils import make_name

_is_cuda_alike = is_cuda_alike()


@cache_once
def _jit_mask_topk_module():
    return load_jit(
        make_name("mask_topk"),
        cuda_files=["deepseek_v4/hash_topk.cuh"],
        cuda_wrappers=[("run", "MaskKernel::run")],
    )


@cache_once
def _jit_hash_topk_module():
    args = make_cpp_args("act_sqrt_softplus", is_arch_support_pdl())
    return load_jit(
        make_name("hash_topk"),
        *args,
        cuda_files=["deepseek_v4/hash_topk.cuh"],
        cuda_wrappers=[("hash_topk", f"HashTopKKernel<{args}>::run")],
    )


@cache_once
def _jit_mega_moe_pre_dispatch_module(quant_group_size: int):
    args = make_cpp_args(quant_group_size, is_arch_support_pdl())
    return load_jit(
        make_name("mega_moe_pre_dispatch"),
        *args,
        cuda_files=["deepseek_v4/mega_moe_pre_dispatch.cuh"],
        cuda_wrappers=[("run", f"MegaMoEPreDispatchKernel<{args}>::run")],
    )


@cache_once
def _jit_silu_mul_quant_varlen_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    swizzle: bool,
    apply_swiglu_limit: bool,
):
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        make_name("silu_mul_quant_varlen"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulMaskedPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_silu_mul_quant_contig_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    swizzle: bool,
    apply_swiglu_limit: bool,
):
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        make_name("silu_mul_quant_contig"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulContigPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_silu_and_mul_clamp_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("silu_and_mul_clamp"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulClampKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


def mask_topk_ids(topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor):
    return _jit_mask_topk_module().run(topk_ids, num_token_non_padded)


def hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert scoring_func == "sqrtsoftplus"
    if is_hip_runtime():
        from sglang.jit_kernel.triton.hash_topk import hash_topk_triton

        return hash_topk_triton(
            router_logits,
            input_ids,
            tid2eid,
            num_fused_shared_experts,
            routed_scaling_factor,
            scoring_func,
        )
    else:
        num_tokens = router_logits.size(0)
        topk_routed = tid2eid.size(1)
        topk_fused = topk_routed + num_fused_shared_experts
        topk_ids = torch.empty(
            (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
        )
        topk_weights = torch.empty(
            (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
        )
        module = _jit_hash_topk_module()
        module.hash_topk(
            router_logits,
            input_ids,
            tid2eid,
            topk_weights,
            topk_ids,
            routed_scaling_factor,
        )
        return topk_weights, topk_ids


def mega_moe_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    quant_group_size: int = 32,
) -> None:
    module = _jit_mega_moe_pre_dispatch_module(quant_group_size)
    module.run(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
    )


def silu_and_mul_clamp(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    if _is_cuda_alike:
        module = _jit_silu_and_mul_clamp_module(input.dtype)
        module.run(input, output, float(swiglu_limit))
    else:
        silu_and_mul_clamp_torch(input, output, float(swiglu_limit))


def silu_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    apply_swiglu_limit = swiglu_limit is not None
    module = _jit_silu_mul_quant_varlen_module(
        quant_group_size, scale_ue8m0, swizzle, apply_swiglu_limit
    )
    module.run(
        input,
        output,
        output_scale,
        masked_m,
        topk,
        transposed,
        float(swiglu_limit) if apply_swiglu_limit else 0.0,
    )


def silu_and_mul_contig_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    scale_ue8m0: bool = False,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    apply_swiglu_limit = swiglu_limit is not None
    module = _jit_silu_mul_quant_contig_module(
        quant_group_size, scale_ue8m0, swizzle, apply_swiglu_limit
    )
    module.run(
        input,
        output,
        output_scale,
        transposed,
        float(swiglu_limit) if apply_swiglu_limit else 0.0,
    )


def silu_and_mul_clamp_torch(
    input: torch.Tensor,  # (M, 2*H)  bf16 or fp16
    output: torch.Tensor,  # (M, H)    same dtype, pre-allocated
    swiglu_limit: float,
) -> None:
    """
    In-place fused SiLU-and-Mul with optional BF16 swiglu clamping.
    Writes result into *output*; returns nothing.

    Args:
        input:        (M, 2*H) tensor — gate and up concatenated on last dim.
        output:       (M, H)   tensor — pre-allocated output buffer.
        swiglu_limit: Clamping bound (positive scalar). Applied in BF16,
                      matching the CUDA __hmin2/__hmax2 semantics.

    Pure-PyTorch reference implementation of SiluAndMulClampKernel::run.

    Matches the CUDA kernel in:
      python/sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh

    Algorithm per token row:
      1. Split input (M, 2*H) into gate (M, H) and up (M, H).
      2. Clamp in BF16 precision  (critical: matches DeepGEMM reference):
           gate = min(gate,  limit)          -- upper-clamp only
           up   = clamp(up, -limit, limit)   -- both sides
      3. SiLU(gate) * up  in fp32, cast back to the input dtype.

    Reference:
      https://github.com/deepseek-ai/DeepGEMM/blob/7f2a703ed51ac1f7af07f5e1453b2d3267d37d50/
      deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L984-L997
    """
    M, D = input.shape
    assert D % 2 == 0, "input last dim must be even (gate || up)"
    H = D // 2
    assert output.shape == (M, H), f"output must be ({M}, {H}), got {output.shape}"
    assert input.dtype in (
        torch.bfloat16,
        torch.float16,
    ), "only bf16/fp16 supported (matches CUDA static_assert sizeof(DType)==2)"

    # ------------------------------------------------------------------
    # Step 1: Split into gate and up halves
    # Matches: gate_vec.load(input, bid*2+0); up_vec.load(input, bid*2+1)
    # in the CTA-tiled kernel where each CTA handles one token row.
    # ------------------------------------------------------------------
    gate = input[:, :H]  # (M, H)
    up = input[:, H:]  # (M, H)

    # ------------------------------------------------------------------
    # Step 2: Clamp in BF16
    # CUDA (kApplySwigluLimit=true):
    #   gate = __hmin2(gate, {limit, limit})      → gate ≤ limit
    #   up   = __hmax2(up,  {-limit, -limit})     → up  ≥ -limit
    #   up   = __hmin2(up,  { limit,  limit})     → up  ≤  limit
    #
    # The comment in the CUDA source stresses that clamping MUST happen
    # in bf16 (not fp32) to match the DeepGEMM reference behaviour, so we
    # cast to bf16 before clamping even when the input is fp16.
    # ------------------------------------------------------------------
    gate_bf16 = gate.to(torch.bfloat16)
    up_bf16 = up.to(torch.bfloat16)

    gate_clamped = gate_bf16.clamp(max=swiglu_limit)  # upper only
    up_clamped = up_bf16.clamp(min=-swiglu_limit, max=swiglu_limit)  # both sides

    # ------------------------------------------------------------------
    # Step 3: SiLU(gate) * up  in fp32
    # CUDA silu_and_mul (kPrecise=true):
    #   silu0 = g0 / (1 + exp(-g0))
    #   val0  = silu0 * u0
    # ------------------------------------------------------------------
    gate_fp32 = gate_clamped.float()
    up_fp32 = up_clamped.float()

    silu_gate = gate_fp32 * torch.sigmoid(
        gate_fp32
    )  # equivalent to torch.nn.functional.silu
    result = silu_gate * up_fp32  # (M, H)  fp32

    # ------------------------------------------------------------------
    # Step 4: Cast back to input dtype and write to output
    # Matches: tile.store(params.output, out, bid)
    # ------------------------------------------------------------------
    output.copy_(result.to(input.dtype))
