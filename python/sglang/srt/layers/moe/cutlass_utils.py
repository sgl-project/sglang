from typing import Optional, Tuple

import torch

from sglang.srt.layers.moe.cutlass_moe_params import (
    CutlassMoEParams,
    CutlassMoEType,
)
from sglang.srt.utils import is_cuda, is_sm90_supported

if is_cuda():
    from sgl_kernel import (
        cutlass_fp4_group_mm,
        cutlass_w4a8_moe_mm,
        es_fp8_blockwise_scaled_grouped_mm,
        es_sm100_mxfp8_blockscaled_grouped_mm,
        es_sm100_mxfp8_blockscaled_grouped_quant,
        fp8_blockwise_scaled_grouped_mm,
        scaled_fp4_experts_quant,
        silu_and_mul,
    )
    from sgl_kernel.gemm import sgl_per_tensor_quant_fp8


def w4a8_moe(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    params: CutlassMoEParams,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    masked_m: Optional[torch.Tensor] = None,
    deepep_ll_or_deepep_normal: Optional[CutlassMoEType] = None,
) -> torch.Tensor:
    from sglang.srt.layers.moe.ep_moe.kernels import (
        silu_and_mul_masked_post_per_tensor_quant_fwd,
        silu_mul_static_tensorwise_quant_for_cutlass_moe,
    )

    """
    This function computes a w4a8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    This unified function handles three MoE types:
    1. DeepEP_LL: Post-per-tensor quantized FP8 input
        - `a` Shape: [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, K]
    2. DeepEP_Normal: Preprocessed FP8 input (permuted and quantized)
        - `a` Shape: [M * topk, K]
    3. W4A8: Standard preprocessed FP8 input
        - `a` Shape: [M * topk, K]

    Parameters:
    - a (torch.Tensor): The preprocessed FP8 input tensor to the MoE layer.
    - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
        Shape: [num_experts, N * 2,  K // 2]
        (the weights are passed transposed and int4-packed)
    - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
        Shape: [num_experts, K, N // 2]
        (the weights are passed transposed and int4-packed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts, K // 512, N * 8]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts, N // 512, K * 4]
    - topk_ids (torch.Tensor): The ids of each token->expert mapping.
    - params (CutlassMoEParams): The initialized Cutlass MoE parameters.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [1, K]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [1, N]
    - masked_m (Optional[torch.Tensor]): The valid tokens mask for DeepEP_LL mode.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer (pre-reordered for W4A8, otherwise fully processed).
    """
    gateup_input = a
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids.size(1)

    device = a.device

    if deepep_ll_or_deepep_normal == CutlassMoEType.DeepEP_LL:
        num_experts = w1_q.size(0)
        m = a.size(1)
        c1_shape = (num_experts, m, n * 2)
        c2_shape = (num_experts, m, k)
    else:
        m_tokens = topk_ids.size(0)
        c1_shape = (m_tokens * topk, n * 2)
        c2_shape = (m_tokens * topk, k)

    c1 = torch.empty(c1_shape, device=device, dtype=torch.bfloat16)
    if deepep_ll_or_deepep_normal == CutlassMoEType.DeepEP_Normal:
        c2 = torch.zeros(c2_shape, device=device, dtype=torch.bfloat16)
    else:
        c2 = torch.empty(c2_shape, device=device, dtype=torch.bfloat16)

    cutlass_w4a8_moe_mm(
        c1,
        gateup_input,
        w1_q,
        a1_scale.float(),
        w1_scale,
        params.expert_offsets[:-1],
        params.problem_sizes1,
        params.a_strides1,
        params.b_strides1,
        params.c_strides1,
        params.s_strides13,
        128,
        topk,
    )

    if deepep_ll_or_deepep_normal == CutlassMoEType.DeepEP_LL:
        intermediate_q = torch.empty(
            (num_experts, m, n), dtype=torch.float8_e4m3fn, device=device
        )
        silu_and_mul_masked_post_per_tensor_quant_fwd(
            c1, intermediate_q, masked_m, a2_scale
        )
    elif deepep_ll_or_deepep_normal == CutlassMoEType.DeepEP_Normal:
        intermediate = torch.empty(c1_shape[0], n, device=device, dtype=torch.bfloat16)
        silu_and_mul(c1, intermediate)

        intermediate_q = torch.empty(
            intermediate.shape, dtype=torch.float8_e4m3fn, device=device
        )
        sgl_per_tensor_quant_fp8(intermediate, intermediate_q, a2_scale.float(), True)
    else:
        intermediate_q = torch.empty(
            (c1_shape[0], n), dtype=torch.float8_e4m3fn, device=device
        )
        silu_mul_static_tensorwise_quant_for_cutlass_moe(
            c1,
            intermediate_q,
            a2_scale.float(),
            params.expert_offsets[-1:],
            c1_shape[0],
            n,
        )

    cutlass_w4a8_moe_mm(
        c2,
        intermediate_q,
        w2_q,
        a2_scale.float(),
        w2_scale,
        params.expert_offsets[:-1],
        params.problem_sizes2,
        params.a_strides2,
        params.b_strides2,
        params.c_strides2,
        params.s_strides2,
        128,
        topk,
    )

    return c2


def cutlass_fused_experts_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    params: CutlassMoEParams,
    rep_a1_scales: torch.Tensor,
    blockscale_offsets: Optional[torch.Tensor] = None,
    max_blockscale: Optional[int] = None,
    use_mxfp8: bool = False,
    enable_es: Tuple[bool, bool] = (False, False),
) -> torch.Tensor:
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    """Performs Fused MoE computation using CUTLASS-like kernels with FP8 weights and activations.

    This function implements a Mixture of Experts (MoE) layer with a SwiGLU/SiLU
    activation, leveraging custom kernels likely derived from CUTLASS principles
    for grouped matrix multiplication (`fp8_blockwise_scaled_grouped_mm`) and
    data preparation (`prepare_moe_input`, `silu_and_mul`).

    It takes shuffled input activations, quantizes them to FP8 with per-token scales,
    performs the expert computations using FP8 GEMMs with pre-quantized FP8 weights
    (per-block scales), applies the SiLU activation, and returns the raw expert outputs
    for combination.

    Args:
        a (torch.Tensor): Preprocessed shuffled input activations. Shape: `(m * topk, k)`,
            where `m` is the number of tokens and `k` is the hidden size. Expected dtype: `torch.half`
            or `torch.bfloat16`.
        w1_q (torch.Tensor): Pre-quantized FP8 weight tensor for the first GEMM
            (up-projection part of SwiGLU). Expected shape: `(E, k, n*2)`, where
            `E` is the number of experts, `k` is the hidden size, and `n*2` is the
            intermediate size (`I`). Expected dtype: `torch.float8_e4m3fn`.
            Note: This shape implies weights are stored as (num_experts, hidden_size, intermediate_size).
        w2_q (torch.Tensor): Pre-quantized FP8 weight tensor for the second GEMM
            (down-projection). Expected shape: `(E, n, k)`, where `n` is half the
            intermediate size (`I // 2`). Expected dtype: `torch.float8_e4m3fn`.
            Note: This shape implies weights are stored as (num_experts, intermediate_size // 2, hidden_size).
        w1_scale (torch.Tensor): Scales corresponding to `w1_q` (per-block scales).
            Shape: `(E, num_blocks_n, num_blocks_k)`. Dtype: `torch.float32`.
        w2_scale (torch.Tensor): Scales corresponding to `w2_q` (per-block scales).
            Shape: `(E, num_blocks_k, num_blocks_n)`. Dtype: `torch.float32`.
        topk_ids (torch.Tensor): Indices of the selected top-k experts for each token.
            Shape: `(m, topk)`. Dtype: `torch.int32`.
        params (CutlassMoEParams): The initialized Cutlass MoE parameters.
        use_fp8_blockscale (bool, optional): Flag indicating usage of FP8 with
            block scaling. Currently, only `True` is supported. Defaults to `True`.
        enable_es (tuple(bool, bool)): Flag indicating usage of expert specialization kernel for (up-projection, down-projection)
    Returns:
        torch.Tensor: Raw expert outputs with shape `(m * topk, k)`, where each token's expert assignments are contiguous.

    Raises:
        AssertionError: If input shapes, dtypes, or flags are inconsistent or unsupported.
        NotImplementedError: If CUDA is not available or `sgl_kernel` is not properly installed.
    """

    es_up, es_down = enable_es
    out_dtype = torch.bfloat16  # Default output dtype for FP8 operations
    num_experts = w1_q.size(0)
    m = a.size(0) // topk_ids.size(1)  # Original number of tokens
    topk = topk_ids.size(1)
    k = w1_q.size(1)
    n = w2_q.size(1)

    device = a.device
    rep_a_q = a

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)

    a_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)
    w_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)

    if is_sm90_supported() and es_up:
        es_fp8_blockwise_scaled_grouped_mm(
            c1,
            rep_a_q,
            w1_q,
            rep_a1_scales,
            w1_scale,
            params.ab_strides_13,
            params.ab_strides_13,
            params.c_strides_13,
            params.problem_sizes1,
            params.expert_offsets[:-1],
            params.workspace,
        )
    elif use_mxfp8 and es_up:
        es_sm100_mxfp8_blockscaled_grouped_mm(
            c1,
            rep_a_q,
            w1_q,
            rep_a1_scales,
            w1_scale,
            params.problem_sizes1,
            params.expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
    else:
        fp8_blockwise_scaled_grouped_mm(
            c1,
            params.a_ptrs,
            params.b_ptrs,
            params.out_ptrs,
            params.a_scales_ptrs,
            params.b_scales_ptrs,
            rep_a_q,
            w1_q,
            rep_a1_scales,
            w1_scale,
            params.ab_strides_13,
            params.ab_strides_13,
            params.c_strides_13,
            a_sf_layout,
            w_sf_layout,
            params.problem_sizes1,
            params.expert_offsets[:-1],
            params.workspace,
        )

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    silu_and_mul(c1, intermediate)

    if use_mxfp8 and es_down:
        intemediate_q = torch.empty_like(intermediate, dtype=torch.float8_e4m3fn)
        a2_scale = torch.empty(
            (max_blockscale, n // 32), dtype=torch.uint8, device=device
        )
        es_sm100_mxfp8_blockscaled_grouped_quant(
            intermediate,
            params.problem_sizes2,
            params.expert_offsets[:-1],
            blockscale_offsets[:-1],
            intemediate_q,
            a2_scale,
        )
    else:
        intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(intermediate, 128)

    if is_sm90_supported() and es_down:
        es_fp8_blockwise_scaled_grouped_mm(
            c2,
            intemediate_q,
            w2_q,
            a2_scale,
            w2_scale,
            params.ab_strides_2,
            params.ab_strides_2,
            params.c_strides_2,
            params.problem_sizes2,
            params.expert_offsets[:-1],
            params.workspace,
        )
    elif use_mxfp8 and es_down:
        es_sm100_mxfp8_blockscaled_grouped_mm(
            c2,
            intemediate_q,
            w2_q,
            a2_scale,
            w2_scale,
            params.problem_sizes2,
            params.expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
    else:
        fp8_blockwise_scaled_grouped_mm(
            c2,
            params.a_ptrs,
            params.b_ptrs,
            params.out_ptrs,
            params.a_scales_ptrs,
            params.b_scales_ptrs,
            intemediate_q,
            w2_q,
            a2_scale,
            w2_scale,
            params.ab_strides_2,
            params.ab_strides_2,
            params.c_strides_2,
            a_sf_layout,
            w_sf_layout,
            params.problem_sizes2,
            params.expert_offsets[:-1],
            params.workspace,
        )

    return c2


def cutlass_moe_fp4(
    a: torch.Tensor,
    rep_aux: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_ids: torch.Tensor,
    params: CutlassMoEParams,
):
    """
    MoE implementation for FP4 Inputs

    # Gemm 1
    a: Preprocessed FP4 input tensor: [m * topk, k // 2] (uint8, FP4 quantized and reordered)
    w1(gate up) (not an argument to cutlass_moe_fp4): [e, 2 * n, k]
    w1_fp4: [e, 2 * n, k // 2], dtype: torch.uint8 (stacked fp4: E2M1)
    (Note: `n` is the up projection output dim, `k` is the input dim in
    full precision)
    w1_blockscale: [e, 2 * n, k // block_size] (float8_e4m3)
                (Block size = 16 for NVFP4)

    # Gemm 2
    a2_gscale: Activation scale per expert: [e]
    w2(down projection) (not an argument to cutlass_moe_fp4): [e, k, n]
    w2_fp4: [e, k, n // 2], dtype: torch.uint8 (stacked E2M1)
    w2_blockscale: [e, k, n // block_size], dtype: float8_e4m3

    Strides for activations, weights and output in logical number of elements.
    The activations & output stride is the number of elements to the next row.
    The weights stride is the number of elements to the next row per expert.
    For example, if the weight is [e, n, k], then the b_stride is a tensor of
    shape [e] with each element being k. Similarly for activations, if the
    shape is [m, k], then the a_stride has shape [e] with each value k.
    Similarly for output, if the output is [m, n], then the c_stride is a
    tensor of shape [e] with each element being k.

    Note: cutlass_fp4_group_mm is designed to accept the strides of
    activations and weights to be the same, so it is passed in as a single
    tensor.
    ab_strides_13: [e] dtype: int64 [Gemm 1: Activation / Weight strides]
    ab_strides_2: [e] dtype: int64 [Gemm 2: Activation / Weight strides]
    c_strides_13: [e] dtype: int64 [Gemm 1: Output Strides]
    c_strides_2: [e] dtype: int64 [Gemm 1: Output Strides]

    topk_weights: [m, topk] dtype: float8
    topk_ids: [m, topk] dtype: float8

    m, n, k: Unquantized weight shapes, dtype: int
    e: number of experts for the current rank, dtype: int
    assumes that topk < k < n to satisfy - up/down projection expectations.
    """

    rep_a_fp4 = a
    rep_a_blockscale = rep_aux

    out_dtype = torch.bfloat16  # Default output dtype for FP4 operations
    num_topk = topk_ids.shape[1]
    device = a.device
    m_a = a.size(0) // num_topk  # Original number of tokens
    c1 = cutlass_fp4_group_mm(
        rep_a_fp4,
        w1_fp4,
        rep_a_blockscale,
        w1_blockscale,
        w1_alphas,
        out_dtype,
        device,
        params.to_gemm1_args(),
    )
    del rep_a_fp4, rep_a_blockscale

    # hidden size dimension is split to one halfpytho sized tensor.
    intermediate = torch.empty(
        (m_a * num_topk, w1_fp4.shape[1] // 2), device=device, dtype=out_dtype
    )
    silu_and_mul(c1, intermediate)

    int_fp4, int_blockscale = scaled_fp4_experts_quant(
        intermediate,
        a2_gscale,
        params.expert_offsets,
        params.blockscale_offsets,
        num_topk,
    )
    c2 = cutlass_fp4_group_mm(
        int_fp4,
        w2_fp4,
        int_blockscale,
        w2_blockscale,
        w2_alphas,
        out_dtype,
        device,
        params.to_gemm2_args(),
    )
    del int_fp4, int_blockscale
    return c2
