from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers import zero_copy_context
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import moe_sum_reduce

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm


@triton.jit
def _tl_tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _situ_and_mul_kernel(
    x_ptr,  # [M, 2N] gate;up halves (non-interleaved)
    out_ptr,  # [M, N]
    N,
    situ_beta,
    linear_beta,
    stride_xm,
    stride_om,
    BLOCK_N: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    base = x_ptr + pid_m * stride_xm
    gate = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + N + offs, mask=mask, other=0.0).to(tl.float32)
    gate = situ_beta * _tl_tanh(gate / situ_beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * _tl_tanh(up / linear_beta)
    out = gate * up
    tl.store(
        out_ptr + pid_m * stride_om + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def situ_and_mul(
    output: torch.Tensor,
    x: torch.Tensor,
    situ_beta: float,
    linear_beta: Optional[float],
) -> None:
    """SiTU gated activation (Kimi K3), fused into one elementwise kernel:
    out = situ_beta*tanh(gate/situ_beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
    where x = [gate; up] halves along the last dim.
    """
    M, N2 = x.shape
    N = N2 // 2
    assert output.shape == (M, N)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _situ_and_mul_kernel[grid](
        x,
        output,
        N,
        float(situ_beta),
        float(linear_beta) if linear_beta is not None else 0.0,
        x.stride(0),
        output.stride(0),
        BLOCK_N=BLOCK_N,
        HAS_LINEAR_BETA=linear_beta is not None,
    )


def get_scalar_type(
    num_bits: int,
    has_zp: bool,
    scales: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
):
    from sgl_kernel.scalar_type import scalar_types

    if (
        not has_zp
        and num_bits == 4
        and scales is not None
        and (scales.dtype == torch.float8_e8m0fnu or global_scale is not None)
    ):
        return scalar_types.float4_e2m1f
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def swiglu_limit_func(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    swiglu_limit: float = 0.0,
) -> None:
    d = input.shape[1] // 2
    gate = input[:, :d]
    up = input[:, d:]

    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)

    output.copy_(F.silu(gate) * up)


def swiglu_gpt_oss_sigmoid_alpha_contiguous(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    gemm1_alpha: float,
    gemm1_limit: float,
) -> None:
    d = input.shape[1] // 2
    gate = input[:, :d].clamp(max=gemm1_limit)
    up = input[:, d:].clamp(min=-gemm1_limit, max=gemm1_limit)
    output.copy_(gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1))


@register_custom_op(out_shape="hidden_states")
def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    w1_global_scale: Optional[torch.Tensor] = None,
    w2_global_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
    clamp_limit: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    activation: str = "silu",
    is_gated: bool = True,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (int): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (num_bits // 2), (
        "Hidden size mismatch w2"
    )
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    is_mxfp4_marlin = (
        num_bits == 4
        and w1_zeros is None
        and w2_zeros is None
        and w1_scale.dtype == torch.float8_e8m0fnu
        and w2_scale.dtype == torch.float8_e8m0fnu
    )
    is_nvfp4_marlin = (
        num_bits == 4
        and w1_zeros is None
        and w2_zeros is None
        and w1_global_scale is not None
        and w2_global_scale is not None
    )
    if is_mxfp4_marlin:
        assert hidden_states.dtype == torch.bfloat16, (
            "MXFP4 Marlin with E8M0 scales is only instantiated for bfloat16 "
            f"activations, got {hidden_states.dtype}"
        )
    elif not is_nvfp4_marlin:
        assert hidden_states.dtype == w1_scale.dtype, (
            f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w1_scale.dtype ({w1_scale.dtype})"
        )
        assert hidden_states.dtype == w2_scale.dtype, (
            f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w2_scale.dtype ({w2_scale.dtype})"
        )
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]
    gemm1_n = 2 * N if is_gated else N

    # M block size selection logic
    # TODO: tune this further for specific models
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if global_num_experts == -1:
        global_num_experts = E
    if (
        M == 1
        and topk <= 32
        and expert_map is None
        # The JIT kernel is int32-only; torch-native topk emits int64 -- let
        # that (test-only) shape take the generic path instead of casting.
        and topk_ids.dtype == torch.int32
    ):
        # Single-token decode: top-k ids are distinct, so alignment is a
        # single-warp sort instead of the align + count_and_sort kernel pair.
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_single_token(
            topk_ids, block_size_m
        )
    else:
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, global_num_experts
        )

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * (
            sorted_token_ids.size(0) // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    scalar_type1 = get_scalar_type(
        num_bits, w1_zeros is not None, w1_scale, w1_global_scale
    )
    scalar_type2 = get_scalar_type(
        num_bits, w2_zeros is not None, w2_scale, w2_global_scale
    )

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # Marlin skips masked expert rows, so their shared cache must start at zero.
    intermediate_cache13 = torch.zeros(
        (M * topk_ids.shape[1] * max(gemm1_n, K),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[: M * topk_ids.shape[1] * gemm1_n]
    intermediate_cache1 = intermediate_cache1.view(-1, gemm1_n)
    intermediate_cache3 = intermediate_cache13[: M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    ) and (not is_mxfp4_marlin)

    intermediate_cache1 = moe_wna16_marlin_gemm(
        hidden_states,
        intermediate_cache1,
        w1,
        w1_bias,
        w1_scale,
        w1_global_scale,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=expert_map is not None,
        b_q_type=scalar_type1,
        size_m=M,
        size_n=gemm1_n,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    if activation == "silu" and is_gated and gemm1_alpha is not None:
        if clamp_limit is None:
            raise ValueError("GPT-OSS Marlin activation requires clamp_limit.")
        swiglu_gpt_oss_sigmoid_alpha_contiguous(
            intermediate_cache2,
            intermediate_cache1.view(-1, gemm1_n),
            gemm1_alpha,
            clamp_limit,
        )
    elif activation == "silu" and is_gated and clamp_limit is not None:
        swiglu_limit_func(
            intermediate_cache2,
            intermediate_cache1.view(-1, gemm1_n),
            clamp_limit,
        )
    elif activation == "silu" and is_gated:
        silu_and_mul(intermediate_cache1.view(-1, gemm1_n), intermediate_cache2)
    elif activation == "situ" and is_gated:
        situ_and_mul(
            intermediate_cache2,
            intermediate_cache1.view(-1, gemm1_n),
            situ_beta=gemm1_alpha if gemm1_alpha is not None else 4.0,
            linear_beta=clamp_limit,
        )
    elif activation == "silu" and not is_gated:
        intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
    elif activation == "relu2" and not is_gated:
        intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
    else:
        raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = moe_wna16_marlin_gemm(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        w2_bias,
        w2_scale,
        w2_global_scale,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type=scalar_type2,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    ).view(-1, topk, K)

    output = zero_copy_context.get_moe_output(hidden_states)
    if output is None:
        output = hidden_states if inplace else torch.empty_like(hidden_states)

    if is_mxfp4_marlin:
        # Top-k weights (incl. routed scaling) are already applied above via
        # mul_topk_weights, so this is a plain sum over the topk dim. The JIT
        # vectorized pass (~1.5us at decode shapes) beats sgl_kernel's
        # moe_sum_reduce_kernel_general (~5.7us) and the generic at::native
        # reduce_kernel torch.sum dispatches to (~6.7us).
        if (
            intermediate_cache3.dtype == torch.bfloat16
            and intermediate_cache3.is_contiguous()
            and output.is_contiguous()
            and intermediate_cache3.shape[-1] % 8 == 0
        ):
            from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum

            moe_topk_sum(intermediate_cache3, output)
        else:
            moe_sum_reduce(intermediate_cache3, output, 1.0)
        return output
    else:
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        moe_sum_reduce(
            intermediate_cache3,
            output,
            routed_scaling_factor,
        )
        return output
