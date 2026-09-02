from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel import silu_and_mul

from sglang.kernels.ops.kimi_k3 import situ_and_mul
from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather, ep_scatter_sail
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    _apply_swiglu_limit,
)
from sglang.srt.utils import get_device_sm
from sglang.srt.utils.custom_op import register_custom_op

if get_device_sm() >= 89:
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
        sglang_per_token_quant_fp8,
    )
    from sglang.srt.layers.quantization.ppu_mxfp4_utils import downcast_to_mxfp4
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_quant_mxfp4,
    )

from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)

# Add for nvtx profiling
SGLANG_PROFILE_NVTX = envs.SGLANG_PROFILE_NVTX.get()
SGLANG_PROFILE_NVTX_PRINT_TOPID = envs.SGLANG_PROFILE_NVTX_PRINT_TOPID.get()
if SGLANG_PROFILE_NVTX:
    try:
        from torch.cuda.nvtx import range_pop as th_nvtx_range_pop
        from torch.cuda.nvtx import range_push as th_nvtx_range_push
    except ImportError as e:
        from sglang.srt.utils import logger

        logger.warning(f"Import NVTX Error! {e}")
        SGLANG_PROFILE_NVTX = False
        SGLANG_PROFILE_NVTX_PRINT_TOPID = False


@torch.compile
def _swiglu_silu_clamp_mul(x, gemm1_limit):
    gate, up = x.chunk(2, dim=-1)
    gate = F.silu(gate)
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * up


def grouped_gemm_nt_i8i8bf16_nopad_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    return


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_i8i8bf16_nopad_fake)
def grouped_gemm_nt_i8i8bf16_nopad(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    deep_gemm_wrapper.grouped_gemm_nt_i8i8bf16_nopad(
        (A, As), (B, Bs), C, m_indices, m_rows
    )


def grouped_gemm_nt_f8f8bf16_nopad_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    return


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_f8f8bf16_nopad_fake)
def grouped_gemm_nt_f8f8bf16_nopad(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_nopad(
        (A, As), (B, Bs), C, m_indices, m_rows
    )


def grouped_gemm_nt_bf16bf16bf16_nopad_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    return


@register_custom_op(
    mutates_args=["C"], fake_impl=grouped_gemm_nt_bf16bf16bf16_nopad_fake
)
def grouped_gemm_nt_bf16bf16bf16_nopad(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    deep_gemm_wrapper.grouped_gemm_nt_bf16_nopad(A, B, C, m_indices, m_rows)


def grouped_gemm_nt_f4f4bf16_nopad_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    return


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_f4f4bf16_nopad_fake)
def grouped_gemm_nt_f4f4bf16_nopad(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_nopad(
        (A, As), (B, Bs), bias, C, m_indices, m_rows
    )


@register_custom_op(mutates_args=["C"])
def grouped_gemm_nt_bf16i4bf16_nopad(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    deep_gemm_wrapper.grouped_gemm_nt_bf16i4bf16_nopad(A, (B, Bs), C, m_indices, m_rows)


def grouped_gemm_nt_bf16i4bf16_fused_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = A.shape[0] * topk_ids.shape[1]
    device = topk_ids.device
    return (
        torch.empty(B.shape[0], dtype=torch.int32, device=device),
        torch.empty(numel, dtype=torch.int32, device=device),
        torch.empty(numel, dtype=torch.int32, device=device),
    )


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_bf16i4bf16_fused_fake)
def grouped_gemm_nt_bf16i4bf16_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        configs_list,
        m_rows,
        expert_ids_and_cumsum,
        sorted_token_ids,
        aligned_num_m_blocks,
        inv_perm,
        expert_ids,
    ) = deep_gemm_wrapper.moe_align_block_size(
        A, B, topk_ids, perchannel_quant=False, configs=configs
    )

    deep_gemm_wrapper.grouped_gemm_nt_bf16i4bf16_fused(
        A,
        (B, Bs),
        C,
        m_rows,
        expert_ids_and_cumsum,
        sorted_token_ids,
        aligned_num_m_blocks,
        configs_list,
    )
    return m_rows, inv_perm, expert_ids


# ---- fused variants: register_custom_op for CUDA graph / torch.compile ----


def grouped_gemm_nt_f8f8bf16_fused_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    is_perchannel: bool,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = A.shape[0] * topk_ids.shape[1]
    device = topk_ids.device
    return (
        torch.empty(B.shape[0], dtype=torch.int32, device=device),
        torch.empty(numel, dtype=torch.int32, device=device),
        torch.empty(numel, dtype=torch.int32, device=device),
    )


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_f8f8bf16_fused_fake)
def grouped_gemm_nt_f8f8bf16_fused(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    is_perchannel: bool,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        configs_list,
        m_rows,
        expert_ids_and_cumsum,
        sorted_token_ids,
        aligned_num_m_blocks,
        inv_perm,
        expert_ids,
    ) = deep_gemm_wrapper.moe_align_block_size(
        A, B, topk_ids, perchannel_quant=is_perchannel, configs=configs
    )

    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_fused(
        (A, As),
        (B, Bs),
        C,
        m_rows,
        expert_ids_and_cumsum,
        sorted_token_ids,
        aligned_num_m_blocks,
        configs_list,
    )

    return m_rows, inv_perm, expert_ids


def grouped_gemm_nt_i8i8bf16_fused_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = A.shape[0] * topk_ids.shape[1]
    device = topk_ids.device
    return (
        torch.empty(B.shape[0], dtype=torch.int32, device=device),
        torch.empty(numel, dtype=torch.int32, device=device),
        torch.empty(numel, dtype=torch.int32, device=device),
    )


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_i8i8bf16_fused_fake)
def grouped_gemm_nt_i8i8bf16_fused(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        configs_list,
        m_rows,
        expert_ids_and_cumsum,
        sorted_token_ids,
        aligned_num_m_blocks,
        inv_perm,
        expert_ids,
    ) = deep_gemm_wrapper.moe_align_block_size(
        A, B, topk_ids, perchannel_quant=True, configs=configs
    )

    deep_gemm_wrapper.grouped_gemm_nt_i8i8bf16_fused(
        (A, As),
        (B, Bs),
        C,
        m_rows,
        expert_ids_and_cumsum,
        sorted_token_ids,
        aligned_num_m_blocks,
        configs_list,
    )

    return m_rows, inv_perm, expert_ids


def grouped_gemm_nt_bf16bf16bf16_fused_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_groups = B.shape[0]
    numel = A.shape[0] * topk_ids.shape[1]
    device = topk_ids.device
    return (
        torch.empty(num_groups, dtype=torch.int32, device=device),
        torch.empty(numel, dtype=torch.int32, device=device),
        torch.empty(numel, dtype=torch.int32, device=device),
    )


@register_custom_op(
    mutates_args=["C"], fake_impl=grouped_gemm_nt_bf16bf16bf16_fused_fake
)
def grouped_gemm_nt_bf16bf16bf16_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        configs_list,
        m_rows,
        expert_ids_and_cumsum,
        sorted_token_ids,
        aligned_num_m_blocks,
        inv_perm,
        expert_ids,
    ) = deep_gemm_wrapper.moe_align_block_size(A, B, topk_ids, configs=configs)

    deep_gemm_wrapper.grouped_gemm_nt_bf16bf16bf16_fused(
        A,
        B,
        C,
        m_rows,
        expert_ids_and_cumsum,
        sorted_token_ids,
        aligned_num_m_blocks,
        configs_list,
    )

    return m_rows, inv_perm, expert_ids


def grouped_gemm_nt_f4f4bf16_fused_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_groups = B.shape[0]
    numel = A.shape[0] * topk_ids.shape[1]
    device = topk_ids.device
    return (
        torch.empty(num_groups, dtype=torch.int32, device=device),
        torch.empty(numel, dtype=torch.int32, device=device),
        torch.empty(numel, dtype=torch.int32, device=device),
    )


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_f4f4bf16_fused_fake)
def grouped_gemm_nt_f4f4bf16_fused(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        configs_list,
        m_rows,
        expert_ids_and_cumsum,
        sorted_token_ids,
        aligned_num_m_blocks,
        inv_perm,
        expert_ids,
    ) = deep_gemm_wrapper.moe_align_block_size(
        A, B, topk_ids, perchannel_quant=False, configs=configs
    )

    deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_fused(
        (A, As),
        (B, Bs),
        C,
        m_rows,
        expert_ids_and_cumsum,
        sorted_token_ids,
        aligned_num_m_blocks,
        configs_list,
    )

    return m_rows, inv_perm, expert_ids


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def compute_aligned_M(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
):
    # expert_num_tokens information is not available on the cpu.
    # compute the max required size.
    M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum


@triton.jit
def round_up_triton(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    expert_valid_num_tokens_ptr,
    num_experts,
    topk_numel,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    curr_expert = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    topk_ids_ptrs = topk_ids_ptr + offsets

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        mask = offsets < (topk_numel - x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptrs, mask=mask, other=-1)

        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert
        topk_ids_ptrs += BLOCK_SIZE

    if curr_expert < num_experts:
        actual_count = tl.sum(acc)
        tl.store(
            expert_num_tokens_ptr + curr_expert, round_up_triton(actual_count, BLOCK_E)
        )
        tl.store(expert_valid_num_tokens_ptr + curr_expert, actual_count)


def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, block_align: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Count the number to tokens assigned to each expert.

    Parameters:
    - topk_ids (torch.Tensor): Tensor mapping each token to its
    list of experts.
    - num_local_experts (int): Number of experts in this rank.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
    from the global expert space to the local expert space of the expert
    parallel shard.

    Returns:
    A tuple of two tensors of size num_local_experts:
    - expert_num_tokens[i] holds the block-aligned (padded) number of tokens
      assigned to the ith expert.
    - expert_valid_num_tokens[i] holds the actual (un-padded) number of tokens
      assigned to the ith expert.
    """
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )
    expert_valid_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )

    grid = num_local_experts
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    _count_expert_num_tokens[(grid,)](
        topk_ids,
        expert_num_tokens,
        expert_valid_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_E=block_align,
    )

    return expert_num_tokens, expert_valid_num_tokens


def deepgemm_moe_permute(
    aq: torch.Tensor,
    aq_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    aq_out: Optional[torch.Tensor] = None,
    block_align: int = 1,
    block_k: int = 1,
    is_block_wise=False,
):
    assert aq.ndim == 2
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    H = aq.size(1)
    device = aq.device

    M_sum = compute_aligned_M(
        M=topk_ids.size(0),
        num_topk=topk_ids.size(1),
        local_num_experts=local_num_experts,
        alignment=block_align,
    )

    expert_start_loc = torch.empty(
        (local_num_experts), device=device, dtype=torch.int32
    )

    assert aq_out is None or aq_out.shape == (M_sum, H)
    if aq_out is None:
        aq_out = torch.empty((M_sum, H), device=device, dtype=aq.dtype)

    # mxfp4: torch.uint16, others: torch.float32
    if aq_scale is not None and aq_scale.dtype == torch.uint16:
        # Fused uint16 scale: physical layout [S//2, M] transposed to [M, S//2] with stride(0)=1.
        # Output must preserve this layout: create [S//2, M_sum] contiguous then .t()
        scale_hidden = aq_scale.shape[1]
        aq_scale_out = torch.empty(
            (scale_hidden, M_sum), device=device, dtype=torch.uint16
        ).t()  # [M_sum, S//2] with stride(0)=1, stride(1)=M_sum
    else:
        scale_hidden = (H + block_k - 1) // block_k
        aq_scale_out = torch.empty(
            (M_sum, scale_hidden), device=device, dtype=torch.float32
        )

    expert_ids = torch.empty((M_sum), device=device, dtype=torch.int32)
    inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)

    expert_num_tokens, expert_valid_num_tokens = count_expert_num_tokens(
        topk_ids, local_num_experts, block_align
    )

    ep_scatter_sail(
        recv_x=aq,
        recv_x_scale=aq_scale,
        recv_topk=topk_ids.to(torch.int32),
        num_recv_tokens_per_expert=expert_num_tokens,
        expert_start_loc=expert_start_loc,
        output_tensor=aq_out,
        output_tensor_scale=aq_scale_out,
        m_indices=expert_ids,
        output_index=inv_perm,
        BLOCK_E=block_align,
        BLOCK_D=block_k,
        is_block_wise=is_block_wise,
        num_valid_tokens_per_expert=expert_valid_num_tokens,
    )
    return aq_out, aq_scale_out, expert_ids, inv_perm, expert_num_tokens


@torch.compile
def swiglu_with_alpha_and_limit(x, gemm1_alpha, gemm1_limit):
    # for gpt-oss models with deepgemm backend
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


@torch.compile
def swiglu_no_interleaved_with_alpha_and_limit(x, gemm1_alpha, gemm1_limit):
    # Same numerics as swiglu_with_alpha_and_limit, for w13 stored as two
    # contiguous [gate; up] halves instead of gpt-oss' row-interleaved pairs.
    # MiniMax-M3 builds its fused MoE with gate_up_interleaved=False.
    # Twin of triton_utils.fused_moe.swiglu_no_interleaved_with_alpha_and_limit.
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


def deep_moe_impl_fused(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: Optional[float] = None,
    per_channel_quant: bool = False,
    block_shape: Optional[List[int]] = None,
    use_fp8: bool = False,
    use_int8: bool = False,
    use_mxfp4: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a16: bool = False,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    swiglu_limit: Optional[float] = None,
    activation: str = "silu",
    gate_up_interleaved: bool = True,
    out_hidden_states: Optional[torch.Tensor] = None,
):
    block_align = 1

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    num_tokens, K = hidden_states.shape
    E, N, _ = w1.shape
    _, top_k = topk_ids.shape

    use_w4a16 = use_int4_w4a16 or use_mxfp4_w4a16
    use_w4a16_mma = use_mxfp4_w4a16 and w1.dtype == torch.uint8
    if use_w4a16 and not use_w4a16_mma:
        N = w2.shape[1] * 32

    if out_hidden_states is None:
        out_hidden_states = hidden_states
    num_tokens_padded = compute_aligned_M(num_tokens, top_k, E, block_align)

    if per_channel_quant:
        w1_scale = w1_scale.unsqueeze(-1) if w1_scale.ndim != w1.ndim else w1_scale
        w2_scale = w2_scale.unsqueeze(-1) if w2_scale.ndim != w2.ndim else w2_scale

    # Hybrid fused path: requires DeepGemm commit 53ea8ff (refactor fused MoE API)
    use_fused_path = envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.get()

    if use_int8:
        assert (
            per_channel_quant and block_shape is None
        ), "int8 quantization only supports per-channel quant."
        hidden_states, hidden_states_scale = per_token_quant_int8(hidden_states)
        # channelwise
        block_k = K
    elif use_fp8:
        # TODO: add fp8 channel-wise support
        if per_channel_quant:
            block_k = K
            hidden_states, hidden_states_scale = sglang_per_token_quant_fp8(
                hidden_states
            )
        else:
            assert (
                len(block_shape) == 2
            ), f"Block-shape:{block_shape} mismatch for FP8 block-wise quant!"
            block_n, block_k = block_shape[0], block_shape[1]
            hidden_states, hidden_states_scale = sglang_per_token_group_quant_fp8(
                hidden_states, block_k, column_major_scales=False
            )
    elif use_mxfp4:
        # hidden_states: torch.uint8, hidden_states_scale: torch.uint16
        hidden_states, hidden_states_scale = downcast_to_mxfp4(
            hidden_states, axis=1, enable_fused_moe=use_fused_path
        )
        block_k = block_shape[1]
    else:
        hidden_states_scale = None
        block_k = K

    out1 = torch.empty(
        (num_tokens_padded, N), device=hidden_states.device, dtype=torch.bfloat16
    )

    if use_fused_path:
        # ── fused path: moe_align + GEMM1 fused (gather read) ──

        if use_int8:
            m_rows, inv_perm, expert_ids = grouped_gemm_nt_i8i8bf16_fused(
                hidden_states, hidden_states_scale, w1, w1_scale, out1, topk_ids
            )
        elif use_fp8:
            m_rows, inv_perm, expert_ids = grouped_gemm_nt_f8f8bf16_fused(
                hidden_states,
                hidden_states_scale,
                w1,
                w1_scale,
                out1,
                topk_ids,
                per_channel_quant,
            )
        elif use_mxfp4:
            m_rows, inv_perm, expert_ids = grouped_gemm_nt_f4f4bf16_fused(
                hidden_states,
                hidden_states_scale,
                w1,
                w1_scale,
                out1,
                topk_ids,
            )
        elif use_w4a16:
            m_rows, inv_perm, expert_ids = grouped_gemm_nt_bf16i4bf16_fused(
                hidden_states,
                w1,
                w1_scale,
                out1,
                topk_ids,
            )
        else:
            m_rows, inv_perm, expert_ids = grouped_gemm_nt_bf16bf16bf16_fused(
                hidden_states, w1, out1, topk_ids
            )

        num_recv_tokens_per_expert = m_rows
        inv_perm = inv_perm.view(num_tokens, top_k)

    else:
        # ── nopad path: deepgemm_moe_permute + GEMM1 nopad ──
        a, a_scale, expert_ids, inv_perm, num_recv_tokens_per_expert = (
            deepgemm_moe_permute(
                aq=hidden_states,
                aq_scale=hidden_states_scale,
                topk_ids=topk_ids,
                local_num_experts=E,
                block_align=block_align,
                block_k=block_k,
                is_block_wise=(block_shape is not None),
            )
        )
        assert a.size(0) == num_tokens_padded

    _nvtx_moe_pushed = False
    if SGLANG_PROFILE_NVTX:
        _moe_tag = (
            f"M_{topk_ids.shape[0]}_E_{w1.shape[0]}_"
            f"H_{w1.shape[2]}_In_{w1.shape[1]}_topk_{top_k}"
        )
        if torch.cuda.is_current_stream_capturing():
            th_nvtx_range_push(f"D_MoE,{_moe_tag}")
        else:
            th_nvtx_range_push(f"P_MoE,{_moe_tag}")
        _nvtx_moe_pushed = True

    # print topid only if SGLANG_PROFILE_NVTX_PRINT_TOPID_TOPID set to avoid impact perf compare
    nvtx_pushed = False
    if SGLANG_PROFILE_NVTX and SGLANG_PROFILE_NVTX_PRINT_TOPID:
        if not torch.cuda.is_current_stream_capturing():
            num_activated_experts = (num_recv_tokens_per_expert > 0).sum().item()
            token_counts_list = num_recv_tokens_per_expert.flatten().cpu().tolist()
            # shape param
            M = hidden_states.shape[0]
            E = w1.shape[0]
            H = w1.shape[2]
            In = w1.shape[1]
            K = topk_ids.shape[1]
            U = num_activated_experts

            nvtx_tag = (
                f"MoE,"
                f"M_{M}_E_{E}_H_{H}_In_{In}_"
                f"topk_{K}_"
                f"topkids{token_counts_list}_"
                f"unique_{U}"
            )
            nvtx_pushed = True
            th_nvtx_range_push(nvtx_tag)

    if not use_fused_path:
        if use_int8:
            grouped_gemm_nt_i8i8bf16_nopad(
                a, a_scale, w1, w1_scale, out1, expert_ids, num_recv_tokens_per_expert
            )
        elif use_fp8:
            grouped_gemm_nt_f8f8bf16_nopad(
                a, a_scale, w1, w1_scale, out1, expert_ids, num_recv_tokens_per_expert
            )
        elif use_mxfp4:
            grouped_gemm_nt_f4f4bf16_nopad(
                a,
                a_scale,
                w1,
                w1_scale,
                b1,
                out1,
                expert_ids,
                num_recv_tokens_per_expert,
            )
        elif use_w4a16:
            grouped_gemm_nt_bf16i4bf16_nopad(
                a, w1, w1_scale, out1, expert_ids, num_recv_tokens_per_expert
            )
        else:
            grouped_gemm_nt_bf16bf16bf16_nopad(
                a, w1, out1, expert_ids, num_recv_tokens_per_expert
            )

    if use_w4a16 and not use_fused_path:
        del a

    if (
        activation != "situ"
        and gemm1_alpha is None
        and gemm1_limit is None
        and use_mxfp4
    ):
        a, a_scale = silu_and_mul_post_quant_mxfp4(out1, swiglu_limit=swiglu_limit)
    else:
        if activation == "situ":
            out2 = situ_and_mul(
                out1,
                None,
                gemm1_alpha if gemm1_alpha is not None else 4.0,
                gemm1_limit,
            )
        elif gemm1_alpha is not None:
            swiglu_alpha_limit = (
                swiglu_with_alpha_and_limit
                if gate_up_interleaved
                else swiglu_no_interleaved_with_alpha_and_limit
            )
            out2 = swiglu_alpha_limit(
                out1,
                gemm1_alpha,
                gemm1_limit,
            )
        elif gemm1_limit is not None:
            out2 = _swiglu_silu_clamp_mul(out1, gemm1_limit)
        else:
            if swiglu_limit is not None:
                out1 = _apply_swiglu_limit(out1, swiglu_limit=swiglu_limit)
            out2 = torch.empty(
                (num_tokens_padded, N // 2),
                device=hidden_states.device,
                dtype=torch.bfloat16,
            )
            silu_and_mul(out1, out2)

        if use_int8:
            a, a_scale = per_token_quant_int8(out2)
        elif use_fp8:
            if per_channel_quant:
                a, a_scale = sglang_per_token_quant_fp8(out2)
            else:
                a, a_scale = sglang_per_token_group_quant_fp8(
                    out2, block_k, column_major_scales=False
                )
        elif use_mxfp4:
            a, a_scale = downcast_to_mxfp4(out2, axis=1)
        else:
            a = out2
            a_scale = None

    if use_w4a16:
        del out1

    out3 = torch.empty(
        (num_tokens_padded, K), device=hidden_states.device, dtype=torch.bfloat16
    )

    if use_int8:
        grouped_gemm_nt_i8i8bf16_nopad(
            a, a_scale, w2, w2_scale, out3, expert_ids, num_recv_tokens_per_expert
        )
    elif use_fp8:
        grouped_gemm_nt_f8f8bf16_nopad(
            a, a_scale, w2, w2_scale, out3, expert_ids, num_recv_tokens_per_expert
        )
    elif use_mxfp4:
        grouped_gemm_nt_f4f4bf16_nopad(
            a, a_scale, w2, w2_scale, b2, out3, expert_ids, num_recv_tokens_per_expert
        )
    elif use_w4a16:
        grouped_gemm_nt_bf16i4bf16_nopad(
            a, w2, w2_scale, out3, expert_ids, num_recv_tokens_per_expert
        )
    else:
        grouped_gemm_nt_bf16bf16bf16_nopad(
            a, w2, out3, expert_ids, num_recv_tokens_per_expert
        )

    if nvtx_pushed:
        th_nvtx_range_pop()

    ep_gather(
        input_tensor=out3,
        recv_topk_ids=topk_ids,
        recv_topk_weight=topk_weights,
        input_index=inv_perm,
        output_tensor=out_hidden_states,
        routed_scaling_factor=routed_scaling_factor,
    )

    if _nvtx_moe_pushed:
        th_nvtx_range_pop()


@register_fused_func("none", "deep_gemm")
def fused_experts_none_to_deep_gemm(
    dispatch_output: StandardDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    hidden_states = dispatch_output.hidden_states
    w1 = quant_info.w13_weight
    w2 = quant_info.w2_weight
    topk_output = dispatch_output.topk_output
    moe_runner_config = runner_config
    b1 = quant_info.b13
    b2 = quant_info.b2
    use_fp8 = quant_info.use_fp8
    use_int8 = quant_info.use_int8
    use_mxfp4 = quant_info.use_mxfp4
    use_int4_w4a16 = quant_info.use_int4_w4a16
    use_mxfp4_w4a16 = quant_info.use_mxfp4_w4a16
    per_channel_quant = quant_info.per_channel_quant
    w1_scale = quant_info.w13_scale
    w2_scale = quant_info.w2_scale
    block_shape = quant_info.block_shape

    topk_weights, topk_ids, _ = topk_output
    routed_scaling_factor = moe_runner_config.routed_scaling_factor
    output = hidden_states if runner_config.inplace else torch.empty_like(hidden_states)

    deep_moe_impl_fused(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        routed_scaling_factor=routed_scaling_factor,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        use_fp8=use_fp8,
        use_int8=use_int8,
        use_mxfp4=use_mxfp4,
        use_int4_w4a16=use_int4_w4a16,
        use_mxfp4_w4a16=use_mxfp4_w4a16,
        b1=b1,
        b2=b2,
        gemm1_alpha=moe_runner_config.gemm1_alpha,
        gemm1_limit=moe_runner_config.gemm1_clamp_limit,
        swiglu_limit=moe_runner_config.swiglu_limit,
        activation=moe_runner_config.activation,
        gate_up_interleaved=moe_runner_config.gate_up_interleaved,
        out_hidden_states=output,
    )

    return StandardCombineInput(hidden_states=output)
