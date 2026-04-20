from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers import asym_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import (
    ceil_div,
    dispose_tensor,
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_npu,
)
from sglang.srt.utils.offloader import get_offloader

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if not (_is_npu or _is_hip) and _is_cuda:
    from sgl_kernel import silu_and_mul


_MASKED_GEMM_FAST_ACT = get_bool_env_var("SGLANG_MASKED_GEMM_FAST_ACT")


# TODO(kaixih@nvidia): ideally we should merge this logic into
# `fill_gateup_input_triton_kernel` to directly generate e8m0 scale.
@torch.compile(disable=_is_hip or _is_npu)
def _cast_to_e8m0_with_rounding_up(x: torch.Tensor) -> torch.Tensor:
    temp = x.to(torch.float32).view(torch.int32)
    exp = torch.bitwise_right_shift(temp, 23)
    mant = torch.bitwise_and(temp, 0x7FFFFF)
    is_ru = torch.logical_and(
        torch.logical_and((mant > 0), (exp != 0xFE)),
        ~torch.logical_and((exp == 0), (mant <= 0x400000)),
    )
    exp = torch.where(is_ru, exp + 1, exp)
    new_x = exp.to(torch.uint8).view(torch.int)
    return new_x.transpose(1, 2).contiguous().transpose(1, 2)


def copy_list_to_gpu_no_ce(arr: List[int]):
    from sgl_kernel.elementwise import copy_to_gpu_no_ce

    tensor_cpu = torch.tensor(arr, dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    return tensor_gpu


# old version : CPU based, not working with cuda graph
# def build_offsets_experts_from_masked_m(
#     masked_m: torch.Tensor, num_groups: int, max_m: int, block_m: int = 128
# ):
#     """
#     Build offsets and experts for sparse m-grouped masked GEMM.

#     Each group gets a fixed allocation of max_m rows in the A matrix.
#     Only groups with masked_m[g] > 0 are included in the output mapping.
#     Each active group produces an (start, end) offset pair where:
#         start = g * max_m
#         end   = start + ceil(masked_m[g] / block_m) * block_m

#     Args:
#         masked_m:   (num_groups,) tensor of actual token counts per group
#         num_groups: number of expert groups
#         max_m:      allocated row stride per group (e.g. m_max)
#         block_m:    block-M alignment for padding (default 128)

#     Returns:
#         offsets:   flat int32 tensor [start_0, end_0, start_1, end_1, ...]
#         experts:   int32 tensor of active expert IDs + terminator (-1)
#         list_size: len(experts), i.e. num_active_experts + 1
#     """
#     offsets = []
#     experts = []

#     for g in range(num_groups):
#         v = masked_m[g].item()
#         if v > 0:
#             start = g * max_m
#             end = start + ((v + block_m - 1) // block_m) * block_m
#             offsets.append(start)
#             offsets.append(end)
#             experts.append(g)

#     experts.append(-1)

#     return (
#         torch.tensor(offsets, dtype=torch.int32, device=masked_m.device),
#         torch.tensor(experts, dtype=torch.int32, device=masked_m.device),
#         len(experts),
#     )


def build_offsets_experts_from_masked_m(
    masked_m: torch.Tensor, num_groups: int, max_m: int, block_m: int = 128
):
    """CUDA-graph-safe builder preserving the old compact ABI.

    Active groups are packed contiguously to the front of the output buffers in
    ascending group-id order. The layout is:

        experts[0..num_active-1] = active group ids (ascending)
        experts[num_active]      = -1 (terminator)
        experts[num_active+1..]  = -1 (sentinel padding)
        offsets[0..2*num_active-1] = packed [start_g0, end_g0, start_g1, end_g1, ...]
        offsets[2*num_active..]  = 0, 0, 0, ... (sentinel pairs)
        list_size = num_active + 1 (1-element int32 tensor)

    Kernel launches grid Y = num_groups (CUDA-graph requirement). For
    blockIdx.y >= num_active the sentinel offsets pair (0, 0) yields
    m_start == m_end and the kernel early-exits.

    Packing is done with a cumsum-based scatter (no boolean indexing, no host
    copies of device scalars, no dynamic shapes), so this is capture-safe.
    """
    device = masked_m.device

    group_ids = torch.arange(num_groups, device=device, dtype=torch.int32)
    masked_m_i32 = masked_m.to(torch.int32)
    mask = masked_m_i32 > 0
    mask_int = mask.to(torch.int32)

    # Packed rank (0-indexed) for active groups: cumsum-1 in ascending group order.
    # The value for inactive groups is meaningless; we redirect those writes
    # to a discard slot at index num_groups (past the kernel's read range).
    rank = mask_int.cumsum(0).to(torch.int32) - 1
    discard_slot = torch.full_like(rank, num_groups)
    write_idx = torch.where(mask, rank, discard_slot).to(torch.int64)

    starts = group_ids * max_m
    ends = starts + ((masked_m_i32 + (block_m - 1)) // block_m) * block_m

    # Fixed-size buffers with one extra slot for inactive-write discard.
    # experts initial fill of -1 doubles as the post-active terminator + sentinel
    # padding; offsets initial fill of 0 makes unused pairs (0, 0) → sentinel.
    experts_fixed = torch.full((num_groups + 1,), -1, device=device, dtype=torch.int32)
    starts_buf = torch.zeros((num_groups + 1,), device=device, dtype=torch.int32)
    ends_buf = torch.zeros((num_groups + 1,), device=device, dtype=torch.int32)

    experts_fixed.scatter_(0, write_idx, group_ids)
    starts_buf.scatter_(0, write_idx, starts)
    ends_buf.scatter_(0, write_idx, ends)

    # Interleave the kernel-visible range [0, num_groups) into pair layout.
    offsets_fixed = torch.stack(
        [starts_buf[:num_groups], ends_buf[:num_groups]], dim=1
    ).flatten()

    list_size = (mask_int.sum() + 1).to(torch.int32).view(1)

    return offsets_fixed, experts_fixed, list_size

@dataclass
class AsymGemmRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    use_masked_gemm: bool
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None
    m_indices: Optional[torch.Tensor] = None
    offsets: Optional[torch.Tensor] = None
    experts: Optional[torch.Tensor] = None
    list_size: Optional[torch.Tensor] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASYM_GEMM


@dataclass
class AsymGemmRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASYM_GEMM


@dataclass
class AsymGemmMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    use_fp8: bool
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None


class AsymGemmRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        assert self.config.activation == "silu"
        assert self.config.is_gated
        # Internal BF16 runner core for dtype-based dispatch
        from sglang.srt.layers.moe.moe_runner.asym_gemm_bf16 import (
            AsymGemmBf16RunnerCore,
        )

        self._bf16_core = AsymGemmBf16RunnerCore(config)

    def run(
        self,
        runner_input,
        quant_info,
        running_state: dict,
    ):
        from sglang.srt.layers.moe.moe_runner.asym_gemm_bf16 import (
            AsymGemmBf16MoeQuantInfo,
        )

        # Dtype-based dispatch: route to BF16 runner core if quant_info is BF16
        if isinstance(quant_info, AsymGemmBf16MoeQuantInfo):
            return self._bf16_core.run(runner_input, quant_info, running_state)

        if not runner_input.use_masked_gemm:
            hidden_states = self._run_contiguous_gemm(
                runner_input, quant_info, running_state
            )
        else:
            hidden_states = self._run_masked_gemm(
                runner_input, quant_info, running_state
            )
        return AsymGemmRunnerOutput(hidden_states=hidden_states)

    def _run_contiguous_gemm(
        self,
        runner_input: AsymGemmRunnerInput,
        quant_info: AsymGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.ep_moe.kernels import tma_align_input_scale
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_dtype = running_state["hidden_states_dtype"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]
        scale_block_size = 128

        w13_weight_fp8 = (
            quant_info.w13_weight,
            quant_info.w13_scale,
        )
        w2_weight_fp8 = (quant_info.w2_weight, quant_info.w2_scale)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if not asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0:
            hidden_states_scale = tma_align_input_scale(hidden_states_scale)

        asym_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (hidden_states, hidden_states_scale),
            w13_weight_fp8,
            gateup_output,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        down_input = torch.empty(
            (
                all_tokens,
                N // 2,
            ),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(gateup_output.view(-1, N), down_input)
        del gateup_output

        down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
            down_input,
            scale_block_size,
            column_major_scales=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
            scale_tma_aligned=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
            scale_ue8m0=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
        )
        del down_input

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if not asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0:
            down_input_scale = tma_align_input_scale(down_input_scale)

        asym_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )

        return down_output

    def _run_masked_gemm(
        self,
        runner_input: AsymGemmRunnerInput,
        quant_info: AsymGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.srt.layers import asym_gemm_wrapper
        from sglang.srt.layers.moe.ep_moe.kernels import (
            silu_and_mul_masked_post_quant_fwd,
        )
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_8bit,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0
        if asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0:
            if hidden_states_scale.dtype != torch.int:
                b, s_mn, s_k = hidden_states_scale.shape
                assert (
                    s_mn % 4 == 0 and s_k % 4 == 0
                ), f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"
                hidden_states_scale = _cast_to_e8m0_with_rounding_up(
                    hidden_states_scale
                )
        else:
            hidden_states_scale = asym_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                hidden_states_scale
            )

        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )

        asym_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
        )
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        # Act
        scale_block_size = 128
        if _MASKED_GEMM_FAST_ACT:
            down_input, down_input_scale = sglang_per_token_group_quant_8bit(
                x=gateup_output,
                dst_dtype=torch.float8_e4m3fn,
                group_size=scale_block_size,
                masked_m=masked_m,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
                fuse_silu_and_mul=True,
                enable_v2=True,
            )
        else:
            down_input = torch.empty(
                (
                    gateup_output.shape[0],
                    gateup_output.shape[1],
                    gateup_output.shape[2] // 2,
                ),
                device=hidden_states_device,
                dtype=torch.float8_e4m3fn,
            )
            down_input_scale = torch.empty(
                (
                    gateup_output.shape[0],
                    gateup_output.shape[1],
                    gateup_output.shape[2] // 2 // scale_block_size,
                ),
                device=hidden_states_device,
                dtype=torch.float32,
            )
            silu_and_mul_masked_post_quant_fwd(
                gateup_output,
                down_input,
                down_input_scale,
                scale_block_size,
                masked_m,
                scale_ue8m0=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
            )
        del gateup_output

        # GroupGemm-1
        n = w2_weight.shape[1]

        if not asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0:
            down_input_scale = asym_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                down_input_scale
            )

        down_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )

        down_gemm_overlap_args = running_state.get("down_gemm_overlap_args", None)
        if down_gemm_overlap_args is None:
            gemm_overlap_args_dict = {}
        else:
            down_gemm_overlap_args.start_event.record()
            max_block_n = 256
            gemm_overlap_args_dict = {
                "overlap_args": down_gemm_overlap_args,
                "max_block_n": max_block_n,
            }

        asym_gemm_return_value = asym_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (down_input, down_input_scale),
            (w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
            runner_input.offsets,
            runner_input.experts,
            runner_input.list_size,
            **gemm_overlap_args_dict,
        )
        meta_overlap_args = running_state.get("meta_overlap_args", None)
        if meta_overlap_args is not None:
            block_m, threshold = asym_gemm_return_value
            meta_overlap_args["block_m"] = block_m
            meta_overlap_args["threshold"] = threshold

        return down_output

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASYM_GEMM


@register_pre_permute("standard", "asym_gemm")
def pre_permute_standard_to_asym_gemm(
    dispatch_output: StandardDispatchOutput,
    quant_info,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    from sglang.srt.layers.moe.moe_runner.asym_gemm_bf16 import (
        AsymGemmBf16MoeQuantInfo,
    )

    # Dtype-based dispatch
    if isinstance(quant_info, AsymGemmBf16MoeQuantInfo):
        return _pre_permute_standard_to_asym_gemm_bf16(
            dispatch_output, quant_info, runner_config, running_state
        )

    return _pre_permute_standard_to_asym_gemm_fp8(
        dispatch_output, quant_info, runner_config, running_state
    )


def _pre_permute_standard_to_asym_gemm_fp8(
    dispatch_output: StandardDispatchOutput,
    quant_info: AsymGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AsymGemmRunnerInput:
    from sglang.srt.layers.moe.ep_moe.kernels import moe_ep_deepgemm_preprocess

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device
    hidden_states_ref = hidden_states

    topk_weights, topk_ids = topk_weights, topk_ids

    # PreReorder
    masked_m, expected_m, src2dst, hidden_states, hidden_states_scale = (
        moe_ep_deepgemm_preprocess(
            topk_ids,
            runner_config.num_local_experts,
            hidden_states,
            runner_config.top_k,
            quant_info.block_shape,
        )
    )

    dispose_tensor(hidden_states_ref)

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["hidden_states_device"] = hidden_states_device
    running_state["src2dst"] = src2dst

    offsets, experts, list_size = build_offsets_experts_from_masked_m(
        masked_m,
        hidden_states.shape[0],
        hidden_states.shape[1],
    )

    return AsymGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
        offsets=offsets,
        experts=experts,
        list_size=list_size,
    )


def _pre_permute_standard_to_asym_gemm_bf16(
    dispatch_output: StandardDispatchOutput,
    quant_info,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    from sglang.srt.layers.moe.ep_moe.kernels import (
        compute_masked_m_triton_kernel,
        compute_seg_indptr_triton_kernel,
        deepgemm_compute_src2dst_triton_kernel,
    )
    from sglang.srt.layers.moe.moe_runner.asym_gemm_bf16 import (
        AsymGemmBf16RunnerInput,
        fill_gateup_input_bf16,
    )

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device
    hidden_states_ref = hidden_states

    num_local_experts = runner_config.num_local_experts
    top_k = runner_config.top_k

    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    seg_indptr = torch.zeros(
        num_local_experts + 1, device=topk_ids.device, dtype=torch.int64
    )
    src2dst = torch.empty(topk_ids.numel(), device=topk_ids.device, dtype=torch.int32)
    masked_m = torch.empty(
        num_local_experts, device=topk_ids.device, dtype=torch.int32
    )

    compute_seg_indptr_triton_kernel[(num_local_experts + 1,)](
        reorder_topk_ids, seg_indptr, topk_ids.numel()
    )
    compute_masked_m_triton_kernel[(num_local_experts,)](seg_indptr, masked_m)

    m_max = (hidden_states.size(0) // 256 + 1) * 256
    expected_m = (topk_ids.numel() - 1) // num_local_experts + 1

    gateup_input = torch.empty(
        (num_local_experts, m_max, hidden_states.size(1)),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    import triton

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(topk_ids.numel(), meta["BLOCK_SIZE"]),
    )
    deepgemm_compute_src2dst_triton_kernel[grid](
        topk_ids,
        reorder_ids,
        seg_indptr,
        src2dst,
        m_max,
        topk_ids.numel(),
        BLOCK_SIZE=256,
    )

    fill_gateup_input_bf16(
        hidden_states,
        gateup_input,
        src2dst,
        topk_ids,
        top_k,
    )

    dispose_tensor(hidden_states_ref)

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["hidden_states_device"] = hidden_states_device
    running_state["src2dst"] = src2dst

    offsets, experts, list_size = build_offsets_experts_from_masked_m(
        masked_m,
        num_local_experts,
        m_max,
    )

    return AsymGemmBf16RunnerInput(
        hidden_states=gateup_input,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
        offsets =  offsets,
        experts = experts,
        list_size = list_size,
    )


@register_post_permute("asym_gemm", "standard")
def post_permute_asym_gemm_to_standard(
    runner_output: AsymGemmRunnerOutput,
    quant_info: AsymGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.ep_moe.kernels import post_reorder_triton_kernel
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states_shape = running_state["hidden_states_shape"]
    hidden_states_dtype = running_state["hidden_states_dtype"]
    hidden_states_device = running_state["hidden_states_device"]
    src2dst = running_state["src2dst"]
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]

    output = torch.empty(
        hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
    )
    post_reorder_triton_kernel[(hidden_states_shape[0],)](
        runner_output.hidden_states,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        runner_config.top_k,
        hidden_states_shape[1],
        BLOCK_SIZE=512,
    )

    dispose_tensor(runner_output.hidden_states)

    if runner_config.routed_scaling_factor is not None:
        output *= runner_config.routed_scaling_factor

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("deepep_ll", "asym_gemm")
def pre_permute_deepep_ll_to_asym_gemm(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    from sglang.srt.layers.moe.moe_runner.asym_gemm_bf16 import (
        AsymGemmBf16MoeQuantInfo,
        AsymGemmBf16RunnerInput,
    )

    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, expected_m = (
        dispatch_output
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["hidden_states_device"] = hidden_states.device

    offsets, experts, list_size = build_offsets_experts_from_masked_m(
        masked_m,
        hidden_states.shape[0],
        hidden_states.shape[1],
    )

    # Dtype-based dispatch
    if isinstance(quant_info, AsymGemmBf16MoeQuantInfo):
        return AsymGemmBf16RunnerInput(
            hidden_states=hidden_states,
            use_masked_gemm=True,
            masked_m=masked_m,
            expected_m=expected_m,
            offsets=offsets,
            experts=experts,
            list_size=list_size,
        )

    return AsymGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
        offsets =  offsets,
        experts = experts,
        list_size = list_size,
    )


@register_post_permute("asym_gemm", "deepep_ll")
def post_permute_asym_gemm_to_deepep_ll(
    runner_output: AsymGemmRunnerOutput,
    quant_info: AsymGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("deepep_normal", "asym_gemm")
def pre_permute_deepep_normal_to_asym_gemm(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    from sglang.srt.layers.moe.moe_runner.asym_gemm_bf16 import (
        AsymGemmBf16MoeQuantInfo,
        AsymGemmBf16RunnerInput,
    )

    # Dtype-based dispatch
    if isinstance(quant_info, AsymGemmBf16MoeQuantInfo):
        return _pre_permute_deepep_normal_to_asym_gemm_bf16(
            dispatch_output, quant_info, runner_config, running_state
        )

    return _pre_permute_deepep_normal_to_asym_gemm_fp8(
        dispatch_output, quant_info, runner_config, running_state
    )


def _pre_permute_deepep_normal_to_asym_gemm_fp8(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: AsymGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AsymGemmRunnerInput:
    from sglang.srt.layers.moe.ep_moe.kernels import ep_scatter

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    assert runner_config.activation == "silu"

    all_tokens = sum(num_recv_tokens_per_expert)
    running_state["all_tokens"] = all_tokens

    K = hidden_states.shape[1]

    hidden_states_shape = hidden_states.shape
    hidden_states_device = hidden_states.device
    hidden_states_dtype = hidden_states.dtype

    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_device"] = hidden_states_device
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    input_tensor = torch.empty(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0:
        input_tensor_scale = torch.zeros(
            (ceil_div(K // 128, 4), all_tokens),
            device=hidden_states.device,
            dtype=torch.int,
        ).transpose(0, 1)
    else:
        input_tensor_scale = torch.empty(
            (all_tokens, K // 128),
            device=hidden_states.device,
            dtype=torch.float32,
        )
    m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)

    if get_offloader().forbid_copy_engine_usage:
        num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
            num_recv_tokens_per_expert
        )
    else:
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

    ep_scatter(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
    )
    dispose_tensor(hidden_states)
    dispose_tensor(hidden_states_scale)

    running_state["output_index"] = output_index

    return AsymGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
    )


def _pre_permute_deepep_normal_to_asym_gemm_bf16(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    from sglang.srt.layers.moe.ep_moe.kernels import ep_scatter
    from sglang.srt.layers.moe.moe_runner.asym_gemm_bf16 import (
        AsymGemmBf16RunnerInput,
    )

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    assert runner_config.activation == "silu"

    all_tokens = sum(num_recv_tokens_per_expert)
    running_state["all_tokens"] = all_tokens

    K = hidden_states.shape[1]

    hidden_states_shape = hidden_states.shape
    hidden_states_device = hidden_states.device
    hidden_states_dtype = hidden_states.dtype

    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_device"] = hidden_states_device
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    input_tensor = torch.empty(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    # Dummy scale for ep_scatter API compatibility
    dummy_scale = torch.empty(
        (all_tokens, K // 128),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)

    if get_offloader().forbid_copy_engine_usage:
        num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
            num_recv_tokens_per_expert
        )
    else:
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

    ep_scatter(
        hidden_states,
        hidden_states_scale if hidden_states_scale is not None else dummy_scale[:hidden_states.shape[0]],
        topk_ids,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor,
        dummy_scale,
        m_indices,
        output_index,
        scale_ue8m0=False,
    )
    dispose_tensor(hidden_states)

    running_state["output_index"] = output_index

    return AsymGemmBf16RunnerInput(
        hidden_states=input_tensor,
        use_masked_gemm=False,
        m_indices=m_indices,
    )


@register_post_permute("asym_gemm", "deepep_normal")
def post_permute_asym_gemm_to_deepep_normal(
    runner_output: AsymGemmRunnerOutput,
    quant_info: AsymGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.ep_moe.kernels import ep_gather
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    hidden_states = runner_output.hidden_states
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]
    output_index = running_state["output_index"]

    gather_out = torch.empty(
        running_state["hidden_states_shape"],
        device=running_state["hidden_states_device"],
        dtype=torch.bfloat16,
    )
    ep_gather(hidden_states, topk_ids, topk_weights, output_index, gather_out)

    return DeepEPNormalCombineInput(
        hidden_states=gather_out,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )
