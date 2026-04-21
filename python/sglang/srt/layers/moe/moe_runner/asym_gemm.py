from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl

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
from sglang.srt.layers.quantization.fp8_utils import (
    FP8_E4M3_MAX,
)

# Cached hot-path function references (populated on first use to avoid circular imports)
_hp = {}  # hot-path cache dict

def _hp_get(name):
    """Get a cached hot-path function reference. Populated lazily on first use."""
    v = _hp.get(name)
    if v is not None:
        return v
    if name == 'preprocess':
        from sglang.srt.layers.moe.ep_moe.kernels import moe_ep_deepgemm_preprocess
        _hp[name] = moe_ep_deepgemm_preprocess
    elif name == 'post_reorder':
        from sglang.srt.layers.moe.ep_moe.kernels import post_reorder_triton_kernel
        _hp[name] = post_reorder_triton_kernel
    elif name == 'silu_masked':
        from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd
        _hp[name] = silu_and_mul_masked_post_quant_fwd
    elif name == 'quant_8bit':
        from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit
        _hp[name] = sglang_per_token_group_quant_8bit
    elif name == 'quant_fp8':
        from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
        _hp[name] = sglang_per_token_group_quant_fp8
    elif name == 'tma_align':
        from sglang.srt.layers.moe.ep_moe.kernels import tma_align_input_scale
        _hp[name] = tma_align_input_scale
    elif name == 'combine_input':
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        _hp[name] = StandardCombineInput
    elif name == 'seg_indptr':
        from sglang.srt.layers.moe.ep_moe.kernels import compute_seg_indptr_triton_kernel
        _hp[name] = compute_seg_indptr_triton_kernel
    return _hp[name]

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


def build_offsets_experts_from_masked_m(
    masked_m: torch.Tensor, num_groups: int, max_m: int, block_m: int = 128
):
    """
    Build offsets and experts for sparse m-grouped masked GEMM.
    Fixed-size output tensors; active experts are compacted to the front
    (unordered), inactive slots are -1.

    Example with num_groups=256, active experts 0, 1, 128:
        experts = [0, 1, 128, -1, -1, ..., -1]  (size num_groups+1)
        offsets = [s0, e0, s1, e1, s128, e128, 0, 0, ...]  (size 2*num_groups)
        list_size = 4  (3 active + 1 terminator)

    Args:
        masked_m:   (num_groups,) tensor of actual token counts per group
        num_groups: number of expert groups
        max_m:      allocated row stride per group (e.g. m_max)
        block_m:    block-M alignment for padding (default 128)

    Returns:
        offsets:   flat int32 tensor of size 2*num_groups
        experts:   int32 tensor of size num_groups+1
        list_size: num_active + 1
    """
    offsets = []
    experts = []
    for g in range(num_groups):
        v = masked_m[g].item()
        if v > 0:
            start = g * max_m
            end = start + ((v + block_m - 1) // block_m) * block_m
            offsets.append(start)
            offsets.append(end)
            experts.append(g)

    experts.append(-1)
    return (
        torch.tensor(offsets, dtype=torch.int32, device=masked_m.device),
        torch.tensor(experts, dtype=torch.int32, device=masked_m.device),
        len(experts),
    )


@triton.jit
def _build_offsets_experts_kernel(
    masked_m_ptr,
    offsets_ptr,
    experts_ptr,
    list_size_ptr,
    max_m: int,
    block_m: int,
):
    g = tl.program_id(0)
    v = tl.load(masked_m_ptr + g)
    if v > 0:
        slot = tl.atomic_add(list_size_ptr, 1)
        start = g * max_m
        end = start + ((v + block_m - 1) // block_m) * block_m
        tl.store(offsets_ptr + slot * 2, start)
        tl.store(offsets_ptr + slot * 2 + 1, end)
        tl.store(experts_ptr + slot, g)


def build_offsets_experts_from_seg_indptr(
    seg_indptr: torch.Tensor, num_groups: int
):
    """
    Build offsets and experts for contiguous m-grouped GEMM from segment index pointer.

    Args:
        seg_indptr: (num_groups + 1,) cumulative token counts per expert
        num_groups: number of expert groups

    Returns:
        offsets:   flat int32 tensor of size 2*num_groups
        experts:   int32 tensor of size num_groups+1
        list_size: num_active + 1
    """
    offsets = torch.zeros(2 * num_groups, dtype=torch.int32, device=seg_indptr.device)
    experts = torch.full(
        (num_groups + 1,), -1, dtype=torch.int32, device=seg_indptr.device
    )
    list_size = torch.zeros(1, 1, dtype=torch.int32, device=seg_indptr.device)

    _build_offsets_experts_contig_kernel[(num_groups,)](
        seg_indptr, offsets, experts, list_size
    )
    list_size.add_(1)
    return offsets, experts, list_size


@triton.jit
def _build_offsets_experts_contig_kernel(
    seg_indptr_ptr,
    offsets_ptr,
    experts_ptr,
    list_size_ptr,
):
    g = tl.program_id(0)
    start = tl.load(seg_indptr_ptr + g)
    end = tl.load(seg_indptr_ptr + g + 1)
    if end > start:
        slot = tl.atomic_add(list_size_ptr, 1)
        tl.store(offsets_ptr + slot * 2, start.to(tl.int32))
        tl.store(offsets_ptr + slot * 2 + 1, end.to(tl.int32))
        tl.store(experts_ptr + slot, g)


def build_offsets_experts_direct(
    sorted_expert_ids: torch.Tensor, num_groups: int, all_tokens: int
):
    """Build offsets/experts/list_size directly from sorted expert IDs via binary search.
    Fuses compute_seg_indptr + build_offsets into a single kernel launch."""
    offsets = torch.zeros(2 * num_groups, dtype=torch.int32, device=sorted_expert_ids.device)
    experts = torch.full(
        (num_groups + 1,), -1, dtype=torch.int32, device=sorted_expert_ids.device
    )
    list_size = torch.zeros(1, 1, dtype=torch.int32, device=sorted_expert_ids.device)

    _build_offsets_direct_kernel[(num_groups,)](
        sorted_expert_ids, offsets, experts, list_size, all_tokens,
    )
    list_size.add_(1)
    return offsets, experts, list_size


@triton.jit
def _build_offsets_direct_kernel(
    sorted_ids_ptr,
    offsets_ptr,
    experts_ptr,
    list_size_ptr,
    all_tokens,
):
    """Binary search for start/end of each expert in sorted IDs, then build offsets."""
    g = tl.program_id(0)

    # Binary search for start: first index where sorted_ids >= g
    lo = 0
    hi = all_tokens
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(sorted_ids_ptr + mid) < g:
            lo = mid + 1
        else:
            hi = mid
    start = lo

    # Binary search for end: first index where sorted_ids >= g+1
    lo2 = start
    hi2 = all_tokens
    while lo2 < hi2:
        mid2 = (lo2 + hi2) // 2
        if tl.load(sorted_ids_ptr + mid2) < g + 1:
            lo2 = mid2 + 1
        else:
            hi2 = mid2
    end = lo2

    if end > start:
        slot = tl.atomic_add(list_size_ptr, 1)
        tl.store(offsets_ptr + slot * 2, start.to(tl.int32))
        tl.store(offsets_ptr + slot * 2 + 1, end.to(tl.int32))
        tl.store(experts_ptr + slot, g)


def _invert_permutation(sorted_indices: torch.Tensor, n: int) -> torch.Tensor:
    """Build inverse permutation: src2dst[sorted_indices[i]] = i, using a single Triton kernel."""
    src2dst = torch.empty(n, device=sorted_indices.device, dtype=torch.int32)
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _invert_permutation_kernel[grid](sorted_indices, src2dst, n, BLOCK=BLOCK)
    return src2dst


@triton.jit
def _invert_permutation_kernel(
    sorted_indices_ptr, src2dst_ptr, n, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(sorted_indices_ptr + offs, mask=mask)
    tl.store(src2dst_ptr + idx, offs.to(tl.int32), mask=mask)


@dataclass
class AsymGemmRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    use_masked_gemm: bool
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None
    m_indices: Optional[torch.Tensor] = None
    offsets: Optional[int] = None
    experts: Optional[int] = None
    list_size: int = 0

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
        # Internal runner cores for dtype-based dispatch
        from sglang.srt.layers.moe.moe_runner.asym_gemm_bf16 import (
            AsymGemmBf16RunnerCore,
        )
        from sglang.srt.layers.moe.moe_runner.asym_gemm_fp4 import (
            AsymGemmFp4RunnerCore,
        )

        self._bf16_core = AsymGemmBf16RunnerCore(config)
        self._fp4_core = AsymGemmFp4RunnerCore(config)

    def run(
        self,
        runner_input,
        quant_info,
        running_state: dict,
    ):
        from sglang.srt.layers.moe.moe_runner.asym_gemm_bf16 import (
            AsymGemmBf16MoeQuantInfo,
        )
        from sglang.srt.layers.moe.moe_runner.asym_gemm_fp4 import (
            AsymGemmFp4MoeQuantInfo,
        )

        # Dtype-based dispatch: route to the matching inner runner core.
        if isinstance(quant_info, AsymGemmBf16MoeQuantInfo):
            return self._bf16_core.run(runner_input, quant_info, running_state)
        
        if isinstance(quant_info, AsymGemmFp4MoeQuantInfo):
            return self._fp4_core.run(runner_input, quant_info, running_state)

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
        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_dtype = running_state["hidden_states_dtype"]
        hidden_states_shape = running_state["hidden_states_shape"]

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
            hidden_states_scale = _hp_get("tma_align")(hidden_states_scale)

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

        # Fuse silu_and_mul + FP8 quantization into a single kernel
        down_input_fp8, down_input_scale = _hp_get("quant_fp8")(
            gateup_output.view(-1, N),
            scale_block_size,
            column_major_scales=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
            scale_tma_aligned=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
            scale_ue8m0=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
            fuse_silu_and_mul=True,
            enable_v2=True,
        )
        del gateup_output

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if not asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0:
            down_input_scale = _hp_get("tma_align")(down_input_scale)

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
        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0: scale conversion
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
            down_input, down_input_scale = _hp_get('quant_8bit')(
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
            _hp_get('silu_masked')(
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
    from sglang.srt.layers.moe.moe_runner.asym_gemm_fp4 import (
        AsymGemmFp4MoeQuantInfo,
    )

    # Dtype-based dispatch
    if isinstance(quant_info, AsymGemmBf16MoeQuantInfo):
        return _pre_permute_standard_to_asym_gemm_bf16(
            dispatch_output, quant_info, runner_config, running_state
        )
    if isinstance(quant_info, AsymGemmFp4MoeQuantInfo):
        return _pre_permute_standard_to_asym_gemm_fp4(
            dispatch_output, quant_info, runner_config, running_state
        )

    return _pre_permute_standard_to_asym_gemm_fp8_contiguous(
        dispatch_output, quant_info, runner_config, running_state
    )


def _pre_permute_standard_to_asym_gemm_fp8(
    dispatch_output: StandardDispatchOutput,
    quant_info: AsymGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AsymGemmRunnerInput:
    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device
    hidden_states_ref = hidden_states

    # PreReorder
    masked_m, expected_m, src2dst, hidden_states, hidden_states_scale = (
        _hp_get('preprocess')(
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


def _pre_permute_standard_to_asym_gemm_fp8_contiguous(
    dispatch_output: "StandardDispatchOutput",
    quant_info: AsymGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AsymGemmRunnerInput:
    """Contiguous pre-permute: sorts tokens by expert into a flat (all_tokens, K) layout.
    More memory-efficient than masked layout for standard dispatcher."""
    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    num_tokens, K = hidden_states.shape
    top_k = runner_config.top_k
    num_local_experts = runner_config.num_local_experts
    all_tokens = num_tokens * top_k
    scale_block_size = 128
    device = hidden_states.device

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device
    hidden_states_ref = hidden_states

    # 1. Sort token-expert pairs by expert ID
    flat_topk_ids = topk_ids.view(-1)  # (all_tokens,)
    sorted_expert_ids, sorted_indices = torch.sort(flat_topk_ids, stable=True)

    # 2. Build src2dst: original flat index → sorted position (fused kernel)
    src2dst = _invert_permutation(sorted_indices, all_tokens)

    # 3. Gather hidden states in expert-sorted order
    original_token_indices = sorted_indices // top_k
    sorted_hidden = hidden_states[original_token_indices]  # (all_tokens, K)

    dispose_tensor(hidden_states_ref)

    # 4. Quantize to FP8 (with ue8m0 scales if on Blackwell)
    sorted_fp8, sorted_scale = _hp_get("quant_fp8")(
        sorted_hidden,
        scale_block_size,
        column_major_scales=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
        scale_tma_aligned=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
        scale_ue8m0=asym_gemm_wrapper.ASYMGEMM_SCALE_UE8M0,
    )
    del sorted_hidden

    # 5. Build offsets/experts/list_size directly via binary search (fused kernel)
    offsets, experts, list_size = build_offsets_experts_direct(
        sorted_expert_ids, num_local_experts, all_tokens
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["hidden_states_device"] = hidden_states_device
    running_state["src2dst"] = src2dst
    running_state["all_tokens"] = all_tokens

    return AsymGemmRunnerInput(
        hidden_states=sorted_fp8,
        hidden_states_scale=sorted_scale,
        use_masked_gemm=False,
        offsets=offsets,
        experts=experts,
        list_size=list_size,
    )


def _pre_permute_standard_to_asym_gemm_fp4_contiguous(
    dispatch_output: "StandardDispatchOutput",
    quant_info,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    """Contiguous pre-permute for NVFP4: sort tokens by expert and quantize to FP4."""
    from sglang.srt.layers.moe.moe_runner.asym_gemm_fp4 import (
        AsymGemmFp4RunnerInput,
        _quantize_bf16_to_nvfp4_e4m3,
    )

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    num_tokens, K = hidden_states.shape
    top_k = runner_config.top_k
    num_local_experts = runner_config.num_local_experts
    all_tokens = num_tokens * top_k

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device
    hidden_states_ref = hidden_states

    # 1. Sort token-expert pairs by expert ID
    flat_topk_ids = topk_ids.view(-1)
    sorted_expert_ids, sorted_indices = torch.sort(flat_topk_ids, stable=True)

    # 2. Build src2dst: original flat index → sorted position
    src2dst = _invert_permutation(sorted_indices, all_tokens)

    # 3. Gather hidden states in expert-sorted order
    original_token_indices = sorted_indices // top_k
    sorted_hidden = hidden_states[original_token_indices]  # (all_tokens, K)

    dispose_tensor(hidden_states_ref)

    # 4. Quantize bf16/fp16 activations to NVFP4 (packed u8) + E4M3 scales
    sorted_fp4, sorted_scale = _quantize_bf16_to_nvfp4_e4m3(sorted_hidden)
    del sorted_hidden

    # 5. Build offsets/experts/list_size directly via binary search
    offsets, experts, list_size = build_offsets_experts_direct(
        sorted_expert_ids, num_local_experts, all_tokens
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["hidden_states_device"] = hidden_states_device
    running_state["src2dst"] = src2dst
    running_state["all_tokens"] = all_tokens

    return AsymGemmFp4RunnerInput(
        hidden_states=sorted_fp4,
        hidden_states_scale=sorted_scale,
        use_masked_gemm=False,
        offsets=offsets,
        experts=experts,
        list_size=list_size,
    )


def _pre_permute_standard_to_asym_gemm_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AsymGemmFp4RunnerInput:
    """Masked pre-permute for NVFP4 — mirrors _pre_permute_standard_to_asym_gemm_fp8.

    Reorders tokens into a padded masked layout (num_experts, m_max, K) as BF16,
    then quantizes to NVFP4 (packed uint8 + E4M3 per-group scales).
    """
    from sglang.srt.layers.moe.ep_moe.kernels import (
        compute_masked_m_triton_kernel,
        compute_seg_indptr_triton_kernel,
        deepgemm_compute_src2dst_triton_kernel,
    )
    from sglang.srt.layers.moe.moe_runner.asym_gemm_fp4 import (
        AsymGemmFp4RunnerInput,
        _quantize_bf16_to_nvfp4_e4m3,
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
    all_tokens = topk_ids.numel()  # num_tokens * top_k
    # m_max must be a multiple of 256 (matches moe_ep_deepgemm_preprocess)
    m_max = (hidden_states.size(0) // 256 + 1) * 256
    expected_m = (all_tokens - 1) // num_local_experts + 1

    # PreReorder: sort tokens by expert, build seg_indptr / masked_m / src2dst
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    seg_indptr = torch.zeros(
        num_local_experts + 1, device=hidden_states_device, dtype=torch.int64
    )
    src2dst = torch.empty(all_tokens, device=hidden_states_device, dtype=torch.int32)
    masked_m = torch.empty(num_local_experts, device=hidden_states_device, dtype=torch.int32)

    compute_seg_indptr_triton_kernel[(num_local_experts + 1,)](
        reorder_topk_ids, seg_indptr, all_tokens
    )
    compute_masked_m_triton_kernel[(num_local_experts,)](seg_indptr, masked_m)

    grid = lambda meta: (triton.cdiv(all_tokens, meta["BLOCK_SIZE"]),)
    deepgemm_compute_src2dst_triton_kernel[grid](
        topk_ids, reorder_ids, seg_indptr, src2dst, m_max, all_tokens, BLOCK_SIZE=256,
    )

    # Scatter tokens into zero-initialised BF16 masked buffer (num_experts, m_max, K).
    # src2dst[flat_idx] = expert_id * m_max + row_within_expert (flattened masked position).
    K = hidden_states.size(1)
    gateup_input = torch.zeros(
        (num_local_experts, m_max, K),
        device=hidden_states_device,
        dtype=hidden_states_dtype,
    )
    flat_topk_ids = topk_ids.view(-1)
    arange = torch.arange(all_tokens, device=hidden_states_device)
    valid_mask = flat_topk_ids >= 0
    valid_masked_pos = src2dst[valid_mask].to(torch.int64)
    valid_token_indices = arange[valid_mask] // top_k
    gateup_input.view(-1, K)[valid_masked_pos] = hidden_states[valid_token_indices]

    dispose_tensor(hidden_states_ref)

    # Quantize masked BF16 buffer to NVFP4
    gateup_fp4, gateup_scale = _quantize_bf16_to_nvfp4_e4m3(gateup_input)
    del gateup_input

    offsets, experts, list_size = build_offsets_experts_from_masked_m(
        masked_m,
        gateup_fp4.shape[0],
        gateup_fp4.shape[1],
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["hidden_states_device"] = hidden_states_device
    running_state["src2dst"] = src2dst

    return AsymGemmFp4RunnerInput(
        hidden_states=gateup_fp4,
        hidden_states_scale=gateup_scale,
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
) -> "StandardCombineInput":
    hidden_states_shape = running_state["hidden_states_shape"]
    hidden_states_dtype = running_state["hidden_states_dtype"]
    hidden_states_device = running_state["hidden_states_device"]
    src2dst = running_state["src2dst"]
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]

    output = torch.empty(
        hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
    )
    _hp_get('post_reorder')[(hidden_states_shape[0],)](
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

    return _hp_get('combine_input')(
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
    from sglang.srt.layers.moe.moe_runner.asym_gemm_fp4 import (
        AsymGemmFp4MoeQuantInfo,
        AsymGemmFp4RunnerInput,
        _quantize_bf16_to_nvfp4_e4m3,
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

    if isinstance(quant_info, AsymGemmFp4MoeQuantInfo):
        # DeepEP dispatcher may pass BF16 hidden states (no scale) or already
        # FP4-quantized states. Quantize here only when scales are absent.
        if hidden_states_scale is None:
            hs_fp4, hs_scale = _quantize_bf16_to_nvfp4_e4m3(hidden_states)
            dispose_tensor(hidden_states)
        else:
            hs_fp4, hs_scale = hidden_states, hidden_states_scale
        return AsymGemmFp4RunnerInput(
            hidden_states=hs_fp4,
            hidden_states_scale=hs_scale,
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
    from sglang.srt.layers.moe.moe_runner.asym_gemm_fp4 import (
        AsymGemmFp4MoeQuantInfo,
    )

    # Dtype-based dispatch
    if isinstance(quant_info, AsymGemmBf16MoeQuantInfo):
        return _pre_permute_deepep_normal_to_asym_gemm_bf16(
            dispatch_output, quant_info, runner_config, running_state
        )
    if isinstance(quant_info, AsymGemmFp4MoeQuantInfo):
        return _pre_permute_deepep_normal_to_asym_gemm_fp4(
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


def _pre_permute_deepep_normal_to_asym_gemm_fp4(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    """DeepEP-normal → AsymGEMM-FP4 pre-permute: scatter BF16 activations then
    quantize to NVFP4 (packed FP4 + per-group E4M3 scales)."""
    from sglang.srt.layers.moe.ep_moe.kernels import ep_scatter
    from sglang.srt.layers.moe.moe_runner.asym_gemm_fp4 import (
        AsymGemmFp4RunnerInput,
        _quantize_bf16_to_nvfp4_e4m3,
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

    # Scatter BF16 into the contiguous layout; activation FP4 quant happens
    # downstream. ep_scatter still wants a scale tensor for layout reasons.
    input_tensor = torch.empty(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
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
        hidden_states_scale
        if hidden_states_scale is not None
        else dummy_scale[: hidden_states.shape[0]],
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

    # Now quantize the expert-permuted BF16 tokens into packed NVFP4.
    input_fp4, input_scale = _quantize_bf16_to_nvfp4_e4m3(input_tensor)
    del input_tensor

    return AsymGemmFp4RunnerInput(
        hidden_states=input_fp4,
        hidden_states_scale=input_scale,
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
