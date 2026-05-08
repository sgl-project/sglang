from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import einops
import torch

from sglang.jit_kernel.deepseek_v4 import silu_and_mul_masked_post_quant
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
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
    is_musa,
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
_is_musa = is_musa()

# Imported only for the SGLANG_OPT_FIX_MEGA_MOE_MEMORY=False fallback path.
if not (_is_npu or _is_hip) and _is_cuda:
    from sglang.jit_kernel.activation import silu_and_mul as _legacy_silu_and_mul
else:
    _legacy_silu_and_mul = None


_MASKED_GEMM_FAST_ACT = get_bool_env_var("SGLANG_MASKED_GEMM_FAST_ACT")
_DEEPGEMM_ON_H20 = get_bool_env_var("SGLANG_DEEPGEMM_ON_H20")


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


@dataclass
class DeepGemmRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    use_masked_gemm: bool
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None
    m_indices: Optional[torch.Tensor] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    use_fp8: bool
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None
    # DSV4 mxfp4 layout flag; selects recipe_a=(1,128)/recipe_b=(1,32) downstream.
    is_fp4_experts: bool = False


class DeepGemmRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        assert self.config.activation == "silu"
        assert self.config.is_gated
        self.swiglu_limit = self.config.swiglu_limit
        self.use_swizzle = False
        if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
            assert envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get()
            assert envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get()
            assert envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get()
            self.use_swizzle = True

    def run(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> DeepGemmRunnerOutput:
        if not runner_input.use_masked_gemm:
            hidden_states = self._run_contiguous_gemm(
                runner_input, quant_info, running_state
            )
        else:
            hidden_states = self._run_masked_gemm(
                runner_input, quant_info, running_state
            )
        return DeepGemmRunnerOutput(hidden_states=hidden_states)

    def _run_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.jit_kernel.deepseek_v4 import silu_and_mul_contig_post_quant
        from sglang.srt.layers.moe.ep_moe.kernels import tma_align_input_scale
        from sglang.srt.layers.quantization.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale,
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

        recipe_a, recipe_b = (
            ((1, 128), (1, 32)) if quant_info.is_fp4_experts else (None, None)
        )

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
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            hidden_states_scale = tma_align_input_scale(hidden_states_scale)

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (hidden_states, hidden_states_scale),
            w13_weight_fp8,
            gateup_output,
            m_indices,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
            swiglu_limit_arg: Optional[float] = self.swiglu_limit

            down_input_fp8 = torch.empty(
                (all_tokens, N // 2),
                device=gateup_output.device,
                dtype=torch.float8_e4m3fn,
            )
            down_input_scale = create_per_token_group_quant_fp8_output_scale(
                x_shape=(all_tokens, N // 2),
                device=gateup_output.device,
                group_size=scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            silu_and_mul_contig_post_quant(
                input=gateup_output,
                output=down_input_fp8,
                output_scale=down_input_scale,
                quant_group_size=scale_block_size,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                transposed=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                swiglu_limit=swiglu_limit_arg,
                swizzle=self.use_swizzle,
            )
            del gateup_output
        else:
            # Hacky byte-equal fallback that reproduces the optimize-branch
            # code path exactly: bf16 silu_and_mul then a separate per-token
            # group fp8 quant. Kept behind the mega-moe-memory flag.
            from sglang.srt.layers.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            if self.swiglu_limit is not None:
                gateup_output = _apply_swiglu_limit(
                    gateup_output, swiglu_limit=self.swiglu_limit
                )

            down_input = torch.empty(
                (all_tokens, N // 2),
                device=gateup_output.device,
                dtype=torch.bfloat16,
            )
            _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
            del gateup_output

            down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
                down_input,
                scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            del down_input

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            down_input_scale = tma_align_input_scale(down_input_scale)

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

        return down_output

    def _run_masked_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.srt.layers import deep_gemm_wrapper

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        recipe_a, recipe_b = (
            ((1, 128), (1, 32)) if quant_info.is_fp4_experts else (None, None)
        )

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            if hidden_states_scale.dtype != torch.int:
                b, s_mn, s_k = hidden_states_scale.shape
                assert (
                    s_mn % 4 == 0 and s_k % 4 == 0
                ), f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"
                hidden_states_scale = _cast_to_e8m0_with_rounding_up(
                    hidden_states_scale
                )
        elif deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            hidden_states_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                hidden_states_scale
            )

        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        swiglu_limit_arg: Optional[float] = None
        if self.swiglu_limit is not None:
            # DeepSeek V4: clamped swiglu requires JIT EP activation; the
            # FAST_ACT fused-quant path doesn't carry a swiglu_limit arg.
            assert (
                not _MASKED_GEMM_FAST_ACT
            ), "DeepSeek V4 does not support SGLANG_MASKED_GEMM_FAST_ACT"
            assert (
                envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get()
            ), "DeepSeek V4 requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"

            if envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get():
                swiglu_limit_arg = self.swiglu_limit
            else:
                gateup_output = einops.rearrange(
                    gateup_output, "grp tok hidden -> (grp tok) hidden"
                )
                gateup_output = _apply_swiglu_limit(
                    gateup_output, swiglu_limit=self.swiglu_limit
                )
                gateup_output = einops.rearrange(
                    gateup_output, "(grp tok) hidden -> grp tok hidden", grp=num_groups
                )

        # Act
        down_input, down_input_scale = _varlen_deep_gemm_silu_mul_quant(
            gateup_output,
            masked_m,
            group_size=128,
            topk=self.config.top_k,
            swiglu_limit=swiglu_limit_arg,
            swizzle=self.use_swizzle,
        )
        del gateup_output

        # GroupGemm-1
        n = w2_weight.shape[1]

        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            down_input_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
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
            max_block_n = (
                160 if (_DEEPGEMM_ON_H20 and runner_input.expected_m <= 64) else 256
            )
            gemm_overlap_args_dict = {
                "overlap_args": down_gemm_overlap_args,
                "max_block_n": max_block_n,
            }

        deep_gemm_return_value = deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (down_input, down_input_scale),
            (w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            **gemm_overlap_args_dict,
        )
        meta_overlap_args = running_state.get("meta_overlap_args", None)
        if meta_overlap_args is not None:
            block_m, threshold = deep_gemm_return_value
            meta_overlap_args["block_m"] = block_m
            meta_overlap_args["threshold"] = threshold

        return down_output

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@register_pre_permute("standard", "deep_gemm")
def pre_permute_standard_to_deep_gemm(
    dispatch_output: StandardDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
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

    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
    )


@register_post_permute("deep_gemm", "standard")
def post_permute_deep_gemm_to_standard(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
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


@register_pre_permute("deepep_ll", "deep_gemm")
def pre_permute_deepep_ll_to_deep_gemm(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, expected_m = (
        dispatch_output
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["hidden_states_device"] = hidden_states.device

    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
    )


@register_post_permute("deep_gemm", "deepep_ll")
def post_permute_deep_gemm_to_deepep_ll(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("deepep_normal", "deep_gemm")
def pre_permute_deepep_normal_to_deep_gemm(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
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
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        # TODO check whether need `zeros`
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
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )
    dispose_tensor(hidden_states)
    dispose_tensor(hidden_states_scale)

    running_state["output_index"] = output_index

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
    )


@register_post_permute("deep_gemm", "deepep_normal")
def post_permute_deep_gemm_to_deepep_normal(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
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


def _varlen_deep_gemm_silu_mul_quant(
    gateup_output: torch.Tensor,
    masked_m: Optional[torch.Tensor],
    group_size: int,
    topk: int,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_8bit,
    )

    if _MASKED_GEMM_FAST_ACT:
        assert not swizzle, (
            "SGLANG_OPT_FIX_MEGA_MOE_MEMORY is incompatible with "
            "SGLANG_MASKED_GEMM_FAST_ACT (swizzled layout only supported by JIT act)"
        )
        assert (
            swiglu_limit is None
        ), "swiglu_limit (DeepSeek V4) is not supported together with SGLANG_MASKED_GEMM_FAST_ACT"
        return sglang_per_token_group_quant_8bit(
            x=gateup_output,
            dst_dtype=torch.float8_e4m3fn,
            group_size=group_size,
            masked_m=masked_m,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            fuse_silu_and_mul=True,
            enable_v2=True,
        )

    assert masked_m is not None
    hidden_states_device = gateup_output.device
    E, N, D_2 = gateup_output.shape
    D = D_2 // 2
    del D_2
    G = D // group_size
    down_input = torch.empty(
        (E, N, D),
        device=hidden_states_device,
        dtype=torch.float8_e4m3fn,
    )

    if envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get():
        assert N % 4 == 0 and G % 4 == 0
        packed_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        down_input_scale = torch.empty(
            (E, G // 4, N) if packed_ue8m0 else (E, N, G),
            device=hidden_states_device,
            dtype=torch.int32 if packed_ue8m0 else torch.float32,
        )
        silu_and_mul_masked_post_quant(
            gateup_output,
            down_input,
            down_input_scale,
            group_size,
            masked_m,
            scale_ue8m0=packed_ue8m0,
            topk=topk,
            transposed=packed_ue8m0,
            swiglu_limit=swiglu_limit,
            swizzle=swizzle,
        )
        if packed_ue8m0:
            down_input_scale = down_input_scale.transpose(-1, -2)
    else:
        assert (
            swiglu_limit is None
        ), "swiglu_limit (DeepSeek V4) requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"
        assert (
            not swizzle
        ), "SGLANG_OPT_FIX_MEGA_MOE_MEMORY requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"
        down_input_scale = torch.empty(
            (E, N, G),
            device=hidden_states_device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            group_size,
            masked_m,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
    return down_input, down_input_scale


def _apply_swiglu_limit(
    gateup_output: torch.Tensor, swiglu_limit: float
) -> torch.Tensor:
    assert swiglu_limit == 10

    num_tokens, hidden_size_x2 = gateup_output.shape
    assert gateup_output.dtype == torch.bfloat16

    gate, up = torch.chunk(gateup_output, chunks=2, dim=-1)
    assert gate.shape == (num_tokens, hidden_size_x2 // 2)
    assert up.shape == (num_tokens, hidden_size_x2 // 2)

    up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    gate = torch.clamp(gate, max=swiglu_limit)

    out = torch.cat([gate, up], dim=-1)
    assert out.shape == (num_tokens, hidden_size_x2)
    return out
