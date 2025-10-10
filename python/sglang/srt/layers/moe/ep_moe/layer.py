from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.parallel_state import (
    get_moe_ep_group,
    get_moe_expert_parallel_world_size,
    get_world_group,
)
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
    get_moriep_mode,
    should_use_flashinfer_trtllm_moe,
)
from sglang.srt.layers.moe.ep_moe.kernels import (
    ep_gather,
    ep_scatter,
    moe_ep_deepgemm_preprocess,
    post_reorder_triton_kernel,
    silu_and_mul_masked_post_quant_fwd,
    tma_align_input_scale,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFusedMoE, FusedMoE
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import (
    is_fp8_fnuz,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.offloader import get_offloader
from sglang.srt.utils import (
    ceil_div,
    dispose_tensor,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLOutput,
        DeepEPNormalOutput,
        DispatchOutput,
        MoRILLOutput,
        MoRINormalOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if not (_is_npu or _is_hip):
    from sgl_kernel import silu_and_mul

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter import dtypes as aiter_dtypes
    from aiter.fused_moe import fused_moe

logger = logging.getLogger(__name__)


# TODO(kaixih@nvidia): ideally we should merge this logic into
# `fill_gateup_input_triton_kernel` to directly generate e8m0 scale.
@torch.compile
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


class EPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl


    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        gemm1_alpha: Optional[float] = None,
        gemm1_clamp_limit: Optional[float] = None,
        with_bias: bool = False,
    ):
        super().__init__(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_fused_shared_experts=num_fused_shared_experts,
            layer_id=layer_id,
            top_k=top_k,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            # apply_router_weight_on_input=apply_router_weight_on_input,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
            with_bias=with_bias,
        )

        self.intermediate_size = intermediate_size

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.activation_scheme = quant_config.activation_scheme
        else:
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.block_shape = None
            self.activation_scheme = None

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8:
            return self.forward_deepgemm(hidden_states, topk_output)
        else:
            return super().forward(hidden_states, topk_output)

    def forward_deepgemm(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        self.w13_weight_fp8 = (
            self.w13_weight,
            (
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            self.w2_weight_scale_inv if self.use_block_quant else self.w2_weight_scale,
        )

        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        hidden_states_shape = hidden_states.shape
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device

        topk_weights, topk_ids, _ = topk_output

        if not self.use_block_quant:
            # Convert per-tensor quant to per-block quant by repeating scales for forward_deepgemm
            scale_block_size = 128
            w13_weight_scale_n = 2 * (
                (self.intermediate_size + scale_block_size - 1) // scale_block_size
            )
            w13_weight_scale_k = (
                hidden_states_shape[-1] + scale_block_size - 1
            ) // scale_block_size
            w13_weight_scale = (
                self.w13_weight_scale.unsqueeze(1)
                .repeat_interleave(w13_weight_scale_n, dim=1)
                .unsqueeze(2)
                .repeat_interleave(w13_weight_scale_k, dim=2)
            )
            self.w13_weight_fp8 = (
                self.w13_weight,
                w13_weight_scale,
            )
            w2_weight_scale_n = (
                hidden_states_shape[-1] + scale_block_size - 1
            ) // scale_block_size
            w2_weight_scale_k = (
                self.intermediate_size + scale_block_size - 1
            ) // scale_block_size
            w2_weight_scale = (
                self.w2_weight_scale.unsqueeze(1)
                .repeat_interleave(w2_weight_scale_n, dim=1)
                .unsqueeze(2)
                .repeat_interleave(w2_weight_scale_k, dim=2)
            )
            self.w2_weight_fp8 = (
                self.w2_weight,
                w2_weight_scale,
            )

        # PreReorder
        m_max, masked_m, expected_m, src2dst, gateup_input, gateup_input_scale = (
            moe_ep_deepgemm_preprocess(
                topk_ids,
                self.num_experts,
                hidden_states,
                self.top_k,
                self.start_expert_id,
                self.end_expert_id,
                self.block_shape,
            )
        )

        dispose_tensor(hidden_states)

        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            b, s_mn, s_k = gateup_input_scale.shape
            assert (
                s_mn % 4 == 0 and s_k % 4 == 0
            ), f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"

        # GroupGemm-0
        gateup_input_fp8 = (
            gateup_input,
            (
                _cast_to_e8m0_with_rounding_up(gateup_input_scale)
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                    gateup_input_scale
                )
            ),
        )
        num_groups, m, k = gateup_input_fp8[0].size()
        n = self.w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            gateup_input_fp8,
            self.w13_weight_fp8,
            gateup_output,
            masked_m,
            expected_m,
        )
        del gateup_input
        del gateup_input_fp8

        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=hidden_states_device,
            dtype=self.fp8_dtype,
        )
        scale_block_size = 128
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
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        del gateup_output

        # GroupGemm-1
        n = self.w2_weight.size(1)
        down_input_fp8 = (
            down_input,
            (
                down_input_scale
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(down_input_scale)
            ),
        )
        down_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            down_input_fp8,
            self.w2_weight_fp8,
            down_output,
            masked_m,
            expected_m,
        )
        del down_input
        del down_input_fp8

        # PostReorder
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
        post_reorder_triton_kernel[(hidden_states_shape[0],)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states_shape[1],
            m_max * self.start_expert_id,
            BLOCK_SIZE=512,
        )
        if self.moe_runner_config.routed_scaling_factor is not None:
            output *= self.moe_runner_config.routed_scaling_factor
        return output


class DeepEPMoE(EPMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.deepep_mode = get_deepep_mode()

        # TODO: move to the beginning of the file
        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.two_batch_overlap import MaybeTboDeepEPDispatcher

        self.deepep_dispatcher = MaybeTboDeepEPDispatcher(
            group=get_tp_group().device_group,
            router_topk=self.top_k,
            permute_fusion=True,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=self.deepep_mode,
            async_finish=True,  # TODO
            return_recv_hook=True,
        )

        if self.deepep_mode.enable_low_latency() and not _is_npu:
            # NPU supports low_latency deepep without deepgemm
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        if _use_aiter:
            # expert_mask is of size (self.num_local_experts + 1),
            # the extra 1 is for invalid rank_id (in original deepep, the invalid rank_id is -1, but aiter does not allow -1, we use a mask to make those ids invalid)
            # for instance, if we have 4 experts on this rank, we would have a expert_mask like:
            #     self.expert_mask = [1, 1, 1, 1, 0]
            # idx from 0-3 is valid and will be processed, while idx == 4 will be masked out
            self.expert_mask = torch.zeros(
                (self.num_local_experts + 1),
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            # the last one is invalid rank_id
            self.expert_mask[:-1] = 1
        elif not _is_npu:
            self.w13_weight_fp8 = (
                self.w13_weight,
                (
                    self.w13_weight_scale_inv
                    if self.use_block_quant
                    else self.w13_weight_scale
                ),
            )
            self.w2_weight_fp8 = (
                self.w2_weight,
                (
                    self.w2_weight_scale_inv
                    if self.use_block_quant
                    else self.w2_weight_scale
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        dispatch_output = self.dispatch(
            hidden_states, topk_idx, topk_weights, forward_batch
        )
        hidden_states = self.moe_impl(dispatch_output)
        hidden_states = self.combine(
            hidden_states,
            dispatch_output.topk_idx,
            dispatch_output.topk_weights,
            forward_batch,
        )
        return hidden_states

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self.deepep_dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            forward_batch=forward_batch,
        )

    def moe_impl(self, dispatch_output: DispatchOutput):
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if _use_aiter:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            return self.forward_aiter(dispatch_output)
        if _is_npu:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            return self.forward_npu(dispatch_output)
        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8
            return self.forward_deepgemm_contiguous(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if get_moe_runner_backend().is_flashinfer_cutedsl():
                return self.forward_flashinfer_cutedsl(dispatch_output)
            assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8
            return self.forward_deepgemm_masked(dispatch_output)
        else:
            raise ValueError(
                f"Dispatch output format {dispatch_output.format} is not supported"
            )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self.deepep_dispatcher.combine(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            forward_batch=forward_batch,
        )

    def forward_aiter(
        self,
        dispatch_output: Union[DeepEPNormalOutput, DeepEPLLOutput],
    ):
        hidden_states, topk_idx, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_idx,
            dispatch_output.topk_weights,
        )
        if hidden_states.shape[0] == 0:
            return hidden_states
        # in original deepep, idx == -1 meaning invalid and will not be processed.
        # aiter does not accept -1, we use a expert mask to make these idx invalid
        # (idx == num_local_experts) meaning not used in aiter fused_moe
        topk_idx_copy = topk_idx.to(torch.int32)
        topk_idx_copy[topk_idx_copy == -1] = self.num_local_experts

        return fused_moe(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_weights,
            topk_idx_copy,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            quant_type=QuantType.per_128x128,
            activation=(
                ActivationType.Silu
                if self.moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            expert_mask=self.expert_mask,
        )

    def forward_deepgemm_contiguous(
        self,
        dispatch_output: DeepEPNormalOutput,
    ):
        hidden_states_fp8, topk_idx, topk_weights, num_recv_tokens_per_expert = (
            dispatch_output
        )
        hidden_states_fp8, hidden_states_scale = hidden_states_fp8
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states_fp8.bfloat16()
        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states_fp8.bfloat16()
        M, K = hidden_states_fp8.size()
        N = self.w13_weight.size(1)
        scale_block_size = 128

        # TODO also unify other branches (e.g. `EPMoE.forward_deepgemm` sets the field on forward pass)
        w13_weight_fp8 = (
            self.w13_weight,
            (
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
        )
        w2_weight_fp8 = (
            self.w2_weight,
            (
                self.w2_weight_scale_inv
                if self.use_block_quant
                else self.w2_weight_scale
            ),
        )

        hidden_states_fp8_shape = hidden_states_fp8.shape
        hidden_states_fp8_device = hidden_states_fp8.device
        hidden_states_fp8_dtype = hidden_states_fp8.dtype

        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states_fp8.device,
                dtype=hidden_states_fp8.dtype,
            ),
            (
                # TODO check whether need `zeros`
                torch.zeros(
                    (ceil_div(K // 128, 4), all_tokens),
                    device=hidden_states_fp8.device,
                    dtype=torch.int,
                ).transpose(0, 1)
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else torch.empty(
                    (all_tokens, K // 128),
                    device=hidden_states_fp8.device,
                    dtype=torch.float32,
                )
            ),
        ]
        m_indices = torch.empty(
            all_tokens, device=hidden_states_fp8.device, dtype=torch.int32
        )
        output_index = torch.empty_like(topk_idx)

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
            hidden_states_fp8,
            hidden_states_scale,
            topk_idx,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        dispose_tensor(hidden_states_fp8)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            input_tensor, w13_weight_fp8, gateup_output, m_indices
        )
        del input_tensor
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
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
            down_input,
            scale_block_size,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        del down_input
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            down_input_scale = tma_align_input_scale(down_input_scale)
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,
        )
        del down_input_fp8, down_input_scale

        gather_out = torch.empty(
            hidden_states_fp8_shape,
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)

        return gather_out

    def forward_flashinfer_cutedsl(
        self,
        dispatch_output: DeepEPLLOutput,
    ):
        hidden_states, _, _, masked_m, _ = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        output = self.quant_method.apply_without_routing_weights(
            layer=self,
            x=hidden_states,
            masked_m=masked_m,
            moe_runner_config=self.moe_runner_config,
        )
        return output

    def forward_deepgemm_masked(
        self,
        dispatch_output: DeepEPLLOutput,
    ):
        hidden_states_fp8, _, _, masked_m, expected_m = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        # GroupGemm-0
        num_groups, m, k = hidden_states_fp8[0].size()
        n = self.w13_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_fp8[0].device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            hidden_states_fp8,
            self.w13_weight_fp8,
            gateup_output,
            masked_m,
            expected_m,
        )
        dispose_tensor(hidden_states_fp8[0])

        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=self.fp8_dtype,
        )
        scale_block_size = 128
        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2 // scale_block_size,
            ),
            device=gateup_output.device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            scale_block_size,
            masked_m,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        del gateup_output

        # GroupGemm-1
        n = self.w2_weight.size(1)
        down_input_fp8 = (
            down_input,
            (
                down_input_scale
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(down_input_scale)
            ),
        )
        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            down_input_fp8,
            self.w2_weight_fp8,
            down_output,
            masked_m,
            expected_m,
        )

        return down_output

    def forward_npu(
        self,
        dispatch_output: Union[DeepEPNormalOutput, DeepEPLLOutput],
    ):
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        import torch_npu

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        def _forward_normal(dispatch_output: DeepEPNormalOutput):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPNormalOutput)
            hidden_states, _, _, num_recv_tokens_per_expert = dispatch_output

            if isinstance(hidden_states, tuple):
                per_token_scale = hidden_states[1]
                hidden_states = hidden_states[0]
            else:
                # dynamic quant
                hidden_states, per_token_scale = torch_npu.npu_dynamic_quant(
                    hidden_states
                )

            group_list = torch.tensor(num_recv_tokens_per_expert, dtype=torch.int64).to(
                hidden_states.device
            )

            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[self.w13_weight],
                scale=[self.w13_weight_scale.to(output_dtype)],
                per_token_scale=[per_token_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]

            # act_fn: swiglu
            hidden_states = torch_npu.npu_swiglu(hidden_states)
            hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)

            # gmm2: down_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[self.w2_weight],
                scale=[self.w2_weight_scale.to(output_dtype)],
                per_token_scale=[swiglu_out_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]

            return hidden_states

        def _forward_ll(dispatch_output: DeepEPLLOutput):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPLLOutput)
            hidden_states, topk_idx, topk_weights, group_list, _ = dispatch_output

            per_token_scale = hidden_states[1]
            hidden_states = hidden_states[0]

            group_list = group_list.to(torch.int64)

            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[self.w13_weight],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32,
            )[0]

            # act_fn: swiglu
            hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=self.w13_weight_scale.to(torch.float32),
                activation_scale=per_token_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=group_list,
                activate_left=True,
                quant_mode=1,
            )

            # gmm2: down_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[self.w2_weight],
                scale=[self.w2_weight_scale.to(output_dtype)],
                per_token_scale=[swiglu_out_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]

            return hidden_states

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            return _forward_normal(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            return _forward_ll(dispatch_output)
        else:
            raise ValueError(f"Not Supported DeepEP format {dispatch_output.format}")


MORI_SHMEM_INITIALIZED = False
MORI_QUANT_CONFIG = None


# NOTE: Currently, the code of MoRIEPMoE is originated from DeepEPMoE.
#       We need to change it more mori-specific way as we can possible.
class MoRIEPMoE(EPMoE):
    """
    MoE Expert Parallel Impl based on mori (https://github.com/ROCm/mori)
    """

    _has_printed = False

    # NOTE: Keep the same interface with DeepEPMoE
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.moriep_mode = get_moriep_mode()
        self.set_mori_quant_config()
        # FIXME: This env var used in MoRIDispatcher. We may change the location where getting this variable.
        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 256
        )

        # TODO: move to the beginning of the file
        from sglang.srt.two_batch_overlap import MaybeTboDeepEPDispatcher

        self._ensure_shmem_initialized()
        self.mori_dispatcher = MaybeTboDeepEPDispatcher(
            group=get_moe_ep_group(),  # NOTE: MoRIDispatcher requires GroupCoordinator
            router_topk=self.top_k,
            permute_fusion=True,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            use_fp8_w8a8=self.use_fp8_w8a8,
            quant_dtype=self.fp8_dtype,
            moriep_mode=self.moriep_mode,
            async_finish=True,  # TODO
            return_recv_hook=False,  # Currently, not used
        )

        if _use_aiter:
            assert not _is_npu, f"MoRI does not support npu devices."
            assert not is_cuda(), f"MoRI does not support cuda environment."

            # expert_mask is of size (self.num_local_experts + 1),
            # the extra 1 is for invalid rank_id (in original deepep, the invalid rank_id is -1,
            # but aiter does not allow -1, we use a mask to make those ids invalid)
            # for instance, if we have 4 experts on this rank, we would have a expert_mask like:
            #     self.expert_mask = [1, 1, 1, 1, 0]
            # idx from 0-3 is valid and will be processed, while idx == 4 will be masked out
            self.expert_mask = torch.zeros(
                (self.num_local_experts + 1),
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            # the last one is invalid rank_id
            self.expert_mask[:-1] = 1

    # TODO: Currently MoRI supports only `low-latency` mode which generates static-sized communication buffer.
    # So, when another mode (like deepep's high throughput mode) be added, we should change the execution path.
    # FIXME: Chunked forward is implemented but it does not distinguish the `forward_batch.forward_mode` now.
    # If we need separating both prefill and decode path, fix the code below later.
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        assert self.quant_method is not None
        if self.moe_ep_size > 1:
            if self.expert_map_cpu is not None and self.expert_map_gpu is None:
                # If we are in EP mode, we need to move the expert map to GPU.
                self.expert_map_gpu = self.expert_map_cpu.to(device="cuda")

        local_num_tokens = hidden_states.shape[0]
        max_num_tokens_over_dp_ranks = max(forward_batch.global_num_tokens_cpu)
        step = self.num_max_dispatch_tokens_per_rank

        num_valid_forward, num_pad_forward = self.get_loop_count(
            local_num_tokens, max_num_tokens_over_dp_ranks, step
        )
        output = torch.zeros_like(hidden_states)

        if num_valid_forward > 0:
            _range = list(range(0, local_num_tokens, step))
            if _range[-1] != local_num_tokens:
                _range = _range + [local_num_tokens]

            for idx, val in enumerate(_range[:-1]):
                _start = _range[idx]
                _end = _range[idx + 1]
                self.forward_single(
                    output=output[_start:_end, :],
                    hidden_states=hidden_states[_start:_end, :],
                    topk_idx=topk_idx[_start:_end, :],
                    topk_weights=topk_weights[_start:_end, :],
                    forward_batch=forward_batch,
                )
        if num_pad_forward > 0:
            _empty_output = torch.empty(
                (0, hidden_states.shape[-1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            _empty_hidden_states = torch.empty(
                (0, hidden_states.shape[-1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            _empty_topk_idx = torch.empty(
                (0, topk_idx.shape[-1]), device=topk_idx.device, dtype=topk_idx.dtype
            )
            _empty_topk_weights = torch.empty(
                (0, topk_weights.shape[-1]),
                device=topk_weights.device,
                dtype=topk_weights.dtype,
            )
        for _ in range(0, num_pad_forward):
            self.forward_single(
                output=_empty_output,
                hidden_states=_empty_hidden_states,
                topk_idx=_empty_topk_idx,
                topk_weights=_empty_topk_weights,
                forward_batch=forward_batch,
            )

        return output

    def forward_single(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        dispatch_output = self.dispatch(
            hidden_states, topk_idx, topk_weights, forward_batch
        )
        hidden_states = self.moe_impl(dispatch_output)
        topk_idx = (
            topk_idx if dispatch_output.topk_idx is None else dispatch_output.topk_idx
        )
        topk_weights = (
            topk_weights
            if dispatch_output.topk_weights is None
            else dispatch_output.topk_weights
        )
        self.combine(
            output,
            hidden_states,
            dispatch_output.topk_idx,
            dispatch_output.topk_weights,
            forward_batch,
        )

    def get_loop_count(self, local_num_tokens, max_num_tokens, step):
        if local_num_tokens == 0:
            num_valid_forward = 0
        else:
            num_valid_forward = ceil_div(local_num_tokens, step)
        num_pad_forward = ceil_div(max_num_tokens, step) - num_valid_forward
        assert (
            num_pad_forward >= 0
        ), f"Invalid configuration for forward_batch: {local_num_tokens=}, {max_num_tokens=}, {step=}"
        return num_valid_forward, num_pad_forward

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self.mori_dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            forward_batch=forward_batch,
        )

    def moe_impl(self, dispatch_output: DispatchOutput):
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if _use_aiter:
            assert DispatchOutputChecker.format_is_mori(dispatch_output)
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            return self.forward_aiter(dispatch_output)
        else:
            raise NotImplementedError(f"Currently, only aiter kernel is supported.")

    def combine(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self.mori_dispatcher.combine(
            output=output,
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            forward_batch=forward_batch,
        )

    def forward_aiter(
        self,
        dispatch_output: Union[MoRINormalOutput, MoRILLOutput],
    ):
        hidden_states, topk_idx, topk_weights, num_local_tokens_per_expert, scales = (
            dispatch_output.hidden_states,
            dispatch_output.topk_idx,
            dispatch_output.topk_weights,
            dispatch_output.num_recv_tokens_per_expert,
            dispatch_output.scales,
        )
        if hidden_states.shape[0] == 0:
            return hidden_states

        topk_idx = topk_idx.to(torch.int32)
        if self.expert_map_gpu is not None:
            topk_idx = self.expert_map_gpu[topk_idx]
        topk_idx[topk_idx == -1] = self.num_local_experts

        return fused_moe(
            hidden_states=hidden_states,
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_weight=topk_weights,
            topk_ids=topk_idx,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            a1_scale=scales,
            num_local_tokens=num_local_tokens_per_expert,
            # NOTE: DSr-1 config follows QuantType.per_128x128 but
            # already 1x128 scale applied to hidden_states due to fp8 dispatch.
            # Also, aiter uses the QuantType.per_128x128 as QuantType.per_1x128
            # internally. So it doesn't matter to use 1x128 rather than 128x128.
            quant_type=QuantType.per_1x128,
            activation=(
                ActivationType.Silu
                if self.moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            dtype=aiter_dtypes.bf16,
            expert_mask=self.expert_mask,
        )

    # NOTE: Maybe move mori config and shmem initialization to where model parallel is initialized
    def _ensure_shmem_initialized(self):
        """Ensure mori's shared memory system is initialized"""
        global MORI_SHMEM_INITIALIZED
        ep_group = get_moe_ep_group()
        world_group = get_world_group()

        if MORI_SHMEM_INITIALIZED:
            return

        import mori.shmem
        import torch.distributed as dist

        try:
            # Wait for PyTorch distributed to be ready
            if not dist.is_initialized():
                raise RuntimeError("PyTorch distributed not initialized yet")

            # Check if we have a valid backend
            backend = dist.get_backend()
            if backend is None:
                raise RuntimeError("No valid distributed backend found")

            logger.debug(
                f"[rank {self.moe_ep_rank}] PyTorch distributed ready with backend: {backend}"
            )
            current_group = (
                ep_group.cpu_group
                if ep_group.cpu_group is not None
                else world_group.cpu_group
            )

            group_name = "default"
            try:
                import torch._C._distributed_c10d as c10d

                # Try to unregister first in case it exists
                try:
                    c10d._unregister_process_group(group_name)
                except:
                    pass

                # Register the current process group
                c10d._register_process_group(group_name, current_group)
                logger.debug(
                    f"[rank {self.moe_ep_rank}] Registered process group '{group_name}'"
                )

                # Initialize mori shmem with the registered group
                mori.shmem.shmem_torch_process_group_init(group_name)
                logger.debug(
                    f"[rank {self.moe_ep_rank}] Torch process group shmem initialization successful"
                )
                MORI_SHMEM_INITIALIZED = True
                return

            except Exception as torch_error:
                logger.debug(
                    f"[rank {self.moe_ep_rank}] Torch process group shmem init failed: {torch_error}"
                )
                raise RuntimeError(
                    f"Failed to register process group for mori shmem initialization: {torch_error}"
                ) from torch_error

        except Exception as e:
            # NOTE: Do we need continuing the shmem initialization even it failed?
            #       I think raising error is more proper in this scenario.
            logger.error(
                f"[rank {self.moe_ep_rank}] mori shmem initialization failed: {e}"
            )
            raise RuntimeError(f"mori shmem initialization failed: {e}") from e

    def set_mori_quant_config(self):
        global MORI_QUANT_CONFIG
        _cfg = {}
        _cfg["use_block_quant"] = self.use_block_quant
        _cfg["block_shape"] = self.block_shape
        _cfg["use_fp8_w8a8"] = self.use_fp8_w8a8
        _cfg["fp8_dtype"] = self.fp8_dtype
        MORI_QUANT_CONFIG = _cfg


def get_mori_quant_config() -> dict:
    global MORI_QUANT_CONFIG
    return MORI_QUANT_CONFIG


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    if get_moe_a2a_backend().is_deepep():
        return DeepEPMoE

    if (
        get_moe_a2a_backend().is_mori()
        and not get_moe_runner_backend().is_flashinfer_trtllm()
    ):
        return MoRIEPMoE

    # NEW: Direct FP4 detection (bypasses EP requirements)
    # Check for FP4 quantization with TRTLLM flag, regardless of EP
    if get_moe_runner_backend().is_flashinfer_trtllm():
        # FlashInferFP4MoE must be paired with ModelOptNvFp4FusedMoEMethod.
        # If UnquantizedFusedMoEMethod is detected, fall back to FusedMoE instead.
        if quant_config is None:
            return FusedMoE
        try:
            # Check the quantization argument directly
            if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
                from sglang.srt.layers.moe.fused_moe_triton.layer import (
                    FlashInferFP4MoE,
                )

                return FlashInferFP4MoE
        except:
            pass

    if should_use_flashinfer_trtllm_moe() and quant_config is not None:
        # FIXME: FlashInferFusedMoE only supports fp8 quant now
        return FlashInferFusedMoE
    if get_moe_runner_backend().is_flashinfer_cutlass():
        return FusedMoE
    if get_moe_expert_parallel_world_size() > 1:
        return EPMoE
    return FusedMoE


def copy_list_to_gpu_no_ce(arr: List[int]):
    from sgl_kernel.elementwise import copy_to_gpu_no_ce

    tensor_cpu = torch.tensor(arr, dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    return tensor_gpu
