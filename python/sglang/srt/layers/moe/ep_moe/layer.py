from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from sglang.srt import single_batch_overlap
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
    should_use_flashinfer_trtllm_moe,
)
from sglang.srt.layers.moe.ep_moe.kernels import (
    ep_gather,
    ep_scatter,
    silu_and_mul_masked_post_quant_fwd,
    tma_align_input_scale,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFusedMoE, FusedMoE
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import (
    is_fp8_fnuz,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.single_batch_overlap import DownGemmOverlapArgs
from sglang.srt.utils import ceil_div, dispose_tensor, get_bool_env_var, is_hip, is_npu
from sglang.srt.utils.offloader import get_offloader

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLOutput,
        DeepEPNormalOutput,
        DispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if not (_is_npu or _is_hip):
    from sgl_kernel import silu_and_mul

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

logger = logging.getLogger(__name__)


class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
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

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
        else:
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.use_w4afp8 = False

        self.deepep_mode = get_deepep_mode()

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
                    if self.use_block_quant or self.use_w4afp8
                    else self.w13_weight_scale
                ),
            )
            self.w2_weight_fp8 = (
                self.w2_weight,
                (
                    self.w2_weight_scale_inv
                    if self.use_block_quant or self.use_w4afp8
                    else self.w2_weight_scale
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        forward_shared_experts=None,
        alt_stream=None,
        disable_sbo=False,
    ):

        # We have to call SBO inside MoE to be compatible with hooks used in offloading
        return single_batch_overlap.execute_sbo(
            hidden_states=hidden_states,
            topk_output=topk_output,
            # SBO args
            experts=self,
            forward_shared_experts=forward_shared_experts,
            alt_stream=alt_stream,
            disable_sbo=disable_sbo,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
        down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None,
    ):
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if _use_aiter:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            return self.forward_aiter(dispatch_output)
        if _is_npu:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            return self.forward_npu(dispatch_output)
        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if self.use_w4afp8:
                return self.forward_cutlass_w4afp8(dispatch_output)
            assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8
            return self.forward_deepgemm_contiguous(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            ):
                return self.forward_flashinfer_cutedsl(
                    dispatch_output, down_gemm_overlap_args=down_gemm_overlap_args
                )
            assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8
            assert down_gemm_overlap_args is None
            return self.forward_deepgemm_masked(dispatch_output)
        else:
            raise ValueError(
                f"Dispatch output format {dispatch_output.format} is not supported"
            )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_aiter(
        self,
        dispatch_output: Union[DeepEPNormalOutput, DeepEPLLOutput],
    ):
        hidden_states, topk_ids, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
        if hidden_states.shape[0] == 0:
            return hidden_states
        # in original deepep, idx == -1 meaning invalid and will not be processed.
        # aiter does not accept -1, we use a expert mask to make these idx invalid
        # (idx == num_local_experts) meaning not used in aiter fused_moe
        topk_ids_copy = topk_ids.to(torch.int32)
        topk_ids_copy[topk_ids_copy == -1] = self.num_local_experts

        return fused_moe(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_weights,
            topk_ids_copy,
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
        (
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        ) = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states.bfloat16()
        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states.bfloat16()
        M, K = hidden_states.size()
        N = self.w13_weight.size(1)
        scale_block_size = 128

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

        hidden_states_shape = hidden_states.shape
        hidden_states_device = hidden_states.device
        hidden_states_dtype = hidden_states.dtype

        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
            (
                # TODO check whether need `zeros`
                torch.zeros(
                    (ceil_div(K // 128, 4), all_tokens),
                    device=hidden_states.device,
                    dtype=torch.int,
                ).transpose(0, 1)
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else torch.empty(
                    (all_tokens, K // 128),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            ),
        ]
        m_indices = torch.empty(
            all_tokens, device=hidden_states.device, dtype=torch.int32
        )
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
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        dispose_tensor(hidden_states)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
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
            device=hidden_states_device,
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
            hidden_states_shape,
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        ep_gather(down_output, topk_ids, topk_weights, output_index, gather_out)

        return gather_out

    def forward_flashinfer_cutedsl(
        self,
        dispatch_output: DeepEPLLOutput,
        down_gemm_overlap_args: Optional[DownGemmOverlapArgs],
    ):
        hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        output = self.quant_method.apply_without_routing_weights(
            layer=self,
            x=(hidden_states, hidden_states_scale),
            masked_m=masked_m,
            moe_runner_config=self.moe_runner_config,
            down_gemm_overlap_args=down_gemm_overlap_args,
        )
        return output

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_deepgemm_masked(
        self,
        dispatch_output: DeepEPLLOutput,
    ):
        hidden_states, hidden_states_scale, _, _, masked_m, expected_m = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        assert (
            hidden_states_scale.dtype == torch.float32
        ), f"hidden_states_scale.dtype: {hidden_states_scale.dtype}"

        # GroupGemm-0
        num_groups, m, k = hidden_states.size()
        n = self.w13_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states.device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (hidden_states, hidden_states_scale),
            self.w13_weight_fp8,
            gateup_output,
            masked_m,
            expected_m,
        )
        dispose_tensor(hidden_states)

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
            hidden_states, hidden_states_scale, _, _, num_recv_tokens_per_expert = (
                dispatch_output
            )

            group_list = torch.tensor(num_recv_tokens_per_expert, dtype=torch.int64).to(
                hidden_states.device
            )
            if self.w13_weight.dtype != torch.int8:
                # gmm1: gate_up_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w13_weight.permute(0, 2, 1)],
                    # per_token_scale=[hidden_states_scale],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                # gmm2: down_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w2_weight.permute(0, 2, 1)],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]
            else:
                if not get_bool_env_var("DEEP_NORMAL_MODE_USE_INT8_QUANT"):
                    hidden_states, hidden_states_scale = torch_npu.npu_dynamic_quant(
                        hidden_states
                    )
                # gmm1: gate_up_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w13_weight],
                    scale=[self.w13_weight_scale.to(output_dtype)],
                    per_token_scale=[hidden_states_scale],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]

                # act_fn: swiglu
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(
                    hidden_states
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

        def _forward_ll(dispatch_output: DeepEPLLOutput):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPLLOutput)
            (
                hidden_states,
                hidden_states_scale,
                topk_ids,
                topk_weights,
                group_list,
                _,
            ) = dispatch_output

            group_list = group_list.to(torch.int64)

            if self.w13_weight.dtype != torch.int8:
                # gmm1: gate_up_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w13_weight.permute(0, 2, 1)],
                    # per_token_scale=[hidden_states_scale],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                # gmm2: down_proj
                hidden_states = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[self.w2_weight.permute(0, 2, 1)],
                    split_item=2,
                    group_list_type=group_list_type,
                    group_type=0,
                    group_list=group_list,
                    output_dtype=output_dtype,
                )[0]
            else:
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
                    activation_scale=hidden_states_scale,
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


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
        return DeepEPMoE

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
    return FusedMoE


def copy_list_to_gpu_no_ce(arr: List[int]):
    from sgl_kernel.elementwise import copy_to_gpu_no_ce

    tensor_cpu = torch.tensor(arr, dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    return tensor_gpu
