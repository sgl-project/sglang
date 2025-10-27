from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch

from sglang.srt import single_batch_overlap
from sglang.srt.distributed.parallel_state import (
    get_moe_ep_group,
    get_moe_expert_parallel_world_size,
    get_world_group,
)
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
    get_moriep_mode,
    should_use_flashinfer_trtllm_moe,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFusedMoE, FusedMoE
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.offloader import get_offloader
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.single_batch_overlap import DownGemmOverlapArgs
from sglang.srt.utils import (
    ceil_div,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
        MoRILLOutput,
        MoRINormalOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if not (_is_npu or _is_hip):
    pass

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter import dtypes as aiter_dtypes
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

        if _use_aiter or _is_npu:
            self.deprecate_flag = False
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        if self.deprecate_flag:
            return

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
            self.use_w4afp8 = False
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

        if self.deprecate_flag:
            assert forward_shared_experts is None
            assert alt_stream is None
            return super().forward(
                hidden_states,
                topk_output,
            )

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

        if self.deprecate_flag:
            assert down_gemm_overlap_args is None
            return super().run_moe_core(
                dispatch_output,
            )

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if _use_aiter:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            output = self.forward_aiter(dispatch_output)
        elif _is_npu:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            output = self.forward_npu(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            else:
                assert False, "forward_deepgemm_contiguous is deprecated"
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            ):
                output = self.forward_flashinfer_cutedsl(
                    dispatch_output, down_gemm_overlap_args=down_gemm_overlap_args
                )
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            else:
                assert False, "forward_deepgemm_masked is deprecated"

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )
        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
            overlap_args=down_gemm_overlap_args,
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
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
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

    def forward_flashinfer_cutedsl(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
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
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        assert get_bool_env_var(
            "SGLANG_DEEPEP_BF16_DISPATCH"
        ), "W4AFP8 does not support FP8 dispatch; please set SGLANG_DEEPEP_BF16_DISPATCH=1."
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_npu(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        import torch_npu

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        def _forward_normal(dispatch_output: DeepEPNormalDispatchOutput):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPNormalDispatchOutput)
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

        def _forward_ll(dispatch_output: DeepEPLLDispatchOutput):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPLLDispatchOutput)
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
    if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
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
    return FusedMoE
