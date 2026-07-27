from __future__ import annotations

import logging
from typing import Optional, Callable, TYPE_CHECKING

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import RoutingMethodType, get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.fp8_utils import is_blackwell_supported
from sglang.srt.layers.quantization.utils import (
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
    replace_parameter,
    swizzle_blockscale,
)
from sglang.srt.utils import next_power_of_2, set_weight_attrs

import torch_npu

from sglang.srt.layers.activation import SituAndMul
from sglang.srt.hardware_backend.npu.utils import situ_and_mul

logger = logging.getLogger(__name__)

__all__ = ["NPUCompressedTensorsW4A8mxfp4MoE"]

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


def _npu_swiglu(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.npu.npu_swiglu(x)


class NPUCompressedTensorsW4A8mxfp4MoE(CompressedTensorsMoEScheme):

    def __init__(self):
        self.group_size = 32
        self.act_fn: Callable = _npu_swiglu

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # From packed to weight
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        layer.w13_weight.data = torch_npu.npu_format_cast(
            layer.w13_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
        layer.w2_weight.data = torch_npu.npu_format_cast(
            layer.w2_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        g, n, k = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)
        g, n, k = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.moe_runner_config.activation == "situ":
            # self.act_fn = SituAndMul(
            #     beta=self.moe_runner_config.activation_situ_beta,
            #     linear_beta=self.moe_runner_config.activation_situ_linear_beta,
            # )
            self.act_fn = situ_and_mul

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        combine_input = npu_apply_w4a8_mxfp4_moe_deepep(layer, dispatch_output, act_fn=self.act_fn)
        if combine_input is not None:
            return combine_input

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        hidden_states = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(hidden_states.dtype)
        top_k = (
            self.moe_runner_config.top_k
            if self.moe_runner_config is not None
            else topk_ids.shape[1]
        )
        output = npu_fused_experts_w4a8_mxfp4(
            hidden_states,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            top_k,
            act_fn=self.act_fn,
        )
        return StandardCombineInput(hidden_states=output)


def _reshape_mxfp4_scale_for_npu(scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 3:
        num_experts, n, k32 = scale.shape
        if k32 % 2 != 0:
            raise ValueError(
                "MXFP4 scale K dimension must be divisible by 2 for "
                "[E, K/64, N, 2] layout."
            )
        scale = scale.view(num_experts, n, k32 // 2, 2).transpose(1, 2)
    return scale


def npu_fused_experts_w4a8_mxfp4(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    act_fn: Callable = _npu_swiglu,
):
    if torch.npu.is_current_stream_capturing():
        return npu_fused_experts_w4a8_mxfp4_decode(
            hidden_states=hidden_states,
            w13=w13,
            w13_weight_scale=w13_weight_scale,
            w2=w2,
            w2_weight_scale=w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=top_k,
            act_fn=act_fn,
        )

    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens,
        )
    )
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)

    rows = hidden_states.shape[0]
    row_ids = torch.arange(rows, device=hidden_states.device, dtype=torch.int64)
    valid_mask = row_ids < expert_tokens[-1]
    valid_mask_2d = valid_mask.unsqueeze(1)

    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w13,
        weight_scale=w13_weight_scale,
        group_list_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )
    hidden_states = act_fn(hidden_states, expert_tokens, 0)
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w2,
        weight_scale=w2_weight_scale,
        group_list_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )

    hidden_states = hidden_states * valid_mask_2d.to(hidden_states.dtype)

    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def npu_fused_experts_w4a8_mxfp4_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    act_fn: Callable = _npu_swiglu,
):
    num_tokens = hidden_states.shape[:-1].numel()
    global_num_experts = w13.shape[0]
    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    group_list_type = 1

    hidden_states, expanded_row_idx, expert_tokens, _ = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=group_list_type,
            expert_tokens_num_flag=True,
            active_expert_range=[0, global_num_experts],
            quant_mode=-1,
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)

    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w13,
        weight_scale=w13_weight_scale,
        group_list_type=group_list_type,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )
    hidden_states = act_fn(hidden_states, expert_tokens, group_list_type)
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w2,
        weight_scale=w2_weight_scale,
        group_list_type=group_list_type,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )

    final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=hidden_states,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def npu_apply_w4a8_mxfp4_moe_deepep(
    layer: torch.nn.Module,
    dispatch_output: "DispatchOutput",
    act_fn: Callable = _npu_swiglu,
) -> Optional["CombineInput"]:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )
    from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker

    if not dispatch_output.format.is_deepep():
        return None

    output_dtype = torch.bfloat16
    group_list_type = 1

    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        hidden_states, hidden_states_scale, _, _, num_recv_tokens_per_expert = (
            dispatch_output
        )
        group_list = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int64,
            device=hidden_states.device,
        )
        combine_cls = DeepEPNormalCombineInput
    else:
        hidden_states, hidden_states_scale, _, _, group_list, _ = dispatch_output
        group_list = group_list.to(torch.int64)
        combine_cls = DeepEPLLCombineInput

    hidden_states = npu_apply_without_routing_weights_w4a8_mxfp4(
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
        act_fn=act_fn,
    )
    return combine_cls(
        hidden_states=hidden_states,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )


def npu_apply_without_routing_weights_w4a8_mxfp4(
    layer,
    hidden_states,
    hidden_states_scale,
    group_list_type,
    group_list,
    output_dtype,
    act_fn: Callable = _npu_swiglu,
):
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=hidden_states_scale,
        weight=layer.w13_weight,
        weight_scale=layer.w13_weight_scale,
        group_list_type=group_list_type,
        group_list=group_list,
        output_dtype=output_dtype,
    )
    hidden_states = act_fn(hidden_states, group_list, group_list_type)
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=layer.w2_weight,
        weight_scale=layer.w2_weight_scale,
        group_list_type=group_list_type,
        group_list=group_list,
        output_dtype=output_dtype,
    )
    return hidden_states


def w4a8_mxfp4_gmm_npu(
    input: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_list_type: int,
    group_list: torch.Tensor,
    output_dtype=torch.bfloat16,
) -> torch.Tensor:
    group_list = group_list.to(torch.int64)

    # return torch.ops.npu.npu_grouped_matmul(
    #     [input],
    #     [weight],
    #     antiquant_scale=[weight_scale],
    #     split_item=2,
    #     group_type=0,
    #     group_list=group_list,
    #     group_list_type=group_list_type,
    #     output_dtype=output_dtype,
    # )[0]

    if input_scale is None:
        x, x_scale = torch.ops.npu.npu_dynamic_mx_quant(
            input,
            dst_type=torch_npu.float8_e4m3fn
        )
    else:
        x, x_scale = input, input_scale

    return torch.ops.npu.npu_grouped_matmul(
        [x],
        [weight],
        antiquant_scale=[weight_scale],
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale=[x_scale],
        split_item=2,
        group_type=0,
        group_list=group_list,
        group_list_type=group_list_type,
        output_dtype=output_dtype,
        x_dtype=torch_npu.float8_e4m3fn,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
    )[0]