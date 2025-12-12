from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


def npu_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    **kwargs,
):
    w13_offset = kwargs.get("w13_offset", None)
    w2_offset = kwargs.get("w2_offset", None)
    use_wna16 = kwargs.get("use_wna16", False)

    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
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
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)
    # gmm1: gate_up_proj
    if not use_wna16:
        hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
        scale_args13 = {
            "scale": [w13_scale.to(scale_dtype)],
            "per_token_scale": [pertoken_scale],
        }
    else:
        scale_args13 = {
            "antiquant_scale": [w13_scale],
            "antiquant_offset": [w13_offset],
        }

    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        **scale_args13,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    # act_fn: swiglu
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    if not use_wna16:
        hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

        scale_args2 = {
            "scale": [w2_scale.to(scale_dtype)],
            "per_token_scale": [pertoken_scale],
        }
    else:
        scale_args2 = {"antiquant_scale": [w2_scale], "antiquant_offset": [w2_offset]}
    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        **scale_args2,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

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


def npu_fused_moe_without_routing_weights_bf16(
    layer, hidden_states, group_list_type, group_list, output_dtype
):
    # gmm1: gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[layer.w13_weight.permute(0, 2, 1)],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[layer.w2_weight.permute(0, 2, 1)],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]
    return hidden_states


class NPUW8A8Int8DynamicMoEMethod(FusedMoEMethodBase):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        # weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        # scale
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        # offset
        w13_weight_offset = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)
        w2_weight_offset = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

    def release_weight_cache(self, weight: torch.Tensor):
        # .contiguous() introduces additional memory overhead and needs to be released using resize_(0)
        origin_weight = weight.data.transpose(1, 2)
        new_weight = origin_weight.contiguous()
        origin_weight.untyped_storage().resize_(0)
        return new_weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_data = self.release_weight_cache(layer.w13_weight.data)
        layer.w13_weight = torch.nn.Parameter(weight_data, requires_grad=False)

        weight_data = self.release_weight_cache(layer.w2_weight.data)
        layer.w2_weight = torch.nn.Parameter(weight_data, requires_grad=False)

        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data.squeeze(-1).contiguous().to(torch.float32),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w13_weight_offset = torch.nn.Parameter(
            layer.w13_weight_offset.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w2_weight_offset = torch.nn.Parameter(
            layer.w2_weight_offset.data.squeeze(-1).contiguous(), requires_grad=False
        )

        layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        output = npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
        )
        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        # gmm1: gate_up_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.int32,
        )[0]

        # act_fn: swiglu
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=layer.w13_weight_scale,
            activation_scale=hidden_states_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,
            activate_left=True,
            quant_mode=1,
        )

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale.to(output_dtype)],
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
        )[0]
        return hidden_states


class NPUW4A8Int4DynamicMoEMethod(FusedMoEMethodBase):

    def __init__(self) -> None:
        self.group_size = 256
        self.tp_size = 1

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        # >> weight
        w13_output_size = intermediate_size_per_partition
        w2_output_size = hidden_size // 2
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_output_size, hidden_size, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w2_output_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # >> scale
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # >> offset
        w13_weight_offset = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)

        w2_weight_offset = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

        # >>> special param for w4a8
        w13_weight_scale_second = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_second", w13_weight_scale_second)
        set_weight_attrs(w13_weight_scale_second, extra_weight_attrs)
        w13_weight_offset_second = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset_second", w13_weight_offset_second)
        set_weight_attrs(w13_weight_offset_second, extra_weight_attrs)

        w2_weight_scale_second = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_second", w2_weight_scale_second)
        set_weight_attrs(w2_weight_scale_second, extra_weight_attrs)

        w2_weight_offset_second = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset_second", w2_weight_offset_second)
        set_weight_attrs(w2_weight_offset_second, extra_weight_attrs)

        w13_scale_bias = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale_bias", w13_scale_bias)
        set_weight_attrs(w13_scale_bias, extra_weight_attrs)

        w2_scale_bias = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, 16 // self.tp_size, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scale_bias", w2_scale_bias)
        set_weight_attrs(w2_scale_bias, extra_weight_attrs)

    def process_scale(self, weight: torch.Tensor, scale, per_group_scale):
        scale = scale.transpose(1, 2).contiguous()
        per_group_scale = per_group_scale.transpose(1, 2).contiguous()
        group_num, k, n = weight.shape
        # the weight of the new version is reduced by half by pack n, so it needs to be restored
        n = n * 2
        per_group_scale = per_group_scale.reshape(group_num, -1, n)
        group_num, quantgroup_num, n = per_group_scale.shape
        bias = None

        scale_fp32 = (scale * per_group_scale).to(torch.float16).to(torch.float32)
        scale_fp32_np = scale_fp32.cpu().numpy()
        scale_fp32_np.dtype = np.uint32
        sscale_uint64 = np.zeros((group_num, quantgroup_num, n * 2), dtype=np.uint32)

        sscale_uint64[..., ::2] = scale_fp32_np

        sscale_uint64_buffer = np.frombuffer(
            sscale_uint64.tobytes(), dtype=np.int64
        ).copy()
        sscale_uint64_tensor = torch.from_numpy(sscale_uint64_buffer).reshape(
            group_num, quantgroup_num, n
        )
        sscale_uint64_tensor = sscale_uint64_tensor.npu()
        return sscale_uint64_tensor, bias

    def update_bias(self, layer, w13_bias, w2_bias):
        layer.w13_scale_bias.data = (
            layer.w13_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        )
        layer.w2_scale_bias.data = (
            layer.w2_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        )

    def pack_to_int32(self, weight: torch.Tensor):
        # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
        assert (
            weight.shape[-1] % 4 == 0
        ), "the last dim of weight needs to be divided by 4"
        return weight.view(torch.int32).contiguous()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )

        w13_weight_scale_second = (
            layer.w13_weight_scale_second.data
            if hasattr(layer, "w13_weight_scale_second")
            else None
        )
        w2_weight_scale_second = (
            layer.w2_weight_scale_second.data
            if hasattr(layer, "w2_weight_scale_second")
            else None
        )
        layer.w13_weight_scale.data, w13_bias = self.process_scale(
            layer.w13_weight, layer.w13_weight_scale.data, w13_weight_scale_second
        )
        layer.w2_weight_scale.data, w2_bias = self.process_scale(
            layer.w2_weight, layer.w2_weight_scale.data, w2_weight_scale_second
        )
        if hasattr(layer, "w13_weight_scale_second"):
            # scale_second is no longer used, release this part of the memory
            del layer.w13_weight_scale_second
            del layer.w2_weight_scale_second
            del layer.w13_weight_offset_second
            del layer.w2_weight_offset_second

        self.update_bias(layer, w13_bias, w2_bias)

        layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)
        layer.w13_weight.data = self.pack_to_int32(layer.w13_weight.data)
        layer.w2_weight.data = self.pack_to_int32(layer.w2_weight.data)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        # FIXME W4A8 only support with deepep
        raise NotImplementedError(
            f"W4A8 only support with deepep for now, please enable --moe-a2a-backend deepep"
        )

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            scale=[layer.w13_weight_scale],
            bias=[layer.w13_scale_bias],
            per_token_scale=[hidden_states_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )[0]

        hidden_states, swiglu_out_scale = swiglu_quant(
            hidden_states, group_list, group_list_type
        )

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale],
            bias=[layer.w2_scale_bias],
            per_token_scale=[swiglu_out_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )[0]

        return hidden_states


class NPUW4A16Int4DynamicMoEMethod(FusedMoEMethodBase):

    def __init__(self, quantization_config) -> None:
        self.pack_factor = 8  # weight dtype is int4,  but use int32 to create
        target = (
            "MoEGMM" if "MoEGMM" in quantization_config.target_scheme_map else "Linear"
        )
        if target in quantization_config.target_scheme_map:
            self.group_size = quantization_config.target_scheme_map[target][
                "weights"
            ].group_size
        else:
            self.group_size = 128

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        if (
            extra_weight_attrs.get(
                "intermediate_size_full", intermediate_size_per_partition
            )
            // intermediate_size_per_partition
            > 1
        ):
            quant_method = FusedMoeWeightScaleSupported.GROUP.value
        else:
            quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        extra_weight_attrs.update({"quant_method": quant_method})
        # weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        weight_scale_dtype = torch.bfloat16
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # offset
        w13_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)

        w2_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

    def pack_to_int32(self, weight: torch.Tensor):
        assert weight.dim() == 3
        if weight.dtype == torch.int32:
            # pack 8 int4 to int32, we use a int32 to represent a int4
            assert (
                weight.shape[-1] % 8 == 0
            ), "the last dim of weight needs to be divided by 8"
            new_weight = torch.ops.npu.npu_convert_weight_to_int4pack(
                weight.flatten(0, 1)
            )
            new_weight = new_weight.view(weight.shape[0], weight.shape[1], -1)
        elif weight.dtype == torch.int8:
            # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
            assert (
                weight.shape[-1] % 4 == 0
            ), "the last dim of weight needs to be divided by 4"
            new_weight = weight.view(torch.int32).contiguous()
        else:
            raise ValueError(f"{weight.dtype=} is not supported !")
        return new_weight

    def unpack_from_int32(
        self,
        value: torch.Tensor,
        num_bits: int,
        shape: torch.Size = None,
        packed_dim=1,
    ) -> torch.Tensor:
        """
        Unpacks a tensor of packed int32 weights into individual int8s, maintaining the
        original bit range.

        Return tensors in int8

        :param value: tensor to unpack
        :param num_bits: number of bits to unpack each data point into
        :param shape: shape to unpack into, used to remove padding
        :returns: unpacked int8 tensor
        """
        if value.dtype is not torch.int32:
            raise ValueError(
                f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
            )

        if num_bits > 8:
            raise ValueError("Unpacking is only supported for less than 8 bits")

        pack_factor = 32 // num_bits

        # unpack
        mask = (1 << num_bits) - 1

        if packed_dim == 1:
            unpacked = torch.zeros(
                (value.shape[0], value.shape[1] * pack_factor),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

            # remove padding
            if shape is not None:
                original_row_size = int(shape[1])
                unpacked = unpacked[:, :original_row_size]
        else:
            unpacked = torch.zeros(
                (value.shape[0] * pack_factor, value.shape[1]),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask

            # remove padding
            original_row_size = int(shape[0])
            unpacked = unpacked[:original_row_size, :]

        # bits are packed in unsigned format, reformat to signed
        # update the value range from unsigned to signed
        offset = pow(2, num_bits) // 2
        unpacked = (unpacked - offset).to(torch.int8)

        return unpacked

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_weight_scale = layer.w13_weight_scale.data.transpose(-1, -2).contiguous()
        w2_weight_scale = layer.w2_weight_scale.data.transpose(-1, -2).contiguous()
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_weight_scale, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)

        layer.w13_weight_offset = torch.nn.Parameter(
            layer.w13_weight_offset.data.transpose(-1, -2).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_offset = torch.nn.Parameter(
            layer.w2_weight_offset.data.transpose(-1, -2).contiguous(),
            requires_grad=False,
        )

        # w = [n, k // 8]  --> [k, n // 8]
        # w13_weight = layer.w13_weight.data.transpose(1, 2).contiguous()
        # w2_weight = layer.w2_weight.data.transpose(1, 2).contiguous()
        unpacked_w13_weight = (
            self.unpack_from_int32(layer.w13_weight.data.flatten(0, 1), 4)
            .view(layer.w13_weight.data.shape[0], layer.w13_weight.data.shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )
        unpacked_w2_weight = (
            self.unpack_from_int32(layer.w2_weight.data.flatten(0, 1), 4)
            .view(layer.w2_weight.data.shape[0], layer.w2_weight.data.shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )

        w13_weight = self.pack_to_int32(unpacked_w13_weight)
        w2_weight = self.pack_to_int32(unpacked_w2_weight)

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        output = npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_offset=layer.w13_weight_offset,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_offset=layer.w2_weight_offset,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
            use_wna16=True,
        )
        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        if hidden_states_scale is None:
            # gmm1: gate_up_proj
            hidden_states = torch.ops.npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w13_weight],
                antiquant_scale=[layer.w13_weight_scale],
                antiquant_offset=[layer.w13_weight_offset],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]

            # act_fn: swiglu
            hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

            # gmm2: down_proj
            out_hidden = torch.ops.npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w2_weight],
                antiquant_scale=[layer.w2_weight_scale],
                antiquant_offset=[layer.w2_weight_offset],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]
        else:
            raise ValueError(
                "when weight is int4, hidden_states only supports non-quant dtype!"
            )

        return out_hidden
