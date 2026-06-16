# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.linear import set_weight_attrs
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)

from .awq_scheme import AWQMoESchemeBase

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.awq.awq import AWQConfig, AWQMarlinConfig

__all__ = ["AWQMoEScheme", "AWQAscendMoEScheme"]


from sglang.srt.layers.moe.moe_runner.torch_npu import (
    TorchNpuQuantInfo,
)

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A16Int4MoEMethod,
)

from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)

class AWQMoEScheme(AWQMoESchemeBase):
    def __init__(self, quant_config: AWQMarlinConfig):
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:
            raise ValueError("AWQMoEScheme only supports 4bit now.")
        self.kernel = self._init_kernel(quant_config)

    def _init_kernel(self, quant_config: AWQMarlinConfig):
        from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (
            AWQMoEKernel,
        )

        return AWQMoEKernel(quant_config)

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

        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )

        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size

        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        assert get_moe_runner_backend().is_auto()
        self.moe_runner_config = moe_runner_config
        self.kernel.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ):
        return self.kernel.apply(layer, dispatch_output)


'''class AWQAscendMoEScheme(AWQMoEScheme):
    def _init_kernel(self, quant_config: AWQConfig):
        from sglang.srt.hardware_backend.npu.quantization.awq_kernels import (
            AWQAscendMoEKernel,
        )

        return AWQAscendMoEKernel(quant_config)

    def create_moe_runner(
        self,
        layer: torch.nn.Module,
        moe_runner_config: "MoeRunnerConfig",
        **extra_weight_attrs,
    ):
        self.moe_runner_config = moe_runner_config
        layer.w13_kernel = NPUW4A16Int4MoEMethod()
        layer.w2_kernel = NPUW4A16Int4MoEMethod()
        moe_runner_config.layer = layer
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.TORCH_NPU
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        backend = self.runner.runner_backend
        quant_info = TorchNpuQuantInfo(
            w13_weight=layer.w13_qweight,
            w2_weight=layer.w2_qweight,
            w13_weight_scale=layer.w13_scales,
            w2_weight_scale=layer.w2_scales,
            w13_weight_offset=layer.w13_qzeros,
            w2_weight_offset=layer.w2_qzeros,
        )
        return self.runner.run(dispatch_output, quant_info)'''

class AWQAscendMoEScheme(AWQMoEScheme):
    def _init_kernel(self, quant_config: AWQConfig):
        from sglang.srt.hardware_backend.npu.quantization.awq_kernels import (
            AWQAscendMoEKernel,
        )
        return AWQAscendMoEKernel(quant_config)

    def create_moe_runner(
        self,
        layer: torch.nn.Module,
        moe_runner_config: "MoeRunnerConfig",
        **extra_weight_attrs,
    ):
        # Keep only the config; no need to instantiate runners
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Ensure topk_ids is int32 and weights match dtype
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        # Direct call to NPU fused expert kernel
        output = self.npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_qweight,          # already processed by AWQAscendMoEKernel
            w13_scale=layer.w13_scales,
            w13_offset=layer.w13_qzeros,    # note: qzeros now holds the offset (8 - zero)
            w2=layer.w2_qweight,
            w2_scale=layer.w2_scales,
            w2_offset=layer.w2_qzeros,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
            use_wna16=True,
        )

        return StandardCombineInput(hidden_states=output)


    def npu_fused_experts(
        self,
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
