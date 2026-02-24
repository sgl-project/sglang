from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from compressed_tensors import CompressionFormat

from sglang.srt.distributed import get_moe_expert_parallel_rank, get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.utils import RoutingMethodType, get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.utils import replace_parameter
from sglang.srt.utils import is_flashinfer_available, next_power_of_2, set_weight_attrs

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsMxInt4MoE"]

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

if is_flashinfer_available():
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe import (
        convert_to_block_layout,
        trtllm_mxint4_block_scale_moe,
    )
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )


class CompressedTensorsMxInt4MoE(CompressedTensorsMoEScheme):
    def __init__(self, quant_config: CompressedTensorsConfig):
        self.quant_config = quant_config
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        self.group_size = config.group_size
        self.actorder = config.actorder
        assert (
            config.strategy == "group"
            and config.group_size == 32
            and config.num_bits == 4
        ), "MxInt4 only supports group strategy with group size 32"
        assert config.symmetric, "Only symmetric quantization is supported for MoE"
        assert (
            get_moe_runner_backend().is_flashinfer_trtllm()
        ), "MxInt4 only supports flashinfer_trtllm backend"
        assert (
            not config.actorder
        ), "Actorder is not supported by flashinfer_trtllm backend"
        self.moe_ep_rank = get_moe_expert_parallel_rank()

        if self.quant_config.quant_format != CompressionFormat.pack_quantized.value:
            raise ValueError(
                f"For Fused MoE layers, only {CompressionFormat.pack_quantized.value} "
                "is supported for the mxint4"
            )
        self._cache_permute_indices = {}

    @classmethod
    def get_min_capability(cls) -> int:
        # Requires sm100(blackwell) architecture
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert (
            params_dtype == torch.bfloat16
        ), f"Params dtype should be torch.bfloat16, but got: {params_dtype}"

        extra_weight_attrs.update({"quant_method": self.strategy})
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.packed_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.packed_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_scales_size = intermediate_size_per_partition
        num_groups_w2 = w2_scales_size // self.group_size
        num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                num_groups_w13,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, num_groups_w2, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)

        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    # Adapted from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/moe/test_trtllm_gen_fused_moe.py
    def prepare_static_weights_for_kernel(
        self,
        gemm1_weights,
        gemm2_weights,
        gemm1_scales,
        gemm2_scales,
        num_experts,
    ):
        """Prepare quantized weights for kernel (done offline with weights)."""

        epilogue_tile_m = 128
        gemm1_weights_mxint4_shuffled = []
        gemm1_scales_shuffled = []
        gemm2_weights_mxint4_shuffled = []
        gemm2_scales_shuffled = []

        def repack(w):
            assert w.dim() == 2 and w.dtype == torch.int32
            shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=w.device)
            w = (w.unsqueeze(2) >> shifts) & 0x0F
            w = (w - 8).to(torch.int8).reshape(w.shape[0], -1, 2)
            w = (w[..., 0] & 0x0F) | ((w[..., 1] & 0x0F) << 4)
            w = w.to(torch.uint8)
            return w

        for i in range(num_experts):
            # NOTE(HandH1998):
            # the huggingface weight format follows (w/s + 8) to pack,
            # however, trtllm requires (w/s) to pack
            # we need to convert the weight to trtllm's format first
            cur_expert_gemm1_weight = repack(gemm1_weights[i])
            cur_expert_gemm2_weight = repack(gemm2_weights[i])

            # Calculate the permute indices for the following:
            # 1. Reorder rows of W1 and scales for fused gated activation
            # 2. Shuffle weights and scaling factors for transposed mma output
            # for both w3_w1 and w2 weights and scale factors
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                cur_expert_gemm1_weight,
                epilogue_tile_m,
            )
            gemm1_weights_shuffled = cur_expert_gemm1_weight[
                permute_indices.to(gemm1_weights.device)
            ].contiguous()
            permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_scales[i].to(torch.bfloat16),
                epilogue_tile_m,
                num_elts_per_sf=32,
            )
            gemm1_scales_shuffled.append(
                block_scale_interleave(
                    gemm1_scales[i]
                    .to(torch.bfloat16)[permute_sf_indices.to(gemm1_scales.device)]
                    .contiguous()
                )
            )

            permute_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                cur_expert_gemm2_weight,
                epilogue_tile_m,
            )
            gemm2_weights_shuffled = cur_expert_gemm2_weight[
                permute_indices.to(gemm2_weights.device)
            ].contiguous()

            permute_sf_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_scales[i].to(torch.bfloat16),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_shuffled.append(
                block_scale_interleave(
                    gemm2_scales[i]
                    .to(torch.bfloat16)[permute_sf_indices.to(gemm2_scales.device)]
                    .contiguous()
                )
            )

            block_k = 128
            gemm1_weights_shuffled = convert_to_block_layout(
                gemm1_weights_shuffled.view(torch.uint8), block_k
            )
            gemm2_weights_shuffled = convert_to_block_layout(
                gemm2_weights_shuffled.view(torch.uint8), block_k
            )

            gemm1_weights_mxint4_shuffled.append(gemm1_weights_shuffled)
            gemm2_weights_mxint4_shuffled.append(gemm2_weights_shuffled)

        gemm1_weights_mxint4_shuffled = torch.stack(gemm1_weights_mxint4_shuffled)
        gemm2_weights_mxint4_shuffled = torch.stack(gemm2_weights_mxint4_shuffled)
        gemm1_scales_shuffled = torch.stack(gemm1_scales_shuffled).view(torch.bfloat16)
        gemm2_scales_shuffled = torch.stack(gemm2_scales_shuffled).view(torch.bfloat16)

        return (
            gemm1_weights_mxint4_shuffled,
            gemm1_scales_shuffled,
            gemm2_weights_mxint4_shuffled,
            gemm2_scales_shuffled,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        num_experts = layer.w13_weight_packed.shape[0]
        (
            gemm1_weights_mxint4_shuffled,
            gemm1_scales_shuffled,
            gemm2_weights_mxint4_shuffled,
            gemm2_scales_shuffled,
        ) = self.prepare_static_weights_for_kernel(
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            num_experts=num_experts,
        )
        replace_parameter(layer, "w13_weight_packed", gemm1_weights_mxint4_shuffled)
        replace_parameter(layer, "w2_weight_packed", gemm2_weights_mxint4_shuffled)
        replace_parameter(layer, "w13_weight_scale", gemm1_scales_shuffled)
        replace_parameter(layer, "w2_weight_scale", gemm2_scales_shuffled)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.is_gated
        ), "Only gated MoEs are supported for flashinfer mxint4"

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        router_logits = topk_output.router_logits
        topk_config = topk_output.topk_config
        correction_bias = (
            None
            if topk_config.correction_bias is None
            else topk_config.correction_bias.to(x.dtype)
        )

        local_num_experts = self.moe_runner_config.num_local_experts
        routing_method_type = layer.routing_method_type
        assert routing_method_type is not None
        # DeepSeekV3 style routing requires float32 router logits,
        # see this PR for details: https://github.com/flashinfer-ai/flashinfer/commit/d84e1d560da0a27961c19ca788d96c19cb9dcfb6
        if routing_method_type == RoutingMethodType.DeepSeekV3:
            router_logits = router_logits.to(torch.float32)
        routed_scaling_factor = self.moe_runner_config.routed_scaling_factor
        routed_scaling_factor = (
            routed_scaling_factor if routed_scaling_factor is not None else 1.0
        )

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            num_tokens = x.shape[0]
            hidden_size = x.shape[-1]
            symm_output = torch.empty(
                num_tokens, hidden_size, dtype=torch.bfloat16, device=x.device
            )

        output = trtllm_mxint4_block_scale_moe(
            routing_logits=router_logits,  # float
            routing_bias=correction_bias,
            hidden_states=x,
            gemm1_weights=layer.w13_weight_packed,
            gemm1_weights_scale=layer.w13_weight_scale,
            gemm1_alpha=self.moe_runner_config.gemm1_alpha,
            gemm1_beta=None,
            gemm1_clamp_limit=self.moe_runner_config.gemm1_clamp_limit,
            gemm2_weights=layer.w2_weight_packed,
            gemm2_weights_scale=layer.w2_weight_scale,
            num_experts=self.moe_runner_config.num_experts,
            top_k=topk_config.top_k,
            n_group=topk_config.num_expert_group,
            topk_group=topk_config.topk_group,
            intermediate_size=self.moe_runner_config.intermediate_size_per_partition,
            local_expert_offset=self.moe_ep_rank * local_num_experts,
            local_num_experts=local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=routing_method_type,
            tune_max_num_tokens=next_power_of_2(x.shape[0]),
            output=symm_output,
        )

        return StandardCombineInput(hidden_states=output)
