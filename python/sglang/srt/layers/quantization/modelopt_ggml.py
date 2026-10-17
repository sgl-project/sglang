# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/9cede34a674c392092134e34218b127d6bbb3679/vllm/model_executor/layers/quantization/modelopt_block_quant.py
"""Load ModelOpt safetensors blocks with the existing GGUF kernels.

Supported formats:
- Linear layers: Q8_0 and IQ2_XXS.
- MoE experts: IQ2_XXS, with gated SiLU or ungated ReLU2.

Weights use [N, K / block_size, block_bytes] per linear layer or expert.
Gated experts store gate/up in separate w13 halves. TP keeps blocks intact.
EP, EPLB, fused shared experts and presharded expert weights are unsupported.
Q8_0 experts are rejected. Other GGML codecs are not enabled by this adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


def ggml_block_layout(quant_algo: str) -> tuple[int, int, int]:
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

    quant_type = GGMLQuantizationType[quant_algo]
    block_size, block_bytes = GGML_QUANT_SIZES[quant_type]
    return block_size, block_bytes, int(quant_type)


def validate_ggml_recipe(quant_algo: str, recipe: dict[str, Any]) -> None:
    block_size, block_bytes, _ = ggml_block_layout(quant_algo)
    if (
        recipe.get("group_size") != block_size
        or recipe.get("block_payload_bytes") != block_bytes
        or recipe.get("packing") != "ggml"
    ):
        raise ValueError(f"Invalid {quant_algo} GGML block layout: {recipe}")


class ModelOptGGMLLinearMethod(LinearMethodBase):
    """Retain checkpoint blocks and the standard linear weight-loader contract."""

    def __init__(self, quant_algo: str) -> None:
        self.block_size, self.block_bytes, self.quant_type = ggml_block_layout(
            quant_algo
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        if input_size_per_partition % self.block_size:
            raise ValueError(
                "GGML input partition must contain whole quantization blocks"
            )
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition // self.block_size,
                    self.block_bytes,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=extra_weight_attrs["weight_loader"],
            ),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf

        output = fused_mul_mat_gguf(
            x.reshape(-1, x.shape[-1]),
            layer.weight.flatten(1),
            self.quant_type,
        )
        if bias is not None:
            output.add_(bias)
        return output.view(*x.shape[:-1], layer.weight.shape[0])


class ModelOptGGMLMoEMethod(FusedMoEMethodBase):
    """Packed expert blocks with tensor-parallel loading and GGUF execution."""

    # GGUF applies the routing weights during the final expert reduction.
    fuse_routed_scaling_factor_in_topk = True

    def __init__(self, quant_algo: str) -> None:
        self.block_size, self.block_bytes, self.quant_type = ggml_block_layout(
            quant_algo
        )

    def create_weights(
        self,
        layer: FusedMoE,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

        if (
            layer.moe_ep_size != 1
            or get_global_expert_location_metadata() is not None
            or layer.num_fused_shared_experts
            or layer.use_presharded_weights
        ):
            raise ValueError(
                "ModelOpt GGML experts require TP without EP, EPLB, fused shared "
                "experts, or presharded weights"
            )
        if extra_weight_attrs.get("with_bias", False):
            raise ValueError("ModelOpt GGML experts do not support bias")
        if (
            hidden_size % self.block_size
            or intermediate_size_per_partition % self.block_size
        ):
            raise ValueError(
                "GGML expert partitions must contain whole quantization blocks"
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        self.tp_rank = layer.moe_tp_rank
        self.tp_size = layer.moe_tp_size
        self.is_gated = layer.moe_runner_config.is_gated
        for name, rows, columns in (
            (
                "w13_weight",
                (2 if self.is_gated else 1) * self.intermediate_size,
                hidden_size,
            ),
            ("w2_weight", hidden_size, self.intermediate_size),
        ):
            layer.register_parameter(
                name,
                ModelWeightParameter(
                    data=torch.empty(
                        num_experts,
                        rows,
                        columns // self.block_size,
                        self.block_bytes,
                        dtype=torch.uint8,
                    ),
                    input_dim=2,
                    output_dim=1,
                    weight_loader=self.weight_loader,
                ),
            )

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int | None,
        return_success: bool = False,
    ) -> bool | None:
        if shard_id not in ("w1", "w2", "w3") or (
            not self.is_gated and shard_id == "w3"
        ):
            raise ValueError(f"Invalid GGML expert projection {shard_id}")
        if expert_id is None or not 0 <= expert_id < param.shape[0]:
            raise ValueError(f"Invalid GGML expert index {expert_id}")
        global_intermediate = self.intermediate_size * self.tp_size
        expected = (
            (self.hidden_size, global_intermediate // self.block_size, self.block_bytes)
            if shard_id == "w2"
            else (
                global_intermediate,
                self.hidden_size // self.block_size,
                self.block_bytes,
            )
        )
        if loaded_weight.dtype != torch.uint8 or tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"Invalid packed tensor {weight_name}: expected uint8{expected}, "
                f"got {loaded_weight.dtype}{tuple(loaded_weight.shape)}"
            )

        destination = param.data[expert_id]
        if shard_id == "w2":
            size = self.intermediate_size // self.block_size
            source = loaded_weight.narrow(1, self.tp_rank * size, size)
        else:
            size = self.intermediate_size
            source = loaded_weight.narrow(0, self.tp_rank * size, size)
            destination = destination.narrow(0, size if shard_id == "w3" else 0, size)
        destination.copy_(source)
        return True if return_success else None

    def create_moe_runner(
        self, layer: FusedMoE, moe_runner_config: MoeRunnerConfig
    ) -> None:
        if (moe_runner_config.is_gated and moe_runner_config.activation != "silu") or (
            not moe_runner_config.is_gated and moe_runner_config.activation != "relu2"
        ):
            raise ValueError(
                "ModelOpt GGML experts require gated SiLU or ungated ReLU2; "
                f"got activation={moe_runner_config.activation!r}, "
                f"is_gated={moe_runner_config.is_gated}"
            )
        if moe_runner_config.no_combine:
            raise ValueError("ModelOpt GGML experts do not support no_combine=True")
        if moe_runner_config.apply_router_weight_on_input:
            raise ValueError(
                "ModelOpt GGML experts do not support applying router weights to inputs"
            )
        custom_parameters = [
            name
            for name in (
                "gemm1_alpha",
                "gemm1_beta",
                "gemm1_clamp_limit",
                "swiglu_limit",
            )
            if getattr(moe_runner_config, name) is not None
        ]
        if custom_parameters:
            raise ValueError(
                "ModelOpt GGML experts do not support custom activation parameters: "
                + ", ".join(custom_parameters)
            )
        self.moe_runner_config = moe_runner_config

    def apply(
        self, layer: FusedMoE, dispatch_output: StandardDispatchOutput
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.quantization.gguf import fused_moe_gguf

        topk_weights, topk_ids, _ = dispatch_output.topk_output
        output = fused_moe_gguf(
            x=dispatch_output.hidden_states,
            w1=layer.w13_weight.flatten(-2),
            w2=layer.w2_weight.flatten(-2),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            qweight_type=self.quant_type,
            qweight_type2=self.quant_type,
            activation=self.moe_runner_config.activation,
            is_gated=self.moe_runner_config.is_gated,
        )
        return StandardCombineInput(hidden_states=output)
