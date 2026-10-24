# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precision and initial-load checks for native K2 Horizon MoVA FP8 weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PretrainedConfig

    from sglang.srt.layers.quantization.base_config import QuantizationConfig


def _excluded_mova_modules(config: PretrainedConfig) -> set[str]:
    required = {"lm_head", "embed_tokens", "norm"}
    for layer_id in range(config.num_hidden_layers):
        prefix = f"layers.{layer_id}"
        required.update(
            f"{prefix}.{name}"
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.o_proj",
                "self_attn.gate_proj",
            )
        )
        if layer_id < config.num_dense_layers:
            required.add(f"{prefix}.self_attn.v_proj")
            mlp = f"{prefix}.mlp"
        else:
            required.update((f"{prefix}.self_attn.v_router", f"{prefix}.mlp.gate"))
            required.update(
                f"{prefix}.self_attn.v_experts.{expert}"
                for expert in range(config.num_values)
            )
            if not config.num_shared_experts:
                continue
            mlp = f"{prefix}.mlp.shared_experts"
        required.update(
            f"{mlp}.{projection}"
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
    return required


def _normalized_ignored_modules(names: object) -> set[str]:
    if not isinstance(names, list) or any(type(name) is not str for name in names):
        raise ValueError("MoVA FP8 requires explicit source module exclusions")
    return {name.removeprefix("model.") for name in names}


def _has_mova_fp8_blocks(blocks: object) -> bool:
    return (
        type(blocks) is list
        and len(blocks) == 2
        and all(type(width) is int and width == 128 for width in blocks)
    )


def validate_mova_fp8(
    config: PretrainedConfig,
    quant_config: QuantizationConfig,
    *,
    moe_tp_size: int,
    moe_ep_size: int,
) -> None:
    """Require the declared and effective expert-only FP8 precision partition."""
    import torch

    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.layers.quantization.utils import is_layer_skipped

    metadata = getattr(config, "quantization_config", None)
    if (
        getattr(config, "model_type", None) != "k2_horizon"
        or getattr(config, "_sglang_xllm_checkpoint_format", None) != "k2_horizon_hf"
        or getattr(config, "num_values", 0) <= 0
        or not isinstance(metadata, dict)
        or metadata.get("quant_method") != "fp8"
        or metadata.get("activation_scheme") != "dynamic"
        or not _has_mova_fp8_blocks(metadata.get("weight_block_size"))
        or type(quant_config) is not Fp8Config
        or quant_config.is_checkpoint_fp8_serialized is not True
        or quant_config.activation_scheme != "dynamic"
        or not _has_mova_fp8_blocks(quant_config.weight_block_size)
        or quant_config.use_mxfp8
        or quant_config.is_fp4_experts
        or quant_config.dequant_fp4_to_fp8
        or torch.get_default_dtype() != torch.bfloat16
    ):
        raise ValueError(
            "Native MoVA requires serialized dynamic block FP8 with BF16 exclusions"
        )
    if (
        type(moe_tp_size) is not int
        or moe_tp_size <= 0
        or type(moe_ep_size) is not int
        or moe_ep_size <= 0
        or config.num_experts % moe_ep_size
        or config.moe_intermediate_size % (128 * moe_tp_size)
        or config.hidden_size % 128
    ):
        raise ValueError(
            "MoVA FP8 requires complete expert and 128-wide block partitions"
        )
    required = _excluded_mova_modules(config)
    if (
        _normalized_ignored_modules(metadata.get("ignored_layers")) != required
        or _normalized_ignored_modules(quant_config.ignored_layers) != required
    ):
        raise ValueError("MoVA FP8 requires the complete source precision exclusions")
    for layer_id in range(config.num_hidden_layers):
        mlp = f"model.layers.{layer_id}.mlp"
        if layer_id >= config.num_dense_layers:
            if is_layer_skipped(
                f"{mlp}.experts",
                quant_config.ignored_layers,
                fused_mapping=quant_config.packed_modules_mapping,
            ):
                raise ValueError("MoVA routed FFN experts must remain quantized")
            if not config.num_shared_experts:
                continue
            mlp += ".shared_experts"
        if not is_layer_skipped(
            f"{mlp}.gate_up_proj",
            quant_config.ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
        ):
            raise ValueError("MoVA dense and shared FFNs must remain unquantized")


def mova_fp8_source_shapes(
    config: PretrainedConfig,
    *,
    start_layer: int,
    end_layer: int,
    is_first_rank: bool,
    is_last_rank: bool,
    owned_experts: dict[int, set[int]],
) -> dict[str, tuple[int, ...] | None]:
    """Describe canonical sources; None marks an existing remote or tied omission."""
    hidden = config.hidden_size
    head_dim = getattr(config, "head_dim", hidden // config.num_attention_heads)
    query_width = config.num_attention_heads * head_dim
    value_width = config.num_key_value_heads * head_dim
    shapes = {}
    if is_first_rank or (is_last_rank and config.tie_word_embeddings):
        shapes["model.embed_tokens.weight"] = (config.vocab_size, hidden)
    if is_last_rank:
        shapes["model.norm.weight"] = (hidden,)
        shapes["lm_head.weight"] = (
            None if config.tie_word_embeddings else (config.vocab_size, hidden)
        )
    for layer_id in range(start_layer, end_layer):
        prefix = f"model.layers.{layer_id}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[f"{prefix}.{norm}.weight"] = (hidden,)
        for projection in ("q_proj", "gate_proj"):
            shapes[f"{prefix}.self_attn.{projection}.weight"] = (query_width, hidden)
        shapes[f"{prefix}.self_attn.k_proj.weight"] = (value_width, hidden)
        shapes[f"{prefix}.self_attn.o_proj.weight"] = (hidden, query_width)
        dense = layer_id < config.num_dense_layers
        if dense:
            shapes[f"{prefix}.self_attn.v_proj.weight"] = (value_width, hidden)
        else:
            for expert in range(config.num_values):
                shapes[f"{prefix}.self_attn.v_experts.{expert}.weight"] = (
                    value_width,
                    hidden,
                )
            for router, count in (
                ("self_attn.v_router", config.num_values),
                ("mlp.gate", config.num_experts),
            ):
                shapes[f"{prefix}.{router}.weight"] = (count, hidden)
                if config.moe_gate_bias:
                    shapes[f"{prefix}.{router}.bias"] = (count,)
            for expert in range(config.num_experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    name = f"{prefix}.mlp.experts.{expert}.{projection}"
                    shape = (
                        (hidden, config.moe_intermediate_size)
                        if projection == "down_proj"
                        else (config.moe_intermediate_size, hidden)
                    )
                    local = expert in owned_experts[layer_id]
                    shapes[f"{name}.weight"] = shape if local else None
                    shapes[f"{name}.weight_scale_inv"] = (
                        tuple(width // 128 for width in shape) if local else None
                    )
        if dense or config.num_shared_experts:
            mlp = f"{prefix}.mlp" + ("" if dense else ".shared_experts")
            width = (
                config.intermediate_size
                if dense
                else config.moe_intermediate_size * config.num_shared_experts
            )
            for projection in ("gate_proj", "up_proj", "down_proj"):
                shapes[f"{mlp}.{projection}.weight"] = (
                    (hidden, width) if projection == "down_proj" else (width, hidden)
                )
    return shapes


def accept_mova_fp8_weight(
    name: str,
    tensor: torch.Tensor,
    source_shapes: dict[str, tuple[int, ...] | None],
    pending: set[str],
) -> bool:
    """Check one owned source before loading it and update the completion ledger."""
    import torch

    if name not in source_shapes:
        raise ValueError(f"MoVA FP8 source tensor is unsupported: {name}")
    shape = source_shapes[name]
    if shape is None:
        return False
    if name not in pending:
        raise ValueError(f"MoVA FP8 source tensor is duplicated: {name}")
    scale = name.endswith(".weight_scale_inv")
    dtype = torch.bfloat16
    if scale:
        dtype = torch.float32
    elif ".mlp.experts." in name:
        dtype = torch.float8_e4m3fn
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(
            f"MoVA FP8 source tensor has an invalid dtype or shape: {name}"
        )
    if scale and not bool((torch.isfinite(tensor) & (tensor > 0)).all()):
        raise ValueError(f"MoVA FP8 source scales must be positive and finite: {name}")
    pending.remove(name)
    return True
