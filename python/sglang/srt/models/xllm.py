# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py
# for the xLLM K2MoE architecture.
# Key differences from Qwen2Moe:
#   - Sigmoid routing (not softmax)
#   - Gate bias used for expert selection only (correction_bias pattern)
#   - Router scaling factor applied after renormalization
#   - No shared_expert_gate (shared expert output added directly)
#   - Dense layers specified via mlp_only_layers config
#   - Partial RoPE (rope_head_dim < head_dim)
"""Inference-only xLLM K2MoE and MoVA models compatible with HF weights."""

import math
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, tensor_model_parallel_all_reduce
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
)
from sglang.srt.layers.mova import RoutedValueExperts, mova_router_topk
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import add_prefix, make_layers

_XLLM_SOURCE_ROUTER_PARTITIONS_CONFIG_KEY = "xllm_source_router_gemm_partitions"
_XLLM_SOURCE_ROUTER_PARTITIONS_MISSING = object()
_XLLM_CHECKPOINT_FORMAT_CONFIG_KEY = "_sglang_xllm_checkpoint_format"
_K2_HORIZON_HF_CHECKPOINT_FORMAT = "k2_horizon_hf"
_CONFIG_ATTR_MISSING = object()


class XllmGroupRMSNorm(nn.Module):
    """Reference grouped RMSNorm used by the xLLM model family."""

    def __init__(
        self,
        hidden_size: int,
        n_groups: int = 1,
        eps: float = 1e-6,
        zero_centered: bool = False,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.hidden_size = hidden_size
        if n_groups <= 0 or hidden_size % n_groups:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by n_groups={n_groups}"
            )
        self.variance_epsilon = eps
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(
            torch.zeros(hidden_size) if zero_centered else torch.ones(hidden_size)
        )

    def forward(self, hidden_states, residual=None, post_residual_addition=None):
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states
        if post_residual_addition is not None:
            hidden_states = hidden_states + post_residual_addition
            residual = hidden_states
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states.reshape(
            *hidden_states.shape[:-1], self.n_groups, -1
        )
        hidden_states = hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        )
        hidden_states = hidden_states.reshape(*hidden_states.shape[:-2], -1)
        weight = self.weight + 1.0 if self.zero_centered else self.weight
        hidden_states = (weight * hidden_states).to(orig_dtype)
        if residual is not None:
            return hidden_states, residual
        return hidden_states


def _is_k2_horizon_hf_checkpoint(config: PretrainedConfig) -> bool:
    return (
        getattr(config, _XLLM_CHECKPOINT_FORMAT_CONFIG_KEY, None)
        == _K2_HORIZON_HF_CHECKPOINT_FORMAT
    )


def _set_k2_horizon_alias(
    config: PretrainedConfig,
    *,
    source_name: str,
    target_name: str,
    value: Any,
) -> None:
    """Set one explicit K2Horizon schema alias, rejecting contradictions."""

    current = getattr(config, target_name, _CONFIG_ATTR_MISSING)
    if current is not _CONFIG_ATTR_MISSING and current is not None:
        if current != value:
            raise ValueError(
                f"K2Horizon config has conflicting {source_name}={value!r} "
                f"and {target_name}={current!r}"
            )
        return
    setattr(config, target_name, value)


def _normalize_k2_horizon_config(config: PretrainedConfig) -> None:
    """Translate the canonical K2Horizon HF schema to the native xLLM path.

    This adapter intentionally maps only fields that K2Horizon spells
    differently. In particular, source router GEMM topology is provenance,
    not an architecture property, so it must be supplied explicitly.
    """

    # Dense K2Horizon artifacts may omit the MoVA fields entirely, while the
    # remote config class supplies zero defaults. Treat both representations
    # identically, but keep malformed/negative values distinct from dense.
    mova_num_experts = getattr(config, "mova_num_experts", 0)
    if isinstance(mova_num_experts, bool) or not isinstance(mova_num_experts, int):
        raise ValueError(
            "K2Horizon mova_num_experts must be a non-negative integer, "
            f"got {mova_num_experts!r}."
        )
    if mova_num_experts < 0:
        raise ValueError(
            "K2Horizon mova_num_experts must be a non-negative integer, "
            f"got {mova_num_experts!r}."
        )
    is_mova = mova_num_experts > 0
    num_experts = getattr(config, "num_experts", 0)
    if (
        isinstance(num_experts, bool)
        or not isinstance(num_experts, int)
        or num_experts < 0
    ):
        raise ValueError(
            "K2Horizon num_experts must be a non-negative integer, "
            f"got {num_experts!r}."
        )
    has_moe_ffn = num_experts > 0

    if is_mova:
        if _get_xllm_source_router_gemm_partitions(config) is None:
            raise ValueError(
                "K2Horizon MoVA requires explicit source router GEMM provenance; "
                "SGLang will not infer it from runtime tensor parallelism. After "
                "confirming the training contract, pass "
                "--json-model-override-args "
                "'{\"xllm_source_router_gemm_partitions\": 2}' (use 1 only for "
                "a confirmed MP1 source checkpoint)."
            )

        _set_k2_horizon_alias(
            config,
            source_name="mova_num_experts",
            target_name="num_values",
            value=mova_num_experts,
        )
        mova_num_experts_per_tok = getattr(
            config, "mova_num_experts_per_tok", _CONFIG_ATTR_MISSING
        )
        if (
            isinstance(mova_num_experts_per_tok, bool)
            or not isinstance(mova_num_experts_per_tok, int)
            or not 0 < mova_num_experts_per_tok <= mova_num_experts
        ):
            raise ValueError(
                "K2Horizon mova_num_experts_per_tok must be a positive integer no "
                f"larger than mova_num_experts, got {mova_num_experts_per_tok!r}"
            )
        _set_k2_horizon_alias(
            config,
            source_name="mova_num_experts_per_tok",
            target_name="num_values_per_tok",
            value=mova_num_experts_per_tok,
        )
    else:
        mova_num_experts_per_tok = getattr(config, "mova_num_experts_per_tok", 0)
        if (
            isinstance(mova_num_experts_per_tok, bool)
            or not isinstance(mova_num_experts_per_tok, int)
            or mova_num_experts_per_tok != 0
        ):
            raise ValueError(
                "Dense K2Horizon requires mova_num_experts_per_tok=0, got "
                f"{mova_num_experts_per_tok!r}"
            )
        _set_k2_horizon_alias(
            config,
            source_name="mova_num_experts",
            target_name="num_values",
            value=0,
        )
        _set_k2_horizon_alias(
            config,
            source_name="mova_num_experts_per_tok",
            target_name="num_values_per_tok",
            value=0,
        )
        if not has_moe_ffn:
            for field in (
                "num_experts",
                "num_experts_per_tok",
                "num_shared_experts",
            ):
                value = getattr(config, field, 0)
                if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                    raise ValueError(
                        f"Dense K2Horizon requires {field}=0, got {value!r}"
                    )
                # Some dense exports omit the MoE-only fields. Downstream model
                # construction reads them directly, so materialize the validated
                # dense defaults instead of relying on getattr fallbacks forever.
                setattr(config, field, 0)
        if getattr(config, "query_key_norm", False):
            raise ValueError(
                "Dense K2Horizon native loading does not support query/key "
                "normalization"
            )
        if getattr(config, "sliding_window", None) is not None or getattr(
            config, "use_sliding_window", False
        ):
            raise ValueError(
                "Dense K2Horizon native loading supports full causal attention only"
            )
        attention_gate_func = getattr(
            config, "attention_gate_func", _CONFIG_ATTR_MISSING
        )
        native_gate_func = getattr(config, "attn_gate_func", _CONFIG_ATTR_MISSING)
        if (
            attention_gate_func not in (_CONFIG_ATTR_MISSING, None)
            or native_gate_func not in (_CONFIG_ATTR_MISSING, None)
            or getattr(config, "apply_attn_gate", False)
        ):
            raise ValueError(
                "Dense K2Horizon native loading does not support gated attention"
            )

    attention_gate_func = getattr(config, "attention_gate_func", _CONFIG_ATTR_MISSING)
    if attention_gate_func is not _CONFIG_ATTR_MISSING:
        _set_k2_horizon_alias(
            config,
            source_name="attention_gate_func",
            target_name="attn_gate_func",
            value=attention_gate_func,
        )
        _set_k2_horizon_alias(
            config,
            source_name="attention_gate_func",
            target_name="apply_attn_gate",
            value=attention_gate_func is not None,
        )

    rope_parameters = getattr(config, "rope_parameters", _CONFIG_ATTR_MISSING)
    if rope_parameters is not _CONFIG_ATTR_MISSING and rope_parameters is not None:
        if not isinstance(rope_parameters, dict):
            raise ValueError(
                "K2Horizon rope_parameters must be a dictionary, got "
                f"{type(rope_parameters).__name__}"
            )
        rope_type = rope_parameters.get(
            "rope_type", rope_parameters.get("type", _CONFIG_ATTR_MISSING)
        )
        if is_mova and rope_type != "default":
            raise ValueError(
                "K2Horizon direct loading supports only explicit default "
                f"rope_parameters, got rope_type={rope_type!r}"
            )
        if not is_mova and rope_type not in ("default", "yarn"):
            raise ValueError(
                "Dense K2Horizon direct loading supports only explicit default "
                f"or yarn rope_parameters, got rope_type={rope_type!r}"
            )
        if (
            "rope_type" in rope_parameters
            and "type" in rope_parameters
            and rope_parameters["rope_type"] != rope_parameters["type"]
        ):
            raise ValueError(
                "K2Horizon rope_parameters has conflicting rope_type and type"
            )
        rope_theta = rope_parameters.get("rope_theta", _CONFIG_ATTR_MISSING)
        if rope_theta is _CONFIG_ATTR_MISSING:
            if is_mova:
                raise ValueError(
                    "K2Horizon default rope_parameters must explicitly provide "
                    "rope_theta"
                )
            # Dense K2Horizon YaRN artifacts generated during the TF5 config
            # transition persisted theta at the legacy top level only.
            rope_theta = getattr(config, "rope_theta", _CONFIG_ATTR_MISSING)
        if (
            rope_theta is _CONFIG_ATTR_MISSING
            or isinstance(rope_theta, bool)
            or not isinstance(rope_theta, (int, float))
            or not math.isfinite(rope_theta)
            or rope_theta <= 0
        ):
            raise ValueError(
                "K2Horizon rope_theta must be a positive finite number, got "
                f"{rope_theta!r}"
            )
        _set_k2_horizon_alias(
            config,
            source_name="rope_parameters.rope_theta",
            target_name="rope_theta",
            value=rope_theta,
        )
        default_rope_scaling = dict(rope_parameters)
        default_rope_scaling.pop("type", None)
        default_rope_scaling["rope_theta"] = rope_theta
        default_rope_scaling["rope_type"] = rope_type
        if rope_type == "yarn":
            supported_yarn_keys = {
                "attention_factor",
                "beta_fast",
                "beta_slow",
                "factor",
                "original_max_position_embeddings",
                "rope_theta",
                "rope_type",
                "truncate",
                "type",
            }
            unknown_yarn_keys = set(rope_parameters) - supported_yarn_keys
            if unknown_yarn_keys:
                raise ValueError(
                    "Dense K2Horizon YaRN has unsupported rope_parameters keys: "
                    f"{sorted(unknown_yarn_keys)}"
                )
            factor = rope_parameters.get("factor", _CONFIG_ATTR_MISSING)
            if (
                isinstance(factor, bool)
                or not isinstance(factor, (int, float))
                or not math.isfinite(factor)
                or factor <= 0
            ):
                raise ValueError(
                    "Dense K2Horizon YaRN factor must be positive and finite, "
                    f"got {factor!r}"
                )
            original_max_position_embeddings = rope_parameters.get(
                "original_max_position_embeddings", _CONFIG_ATTR_MISSING
            )
            if (
                isinstance(original_max_position_embeddings, bool)
                or not isinstance(original_max_position_embeddings, int)
                or original_max_position_embeddings <= 0
            ):
                raise ValueError(
                    "Dense K2Horizon YaRN original_max_position_embeddings must "
                    f"be a positive integer, got {original_max_position_embeddings!r}"
                )
            _set_k2_horizon_alias(
                config,
                source_name="rope_parameters.original_max_position_embeddings",
                target_name="original_max_position_embeddings",
                value=original_max_position_embeddings,
            )
            max_position_embeddings = getattr(
                config, "max_position_embeddings", _CONFIG_ATTR_MISSING
            )
            expected_max_position_embeddings = factor * original_max_position_embeddings
            if (
                isinstance(max_position_embeddings, bool)
                or not isinstance(max_position_embeddings, int)
                or not math.isclose(
                    max_position_embeddings,
                    expected_max_position_embeddings,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "Dense K2Horizon YaRN requires max_position_embeddings == "
                    "factor * original_max_position_embeddings; got "
                    f"{max_position_embeddings!r} != "
                    f"{expected_max_position_embeddings!r}"
                )
            for field, default in (("beta_fast", 32), ("beta_slow", 1)):
                value = rope_parameters.get(field, default)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        f"Dense K2Horizon YaRN {field} must be positive and "
                        f"finite, got {value!r}"
                    )
            truncate = rope_parameters.get("truncate", True)
            if not isinstance(truncate, bool):
                raise ValueError(
                    f"Dense K2Horizon YaRN truncate must be a bool, got {truncate!r}"
                )
            attention_factor = default_rope_scaling.pop("attention_factor", None)
            if attention_factor is not None:
                if (
                    isinstance(attention_factor, bool)
                    or not isinstance(attention_factor, (int, float))
                    or not math.isfinite(attention_factor)
                    or attention_factor <= 0
                ):
                    raise ValueError(
                        "Dense K2Horizon YaRN attention_factor must be positive "
                        f"and finite, got {attention_factor!r}"
                    )
                # HF's attention_factor is the final multiplier applied to
                # cos/sin. SGLang's attn_factor multiplies its own standard
                # YaRN mscale, so translate between those two conventions.
                default_attention_factor = (
                    1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
                )
                default_rope_scaling["attn_factor"] = (
                    attention_factor / default_attention_factor
                )
        current_rope_scaling = getattr(config, "rope_scaling", _CONFIG_ATTR_MISSING)
        # Transformers 5 exposes rope_scaling as a property alias for the
        # original rope_parameters dictionary. That is the same source field,
        # not a second independently specified value.
        if current_rope_scaling == rope_parameters:
            setattr(config, "rope_scaling", default_rope_scaling)
        else:
            _set_k2_horizon_alias(
                config,
                source_name="rope_parameters",
                target_name="rope_scaling",
                value=default_rope_scaling,
            )
    elif not is_mova:
        raise ValueError(
            "Dense K2Horizon native loading requires explicit rope_parameters"
        )

    mlp_only_layers = getattr(config, "mlp_only_layers", _CONFIG_ATTR_MISSING)
    if mlp_only_layers is not _CONFIG_ATTR_MISSING:
        if not isinstance(mlp_only_layers, (list, tuple)) or any(
            isinstance(layer_id, bool) or not isinstance(layer_id, int)
            for layer_id in mlp_only_layers
        ):
            raise ValueError("K2Horizon mlp_only_layers must be a list of integers")
        expected_prefix = list(range(len(mlp_only_layers)))
        if list(mlp_only_layers) != expected_prefix:
            raise ValueError(
                "K2Horizon MoVA requires mlp_only_layers to be a contiguous "
                f"prefix starting at zero, got {list(mlp_only_layers)}"
            )
        if not has_moe_ffn and list(mlp_only_layers) != list(
            range(config.num_hidden_layers)
        ):
            raise ValueError(
                "Dense K2Horizon native loading requires every layer in mlp_only_layers"
            )
        _set_k2_horizon_alias(
            config,
            source_name="mlp_only_layers",
            target_name="num_dense_layers",
            value=len(mlp_only_layers),
        )
    elif not has_moe_ffn:
        raise ValueError(
            "Dense K2Horizon native loading requires explicit mlp_only_layers"
        )

    current_format = getattr(
        config, _XLLM_CHECKPOINT_FORMAT_CONFIG_KEY, _CONFIG_ATTR_MISSING
    )
    if current_format not in (
        _CONFIG_ATTR_MISSING,
        _K2_HORIZON_HF_CHECKPOINT_FORMAT,
    ):
        raise ValueError(
            "K2Horizon native adapter requires checkpoint format "
            f"{_K2_HORIZON_HF_CHECKPOINT_FORMAT!r}, got {current_format!r}"
        )
    setattr(
        config,
        _XLLM_CHECKPOINT_FORMAT_CONFIG_KEY,
        _K2_HORIZON_HF_CHECKPOINT_FORMAT,
    )


def _make_norm(config):
    """Create the appropriate RMSNorm for this config."""
    n_groups = getattr(config, "layernorm_num_groups", 1)
    is_mova = getattr(config, "num_values", 0) > 0
    if (n_groups is None or n_groups <= 1) and not is_mova:
        return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    return XllmGroupRMSNorm(
        config.hidden_size,
        n_groups=n_groups or 1,
        eps=config.rms_norm_eps,
        # Converted xLLM MoVA stores zero-centered norm deltas. Canonical
        # K2Horizon HF stores ordinary one-centered RMSNorm weights directly.
        zero_centered=is_mova and not _is_k2_horizon_hf_checkpoint(config),
    )


def _get_xllm_source_router_gemm_partitions(
    config: PretrainedConfig,
) -> Optional[int]:
    """Read optional source-router provenance without inferring it from TP."""

    partitions = getattr(
        config,
        _XLLM_SOURCE_ROUTER_PARTITIONS_CONFIG_KEY,
        _XLLM_SOURCE_ROUTER_PARTITIONS_MISSING,
    )
    if partitions is _XLLM_SOURCE_ROUTER_PARTITIONS_MISSING:
        return None
    if (
        isinstance(partitions, bool)
        or not isinstance(partitions, int)
        or partitions not in (1, 2)
    ):
        raise ValueError(
            f"{_XLLM_SOURCE_ROUTER_PARTITIONS_CONFIG_KEY}={partitions!r} is "
            f"invalid (type={type(partitions).__name__}); when present it must "
            "be the integer 1 or 2. Omit the key to preserve legacy router "
            "GEMM behavior."
        )
    if config.hidden_size % partitions:
        raise ValueError(
            f"explicit {_XLLM_SOURCE_ROUTER_PARTITIONS_CONFIG_KEY}={partitions} "
            f"requires hidden_size divisible by {partitions}; got "
            f"hidden_size={config.hidden_size}"
        )
    return partitions


def _xllm_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    source_partitions: Optional[int],
) -> torch.Tensor:
    """Reproduce the source xLLM router GEMM's partition rounding contract."""

    # Old xLLM artifacts have no source-topology provenance. Preserve their
    # exact pre-contract behavior instead of guessing how the router was run.
    if source_partitions is None:
        return F.linear(hidden_states, weight)
    if isinstance(source_partitions, bool) or not isinstance(source_partitions, int):
        raise ValueError(
            "explicit xLLM router source partitions must be the integer 1 or "
            f"2; got {source_partitions!r} "
            f"(type={type(source_partitions).__name__})"
        )
    if source_partitions not in (1, 2):
        raise ValueError(
            "explicit xLLM router source partitions must be 1 or 2, got "
            f"{source_partitions}"
        )
    if hidden_states.ndim < 1 or weight.ndim != 2:
        raise ValueError(
            "xLLM router GEMM expects input [..., hidden] and weight "
            f"[routes, hidden]; got input={tuple(hidden_states.shape)}, "
            f"weight={tuple(weight.shape)}"
        )
    if hidden_states.shape[-1] != weight.shape[-1]:
        raise ValueError(
            "xLLM router input and weight hidden dimensions differ; got "
            f"input={tuple(hidden_states.shape)}, weight={tuple(weight.shape)}"
        )
    if hidden_states.shape[-1] % source_partitions:
        raise ValueError(
            f"explicit xLLM router partitions={source_partitions} requires "
            f"hidden size divisible by {source_partitions}; got hidden_size="
            f"{hidden_states.shape[-1]}"
        )
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError(
            "Explicit xLLM source router GEMM provenance requires BF16 input "
            f"and weight; got input={hidden_states.dtype}, weight={weight.dtype}"
        )

    if source_partitions == 1:
        return F.linear(hidden_states, weight).float()

    # Native xLLM row-shards each router across MP2. Each rank performs a BF16
    # partial GEMM, rounds that result to BF16, casts it to FP32, and then the
    # FP32 all-reduce adds the two partials. Emulate that ordering locally.
    input_parts = hidden_states.chunk(source_partitions, dim=-1)
    weight_parts = weight.chunk(source_partitions, dim=-1)
    first = F.linear(input_parts[0].contiguous(), weight_parts[0].contiguous())
    second = F.linear(input_parts[1].contiguous(), weight_parts[1].contiguous())
    return first.float() + second.float()


def _validate_mova_config(
    config: PretrainedConfig,
    quant_config: Optional[QuantizationConfig],
) -> None:
    """Fail early for native runtime combinations that cannot be served exactly."""

    if getattr(config, "model_type", None) in ("xllm", "k2_horizon"):
        if torch.get_default_dtype() != torch.bfloat16:
            raise ValueError(
                "Native xLLM/K2 Horizon serving requires --dtype bfloat16: "
                "the released checkpoints persist float32 dtype metadata but "
                "their weights and validated runtime contract are BF16."
            )
        if quant_config is not None and quant_config.get_name() != "compressed_tensors":
            raise ValueError(
                "Native xLLM/K2 Horizon serving supports only "
                "compressed-tensors quantized model weights"
            )

        runtime = get_exec()
        if runtime.overlap.enable_two_batch_overlap:
            raise ValueError(
                "Native xLLM/K2 Horizon serving does not yet support "
                "--enable-two-batch-overlap"
            )

        moe_runtime = runtime.moe
        unsupported_expert_remap = (
            moe_runtime.enable_eplb
            or moe_runtime.init_expert_location != "trivial"
            or moe_runtime.ep_num_redundant_experts > 0
        )
        if unsupported_expert_remap:
            raise ValueError(
                "Native xLLM/K2 Horizon serving does not yet support EPLB, "
                "non-trivial initial expert placement, or redundant experts; "
                "these modes require logical-to-physical expert remapping on "
                "every MoE backend."
            )

    if getattr(config, "num_values", 0) <= 0:
        # Legacy K2 checkpoints use model_type="xllm" for the ordinary
        # attention path.  XllmAttention implements partial RoPE and biased
        # QKV projections, but it does not implement the original xLLM
        # query/key normalization, sliding-window attention, or attention
        # gating variants.  Reject those layouts here instead of silently
        # loading them with different attention math.
        if getattr(config, "query_key_norm", False):
            raise ValueError(
                "Native dense xLLM attention does not support query/key normalization"
            )
        if getattr(config, "sliding_window", None) is not None or getattr(
            config, "use_sliding_window", False
        ):
            raise ValueError(
                "Native dense xLLM attention supports full causal attention only"
            )
        if getattr(config, "apply_attn_gate", False):
            raise ValueError(
                "Native dense xLLM attention does not support gated attention"
            )
        return
    _get_xllm_source_router_gemm_partitions(config)
    if getattr(config, "attention_bias", False):
        raise ValueError("K2 Horizon MoVA requires bias-free Q/K/V/O projections")
    if getattr(config, "query_key_norm", False):
        raise ValueError("K2 Horizon MoVA does not support query/key normalization")
    if not getattr(config, "apply_attn_gate", False):
        raise ValueError("K2 Horizon MoVA requires the xLLM attention gate")
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    if head_dim % 2:
        raise ValueError(f"MoVA requires an even RoPE head dimension, got {head_dim}")
    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("MoVA requires query heads to be divisible by KV heads")
    if getattr(config, "rope_head_dim", head_dim) != head_dim:
        raise ValueError("K2 Horizon MoVA requires full-head interleaved RoPE")
    rope_scaling = getattr(config, "rope_scaling", None)
    # Transformers 5 normalizes a JSON ``rope_scaling: null`` into an
    # explicit default-RoPE dictionary.  That representation does not change
    # the rotary math and must not be confused with linear/dynamic scaling.
    if rope_scaling is not None and not (
        isinstance(rope_scaling, dict)
        and rope_scaling.get("rope_type", rope_scaling.get("type")) == "default"
    ):
        raise ValueError("K2 Horizon MoVA does not support non-default RoPE scaling")
    if getattr(config, "sliding_window", None) is not None or getattr(
        config, "use_sliding_window", False
    ):
        raise ValueError("K2 Horizon MoVA uses full causal RadixAttention only")
    if getattr(config, "attn_gate_func", "silu") not in ("silu", "softplus"):
        raise ValueError("MoVA supports only silu and softplus attention gates")
    if getattr(config, "router_score_func", "sigmoid") not in ("sigmoid", "softmax"):
        raise ValueError("MoVA supports only sigmoid and softmax value routing")
    router_scale = getattr(config, "router_scaling_factor", 1.0)
    if router_scale is None or not math.isfinite(router_scale) or router_scale <= 0:
        raise ValueError(
            f"MoVA requires a positive finite router scaling factor, got {router_scale}"
        )
    num_dense_layers = getattr(config, "num_dense_layers", None)
    if (
        num_dense_layers is None
        or not 0 <= num_dense_layers <= config.num_hidden_layers
    ):
        raise ValueError(
            "MoVA requires num_dense_layers in [0, num_hidden_layers], got "
            f"{num_dense_layers}"
        )
    expected_dense_layers = list(range(num_dense_layers))
    if list(getattr(config, "mlp_only_layers", [])) != expected_dense_layers:
        raise ValueError(
            "K2 Horizon MoVA requires dense attention and dense FFN prefix layers to "
            f"match exactly; expected mlp_only_layers={expected_dense_layers}"
        )
    if getattr(config, "decoder_sparse_step", 1) != 1:
        raise ValueError("K2 Horizon MoVA requires decoder_sparse_step=1")
    num_values = config.num_values
    top_k = getattr(config, "num_values_per_tok", 0)
    if not 0 < top_k <= num_values:
        raise ValueError(
            f"num_values_per_tok must be in [1, {num_values}], got {top_k}"
        )
    if getattr(config, "num_experts", 0) <= 0:
        raise ValueError("MoVA requires sparse MoE feed-forward layers")
    n_groups = getattr(config, "layernorm_num_groups", 1) or 1
    if config.hidden_size % n_groups:
        raise ValueError(
            f"hidden size {config.hidden_size} is not divisible by {n_groups} norm groups"
        )
    attn_tp_size = get_parallel().attn_tp_size
    if config.num_attention_heads % attn_tp_size:
        raise ValueError(f"MoVA query heads must be divisible by TP={attn_tp_size}")
    if config.num_key_value_heads % attn_tp_size:
        raise ValueError(
            "K2 Horizon MoVA requires TP <= KV heads and KV heads divisible by TP; "
            f"got TP={attn_tp_size}, KV heads={config.num_key_value_heads}"
        )


def _xllm_stacked_params_mapping(config: PretrainedConfig):
    if getattr(config, "num_values", 0) <= 0:
        return [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

    mapping = [
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    mapping.extend(
        (".v_experts.weight", f".v_experts.{expert_id}.weight", expert_id)
        for expert_id in range(config.num_values)
    )
    return mapping


def permute_to_xllm(x):
    """Interleave first half and second half: [0,1,...,63,64,...,127] -> [0,64,1,65,...,63,127]"""
    return x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def permute_to_hf(x):
    """Inverse of permute_to_xllm: [0,64,1,65,...,63,127] -> [0,1,...,63,64,...,127]"""
    return x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).reshape(*x.shape[:-1], -1)


class XllmMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, skip_all_reduce=use_reduce_scatter)
        return x


class XllmMoEGate(nn.Module):
    """Router gate for xllm.

    Stores weight and bias separately. The bias is used as correction_bias
    for expert selection (added to sigmoid scores) but not in the linear
    computation of router logits.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.source_router_gemm_partitions = _get_xllm_source_router_gemm_partitions(
            config
        )
        self.weight = nn.Parameter(
            torch.empty((config.num_experts, config.hidden_size))
        )
        if getattr(config, "moe_gate_bias", False):
            # topk_sigmoid kernel requires correction_bias in float32
            self.bias = nn.Parameter(
                torch.empty(config.num_experts, dtype=torch.float32)
            )
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor):
        # The reference router applies sigmoid/softmax in FP32 after the BF16
        # GEMM (or after the explicit source-partition reduction).
        return _xllm_router_gemm(
            hidden_states, self.weight, self.source_router_gemm_partitions
        ).float()


class XllmSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_parallel().tp_size
        self.layer_id = layer_id
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.router_scaling_factor = getattr(config, "router_scaling_factor", 1.0)

        self.gate = XllmMoEGate(config)

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
            scoring_func=getattr(config, "router_score_func", "sigmoid"),
            correction_bias=self.gate.bias,
            # xLLM needs explicit ids and weights so correction-bias routing,
            # EPLB remapping, and post-renormalization scaling keep identical
            # semantics on every MoE runner backend.
            output_format=TopKOutputFormat.STANDARD,
        )

        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts + get_exec().moe.ep_num_redundant_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.RenormalizeNaive,
        )

        # Shared expert (no gating — output added directly)
        num_shared_experts = getattr(config, "num_shared_experts", 0)
        if num_shared_experts > 0:
            shared_intermediate_size = config.moe_intermediate_size * num_shared_experts
            self.shared_experts = XllmMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if (
                        get_moe_a2a_backend().is_deepep()
                        or get_moe_a2a_backend().is_mori()
                        or get_moe_a2a_backend().is_flashinfer()
                    )
                    else {}
                ),
            )
        else:
            self.shared_experts = None

        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mori():
            self.ep_size = get_parallel().moe_ep_size
            self.num_experts = (
                config.num_experts + get_exec().moe.ep_num_redundant_experts
            )
            self.top_k = config.num_experts_per_tok

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        if self.shared_experts is not None:
            return self.shared_experts(hidden_states)
        return None

    def _forward_deepep(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        shared_output = None
        if hidden_states.shape[0] > 0:
            router_logits = self.gate(hidden_states)
            shared_output = self._forward_shared_experts(hidden_states)
            # DeepEP/EPLB requires the current dispatched TopK path so logical
            # expert ids can be remapped and padded rows can be masked.
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=(
                    ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
                ),
            )
            # Apply router scaling factor after renormalization
            if self.router_scaling_factor != 1.0:
                scaled_weights = topk_output.topk_weights * self.router_scaling_factor
                if hasattr(topk_output, "_replace"):
                    topk_output = topk_output._replace(topk_weights=scaled_weights)
                else:
                    topk_output.topk_weights = scaled_weights
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device, layer_id=self.layer_id
            )
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if shared_output is not None:
            final_hidden_states.add_(shared_output)

        return final_hidden_states

    def _forward_router_experts(self, hidden_states: torch.Tensor):
        router_logits = self.gate(hidden_states)
        topk_output = self.topk.forward_native(hidden_states, router_logits)
        # Apply router scaling factor after renormalization
        # TopK output is a NamedTuple (immutable), so we must replace it
        if self.router_scaling_factor != 1.0:
            topk_output = topk_output._replace(
                topk_weights=topk_output.topk_weights * self.router_scaling_factor
            )
        return self.experts(hidden_states, topk_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mori():
            return self._forward_deepep(hidden_states, forward_batch)

        if hidden_states.shape[0] == 0:
            shared_output = None
            topk_output = self.topk.empty_topk_output(
                hidden_states.device, layer_id=self.layer_id
            )
            final_hidden_states = self.experts(hidden_states, topk_output)
        else:
            shared_output = self._forward_shared_experts(hidden_states)
            final_hidden_states = self._forward_router_experts(hidden_states)

        if shared_output is not None:
            final_hidden_states += shared_output
        if (
            self.tp_size > 1
            and not use_reduce_scatter
            and not should_skip_post_experts_all_reduce(is_tp_path=True)
            and not get_moe_a2a_backend().is_flashinfer()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class XllmAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        qkv_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        # Partial RoPE: xLLM/HF stores each head in neox ordering, where the
        # rotary dimensions are not contiguous when rope_head_dim < head_dim.
        # Mirror HF exactly: permute to interleaved, split rope/nope, apply RoPE
        # on the rope slice, then recombine and permute back.
        self.rope_head_dim = rope_head_dim
        self.use_xllm_partial_rope = rope_head_dim < head_dim
        self.rotary_emb = get_rope(
            self.rope_head_dim if self.use_xllm_partial_rope else self.head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _apply_partial_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_heads = q.reshape(-1, self.num_heads, self.head_dim)
        k_heads = k.reshape(-1, self.num_kv_heads, self.head_dim)

        q_interleaved = permute_to_xllm(q_heads)
        k_interleaved = permute_to_xllm(k_heads)

        nope_dim = self.head_dim - self.rope_head_dim
        q_rope, q_nope = q_interleaved.split([self.rope_head_dim, nope_dim], dim=-1)
        k_rope, k_nope = k_interleaved.split([self.rope_head_dim, nope_dim], dim=-1)

        q_rope_flat = permute_to_hf(q_rope).reshape(
            -1, self.num_heads * self.rope_head_dim
        )
        k_rope_flat = permute_to_hf(k_rope).reshape(
            -1, self.num_kv_heads * self.rope_head_dim
        )
        q_rope_flat, k_rope_flat = self.rotary_emb(positions, q_rope_flat, k_rope_flat)

        q_rope = permute_to_xllm(
            q_rope_flat.reshape(-1, self.num_heads, self.rope_head_dim)
        )
        k_rope = permute_to_xllm(
            k_rope_flat.reshape(-1, self.num_kv_heads, self.rope_head_dim)
        )

        q = permute_to_hf(torch.cat([q_rope, q_nope], dim=-1)).reshape(
            -1, self.num_heads * self.head_dim
        )
        k = permute_to_hf(torch.cat([k_rope, k_nope], dim=-1)).reshape(
            -1, self.num_kv_heads * self.head_dim
        )
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_xllm_partial_rope:
            q, k = self._apply_partial_rope(positions, q, k)
        else:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class _XllmMoVAAttentionBase(nn.Module):
    """Shared gated-GQA path for dense and routed-value MoVA layers."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError(
                "K2 Horizon MoVA supports unquantized bf16/fp16 weights only"
            )

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.rope_head_dim = getattr(config, "rope_head_dim", self.head_dim)
        self.apply_attn_gate = getattr(config, "apply_attn_gate", False)
        self.attn_gate_func = getattr(config, "attn_gate_func", "silu")
        self.scaling = self.head_dim**-0.5

        self.tp_rank = get_parallel().attn_tp_rank
        self.tp_size = get_parallel().attn_tp_size
        if self.total_num_heads % self.tp_size:
            raise ValueError(
                f"Attention heads {self.total_num_heads} are not divisible by TP={self.tp_size}"
            )
        if self.total_num_kv_heads % self.tp_size:
            raise ValueError(
                "K2 Horizon MoVA requires TP <= KV heads and KV heads divisible by TP; "
                f"got TP={self.tp_size}, KV heads={self.total_num_kv_heads}"
            )
        self.num_heads = self.total_num_heads // self.tp_size
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Keep the public correctness path as ordinary checkpoint-shaped
        # projections. Packing Q/K/gate is a performance optimization and is
        # deliberately outside the initial K2 Horizon integration.
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("k_proj", prefix),
        )
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=None,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=None,
            prefix=add_prefix("attn", prefix),
        )

    def _project_value(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _activate_gate(self, gate: torch.Tensor) -> torch.Tensor:
        if self.attn_gate_func == "silu":
            return F.silu(gate)
        if self.attn_gate_func == "softplus":
            return F.softplus(gate, beta=math.log(2))
        raise ValueError(
            f"Unsupported xLLM attention gate function: {self.attn_gate_func}"
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        gate, _ = self.gate_proj(hidden_states)
        value = self._project_value(hidden_states)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, value, forward_batch)
        if self.apply_attn_gate:
            attn_output = attn_output * self._activate_gate(gate)
        output, _ = self.o_proj(attn_output)
        return output


class XllmGatedAttention(_XllmMoVAAttentionBase):
    """Dense GQA used by the prefix layers of a MoVA checkpoint."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("v_proj", prefix),
        )

    def _project_value(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value, _ = self.v_proj(hidden_states)
        return value


class XllmMoVAAttention(_XllmMoVAAttentionBase):
    """Sparse MoVA attention with output-sharded routed value experts."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)
        self.num_values = config.num_values
        self.num_values_per_tok = config.num_values_per_tok
        self.router_score_func = getattr(config, "router_score_func", "sigmoid")
        self.router_scaling_factor = getattr(config, "router_scaling_factor", 1.0)
        self.renormalize = getattr(config, "norm_topk_prob", True)
        self.source_router_gemm_partitions = _get_xllm_source_router_gemm_partitions(
            config
        )
        self.v_router = ReplicatedLinear(
            config.hidden_size,
            self.num_values,
            bias=False,
            quant_config=None,
            prefix=add_prefix("v_router", prefix),
        )
        if getattr(config, "moe_gate_bias", False):
            # SGLang's fused sigmoid top-k requires correction bias in fp32.
            # It remains a loadable parameter for Miles weight updates, but is
            # never included in the router logits matmul.
            self.v_router.bias = nn.Parameter(
                torch.empty(self.num_values, dtype=torch.float32),
                requires_grad=False,
            )
        self.v_experts = RoutedValueExperts(
            self.num_values,
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )

    def _project_value(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Router bias is deliberately omitted from the logits matmul. It only
        # changes route selection inside ``mova_router_topk``.
        router_logits = _xllm_router_gemm(
            hidden_states,
            self.v_router.weight,
            self.source_router_gemm_partitions,
        ).float()
        routing_weights, selected_values = mova_router_topk(
            router_logits,
            self.v_router.bias,
            score_func=self.router_score_func,
            top_k=self.num_values_per_tok,
            scaling_factor=self.router_scaling_factor,
            renormalize=self.renormalize,
        )
        return self.v_experts(hidden_states, routing_weights, selected_values)


class XllmDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        qkv_bias = getattr(config, "attention_bias", False)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rope_head_dim = getattr(config, "rope_head_dim", head_dim)

        self.layer_id = layer_id

        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank

        # Determine if this layer is sparse (MoE) or dense
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        if (layer_id not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_id + 1) % decoder_sparse_step == 0
        ):
            self.is_layer_sparse = True
        else:
            self.is_layer_sparse = False

        is_mova_config = getattr(config, "num_values", 0) > 0
        is_mova_attention = is_mova_config and layer_id >= config.num_dense_layers
        if is_mova_attention:
            self.self_attn = XllmMoVAAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        elif is_mova_config:
            self.self_attn = XllmGatedAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.self_attn = XllmAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                layer_id=layer_id,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                qkv_bias=qkv_bias,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )

        # Check neighbors for scatter modes
        def _is_sparse(lid):
            if lid < 0 or lid >= config.num_hidden_layers:
                return False
            return (lid not in mlp_only_layers) and (
                config.num_experts > 0 and (lid + 1) % decoder_sparse_step == 0
            )

        is_previous_layer_sparse = _is_sparse(layer_id - 1)
        is_next_layer_sparse = _is_sparse(layer_id + 1)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = XllmSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = XllmMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = _make_norm(config)
        self.post_attention_layernorm = _make_norm(config)
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(self.layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        if isinstance(self.mlp, XllmMLP):
            hidden_states = self.mlp(
                hidden_states, use_reduce_scatter=use_reduce_scatter
            )
        else:
            hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class XllmModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: XllmDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = _make_norm(config)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            ctx = (
                nullcontext()
                if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class XllmForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    # Quantized checkpoints store these projections separately. This mapping
    # lets quantization configs resolve fused runtime modules and their ignore
    # lists consistently.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        _validate_mova_config(config, quant_config)
        self.model = XllmModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        # Value experts are shards of one attention-TP parameter, not FFN/EP
        # experts. ParameterMapper therefore stages all 64 canonical HF shards
        # before writing the persistent packed tensor during live updates.
        self.stacked_params_mapping = _xllm_stacked_params_mapping(config)
        self.expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            logits_output = self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
            return logits_output
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = self.stacked_params_mapping
        expert_params_mapping = self.expert_params_mapping
        strict_checkpoint = getattr(self.config, "model_type", None) in (
            "xllm",
            "k2_horizon",
        )

        def is_pipeline_missing_weight(name: str) -> bool:
            if not strict_checkpoint:
                return False
            pp_group = getattr(self, "pp_group", None)
            if pp_group is None:
                return False
            return (
                name == "model.embed_tokens.weight" and not pp_group.is_first_rank
            ) or (
                name in ("model.norm.weight", "lm_head.weight")
                and not pp_group.is_last_rank
            )

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            checkpoint_name = name
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if name == "model.embed_tokens.weight" and self.config.tie_word_embeddings:
                # With PP>1, the final stage has no embedding table to alias,
                # so initialize its separate head from the checkpoint embedding.
                if self.pp_group.is_last_rank and "lm_head.weight" in params_dict:
                    param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if weight_name in (".gate_proj", ".up_proj") and ".mlp." not in name:
                    continue
                # Skip experts (handled below in expert_params_mapping)
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    if strict_checkpoint:
                        raise RuntimeError(
                            "xLLM-family checkpoint weight did not resolve to "
                            "a native model parameter: "
                            f"checkpoint={checkpoint_name!r}, mapped={name!r}"
                        )
                    continue
                if name not in params_dict:
                    if strict_checkpoint:
                        raise RuntimeError(
                            "xLLM-family checkpoint weight did not resolve to "
                            "a native model parameter: "
                            f"checkpoint={checkpoint_name!r}, mapped={name!r}"
                        )
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if is_pipeline_missing_weight(name):
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        if strict_checkpoint:
                            raise RuntimeError(
                                "xLLM-family checkpoint weight did not resolve "
                                "to a native model parameter: "
                                f"checkpoint={checkpoint_name!r}, mapped={name!r}"
                            )
                        continue
                    if name not in params_dict:
                        if strict_checkpoint:
                            raise RuntimeError(
                                "xLLM-family checkpoint weight did not resolve "
                                "to a native model parameter: "
                                f"checkpoint={checkpoint_name!r}, mapped={name!r}"
                            )
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        if getattr(config, "num_experts", 0) <= 0:
            return None
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


class K2HorizonForCausalLM(XllmForCausalLM):
    """Load canonical K2Horizon HF checkpoints through the native xLLM path."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        _normalize_k2_horizon_config(config)
        super().__init__(config, quant_config=quant_config, prefix=prefix)


EntryClass = [XllmForCausalLM, K2HorizonForCausalLM]
