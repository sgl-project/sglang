# Copyright 2025 SGLang Team
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
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/transformers
"""Wrapper around `transformers` models."""

import inspect
import logging
import re
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import List, Literal, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sglang.srt.distributed import (
    divide,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_pp_indices,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import direct_register_custom_op
from sglang.srt.utils.hf_transformers_utils import get_hf_text_config


def can_enable_torch_compile(config: PretrainedConfig) -> bool:
    """Check whether the model config is compatible with torch.compile.

    Dynamic rope scaling triggers data-dependent control flow that prevents
    capturing a single computation graph, so we disable compilation for it.
    """
    text_config = getattr(config, "text_config", config)
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        if rope_type == "dynamic":
            return False
    rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        if isinstance(next(iter(rope_params.values()), None), dict):
            return not any(
                rp.get("rope_type") == "dynamic" for rp in rope_params.values()
            )
        if rope_params.get("rope_type") == "dynamic":
            return False
    return True


logger = logging.getLogger(__name__)

_TRANSFORMERS_MOE_LAYERS: dict[str, "TransformersFusedMoE"] = {}


def maybe_prefix(prefix: str, name: str) -> str:
    return name if not prefix else f"{prefix}.{name}"


def log_replacement(name: str, old_module: nn.Module, new_module: nn.Module):
    logger.debug("%s: %s -> %s", name, old_module, new_module)


def _getattr_first(obj, names, default=None):
    """Return the first existing attribute from *names*, else *default*."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _resolve_attention_backend_model_cls(config: PretrainedConfig):
    model_cls = getattr(transformers, getattr(config, "architectures", [""])[0], None)
    if model_cls is not None:
        return model_cls

    auto_map = getattr(config, "auto_map", {}) or {}
    for key in ("AutoModel", "AutoModelForCausalLM"):
        if key not in auto_map:
            continue
        try:
            return get_class_from_dynamic_module(
                auto_map[key],
                getattr(config, "_name_or_path", ""),
            )
        except Exception as e:
            logger.warning(
                "Failed to load dynamic module from auto_map[%s]: %s.",
                key,
                e,
            )
    return None


def _encoder_accepts_feature_kwarg(encoder, feature_kwarg: str) -> bool:
    try:
        sig = inspect.signature(encoder)
    except (TypeError, ValueError):
        return False

    if feature_kwarg in sig.parameters:
        return True

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_var_keyword:
        return False

    required_positional_params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    return len(required_positional_params) == 0


@contextmanager
def _init_on_device_without_buffers(device: torch.device):
    """Initialize model parameters on *device* while leaving buffers on CPU.
    Adapted from ``accelerate``."""
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


Style = Literal["colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"]


def replace_linear_class(
    linear: nn.Linear,
    style: Style = "replicate",
    quant_config: Optional[QuantizationConfig] = None,
    *,
    prefix: str = "",
) -> Union[ColumnParallelLinear, RowParallelLinear, ReplicatedLinear]:
    if not isinstance(style, str):
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")

    sglang_linear_cls, linear_kwargs = {
        "colwise": (ColumnParallelLinear, {}),
        "colwise_rep": (ColumnParallelLinear, {"gather_output": True}),
        "rowwise": (RowParallelLinear, {}),
        "rowwise_rep": (RowParallelLinear, {"input_is_parallel": False}),
        "replicate": (ReplicatedLinear, {}),
    }.get(style, (ReplicatedLinear, {}))

    class HFCompatibleLinear(sglang_linear_cls):
        @property
        def parent_cls(self) -> type:
            return sglang_linear_cls

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return super().forward(input)[0]

    return HFCompatibleLinear(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
        quant_config=quant_config,
        prefix=prefix,
        **linear_kwargs,
    )


def _normalize_tp_style(style: str) -> Style:
    style = style.lower().replace("-", "_")
    style = {
        "colwiseparallel": "colwise",
        "packed_colwise": "colwise",
        "local_colwise": "colwise",
        "rowwiseparallel": "rowwise",
        "packed_rowwise": "rowwise",
        "local_rowwise": "rowwise",
        "local_packed_rowwise": "rowwise",
        "isolated": "replicate",
        "local": "replicate",
        "replicated_with_grad_allreduce": "replicate",
        "moe_tp_experts": "replicate",
    }.get(style, style)
    if style not in {"colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"}:
        raise ValueError(f"Unsupported TP style '{style}' for Transformers backend.")
    return style


def replace_rms_norm_class(rms_norm: nn.Module, hidden_size: int) -> nn.Module:
    eps = _getattr_first(rms_norm, ("eps", "variance_epsilon"), 1e-6)
    kwargs = {"hidden_size": hidden_size, "eps": eps}
    weight_meta = getattr(rms_norm, "weight", None)
    if weight_meta is not None:
        kwargs["hidden_size"] = weight_meta.size(0)

    try:
        with torch.device("cpu"):
            weight_test = getattr(rms_norm.__class__(1), "weight", None)
    except Exception:
        weight_test = None
    is_gemma = weight_test is not None and torch.all(weight_test == 0)

    if is_gemma:
        base_cls = GemmaRMSNorm
        norm = base_cls(
            **{k: v for k, v in kwargs.items() if k in ("hidden_size", "eps")}
        )
    else:
        kwargs["has_weight"] = getattr(rms_norm, "with_scale", True)
        if weight_meta is not None:
            kwargs["weight_dtype"] = weight_meta.dtype
        else:
            kwargs["has_weight"] = False
        base_cls = RMSNorm
        norm = base_cls(**kwargs)

    # Wrap to handle 3D inputs from Transformers backbone (batch dim)
    class HFCompatibleRMSNorm(norm.__class__):
        def forward(self, x, *args, **kwargs):
            orig_shape = x.shape
            if x.ndim > 2:
                x = x.reshape(-1, x.shape[-1]).contiguous()
            result = super().forward(x, *args, **kwargs)
            if isinstance(result, tuple):
                return tuple(
                    (
                        r.reshape(orig_shape)
                        if torch.is_tensor(r) and r.shape != orig_shape
                        else r
                    )
                    for r in result
                )
            if torch.is_tensor(result) and result.shape != orig_shape:
                return result.reshape(orig_shape)
            return result

    norm.__class__ = HFCompatibleRMSNorm
    return norm


def sglang_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float = None,
    attention_instances: Optional[Mapping[int, RadixAttention]] = None,
    forward_batch: Optional[ForwardBatch] = None,
    **kwargs,
):
    self_attn: RadixAttention = attention_instances[module.layer_idx]
    if scaling is not None:
        self_attn.scaling = float(scaling)
    hidden = query.shape[-2]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    return self_attn.forward(query, key, value, forward_batch=forward_batch), None


ALL_ATTENTION_FUNCTIONS["sglang"] = sglang_flash_attention_forward


class TransformersFusedMoE(nn.Module):
    """FusedMoE wrapper for the Transformers modeling backend.

    Wraps SGLang's native MoE implementation and exposes the
    ``(hidden_states, topk_ids, topk_weights)`` signature expected by
    Transformers' ``experts.forward()``.  A registered custom op
    (``torch.ops.sglang.transformers_moe_forward``) is used so that
    ``torch.compile`` can properly graph-break around the MoE kernel.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        reduce_results: bool,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        activation: str,
        with_bias: bool,
        expert_mapping: list,
    ) -> None:
        super().__init__()
        num_redundant = get_global_server_args().ep_num_redundant_experts
        experts_cls = get_moe_impl_class(quant_config)
        self.experts = experts_cls(
            num_experts=num_experts + num_redundant,
            top_k=top_k,
            layer_id=layer_id,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=reduce_results,
            quant_config=quant_config,
            activation=activation,
            with_bias=with_bias,
            prefix=prefix,
        )
        self.layer_name = prefix
        self.num_experts = num_experts
        self.top_k = top_k
        self._expert_mapping = expert_mapping
        _TRANSFORMERS_MOE_LAYERS[prefix] = self

    @property
    def tp_size(self) -> int:
        return getattr(self.experts, "moe_tp_size", 1)

    @property
    def ep_size(self) -> int:
        return getattr(self.experts, "moe_ep_size", 1)

    def maybe_all_reduce_tensor_model_parallel(
        self, output: torch.Tensor
    ) -> torch.Tensor:
        if self.tp_size > 1:
            return tensor_model_parallel_all_reduce(output)
        return output

    def get_expert_weights(self):
        return getattr(self.experts, "get_expert_weights", lambda: None)()

    def get_moe_weights(self) -> list[torch.Tensor]:
        num_local = getattr(self.experts, "num_local_experts", self.num_experts)
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ("correction_bias",)
            and filter_moe_weight_param_global_expert(name, x, num_local)
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(torch.float32)
        if hidden_states.is_cuda:
            return torch.ops.sglang.transformers_moe_forward(
                hidden_states,
                topk_ids,
                topk_weights,
                self.layer_name,
            )
        return _transformers_moe_forward(
            hidden_states,
            topk_ids,
            topk_weights,
            self.layer_name,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        param_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            matched = False
            for param_name, weight_name, expert_id, shard_id in self._expert_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                param = param_dict.get(mapped_name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                except TypeError:
                    weight_loader(param, loaded_weight)
                loaded.add(name)
                matched = True
                break
            if not matched:
                direct_name = name if name in param_dict else f"experts.{name}"
                if direct_name in param_dict:
                    param = param_dict[direct_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    try:
                        weight_loader(param, loaded_weight)
                    except TypeError:
                        default_weight_loader(param, loaded_weight)
                    loaded.add(name)
                else:
                    logger.warning(
                        "MoE weight '%s' in layer '%s' could not be matched to any "
                        "parameter and will be skipped.",
                        name,
                        self.layer_name,
                    )
        return loaded


def _transformers_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    self = _TRANSFORMERS_MOE_LAYERS[layer_name]
    # Record expert distribution for EPLB
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )

    recorder = get_global_expert_distribution_recorder()
    with recorder.with_current_layer(self.experts.layer_id):
        recorder.on_select_experts(topk_ids=topk_ids)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=topk_weights,
    )
    return self.experts(hidden_states.clone(), topk_output)


def _transformers_moe_forward_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="transformers_moe_forward",
    op_func=_transformers_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_transformers_moe_forward_fake,
)

try:
    from sglang.srt.compilation.compilation_config import SPLIT_OPS

    _MOE_SPLIT_OP = "sglang.transformers_moe_forward"
    if _MOE_SPLIT_OP not in SPLIT_OPS:
        SPLIT_OPS.append(_MOE_SPLIT_OP)
except ImportError:
    pass


_BASE_DYNAMIC_ARG_DIMS: dict[str, int] = {
    "input_ids": 0,
    "positions": 0,
    "input_embeds": 0,
}

_MULTIMODAL_DYNAMIC_ARG_DIMS: dict[str, int] = {
    "input_ids": 0,
    "positions": -1,  # last dim to support M-RoPE (Qwen2.5-VL 3×seq layout)
    "input_embeds": 0,
}


class TransformersBase(nn.Module):
    torch_compile_dynamic_arg_dims: dict[str, int] = _BASE_DYNAMIC_ARG_DIMS

    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model.": "model.language_model.",
            "model.transformer.": "model.",
            "model.model.": "model.",
            "model.lm_head.": "lm_head.",
            "model.score.": "classifier.",
            "model.classifier.": "classifier.",
            "transformer.": "model.",
            "model.": "model.",
            "lm_head.": "lm_head.",
            "score.": "classifier.",
            "classifier.": "classifier.",
            "": "model.",
        }
    )

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        mapper = WeightsMapper()
        for base in cls.__mro__:
            base_mapper = getattr(base, "hf_to_sglang_mapper", None)
            if base_mapper is not None:
                mapper = mapper | base_mapper
        cls.hf_to_sglang_mapper = mapper

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Using Transformers backend.")

        self.quant_config = quant_config
        self.config = config
        self.text_config = get_hf_text_config(config)
        self.weight_mapper = self.hf_to_sglang_mapper
        self.pp_group = get_pp_group()

        # Weight loading attrs
        self.skip_prefixes: list[str] = []
        self.skip_substrs: list[str] = []
        self.ignore_unexpected_prefixes: list[str] = []
        self.ignore_unexpected_suffixes: list[str] = []
        self.skip_substrs.extend([".attn.bias", ".attn.masked_bias", ".masked_bias"])
        self.ignore_unexpected_prefixes.extend(["classifier.", "score."])

        if self.quant_config is not None:
            quant_method_name = self.quant_config.get_name()
            if "gptq" in quant_method_name:
                self.ignore_unexpected_suffixes.append(".bias")
            if "fp8" in quant_method_name:
                fp8_suffix_map = {".activation_scale": ".input_scale"}
                use_mxfp8 = bool(getattr(self.quant_config, "use_mxfp8", False))
                weight_block_size = getattr(
                    self.quant_config, "weight_block_size", None
                )
                if not use_mxfp8 and weight_block_size is None:
                    fp8_suffix_map[".weight_scale_inv"] = ".weight_scale"
                self.weight_mapper = self.weight_mapper | WeightsMapper(
                    orig_to_new_suffix=fp8_suffix_map
                )

        # Resolve model class for _supports_attention_backend check
        model_cls = _resolve_attention_backend_model_cls(config)

        supports_backend = (
            getattr(model_cls, "_supports_attention_backend", True)
            if model_cls
            else True
        )

        # Initialize on meta device to avoid premature GPU allocation
        self.text_config._attn_implementation = "sglang"
        if supports_backend:
            with _init_on_device_without_buffers(torch.device("meta")):
                self.model: PreTrainedModel = AutoModel.from_config(
                    self.config,
                    torch_dtype=torch.get_default_dtype(),
                    trust_remote_code=True,
                )
        else:
            raise ValueError(
                f"Model {model_cls} does not support custom attention backends "
                "(_supports_attention_backend=False). The Transformers backend "
                "requires custom attention support."
            )

        self.vocab_size = getattr(
            self.text_config,
            "vocab_size",
            self.model.get_input_embeddings().num_embeddings,
        )
        self.unpadded_vocab_size = self.vocab_size

        # Embedding scale (e.g. Whisper)
        input_embeddings = self.model.get_input_embeddings()
        self.embed_scale = getattr(input_embeddings, "embed_scale", None)

        self.start_layer = 0
        self.end_layer = getattr(self.text_config, "num_hidden_layers", 0)

        # Pipeline parallel
        self.pipeline_parallel()
        # Module replacement (Linear → TP, RMSNorm → fused, MoE overridden by MoEMixin)
        tp_size = get_tensor_model_parallel_world_size()
        self.recursive_replace()
        # Attention instances
        self.attention_instances = self._create_attention_instances(tp_size)
        # Vocab embeddings
        self.replace_vocab_embed_class(self.model)

        # Initialize remaining meta-device parameters to real device tensors
        self._init_parameters(self.model)

        self.lm_head: Optional[ParallelLMHead] = None
        self.logits_processor: Optional[LogitsProcessor] = None
        self.pooler: Optional[Pooler] = None

        self._compile_compatible = can_enable_torch_compile(config)

    @property
    def _can_torch_compile(self) -> bool:
        """Whether this model instance is safe to wrap with torch.compile."""
        return self._compile_compatible

    def _init_parameters(self, module: nn.Module):
        """Materialize any parameters still on the meta device."""
        for name, param in module.named_parameters(recurse=False):
            if param.device == torch.device("meta"):
                new_param = nn.Parameter(
                    torch.empty_like(
                        param.data,
                        device="cuda",
                    )
                )
                setattr(module, name, new_param)
        for child in module.children():
            self._init_parameters(child)

    def log_replacement(self, name: str, old_module: nn.Module, new_module: nn.Module):
        logger.debug("%s: %s -> %s", name, old_module, new_module)

    # -- TP plan handling ---------------------------------------------------
    def _get_model_tp_plan(self) -> Mapping[str, str]:
        plan = (
            getattr(self.model, "tp_plan", None)
            or getattr(self.model, "_tp_plan", None)
            or getattr(self.model.config, "base_model_tp_plan", None)
            or getattr(self.text_config, "base_model_tp_plan", None)
        )
        if plan:
            return plan

        plan = self._infer_tp_plan_from_children()
        return plan if plan else {}

    _LANGUAGE_MODEL_CHILD_NAMES = frozenset(
        {"language_model", "text_model", "model", "lm"}
    )

    def _infer_tp_plan_from_children(self) -> dict[str, str]:
        plan: dict[str, str] = {}
        for child_name, child_module in self.model.named_children():
            child_plan = getattr(child_module, "_tp_plan", None)
            if child_plan:
                plan.update({f"{child_name}.{k}": v for k, v in child_plan.items()})
                continue

            child_config = getattr(child_module, "config", None)
            if child_config is not None:
                child_tp = getattr(child_config, "base_model_tp_plan", None)
                if child_tp:
                    plan.update({f"{child_name}.{k}": v for k, v in child_tp.items()})
                    continue

            if child_name not in self._LANGUAGE_MODEL_CHILD_NAMES:
                continue
            if child_config is None:
                continue
            model_type = getattr(child_config, "model_type", "")
            base_type = (
                model_type.replace("_vl_text", "")
                .replace("_vl", "")
                .replace("_text", "")
            )
            if base_type and base_type != model_type:
                try:
                    from transformers import AutoConfig

                    base_cfg = AutoConfig.for_model(base_type)
                    base_tp = getattr(base_cfg, "base_model_tp_plan", None)
                    if base_tp:
                        plan.update(
                            {f"{child_name}.{k}": v for k, v in base_tp.items()}
                        )
                except Exception as e:
                    logger.debug(
                        "Could not infer TP plan from base model type '%s': %s",
                        base_type,
                        e,
                    )
        return plan

    def _normalize_tp_plan(self, tp_plan: Mapping[str, str]) -> dict[str, Style]:
        normalized = {}
        for pattern, style in tp_plan.items():
            if pattern.startswith("^model\\."):
                pattern = "^" + pattern[len("^model\\.") :]
            elif pattern.startswith("model\\."):
                pattern = pattern[len("model\\.") :]
            elif pattern.startswith("model."):
                pattern = pattern[len("model.") :]
            normalized[pattern] = _normalize_tp_style(style)
        return normalized

    # -- Recursive module replacement (Linear + RMSNorm) --------------------
    def recursive_replace(self):
        tp_size = get_tensor_model_parallel_world_size()
        tp_plan = self._normalize_tp_plan(self._get_model_tp_plan())

        if not tp_plan and tp_size > 1:
            raise ValueError(
                f"{type(self.model)} does not support tensor parallel yet!"
            )

        # Prefix patterns to match from `self.model`
        prefixed_plan = {maybe_prefix("model", k): v for k, v in tp_plan.items()}

        def _recursive_replace(module: nn.Module, prefix: str):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                new_module = child_module

                if isinstance(child_module, nn.Linear):
                    pattern = next(
                        (p for p in prefixed_plan if re.match(p, qual_name)),
                        None,
                    )
                    style = prefixed_plan.get(pattern, "replicate")
                    new_module = replace_linear_class(
                        child_module,
                        style,
                        self.quant_config,
                        prefix=qual_name,
                    )
                elif child_module.__class__.__name__.endswith("RMSNorm"):
                    new_module = replace_rms_norm_class(
                        child_module,
                        self.text_config.hidden_size,
                    )
                else:
                    _recursive_replace(child_module, prefix=qual_name)

                if new_module is not child_module:
                    setattr(module, child_name, new_module)
                    log_replacement(qual_name, child_module, new_module)

        _recursive_replace(self.model, prefix="model")

    # -- Pipeline parallel --------------------------------------------------
    def _get_model_pp_plan(self) -> Mapping[str, object]:
        return (
            getattr(self.model, "_pp_plan", None)
            or getattr(self.model, "pp_plan", None)
            or getattr(self.model.config, "base_model_pp_plan", None)
            or getattr(self.text_config, "base_model_pp_plan", None)
            or {}
        )

    def _register_missing_prefix(self, prefix: str):
        if not prefix.endswith("."):
            prefix += "."
        if prefix not in self.skip_prefixes:
            self.skip_prefixes.append(prefix)

    @staticmethod
    def _make_pp_missing_layer(original: nn.Module) -> PPMissingLayer:
        """Create a PPMissingLayer that preserves plain attributes from
        *original* so that the HF forward loop can still access per-layer
        metadata (e.g. ``attention_type`` on Qwen2 decoder layers)."""
        replacement = PPMissingLayer()
        for key, value in original.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (nn.Module, nn.Parameter, torch.Tensor)):
                continue
            setattr(replacement, key, value)
        return replacement

    def _get_submodule_or_none(self, name: str) -> Optional[nn.Module]:
        try:
            return self.model.get_submodule(name)
        except AttributeError:
            return None

    def _set_submodule(self, name: str, module: nn.Module):
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent_module = self.model.get_submodule(parent_name)
        else:
            parent_module = self.model
            child_name = name
        setattr(parent_module, child_name, module)

    def pipeline_parallel(self):
        if self.pp_group.world_size <= 1:
            return

        pp_plan = self._get_model_pp_plan()
        if not pp_plan:
            raise ValueError(
                f"{type(self.model)} does not support pipeline parallel yet!"
            )

        pp_keys = [re.sub(r"^model\.", "", name) for name in pp_plan.keys()]
        module_list_idx = None
        module_list_name = None
        for idx, name in enumerate(pp_keys):
            if isinstance(self._get_submodule_or_none(name), nn.ModuleList):
                if module_list_idx is not None:
                    raise ValueError(
                        "Pipeline parallel with multiple ModuleList blocks is not supported."
                    )
                module_list_idx = idx
                module_list_name = name

        if module_list_idx is None or module_list_name is None:
            raise ValueError(f"Could not find ModuleList in {type(self.model)}.")

        keep_prefix_modules = self.pp_group.is_first_rank or (
            getattr(self.text_config, "tie_word_embeddings", False)
            and self.pp_group.is_last_rank
        )
        for name in pp_keys[:module_list_idx]:
            if keep_prefix_modules:
                continue
            self._set_submodule(name, PPMissingLayer())
            self._register_missing_prefix(maybe_prefix("model", name))

        layers = self.model.get_submodule(module_list_name)
        self.start_layer, self.end_layer = get_pp_indices(
            len(layers),
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        for idx in range(len(layers)):
            if self.start_layer <= idx < self.end_layer:
                continue
            layers[idx] = self._make_pp_missing_layer(layers[idx])
            self._register_missing_prefix(
                maybe_prefix("model", f"{module_list_name}.{idx}")
            )

        for name in pp_keys[module_list_idx + 1 :]:
            if self.pp_group.is_last_rank:
                continue
            self._set_submodule(name, PPMissingLayer())
            self._register_missing_prefix(maybe_prefix("model", name))

    # -- Attention instances ------------------------------------------------
    def _create_attention_instances(self, tp_size: int) -> dict[int, RadixAttention]:
        num_heads = self.text_config.num_attention_heads
        num_kv_heads = getattr(self.text_config, "num_key_value_heads", num_heads)
        hidden_size = self.text_config.hidden_size
        head_dim = getattr(self.text_config, "head_dim", hidden_size // num_heads)

        layer_types = getattr(self.text_config, "layer_types", None) or getattr(
            self.config, "layer_types", None
        )
        global_sliding_window = getattr(
            self.text_config, "sliding_window", None
        ) or getattr(self.config, "sliding_window", None)

        # Detect encoder-only models (non-causal attention everywhere)
        is_encoder_only = any(
            not getattr(m, "is_causal", True)
            for m in self.model.modules()
            if hasattr(m, "is_causal")
        )
        if is_encoder_only and self.config != self.text_config:
            is_encoder_only = False
        if is_encoder_only:
            logger.info(
                "Detected encoder-only model (non-causal attention). "
                "Using RadixAttention with is_cross_attention=True."
            )

        instances = {}
        for idx in range(self.start_layer, self.end_layer):
            # Per-layer sliding window (e.g. Gemma2, Cohere)
            per_layer_sliding_window = -1
            if (
                layer_types is not None
                and idx < len(layer_types)
                and layer_types[idx] == "sliding_attention"
                and global_sliding_window is not None
            ):
                per_layer_sliding_window = global_sliding_window

            instances[idx] = RadixAttention(
                num_heads=divide(num_heads, tp_size),
                head_dim=head_dim,
                scaling=head_dim**-0.5,
                num_kv_heads=divide(num_kv_heads, tp_size),
                layer_id=idx,
                quant_config=self.quant_config,
                sliding_window_size=per_layer_sliding_window,
                is_cross_attention=is_encoder_only,
                prefix=f"{idx}.attn",
            )
        return instances

    # -- Vocab embedding replacement ----------------------------------------
    def replace_vocab_embed_class(self, module: nn.Module):
        old_module = self.model.get_input_embeddings()
        if old_module is None or isinstance(old_module, PPMissingLayer):
            return
        embedding_dim = getattr(old_module, "embedding_dim", None)
        if embedding_dim is None:
            embedding_dim = _getattr_first(
                self.text_config,
                ("embedding_size", "hidden_size"),
                None,
            )
        assert embedding_dim is not None
        new_module = VocabParallelEmbedding(
            self.vocab_size,
            embedding_dim,
            org_num_embeddings=self.vocab_size,
            quant_config=None,
        )

        old_embed_scale = getattr(old_module, "embed_scale", None)
        if old_embed_scale is not None:
            base_cls = new_module.__class__

            class ScaledEmbedding(base_cls):
                def forward(self, input_):
                    return base_cls.forward(self, input_) * self.embed_scale

            new_module.__class__ = ScaledEmbedding
            new_module.embed_scale = old_embed_scale
            self.embed_scale = None

        self.log_replacement("input embedding", old_module, new_module)
        self.model.set_input_embeddings(new_module)

    # -- Forward ------------------------------------------------------------
    def _format_position_ids(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim == 2 and positions.shape[0] == 3:
            return positions[:, None, ...]
        if positions.ndim == 1:
            return positions[None, ...]
        return positions

    def _run_hf_backbone(
        self,
        input_ids: Optional[torch.Tensor],
        input_embeds: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        hf_input_ids = None if input_ids is None else input_ids[None, ...]
        hf_input_embeds = None
        if input_embeds is not None:
            hf_input_embeds = input_embeds[None, ...]
            hf_input_ids = None

        # Scale embeddings if needed
        if (
            self.embed_scale is not None
            and hf_input_ids is not None
            and hf_input_embeds is None
        ):
            hf_input_embeds = (
                self.model.get_input_embeddings()(hf_input_ids) * self.embed_scale
            )
            hf_input_ids = None

        return self.model(
            input_ids=hf_input_ids,
            inputs_embeds=hf_input_embeds,
            use_cache=False,
            position_ids=self._format_position_ids(positions),
            return_dict=False,
            forward_batch=forward_batch,
            attention_instances=self.attention_instances,
            **kwargs,
        )[0][0, ...]

    def _forward_hidden_states(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._run_hf_backbone(
            input_ids=input_ids,
            input_embeds=input_embeds,
            positions=positions,
            forward_batch=forward_batch,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> Union[LogitsProcessorOutput, EmbeddingPoolerOutput, PPProxyTensors]:
        runtime_input_ids: Optional[torch.Tensor] = input_ids
        runtime_input_embeds = input_embeds
        if not self.pp_group.is_first_rank:
            assert pp_proxy_tensors is not None
            runtime_input_ids = None
            runtime_input_embeds = pp_proxy_tensors["hidden_states"]

        hidden_states = self._forward_hidden_states(
            input_ids=runtime_input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=runtime_input_embeds,
        )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": hidden_states}
            )

        if get_embedding:
            assert (
                self.pooler is not None
            ), "pooling is not enabled for this model class"
            return self.pooler(hidden_states, forward_batch)

        assert self.logits_processor is not None and self.lm_head is not None
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, None
        )

    # -- Weight loading -----------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=self.skip_prefixes,
            skip_substrs=self.skip_substrs,
            ignore_unexpected_prefixes=self.ignore_unexpected_prefixes,
            ignore_unexpected_suffixes=self.ignore_unexpected_suffixes,
        )
        return loader.load_weights(weights, mapper=self.weight_mapper)


class CausalMixin:

    def __init__(self, *args, prefix: str = "", **kwargs):
        super().__init__(*args, prefix=prefix, **kwargs)

        tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)
        if tie_word_embeddings:
            self.skip_prefixes.append("lm_head.")

        if not self.pp_group.is_last_rank:
            self._register_missing_prefix("lm_head")
            return

        self.lm_head = ParallelLMHead(
            self.vocab_size,
            self.text_config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight

        logit_scale = getattr(self.text_config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.text_config, logit_scale=logit_scale
        )


class EmbeddingMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_unexpected_prefixes.append("lm_head.")
        if not self.pp_group.is_last_rank:
            return
        pooling_name = str(getattr(self.config, "pooling_type", "LAST")).upper()
        pooling_type = PoolingType.CLS if pooling_name == "CLS" else PoolingType.LAST
        normalize = bool(getattr(self.config, "normalize", True))
        self.pooler = Pooler(pooling_type=pooling_type, normalize=normalize)


class MoEMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_model_config_for_expert_location(
        cls, config
    ) -> Optional[ModelConfigForExpertLocation]:
        text_config = getattr(config, "text_config", config)
        num_experts = _getattr_first(
            text_config,
            ("num_local_experts", "num_experts", "n_routed_experts"),
        )
        if num_experts is None:
            return None
        num_groups = getattr(text_config, "n_group", None)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=num_experts,
            num_groups=num_groups,
        )

    @property
    def routed_experts_weights_of_layer(self) -> dict[int, list[torch.Tensor]]:
        return {
            fused.experts.layer_id: fused.get_moe_weights() for fused in self.moe_layers
        }

    def _get_expert_mapping(self, num_experts: int) -> List[Tuple[str, str, int, str]]:
        ckpt_names = [
            ("gate_proj", "down_proj", "up_proj"),
            ("w1", "w2", "w3"),
            ("linear", "linear_1", "linear_v"),
        ]
        mapping: list = []
        for gate, down, up in ckpt_names:
            mapping.extend(
                FusedMoE.make_expert_params_mapping(
                    ckpt_gate_proj_name=gate,
                    ckpt_down_proj_name=down,
                    ckpt_up_proj_name=up,
                    num_experts=num_experts,
                )
            )
        # AutoWeightsLoader dispatches to TransformersFusedMoE (which IS the
        # ``experts`` module) so the incoming weight names have the "experts."
        # prefix already stripped.  Remove it from weight_name in the mapping.
        mapping = [
            (pn, wn.removeprefix("experts."), eid, sid) for pn, wn, eid, sid in mapping
        ]
        return mapping

    def recursive_replace(self):
        """Replace experts modules with TransformersFusedMoE, then call
        super().recursive_replace() for Linear/RMSNorm replacement."""
        text_config = self.text_config

        num_experts = _getattr_first(
            text_config,
            ("num_local_experts", "num_experts", "n_routed_experts"),
        )
        assert num_experts is not None, "Cannot determine num_experts from config."

        top_k = _getattr_first(text_config, ("num_experts_per_tok", "top_k"))
        assert top_k is not None, "Cannot determine top_k from config."

        hidden_size = text_config.hidden_size
        intermediate_size = _getattr_first(
            text_config,
            ("moe_intermediate_size", "intermediate_size"),
        )
        assert intermediate_size is not None, "Cannot determine intermediate_size."

        num_shared_experts = _getattr_first(
            text_config,
            ("n_shared_experts", "moe_num_shared_experts"),
            0,
        )
        reduce_results = num_shared_experts == 0

        renormalize = getattr(text_config, "norm_topk_prob", top_k > 1)

        # Activation function
        activation = "silu"
        wrapped_arch = self.config.architectures[0].lower()
        if "gptoss" in wrapped_arch:
            activation = "swigluoai"
        elif "grok1" in wrapped_arch:
            activation = "gelu"

        # Expert mapping for AutoWeightsLoader
        expert_mapping = self._get_expert_mapping(num_experts)

        # EPLB / EP tracking
        num_redundant = get_global_server_args().ep_num_redundant_experts
        ep_size = get_moe_expert_parallel_world_size()

        self.mlp_moe_layers: list[nn.Module] = []
        self.moe_layers: list[TransformersFusedMoE] = []
        self.num_moe_layers = 0
        self.num_logical_experts = num_experts
        self.num_physical_experts = num_experts + num_redundant
        self.num_local_physical_experts = self.num_physical_experts // max(ep_size, 1)
        self.num_shared_experts = num_shared_experts
        self.num_redundant_experts = num_redundant

        def _add_all_reduce(mlp: nn.Module):
            class MLPWithAllReduce(mlp.__class__):
                def forward(self, *args, **kwargs):
                    output = super().forward(*args, **kwargs)
                    return self.experts.maybe_all_reduce_tensor_model_parallel(output)

            mlp.__class__ = MLPWithAllReduce

        def _recursive_replace(module: nn.Module, prefix: str):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)

                is_modulelist = isinstance(child_module, nn.ModuleList)
                params = list(child_module.parameters())
                is_3d = len(params) > 0 and all(p.ndim == 3 for p in params)

                if child_name == "experts" and (is_modulelist or is_3d):
                    mlp = module
                    experts = child_module

                    has_bias = any("bias" in n for n, _ in experts.named_parameters())

                    nonlocal reduce_results
                    if reduce_results:
                        if any("shared_expert" in n for n, _ in mlp.named_parameters()):
                            reduce_results = False
                            self.num_shared_experts = 1

                    layer_id = self.num_moe_layers

                    fused_experts = TransformersFusedMoE(
                        num_experts=num_experts,
                        top_k=top_k,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        layer_id=layer_id,
                        reduce_results=reduce_results,
                        quant_config=self.quant_config,
                        prefix=qual_name,
                        activation=activation,
                        with_bias=has_bias,
                        expert_mapping=expert_mapping,
                    )
                    mlp.experts = fused_experts
                    log_replacement(qual_name, experts, fused_experts)

                    self.mlp_moe_layers.append(mlp)
                    self.moe_layers.append(fused_experts)
                    self.num_moe_layers += 1

                    if not reduce_results and (
                        fused_experts.tp_size > 1 or fused_experts.ep_size > 1
                    ):
                        _add_all_reduce(mlp)
                else:
                    _recursive_replace(child_module, prefix=qual_name)

        _recursive_replace(self.model, prefix="model")
        super().recursive_replace()


class MultiModalMixin:
    torch_compile_dynamic_arg_dims: dict[str, int] = _MULTIMODAL_DYNAMIC_ARG_DIMS

    # Older VL checkpoints (e.g. Qwen2.5-VL) store text weights as
    # "model.layers.*" but transformers >=5.0 nests the text model under
    # "model.language_model.*".  Map explicitly so these load correctly.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model.": "model.language_model.",
            "text_model.model.": "model.text_model.",
            "text_model.lm_head.": "lm_head.",
            "language_model.lm_head.": "lm_head.",
            "vision_tower.": "model.vision_tower.",
            "vision_model.": "model.vision_model.",
            "vision_embed_tokens.": "model.vision_embed_tokens.",
            "image_newline.": "model.image_newline.",
            "vqmodel.": "model.vqmodel.",
            "multi_modal_projector.": "model.multi_modal_projector.",
            "visual.": "model.visual.",
            "model.layers.": "model.language_model.layers.",
            "model.embed_tokens.": "model.language_model.embed_tokens.",
            "model.norm.": "model.language_model.norm.",
            "model.rotary_emb.": "model.language_model.rotary_emb.",
        }
    )

    _mm_feature_kwarg = {
        "image": "pixel_values",
        "video": "pixel_values_videos",
        "audio": "input_features",
    }
    _mm_encoder_candidates = {
        "image": ("get_image_features", "get_image_feature"),
        "video": ("get_video_features", "get_video_feature"),
        "audio": ("get_audio_features", "get_audio_feature"),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mm_padding_pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def _uses_mrope_positions(self) -> bool:
        rope_scaling = getattr(self.text_config, "rope_scaling", None)
        if isinstance(rope_scaling, Mapping) and "mrope_section" in rope_scaling:
            return True
        rope_type = str(getattr(self.text_config, "rope_type", "")).lower()
        return "mrope" in rope_type

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        return input_ids

    def _get_modality_encoder(self, modality_name: str):
        for name in self._mm_encoder_candidates[modality_name]:
            fn = getattr(self.model, name, None)
            if fn is not None:
                return fn
        raise AttributeError(f"No encoder method found for modality '{modality_name}'")

    def _get_modality_dtype_device(
        self, modality_name: str
    ) -> tuple[Optional[torch.dtype], Optional[torch.device]]:
        module_candidates = {
            "image": ("vision_tower", "vision_model"),
            "video": ("video_tower", "vision_tower", "vision_model"),
            "audio": ("audio_tower", "audio_model", "audio_encoder"),
        }
        modules = []
        for name in module_candidates.get(modality_name, ()):
            module = getattr(self.model, name, None)
            if module is not None:
                modules.append(module)
        modules.append(self.model)

        for module in modules:
            for param in module.parameters():
                if torch.is_floating_point(param):
                    return param.dtype, param.device
            for buf in module.buffers():
                if torch.is_floating_point(buf):
                    return buf.dtype, buf.device
        return None, None

    def _cast_mm_value(self, value, dtype, device):
        if torch.is_tensor(value):
            if value.is_floating_point() and dtype is not None:
                return value.to(dtype=dtype, device=device)
            return value
        if isinstance(value, dict):
            return {k: self._cast_mm_value(v, dtype, device) for k, v in value.items()}
        if isinstance(value, list):
            return [self._cast_mm_value(v, dtype, device) for v in value]
        if isinstance(value, tuple):
            return tuple(self._cast_mm_value(v, dtype, device) for v in value)
        return value

    def _to_tensor_output(self, output) -> torch.Tensor:
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            output = output.pooler_output
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                raise ValueError("Empty multimodal encoder output.")
            if all(torch.is_tensor(x) for x in output):
                output = torch.cat(
                    [x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x for x in output],
                    dim=0,
                )
            else:
                output = output[0]
        elif hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        elif isinstance(output, dict):
            if output.get("pooler_output", None) is not None:
                output = output["pooler_output"]
            else:
                output = next(v for v in output.values() if torch.is_tensor(v))
            if isinstance(output, (list, tuple)):
                if len(output) == 0:
                    raise ValueError("Empty multimodal encoder output.")
                if all(torch.is_tensor(x) for x in output):
                    output = torch.cat(
                        [
                            x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
                            for x in output
                        ],
                        dim=0,
                    )
                else:
                    output = output[0]

        if output.ndim > 2:
            output = output.reshape(-1, output.shape[-1])
        return output

    def _encode_modality_items(
        self, modality_name: str, items: list[MultimodalDataItem]
    ) -> torch.Tensor:
        encoder = self._get_modality_encoder(modality_name)
        feature_kwarg = self._mm_feature_kwarg[modality_name]
        target_dtype, target_device = self._get_modality_dtype_device(modality_name)
        outputs = []
        for item in items:
            kwargs = self._cast_mm_value(
                dict(item.model_specific_data),
                dtype=target_dtype,
                device=target_device,
            )
            feature = self._cast_mm_value(
                item.feature,
                dtype=target_dtype,
                device=target_device,
            )
            if _encoder_accepts_feature_kwarg(encoder, feature_kwarg):
                kwargs[feature_kwarg] = feature
                result = encoder(**kwargs)
            else:
                result = encoder(feature, **kwargs)
            outputs.append(self._to_tensor_output(result))
        return torch.cat(outputs, dim=0)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._encode_modality_items("image", items)

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._encode_modality_items("video", items)

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._encode_modality_items("audio", items)

    def _collect_mm_kwargs(self, forward_batch: ForwardBatch) -> dict:
        """Collect multimodal tensors from the forward batch and return them
        as kwargs suitable for the HF model's forward method."""
        kwargs = {}

        if getattr(forward_batch, "token_type_ids", None) is not None:
            tti = forward_batch.token_type_ids
            if tti.ndim == 1:
                tti = tti.unsqueeze(0)
            token_type_key = (
                "mm_token_type_ids"
                if "mm_token_type_ids"
                in inspect.signature(self.model.forward).parameters
                else "token_type_ids"
            )
            kwargs[token_type_key] = tti

        if (
            not forward_batch.forward_mode.is_decode()
            and forward_batch.contains_mm_inputs()
        ):
            mm_inputs = forward_batch.mm_inputs
            target_device = next(self.model.parameters()).device

            for batch_idx in range(len(mm_inputs or [])):
                mm_input = mm_inputs[batch_idx]
                if mm_input is None:
                    continue
                for item in mm_input.mm_items or []:
                    for key, value in (item.model_specific_data or {}).items():
                        if isinstance(value, torch.Tensor):
                            value = value.to(device=target_device)
                        if key not in kwargs:
                            kwargs[key] = value
                        elif isinstance(value, torch.Tensor) and isinstance(
                            kwargs[key], torch.Tensor
                        ):
                            kwargs[key] = torch.cat([kwargs[key], value], dim=0)
                    if item.feature is not None:
                        feature_key = self._mm_feature_kwarg.get(
                            item.modality.name.lower(), "pixel_values"
                        )
                        feature = item.feature
                        if isinstance(feature, torch.Tensor):
                            feature = feature.to(device=target_device)
                        if feature_key not in kwargs:
                            kwargs[feature_key] = feature
                        elif isinstance(feature, torch.Tensor) and isinstance(
                            kwargs[feature_key], torch.Tensor
                        ):
                            kwargs[feature_key] = torch.cat(
                                [kwargs[feature_key], feature], dim=0
                            )

        return kwargs

    def _forward_hidden_states(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            return super()._forward_hidden_states(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )

        if (
            self._uses_mrope_positions()
            and getattr(forward_batch, "mrope_positions", None) is not None
        ):
            positions = forward_batch.mrope_positions

        mm_kwargs = self._collect_mm_kwargs(forward_batch)

        return self._run_hf_backbone(
            input_ids=input_ids,
            input_embeds=None,
            positions=positions,
            forward_batch=forward_batch,
            **mm_kwargs,
        )


class TransformersForCausalLM(CausalMixin, TransformersBase):
    pass


class TransformersMoEForCausalLM(MoEMixin, CausalMixin, TransformersBase):
    pass


class TransformersMultiModalForCausalLM(MultiModalMixin, CausalMixin, TransformersBase):
    pass


class TransformersMultiModalMoEForCausalLM(
    MultiModalMixin, MoEMixin, CausalMixin, TransformersBase
):
    pass


class TransformersEmbeddingModel(EmbeddingMixin, TransformersBase):
    pass


class TransformersMoEEmbeddingModel(MoEMixin, EmbeddingMixin, TransformersBase):
    pass


class TransformersMultiModalEmbeddingModel(
    MultiModalMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersMultiModalMoEEmbeddingModel(
    MultiModalMixin, MoEMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersForSequenceClassification(EmbeddingMixin, TransformersBase):
    pass


class TransformersMoEForSequenceClassification(
    MoEMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersMultiModalForSequenceClassification(
    MultiModalMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersMultiModalMoEForSequenceClassification(
    MultiModalMixin, MoEMixin, EmbeddingMixin, TransformersBase
):
    pass


EntryClass = [
    TransformersForCausalLM,
    TransformersMoEForCausalLM,
    TransformersMultiModalForCausalLM,
    TransformersMultiModalMoEForCausalLM,
    TransformersEmbeddingModel,
    TransformersMoEEmbeddingModel,
    TransformersMultiModalEmbeddingModel,
    TransformersMultiModalMoEEmbeddingModel,
    TransformersForSequenceClassification,
    TransformersMoEForSequenceClassification,
    TransformersMultiModalForSequenceClassification,
    TransformersMultiModalMoEForSequenceClassification,
]
