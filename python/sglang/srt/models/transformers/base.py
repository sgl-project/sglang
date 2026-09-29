# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team


import logging
import re
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel

from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device
from sglang.srt.utils.hf_transformers_utils import get_hf_text_config

from .attention import AttentionMixin
from .execution_context import wrap_forward_with_context
from .graph import TransformersGraphMixin
from .layers import replace_linear_class, replace_rms_norm_class
from .lora import TransformersLoRAMixin
from .parallel import ParallelMixin
from .speculative import SpeculativeMixin
from .utils import (
    _getattr_first,
    _init_on_device_without_buffers,
    _resolve_attention_backend_model_cls,
    can_enable_torch_compile,
    log_replacement,
    maybe_prefix,
)

logger = logging.getLogger(__name__)

_BASE_DYNAMIC_ARG_DIMS = {"input_ids": 0, "positions": 0, "input_embeds": 0}


class TransformersBase(
    AttentionMixin,
    ParallelMixin,
    SpeculativeMixin,
    TransformersLoRAMixin,
    TransformersGraphMixin,
    nn.Module,
):
    supports_model_config = True
    wrap_forward_context = staticmethod(wrap_forward_with_context)
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
            "gpt_neox.": "model.",
            "embed_out.": "lm_head.",
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
        model_config=None,
    ) -> None:
        super().__init__()
        logger.info("Using Transformers backend.")
        self.quant_config = quant_config
        self.config = config
        self.model_config = model_config
        self.trust_remote_code = bool(getattr(model_config, "trust_remote_code", False))
        self.text_config = get_hf_text_config(config)
        self.weight_mapper = self.hf_to_sglang_mapper
        self.pp_group = get_parallel().pp_group
        if get_parallel().attn_tp_size != get_parallel().tp_size:
            raise ValueError(
                "Transformers models require matching attention and model TP layouts"
            )
        if getattr(config, "is_encoder_decoder", False):
            raise ValueError(
                "Encoder-decoder Transformers models require a separate cache adapter"
            )
        self._stacked_mapping = {}
        self._weights_loaded = False
        self._aux_capture = None
        self.capture_aux_hidden_states = False
        self.packed_modules_mapping = {}
        self.skip_prefixes: list[str] = []
        self.skip_substrs: list[str] = []
        self.ignore_unexpected_prefixes: list[str] = []
        self.ignore_unexpected_suffixes: list[str] = []
        self.skip_substrs.extend(
            [".attn.bias", ".attn.masked_bias", ".attention.bias", ".masked_bias"]
        )
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
        model_cls = _resolve_attention_backend_model_cls(
            config, self.trust_remote_code, getattr(model_config, "revision", None)
        )
        supports_backend = (
            getattr(model_cls, "_supports_attention_backend", True)
            if model_cls
            else True
        )
        self.text_config._attn_implementation = "sglang"
        if supports_backend:
            with _init_on_device_without_buffers(torch.device("meta")):
                self.model: PreTrainedModel = self._build_model(self.config)
        else:
            raise ValueError(
                f"Model {model_cls} does not support custom attention backends (_supports_attention_backend=False). The Transformers backend requires custom attention support."
            )
        self.vocab_size = getattr(
            self.text_config,
            "vocab_size",
            self.model.get_input_embeddings().num_embeddings,
        )
        self.unpadded_vocab_size = self.vocab_size
        self.start_layer = 0
        self.end_layer = getattr(self.text_config, "num_hidden_layers", 0)
        self.pipeline_parallel()
        self.recursive_replace()
        self.attention_instances = self._create_attention_instances()
        self.uses_native_mla = (
            getattr(getattr(model_config, "attention_arch", None), "name", None)
            == "MLA"
        )
        if self.uses_native_mla:
            from .mla import install_mla_adapters

            install_mla_adapters(
                self.model, self.attention_instances, self.quant_config
            )
        self.replace_vocab_embed_class(self.model)
        self._init_parameters(self.model)
        self.lm_head: Optional[ParallelLMHead] = None
        self.logits_processor: Optional[LogitsProcessor] = None
        self.pooler: Optional[Pooler] = None
        self._configure_task()
        for name in ("task_head", "pooler"):
            module = getattr(self, name, None)
            if module is not None:
                self._init_parameters(module)
        self._compile_compatible = can_enable_torch_compile(config)
        self.model.torch_compile_dynamic_arg_dims = {
            "input_ids": 1,
            "inputs_embeds": 1,
            "position_ids": -1,
            "token_type_ids": 1,
        }
        self.eval()

    def _build_model(self, config):
        return AutoModel.from_config(
            config,
            torch_dtype=torch.get_default_dtype(),
            trust_remote_code=self.trust_remote_code,
        )

    def _configure_task(self):
        pass

    def _validate_task_weights(self, loaded):
        pass

    def _pool_output(self, hidden_states, forward_batch):
        if self.pooler is None:
            raise ValueError("Pooling is not enabled for this model")
        return self.pooler(hidden_states, forward_batch)

    @property
    def _can_torch_compile(self) -> bool:
        """Whether this model instance is safe to wrap with torch.compile."""
        return self._compile_compatible

    @property
    def _can_cuda_graph(self) -> bool:
        return self._compile_compatible

    def _init_parameters(self, module: nn.Module):
        """Materialize any parameters still on the meta device."""
        for name, param in module.named_parameters(recurse=False):
            if param.device == torch.device("meta"):
                new_param = nn.Parameter(
                    torch.empty_like(param.data, device=get_device())
                )
                setattr(module, name, new_param)
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.is_meta:
                raise ValueError(
                    f"Cannot initialize model buffer {name} without its values"
                )
            setattr(module, name, buffer.to(device=get_device()))
        for child in module.children():
            self._init_parameters(child)

    def recursive_replace(self):
        from sglang.srt.environ import envs

        from .fusers import fuse_module
        from .fusers.residual import fuse_residual_norm
        from .fusers.rope import fuse_rotary_embedding, replace_rotary_embedding
        from .layers import get_attention_projection_shards

        enable_fusions = envs.SGLANG_ENABLE_TRANSFORMERS_FUSIONS.get()
        disabled_fusions = set(envs.SGLANG_TRANSFORMERS_DISABLED_FUSIONS.get())
        unknown = disabled_fusions - {"qkv", "mlp", "norm", "residual", "rope"}
        if unknown:
            raise ValueError(f"Unknown Transformers fusions: {sorted(unknown)}")
        if not enable_fusions:
            disabled_fusions.update({"qkv", "mlp", "residual", "rope"})
        self.transformers_fusion_counts = {
            name: 0 for name in ("qkv", "mlp", "norm", "residual", "rope")
        }
        tp_size = get_parallel().tp_size
        tp_plan = self._normalize_tp_plan(self._get_model_tp_plan())
        if (
            getattr(getattr(self.model_config, "attention_arch", None), "name", None)
            == "MLA"
        ):
            from .mla import mla_attention_tp_plan

            tp_plan.update(mla_attention_tp_plan(self.model))
        if not tp_plan and tp_size > 1:
            raise ValueError(
                f"{type(self.model)} does not support tensor parallel yet!"
            )
        prefixed_plan = {maybe_prefix("model", k): v for k, v in tp_plan.items()}

        def _recursive_replace(module: nn.Module, prefix: str):
            projection_shards = get_attention_projection_shards(
                module, prefix, prefixed_plan
            )
            result = (
                fuse_module(
                    module,
                    prefix,
                    self.quant_config,
                    prefixed_plan,
                    packed_modules_mapping=self.packed_modules_mapping,
                    disabled_fusions=disabled_fusions,
                )
                if enable_fusions
                else None
            )
            if result is not None:
                self._stacked_mapping.update(result.stacked_mapping)
                self.packed_modules_mapping.update(result.packed_modules_mapping)
                kind = "qkv" if "qkv_proj" in result.packed_modules_mapping else "mlp"
                self.transformers_fusion_counts[kind] += 1
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                new_module = child_module
                if isinstance(child_module, nn.Linear):
                    pattern = next(
                        (p for p in prefixed_plan if re.fullmatch(p, qual_name)), None
                    )
                    style = prefixed_plan.get(pattern, "replicate")
                    new_module = replace_linear_class(
                        child_module,
                        style,
                        self.quant_config,
                        prefix=qual_name,
                        **projection_shards.get(qual_name, {}),
                    )
                elif (
                    child_module.__class__.__name__.endswith("RMSNorm")
                    and "norm" not in disabled_fusions
                ):
                    new_module = replace_rms_norm_class(
                        child_module, self.text_config.hidden_size
                    )
                    self.transformers_fusion_counts["norm"] += int(
                        new_module is not child_module
                    )
                else:
                    if "rope" not in disabled_fusions:
                        new_module = replace_rotary_embedding(child_module)
                    _recursive_replace(new_module, prefix=qual_name)
                if new_module is not child_module:
                    setattr(module, child_name, new_module)
                    log_replacement(qual_name, child_module, new_module)
            if "residual" not in disabled_fusions:
                self.transformers_fusion_counts["residual"] += int(
                    fuse_residual_norm(module)
                )
            if "rope" not in disabled_fusions:
                self.transformers_fusion_counts["rope"] += int(
                    fuse_rotary_embedding(module)
                )

        _recursive_replace(self.model, prefix="model")

    def replace_vocab_embed_class(self, module: nn.Module):
        from .embedding import ScaledVocabParallelEmbedding, scaled_embedding_contract

        old_module = self.model.get_input_embeddings()
        if old_module is None or isinstance(old_module, PPMissingLayer):
            return
        scaled = scaled_embedding_contract(old_module)
        if type(old_module).forward is not nn.Embedding.forward and scaled is None:
            if get_parallel().tp_size > 1:
                raise ValueError(
                    f"Custom embedding {type(old_module).__name__} has no tensor-parallel adapter"
                )
            return
        embedding_dim = getattr(old_module, "embedding_dim", None)
        if embedding_dim is None:
            embedding_dim = _getattr_first(
                self.text_config, ("embedding_size", "hidden_size"), None
            )
        assert embedding_dim is not None
        if getattr(old_module, "max_norm", None) is not None:
            raise ValueError(
                "Transformers embedding max_norm mutates weights during inference"
            )
        embedding_class = (
            VocabParallelEmbedding if scaled is None else ScaledVocabParallelEmbedding
        )
        new_module = embedding_class(
            self.vocab_size,
            embedding_dim,
            org_num_embeddings=self.vocab_size,
            quant_config=None,
        )
        if scaled is not None:
            new_module.set_scale(old_module.embed_scale, scaled)
        log_replacement("input embedding", old_module, new_module)
        self.model.set_input_embeddings(new_module)

    def _format_position_ids(
        self, positions: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.text_config.model_type in {"roberta", "xlm-roberta", "camembert"}:
            padding = self.model.embeddings.padding_idx
            if input_ids is None:
                positions = positions + padding + 1
            else:
                mask = (input_ids != padding).long()
                cumulative = mask.cumsum(0)
                offsets = (
                    torch.where(positions == 0, cumulative - mask, 0).cummax(0).values
                )
                positions = (cumulative - offsets) * mask + padding
        if positions.ndim == 2 and positions.shape[0] == 3:
            return positions[:, None, ...]
        if positions.ndim == 1:
            return positions[None, ...]
        return positions

    def _run_hf_backbone_eager(
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
        if (
            "token_type_ids" not in kwargs
            and getattr(forward_batch, "token_type_ids", None) is not None
        ):
            token_types = forward_batch.token_type_ids
            kwargs["token_type_ids"] = (
                token_types[None, ...] if token_types.ndim == 1 else token_types
            )
        from .execution_context import transformers_execution_context

        with transformers_execution_context(forward_batch):
            return self.model(
                input_ids=hf_input_ids,
                inputs_embeds=hf_input_embeds,
                use_cache=False,
                position_ids=self._format_position_ids(positions, input_ids),
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
        if self._aux_capture is not None:
            self._aux_capture.reset()
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
            return self._pool_output(hidden_states, forward_batch)
        assert self.logits_processor is not None and self.lm_head is not None
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            self._aux_capture.collect() if self._aux_capture is not None else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from .fusers import load_fused_weights

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=self.skip_prefixes,
            skip_substrs=self.skip_substrs,
            ignore_unexpected_prefixes=self.ignore_unexpected_prefixes,
            ignore_unexpected_suffixes=self.ignore_unexpected_suffixes,
        )
        loaded = set()
        weights = self.weight_mapper.apply(weights)
        weights = (
            (name, value) for name, value in weights if not loader._can_skip(name)
        )
        weights = load_fused_weights(
            self,
            weights,
            self._stacked_mapping,
            loaded,
            ignore_unexpected_suffixes=self.ignore_unexpected_suffixes,
            require_complete=not self._weights_loaded,
        )
        loaded.update(loader.load_weights(weights))
        self._validate_task_weights(loaded)
        if hasattr(self, "_validate_expert_weights"):
            self._validate_expert_weights()
        if not self._weights_loaded and self.quant_config is None:
            required = {
                name
                for name, _ in self.named_parameters()
                if not loader._can_skip(name)
            }
            missing = required - loaded
            if missing:
                raise ValueError(
                    f"Checkpoint is missing required parameters: {sorted(missing)}"
                )
        self.post_load_weights()
        if not self._weights_loaded:
            logger.info(
                "Transformers backend: wrapper=%s, backbone=%s, fusions=%s, native_mla=%s",
                type(self).__name__,
                type(self.model).__name__,
                self.transformers_fusion_counts,
                self.uses_native_mla,
            )
        self._weights_loaded = True
        return loaded

    def post_load_weights(self):
        if self.uses_native_mla:
            from .mla import refresh_mla_weights

            refresh_mla_weights(self)
