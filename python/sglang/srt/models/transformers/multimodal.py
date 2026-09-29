# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team

import inspect
import logging
from array import array
from collections.abc import Mapping
from typing import Optional

import torch

from sglang.srt.configs.transformers_backend import (
    supports_transformers_multimodal_cache,
)
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.utils import WeightsMapper

from .multimodal_utils import flatten_encoder_features

logger = logging.getLogger(__name__)


_MULTIMODAL_DYNAMIC_ARG_DIMS = {"input_ids": 0, "positions": -1, "input_embeds": 0}


def _encoder_accepts_feature_kwarg(encoder, feature_kwarg: str) -> bool:
    try:
        sig = inspect.signature(encoder)
    except (TypeError, ValueError):
        return False
    if feature_kwarg in sig.parameters:
        return True
    has_var_keyword = any(
        (p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
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


class MultiModalMixin:
    torch_compile_dynamic_arg_dims: dict[str, int] = _MULTIMODAL_DYNAMIC_ARG_DIMS
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
        self._mm_cache_enabled = supports_transformers_multimodal_cache(self.config)
        if self._mm_cache_enabled and not type(self.model).__module__.startswith(
            f"transformers.models.{self.config.model_type}."
        ):
            raise ValueError(
                "Multimodal caching requires the registered Transformers model implementation"
            )
        vt = getattr(self.model, "vision_tower", None)
        if vt is not None and (
            not any((name == "vision_model" for name, _ in vt.named_children()))
        ):
            self.weight_mapper = (
                WeightsMapper(
                    orig_to_new_prefix={
                        "vision_tower.vision_model.": "model.vision_tower."
                    }
                )
                | self.weight_mapper
            )

    def _uses_mrope_positions(self) -> bool:
        rope_scaling = getattr(self.text_config, "rope_parameters", None) or getattr(
            self.text_config, "rope_scaling", None
        )
        if isinstance(rope_scaling, Mapping) and "mrope_section" in rope_scaling:
            return True
        rope_type = str(getattr(self.text_config, "rope_type", "")).lower()
        return "mrope" in rope_type

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        if self._mm_cache_enabled:
            return self._mm_padding_pattern.pad_input_tokens(input_ids, mm_inputs)
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
            "image": ("vision_tower", "vision_model", "visual"),
            "video": ("video_tower", "vision_tower", "vision_model", "visual"),
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
                    return (param.dtype, param.device)
            for buf in module.buffers():
                if torch.is_floating_point(buf):
                    return (buf.dtype, buf.device)
        return (None, None)

    def _cast_mm_value(self, value, dtype, device):
        if torch.is_tensor(value):
            if value.is_floating_point() and dtype is not None:
                return value.to(dtype=dtype, device=device)
            return value.to(device=device)
        if isinstance(value, dict):
            return {k: self._cast_mm_value(v, dtype, device) for k, v in value.items()}
        if isinstance(value, list):
            return [self._cast_mm_value(v, dtype, device) for v in value]
        if isinstance(value, tuple):
            return tuple((self._cast_mm_value(v, dtype, device) for v in value))
        return value

    def _to_tensor_output(self, output) -> torch.Tensor:
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            output = output.pooler_output
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                raise ValueError("Empty multimodal encoder output.")
            if all((torch.is_tensor(x) for x in output)):
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
                output = next((v for v in output.values() if torch.is_tensor(v)))
            if isinstance(output, (list, tuple)):
                if len(output) == 0:
                    raise ValueError("Empty multimodal encoder output.")
                if all((torch.is_tensor(x) for x in output)):
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
                dict(item.model_specific_data), dtype=target_dtype, device=target_device
            )
            feature = self._cast_mm_value(
                item.feature, dtype=target_dtype, device=target_device
            )
            if _encoder_accepts_feature_kwarg(encoder, feature_kwarg):
                kwargs[feature_kwarg] = feature
                result = encoder(**kwargs)
            else:
                result = encoder(feature, **kwargs)
            features = flatten_encoder_features(result)
            expected_tokens = sum(end - start + 1 for start, end in item.offsets or [])
            if item.offsets and features.shape[0] != expected_tokens:
                raise ValueError(
                    f"{modality_name} encoder returned {features.shape[0]} tokens for {expected_tokens} placeholders"
                )
            outputs.append(features)
        return torch.cat(outputs, dim=0)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._encode_modality_items("image", items)

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._encode_modality_items("video", items)

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._encode_modality_items("audio", items)

    def _collect_mm_kwargs(self, forward_batch: ForwardBatch) -> dict:
        """Collect multimodal tensors from the forward batch and return them"""
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
            pending_5d_features: dict = {}
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
                            if feature.dim() == 5:
                                pending_5d_features.setdefault(feature_key, []).append(
                                    feature
                                )
                                continue
                        if feature_key not in kwargs:
                            kwargs[feature_key] = feature
                        elif isinstance(feature, torch.Tensor) and isinstance(
                            kwargs[feature_key], torch.Tensor
                        ):
                            kwargs[feature_key] = torch.cat(
                                [kwargs[feature_key], feature], dim=0
                            )
            for feature_key, tensors in pending_5d_features.items():
                if len({tuple(t.shape[1:]) for t in tensors}) != 1:
                    raise ValueError(
                        "This multimodal adapter cannot batch unequal patch shapes"
                    )
                combined = torch.cat(tensors, dim=0)
                if feature_key in kwargs:
                    kwargs[feature_key] = torch.cat(
                        [kwargs[feature_key], combined], dim=0
                    )
                else:
                    kwargs[feature_key] = combined
        return kwargs

    def _forward_hidden_states(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            self._uses_mrope_positions()
            and getattr(forward_batch, "mrope_positions", None) is not None
        ):
            positions = forward_batch.mrope_positions
        if input_embeds is not None:
            return super()._forward_hidden_states(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )
        if self._mm_cache_enabled:
            from sglang.srt.managers.mm_utils import embed_mm_inputs

            if (
                not forward_batch.forward_mode.is_decode()
                and not forward_batch.forward_mode.is_target_verify()
                and forward_batch.contains_mm_inputs()
            ):
                active = [
                    index
                    for index, item in enumerate(forward_batch.mm_inputs)
                    if item is not None
                ]
                input_embeds, _ = embed_mm_inputs(
                    mm_inputs_list=[forward_batch.mm_inputs[index] for index in active],
                    extend_prefix_lens=[
                        forward_batch.extend_prefix_lens_cpu[index] for index in active
                    ],
                    extend_seq_lens=[
                        forward_batch.extend_seq_lens_cpu[index] for index in active
                    ],
                    input_ids=input_ids.clone(),
                    input_embedding=self.model.get_input_embeddings(),
                    multimodal_model=self,
                )
                forward_batch.mm_input_embeds = input_embeds
            else:
                input_embeds = self.model.get_input_embeddings()(input_ids)
            return self._run_hf_backbone(
                input_ids=None,
                input_embeds=input_embeds,
                positions=positions,
                forward_batch=forward_batch,
            )
        mm_kwargs = self._collect_mm_kwargs(forward_batch)
        return self._run_hf_backbone(
            input_ids=input_ids,
            input_embeds=None,
            positions=positions,
            forward_batch=forward_batch,
            **mm_kwargs,
        )
