# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""
Inference-only pi0.5 VLA model for SGLang.

Compared with pi0:
- robot state is expected to be serialized into prompt tokens upstream
- action expert uses AdaRMS conditioning
- timestep conditioning uses time_mlp_{in,out}, not action_time_mlp_{in,out}
"""

import copy
import logging
import math
import re
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaConfig,
    GemmaForCausalLM,
    GemmaMLP,
    GemmaModel,
    apply_rotary_pos_emb,
)
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaModel,
)

from sglang.srt.configs.pi05 import Pi05Config  # noqa: F401
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils.common import print_warning_once

from .pi0 import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ACTION_HORIZON,
    DEFAULT_NUM_INFERENCE_STEPS,
    OPENPI_ATTENTION_MASK_VALUE,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)

logger = logging.getLogger(__name__)


def _gated_residual(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
) -> torch.Tensor | None:
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def layernorm_forward(
    layernorm: nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor | None = None,
):
    if cond is not None:
        return layernorm(x, cond=cond)
    return layernorm(x)


class PiGemmaRMSNorm(nn.Module):
    """AdaRMS used by pi0.5."""

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = x.dtype
        normed = self._norm(x)
        if cond is None or self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.to(dtype), None

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")

        modulation = self.dense(cond)
        if x.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


def _get_pi_gemma_decoder_layer_base():
    class _PiGemmaDecoderLayerBase(GradientCheckpointingLayer):
        def __init__(self, config: GemmaConfig, layer_idx: int):
            super().__init__()
            self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
            self.mlp = GemmaMLP(config)
            cond_dim = (
                getattr(config, "adarms_cond_dim", None)
                if getattr(config, "use_adarms", False)
                else None
            )
            self.input_layernorm = PiGemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )
            self.post_attention_layernorm = PiGemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values=None,
            use_cache: bool = False,
            cache_position: torch.LongTensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            adarms_cond: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            residual = hidden_states
            hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)
            hidden_states, _ = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = _gated_residual(residual, hidden_states, gate)

            residual = hidden_states
            hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
            hidden_states = self.mlp(hidden_states)
            hidden_states = _gated_residual(residual, hidden_states, gate)
            return hidden_states

    return _PiGemmaDecoderLayerBase


class PiGemmaModel(GemmaModel):  # type: ignore[misc]
    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        cond_dim = getattr(config, "adarms_cond_dim", None)
        pi_layer = _get_pi_gemma_decoder_layer_base()
        self.layers = nn.ModuleList(
            [pi_layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PiGemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        if len(self.layers) > 0 and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, adarms_cond)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PiGemmaForCausalLM(GemmaForCausalLM):  # type: ignore[misc]
    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = PiGemmaModel(config)


class PaliGemmaModelWithPiGemma(PaliGemmaModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = PiGemmaModel(config.text_config)


class PaliGemmaForConditionalGenerationWithPiGemma(PaliGemmaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = PaliGemmaModelWithPiGemma(config)

    @property
    def language_model(self):
        return self.model.language_model


def _compute_layer_joint(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
):
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []

    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    scaling = 1.0 / math.sqrt(head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    att_output = torch.matmul(attn_weights, value_states)

    batch_size = query_states.shape[0]
    num_heads = query_states.shape[1]
    att_output = att_output.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]

        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb

        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        if out_emb.dtype != layer.mlp.up_proj.weight.dtype:
            out_emb = out_emb.to(dtype=layer.mlp.up_proj.weight.dtype)
        out_emb = layer.mlp(out_emb)
        out_emb = _gated_residual(after_first_residual, out_emb, gate)

        outputs_embeds.append(out_emb)
        start_pos = end_pos

    return outputs_embeds


class PaliGemmaWithActionExpertPi05(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        image_size: int = 224,
        vocab_size: int = 257152,
        image_token_index: int = 257152,
    ):
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = vocab_size
        vlm_config_hf.image_token_index = image_token_index
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = vocab_size
        vlm_config_hf.text_config.use_adarms = False
        vlm_config_hf.text_config.adarms_cond_dim = None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=vocab_size,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=True,
            adarms_cond_dim=action_expert_config.width,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out_dtype = pixel_values.dtype
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)
        if hasattr(self.paligemma, "get_image_features"):
            image_outputs = self.paligemma.get_image_features(pixel_values)
            if hasattr(image_outputs, "pooler_output"):
                features = image_outputs.pooler_output
            elif isinstance(image_outputs, (tuple, list)):
                features = image_outputs[0]
            else:
                features = image_outputs
        else:
            vision_outputs = self.paligemma.model.vision_tower(pixel_values)
            features = vision_outputs.last_hidden_state if hasattr(vision_outputs, "last_hidden_state") else vision_outputs[0]
            features = self.paligemma.model.multi_modal_projector(features)
        features = features * (self.paligemma.config.text_config.hidden_size**0.5)
        return features.to(out_dtype) if features.dtype != out_dtype else features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]

        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            return [prefix_output.last_hidden_state, None], prefix_output.past_key_values

        if inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1],
            )
            return [None, suffix_output.last_hidden_state], None

        models = [self.paligemma.model.language_model, self.gemma_expert.model]
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        for layer_idx in range(num_layers):
            inputs_embeds = _compute_layer_joint(
                layer_idx,
                inputs_embeds,
                attention_mask,
                position_ids,
                adarms_cond,
                paligemma=self.paligemma,
                gemma_expert=self.gemma_expert,
            )

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
            outputs_embeds.append(out_emb)

        return outputs_embeds, None


class Pi05ForActionPrediction(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.action_dim = getattr(config, "max_action_dim", DEFAULT_ACTION_DIM)
        self.action_horizon = getattr(config, "chunk_size", DEFAULT_ACTION_HORIZON)
        self.num_inference_steps = getattr(config, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)

        paligemma_variant = getattr(config, "paligemma_variant", "gemma_2b")
        action_expert_variant = getattr(config, "action_expert_variant", "gemma_300m")
        vlm_config = get_gemma_config(paligemma_variant)
        expert_config = get_gemma_config(action_expert_variant)
        self.expert_width = expert_config.width

        image_resolution = getattr(config, "image_resolution", (224, 224))
        image_size = image_resolution[0] if isinstance(image_resolution, (tuple, list)) else 224
        vocab_size = getattr(config, "vocab_size", 257152)
        image_token_index = getattr(config, "image_token_index", vocab_size)
        self.paligemma_with_expert = PaliGemmaWithActionExpertPi05(
            vlm_config,
            expert_config,
            image_size=image_size,
            vocab_size=vocab_size,
            image_token_index=image_token_index,
        )

        self.action_in_proj = nn.Linear(self.action_dim, self.expert_width)
        self.action_out_proj = nn.Linear(self.expert_width, self.action_dim)

        self.time_mlp_in = nn.Linear(self.expert_width, self.expert_width)
        self.time_mlp_out = nn.Linear(self.expert_width, self.expert_width)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def embed_prefix(
        self,
        images: List[torch.Tensor],
        image_masks: List[torch.Tensor],
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, image_masks):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], -1)
        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []

        model_dtype = self.action_in_proj.weight.dtype
        noisy_actions = noisy_actions.to(dtype=model_dtype)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        ).to(dtype=model_dtype)

        action_emb = self.action_in_proj(noisy_actions)

        time_cond = self.time_mlp_in(time_emb)
        time_cond = F.silu(time_cond)
        time_cond = self.time_mlp_out(time_cond)
        time_cond = F.silu(time_cond)

        embs.append(action_emb)
        bsize, action_len = action_emb.shape[:2]
        pad_masks.append(torch.ones(bsize, action_len, dtype=torch.bool, device=action_emb.device))
        att_masks += [1] + [0] * (self.action_horizon - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, -1)

        return embs, pad_masks, att_masks, time_cond

    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
        past_key_values = copy.deepcopy(past_key_values)

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions(
        self,
        images: List[torch.Tensor],
        image_masks: List[torch.Tensor],
        tokens: torch.Tensor,
        masks: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            noise = torch.randn(
                bsize,
                self.action_horizon,
                self.action_dim,
                dtype=torch.float32,
                device=device,
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, image_masks, tokens, masks
        )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t

    def _extract_request_inputs(self, mm_input, device: torch.device):
        if mm_input is None:
            raise ValueError("pi0.5 requires processor-backed mm_inputs; the old embed_mm_inputs text path is unsupported.")
        if not mm_input.mm_items:
            raise ValueError("pi0.5 requires at least one multimodal item in mm_inputs.")
        item = mm_input.mm_items[0]
        meta = item.model_specific_data or {}
        missing = [k for k in ("lang_tokens", "lang_attention_mask") if k not in meta]
        if missing:
            raise ValueError(f"pi0.5 processor metadata is incomplete; missing fields: {missing}.")
        pixel_values = item.feature if isinstance(item.feature, torch.Tensor) else torch.as_tensor(item.feature)
        pixel_values = pixel_values.to(device=device)
        image_masks = torch.as_tensor(
            meta.get("image_masks", torch.ones(pixel_values.shape[0])),
            dtype=torch.bool,
            device=device,
        )
        lang_tokens = torch.as_tensor(meta["lang_tokens"], dtype=torch.long, device=device)
        lang_masks = torch.as_tensor(meta["lang_attention_mask"], dtype=torch.bool, device=device)
        if lang_tokens.dim() == 1:
            lang_tokens = lang_tokens.unsqueeze(0)
        if lang_masks.dim() == 1:
            lang_masks = lang_masks.unsqueeze(0)
        return pixel_values, image_masks, lang_tokens, lang_masks, meta.get("num_inference_steps")

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_extend():
            all_actions = []
            device = input_ids.device

            for i in range(forward_batch.batch_size):
                mm_input = forward_batch.mm_inputs[i] if forward_batch.mm_inputs else None
                if mm_input is None or not getattr(mm_input, "mm_items", None):
                    print_warning_once(
                        "pi0.5: extend forward got no mm_input. Expected during warmup/CUDA-graph capture; if seen repeatedly on real traffic, the client is sending a non-multimodal request to a VLA model."
                    )
                    all_actions.append(
                        torch.zeros(
                            1,
                            self.action_horizon,
                            self.action_dim,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    continue
                pixel_values, image_masks, lang_tokens, lang_masks, num_steps = self._extract_request_inputs(
                    mm_input, device
                )
                all_actions.append(
                    self.sample_actions(
                        images=[pixel_values[c : c + 1] for c in range(pixel_values.shape[0])],
                        image_masks=[image_masks[c : c + 1] for c in range(image_masks.shape[0])],
                        tokens=lang_tokens,
                        masks=lang_masks,
                        num_steps=num_steps,
                    )
                )

            if all_actions:
                actions = torch.cat(all_actions, dim=0)
                actions_flat = actions.reshape(actions.shape[0], -1)
            else:
                actions_flat = torch.zeros(
                    1, self.action_horizon * self.action_dim, device=device
                )

            return LogitsProcessorOutput(next_token_logits=actions_flat)

        if forward_batch.forward_mode.is_decode():
            dummy = torch.zeros(
                forward_batch.batch_size,
                self.action_horizon * self.action_dim,
                device=input_ids.device,
            )
            return LogitsProcessorOutput(next_token_logits=dummy)

        raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        paligemma_submodules = ("vision_tower", "multi_modal_projector", "language_model")

        def _remap(name: str) -> str:
            if name.startswith("model."):
                name = name[len("model."):]
            if name.startswith("action_time_mlp_in."):
                name = "time_mlp_in." + name[len("action_time_mlp_in."):]
            elif name.startswith("action_time_mlp_out."):
                name = "time_mlp_out." + name[len("action_time_mlp_out."):]
            for sub in paligemma_submodules:
                flat = f"paligemma_with_expert.paligemma.{sub}."
                nested = f"paligemma_with_expert.paligemma.model.{sub}."
                if name.startswith(flat) and not name.startswith(nested):
                    return nested + name[len(flat):]
            if name == "paligemma_with_expert.paligemma.lm_head.weight":
                return "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            return name

        for name, loaded_weight in weights:
            name = _remap(name)

            if name.startswith("state_proj."):
                logger.debug("Skipping state_proj weight in pi0.5: %s", name)
                continue

            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                name,
            ):
                logger.debug("Skipping old norm.weight key for AdaRMS expert: %s", name)
                continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", name):
                logger.debug("Skipping old final norm.weight key for AdaRMS expert: %s", name)
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            elif name in buffers_dict:
                buffers_dict[name].copy_(loaded_weight)
            else:
                logger.debug("Skipping weight: %s", name)


EntryClass = Pi05ForActionPrediction
