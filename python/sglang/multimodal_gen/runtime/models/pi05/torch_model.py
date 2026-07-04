# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenPI and LeRobot Pi0.5 PyTorch inference semantics.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.pi05.pi_gemma import (
    PaliGemmaForConditionalGenerationWithPiGemma,
    PiGemmaForCausalLM,
    gated_residual,
    layernorm_forward,
)

OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38
PALIGEMMA_VOCAB_SIZE = 257_152


@dataclass(frozen=True)
class GemmaVariantConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_gemma_variant_config(variant: str) -> GemmaVariantConfig:
    if variant == "gemma_300m":
        return GemmaVariantConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return GemmaVariantConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    raise ValueError(f"Unknown Pi05 Gemma variant: {variant}")


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape [batch]")
    fraction = torch.linspace(
        0.0,
        1.0,
        dimension // 2,
        dtype=torch.float64,
        device=time.device,
    )
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * time[:, None].to(torch.float64)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
) -> torch.Tensor:
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError("pad_masks and att_masks must be [batch, seq]")
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def trim_trailing_padding_tokens(
    tokens: torch.Tensor,
    token_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_len = int(token_masks.sum(dim=1).max().item())
    if token_len <= 0 or token_len >= tokens.shape[1]:
        return tokens, token_masks
    return tokens[:, :token_len], token_masks[:, :token_len]


def prepare_optional_full_attention_mask(
    att_2d_masks: torch.Tensor,
    *,
    full_attention: bool | None = None,
) -> torch.Tensor | None:
    if full_attention is None:
        full_attention = bool(att_2d_masks.all().item())
    if full_attention:
        return None
    masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


class ReadOnlyPrefixCache:
    def __init__(self, past_key_values):
        self.layers = tuple(past_key_values)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return int(self.layers[layer_idx][0].shape[-2])

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int,
    ) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + int(cache_position.shape[0]), 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_keys, prefix_values, _ = self.layers[layer_idx]
        return (
            torch.cat([prefix_keys, key_states], dim=-2),
            torch.cat([prefix_values, value_states], dim=-2),
        )


def compute_layer_complete(
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    *,
    layers,
    rotary_emb,
):
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        hidden_states, gate = layernorm_forward(
            layer.input_layernorm, hidden_states, adarms_cond[i]
        )
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)
        query_states.append(query_state.transpose(1, 2))
        key_states.append(key_state.transpose(1, 2))
        value_states.append(value_state.transpose(1, 2))

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
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    paligemma_layer = layers[0]
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        paligemma_layer.self_attn.scaling,
    )
    batch_size = query_states.shape[0]
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 8 * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(
            layer.post_attention_layernorm, out_emb, adarms_cond[i]
        )
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config: GemmaVariantConfig,
        action_expert_config: GemmaVariantConfig,
        *,
        use_adarms: list[bool],
        precision: Literal["bfloat16", "float32"],
        image_size: int,
        runtime_role: Literal["all", "prefix", "action", "idle"] = "all",
    ):
        super().__init__()
        self.paligemma = None
        self.gemma_expert = None
        self.prefix_output_device: torch.device | None = None

        if runtime_role in ("all", "prefix"):
            vlm_config_hf = CONFIG_MAPPING["paligemma"]()
            vlm_config_hf._vocab_size = PALIGEMMA_VOCAB_SIZE
            vlm_config_hf.image_token_index = PALIGEMMA_VOCAB_SIZE
            vlm_config_hf.text_config.hidden_size = vlm_config.width
            vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
            vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
            vlm_config_hf.text_config.head_dim = vlm_config.head_dim
            vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
            vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
            vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
            vlm_config_hf.text_config.dtype = precision
            vlm_config_hf.text_config.vocab_size = PALIGEMMA_VOCAB_SIZE
            vlm_config_hf.text_config.use_adarms = use_adarms[0]
            vlm_config_hf.text_config.is_causal = False
            vlm_config_hf.text_config.use_bidirectional_attention = True
            vlm_config_hf.text_config.adarms_cond_dim = (
                vlm_config.width if use_adarms[0] else None
            )
            vlm_config_hf.vision_config.image_size = image_size
            vlm_config_hf.vision_config.intermediate_size = 4304
            vlm_config_hf.vision_config.projection_dim = 2048
            vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
            vlm_config_hf.vision_config.dtype = "float32"
            self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(
                config=vlm_config_hf
            )
            self.paligemma.lm_head = None

        if runtime_role in ("all", "action"):
            action_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=action_expert_config.head_dim,
                hidden_size=action_expert_config.width,
                intermediate_size=action_expert_config.mlp_dim,
                num_attention_heads=action_expert_config.num_heads,
                num_hidden_layers=action_expert_config.depth,
                num_key_value_heads=action_expert_config.num_kv_heads,
                vocab_size=PALIGEMMA_VOCAB_SIZE,
                hidden_activation="gelu_pytorch_tanh",
                dtype=precision,
                use_adarms=use_adarms[1],
                is_causal=False,
                use_bidirectional_attention=True,
                adarms_cond_dim=(action_expert_config.width if use_adarms[1] else None),
            )
            self.gemma_expert = PiGemmaForCausalLM(config=action_config_hf)
            self.gemma_expert.lm_head = None
            self.gemma_expert.model.embed_tokens = None
        self.to_selected_dtype(precision)

    def to_selected_dtype(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ) -> None:
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid Pi05 precision: {precision}")
        self.to(dtype=torch.bfloat16)
        keep_fp32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in keep_fp32):
                param.data = param.data.to(dtype=torch.float32)

    def set_prefix_output_device(self, device: torch.device) -> None:
        self.prefix_output_device = torch.device(device)

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        return next(module.parameters()).device

    def _prefix_transformer_device(self) -> torch.device:
        if self.prefix_output_device is not None:
            return self.prefix_output_device
        language_model = self.paligemma.model.language_model
        return self._module_device(language_model.layers[0])

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        out_dtype = image.dtype
        vision_device = self._module_device(self.paligemma.model.vision_tower)
        output_device = self._prefix_transformer_device()
        if image.device != vision_device or image.dtype != torch.float32:
            image = image.to(device=vision_device, dtype=torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output
        if features.device != output_device or features.dtype != out_dtype:
            features = features.to(device=output_device, dtype=out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        embedding = self.paligemma.model.language_model.get_input_embeddings()
        embedding_device = self._module_device(embedding)
        output_device = self._prefix_transformer_device()
        if tokens.device != embedding_device:
            tokens = tokens.to(device=embedding_device)
        embeds = embedding(tokens)
        if embeds.device != output_device:
            embeds = embeds.to(device=output_device)
        return embeds

    def forward(
        self,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values,
        inputs_embeds: list[torch.FloatTensor | None],
        use_cache: bool | None,
        adarms_cond: list[torch.Tensor | None] | None = None,
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
            return [
                prefix_output.last_hidden_state,
                None,
            ], prefix_output.past_key_values

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

        paligemma_layers = self.paligemma.model.language_model.layers
        expert_layers = self.gemma_expert.model.layers
        rotary_emb = self.paligemma.model.language_model.rotary_emb
        for layers in zip(paligemma_layers, expert_layers, strict=True):
            inputs_embeds = compute_layer_complete(
                inputs_embeds,
                attention_mask,
                position_ids,
                adarms_cond,
                layers=layers,
                rotary_emb=rotary_emb,
            )

        final_norms = (
            self.paligemma.model.language_model.norm,
            self.gemma_expert.model.norm,
        )
        outputs = []
        for i, hidden_states in enumerate(inputs_embeds):
            out_emb, _ = layernorm_forward(
                final_norms[i], hidden_states, adarms_cond[i]
            )
            outputs.append(out_emb)
        return outputs, None


class Pi05TorchModel(nn.Module):
    def __init__(
        self,
        config: Pi05PipelineConfig,
        runtime_role: Literal["all", "prefix", "action", "idle"] = "all",
    ):
        super().__init__()
        self.config = config
        vlm_config = get_gemma_variant_config(config.paligemma_variant)
        action_config = get_gemma_variant_config(config.action_expert_variant)
        precision = (
            "bfloat16"
            if config.materialize_dtype in ("bf16", "bfloat16")
            else "float32"
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            vlm_config,
            action_config,
            use_adarms=[False, True],
            precision=precision,
            image_size=config.image_size[0],
            runtime_role=runtime_role,
        )
        if runtime_role in ("all", "action"):
            self.action_in_proj = nn.Linear(config.action_dim, action_config.width)
            self.action_out_proj = nn.Linear(action_config.width, config.action_dim)
            self.time_mlp_in = nn.Linear(action_config.width, action_config.width)
            self.time_mlp_out = nn.Linear(action_config.width, action_config.width)
        else:
            self.action_in_proj = None
            self.action_out_proj = None
            self.time_mlp_in = None
            self.time_mlp_out = None

    def retain_runtime_components(
        self,
        role: Literal["all", "prefix", "action", "idle"],
    ) -> None:
        if role == "all":
            return
        if role in ("prefix", "idle"):
            self.paligemma_with_expert.gemma_expert = None
            self.action_in_proj = None
            self.action_out_proj = None
            self.time_mlp_in = None
            self.time_mlp_out = None
        if role in ("action", "idle"):
            self.paligemma_with_expert.paligemma = None

    def prepare_attention_masks_4d(
        self,
        att_2d_masks: torch.Tensor,
        *,
        full_attention: bool | None = None,
    ) -> torch.Tensor | None:
        return prepare_optional_full_attention_mask(
            att_2d_masks,
            full_attention=full_attention,
        )

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []
        for image, image_mask in zip(images, image_masks, strict=True):
            image_emb = self.paligemma_with_expert.embed_image(image)
            batch_size, num_image_embs = image_emb.shape[:2]
            embs.append(image_emb)
            pad_masks.append(image_mask[:, None].expand(batch_size, num_image_embs))
            att_masks += [0] * num_image_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
        embs.append(lang_emb)
        pad_masks.append(token_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks_t = att_masks_t[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks_t

    def embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.time_embedding_min_period,
            max_period=self.config.time_embedding_max_period,
        )
        action_emb = self.action_in_proj(
            noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
        )
        time_emb = time_emb.to(dtype=self.time_mlp_in.weight.dtype)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        adarms_cond = F.silu(time_emb)

        batch_size, action_len = action_emb.shape[:2]
        pad_masks = torch.ones(
            batch_size,
            action_len,
            dtype=torch.bool,
            device=noisy_actions.device,
        )
        att_masks_t = torch.zeros(
            batch_size,
            action_len,
            dtype=action_emb.dtype,
            device=noisy_actions.device,
        )
        att_masks_t[:, 0] = 1
        return action_emb, pad_masks, att_masks_t, adarms_cond

    def _move_prefix_image_encoder_to_device(self, device: torch.device) -> None:
        paligemma = self.paligemma_with_expert.paligemma
        paligemma.model.vision_tower.to(device)
        paligemma.model.multi_modal_projector.to(device)

    def _prefix_language_phase_offload_layer_count(self) -> int:
        language_model = self.paligemma_with_expert.paligemma.model.language_model
        if self.config.offload_prefix_language_layers_after_prefix:
            return len(language_model.layers)
        return min(
            max(self.config.offload_prefix_language_layer_count_after_prefix, 0),
            len(language_model.layers),
        )

    def _move_prefix_language_layers_to_device(self, device: torch.device) -> None:
        language_model = self.paligemma_with_expert.paligemma.model.language_model
        layer_count = self._prefix_language_phase_offload_layer_count()
        for layer in language_model.layers[:layer_count]:
            layer.to(device)

    def _prepare_prefix_image_encoder_for_embed(self) -> None:
        if not self.config.offload_prefix_image_encoder_after_embed:
            return
        self._move_prefix_image_encoder_to_device(
            self.paligemma_with_expert._prefix_transformer_device()
        )

    def _offload_prefix_image_encoder_after_embed(self) -> None:
        if not self.config.offload_prefix_image_encoder_after_embed:
            return
        self._move_prefix_image_encoder_to_device(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_prefix_language_layers_for_forward(self) -> None:
        if (
            not self._prefix_language_phase_offload_layer_count()
            or self.config.offload_prefix_language_layers
        ):
            return
        self._move_prefix_language_layers_to_device(
            self.paligemma_with_expert._prefix_transformer_device()
        )

    def _offload_prefix_language_layers_after_prefix(self) -> None:
        if (
            not self._prefix_language_phase_offload_layer_count()
            or self.config.offload_prefix_language_layers
        ):
            return
        self._move_prefix_language_layers_to_device(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        token_masks: torch.Tensor,
        prefix_full_attention_hint: bool | None = None,
    ):
        tokens, token_masks = trim_trailing_padding_tokens(tokens, token_masks)
        self._prepare_prefix_image_encoder_for_embed()
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, image_masks, tokens, token_masks
        )
        self._offload_prefix_image_encoder_after_embed()
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_full_attention = bool(prefix_full_attention_hint)
        if prefix_full_attention:
            attention_mask = None
        else:
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_full_attention = bool(prefix_att_2d_masks.all().item())
            attention_mask = self.prepare_attention_masks_4d(
                prefix_att_2d_masks,
                full_attention=prefix_full_attention,
            )
        self._prepare_prefix_language_layers_for_forward()
        with set_forward_context(current_timestep=0, attn_metadata=None):
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        self._offload_prefix_language_layers_after_prefix()
        return (
            past_key_values,
            prefix_pad_masks,
            prefix_position_ids,
            prefix_full_attention,
        )

    @torch.no_grad()
    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_full_attention: bool = False,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(x_t, timestep)
        )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        if prefix_full_attention:
            attention_mask = None
        else:
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks],
                dim=2,
            )
            attention_mask = self.prepare_attention_masks_4d(full_att_2d_masks)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=ReadOnlyPrefixCache(past_key_values),
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :]
        return self.action_out_proj(
            suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        ).to(dtype=torch.float32)
