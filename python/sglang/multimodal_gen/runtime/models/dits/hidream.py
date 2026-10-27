"""HiDream-O1-Image: unified VLM diffusion model (modified Qwen3VL).

Ports the model from the official HiDream-O1-Image repository with NPU-optimized
changes: SDPA-based two-pass attention, SGLang distributed linears for TP,
and platform-agnostic device references.
"""

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.hidream import HiDreamDitConfig
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum, current_platform
from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.models.qwen2 import Qwen2MLP
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel



class HiDreamTextRotaryEmbedding(nn.Module):
    """3D interleaved RoPE for the text decoder."""

    def __init__(self, head_dim: int, rope_theta: float, mrope_section: list[int],
                 max_position_embeddings: int):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section
        self.max_seq_len_cached = max_position_embeddings
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        """Reorganize from chunked [TTT...HHH...WWW] to interleaved [THTHWHTHW...TT]."""
        freqs_t = freqs[0]  # overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for the given 3D position_ids [3, batch, seq_len]."""
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        with torch.autocast(device_type=current_platform.device_type if hasattr(current_platform, 'device_type') else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos, sin


class HiDreamTextAttention(nn.Module):
    """GQA attention with q_norm/k_norm, TP-optimized Q/K/V/O."""

    def __init__(self, cfg, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            cfg.hidden_size, cfg.head_dim,
            cfg.num_attention_heads, cfg.num_key_value_heads,
            bias=cfg.attention_bias, prefix=f"{prefix}.qkv_proj",
        )
        self.q_size = cfg.num_attention_heads * cfg.head_dim
        self.kv_size = cfg.num_key_value_heads * cfg.head_dim
        self.o_proj = RowParallelLinear(
            cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size,
            bias=cfg.attention_bias, prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        q = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
        v = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v


class HiDreamTextDecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.self_attn = HiDreamTextAttention(cfg, layer_idx, prefix=f"{prefix}.self_attn")
        self.mlp = Qwen2MLP(
            cfg.hidden_size, cfg.intermediate_size, "silu",
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        q, k, v = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = self._attention(q, k, v, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states

    def _attention(self, q, k, v, attention_mask):
        """SDPA attention with optional 4D mask and GQA head repetition."""
        batch_size = q.shape[0]
        seq_len = q.shape[2]
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[3]
        
        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            scale=self.self_attn.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, num_heads * head_dim)
        out, _ = self.self_attn.o_proj(attn_output)
        return out


class HiDreamTextModel(nn.Module):
    """Text decoder with TP-optimized layers and DeepStack visual injection."""

    def __init__(self, cfg, prefix: str = ""):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size, prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList([
            HiDreamTextDecoderLayer(cfg, i, prefix=f"{prefix}.layers.{i}")
            for i in range(cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = HiDreamTextRotaryEmbedding(
            cfg.head_dim, cfg.rope_theta, list(cfg.mrope_section),
            cfg.max_position_embeddings,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local
        return hidden_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(position_ids)
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states, position_embeddings, attention_mask
            )
            if (deepstack_visual_embeds is not None
                    and visual_pos_masks is not None
                    and layer_idx < len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
        return self.norm(hidden_states)


class BottleneckPatchEmbed(nn.Module):
    """Two-layer linear with PCA bottleneck for noise patch embedding."""

    def __init__(self, patch_size: int, in_chans: int, pca_dim: int, embed_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(patch_size * patch_size * in_chans, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t * 1000, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class FinalLayer(nn.Module):
    """Linear projection from hidden to patch predictions."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class HiDreamForCausalLM(BaseDiT):
    """HiDream-O1-Image unified VLM diffusion model.

    Wraps vision encoder, text decoder, and diffusion heads into a single
    model that performs text-to-image and image-to-image generation via
    flow matching denoising.
    """

    _fsdp_shard_conditions = []
    param_names_mapping = {}
    _compile_conditions = []
    _supported_attention_backends = {
        AttentionBackendEnum.TORCH_SDPA,
    }

    def __init__(self, config: HiDreamDitConfig, hf_config: dict[str, Any], **kwargs):
        super().__init__(config=config, hf_config=hf_config, **kwargs)
        ac = self.config

        self.hidden_size = ac.hidden_size
        self.num_attention_heads = ac.num_attention_heads
        self.num_channels_latents = ac.num_channels_latents

        self.visual = Qwen3VLMoeVisionModel(
            Qwen3VLVisionConfig(**hf_config["vision_config"]),
            prefix="visual",
        )

        self.language_model = HiDreamTextModel(ac, prefix="language_model")

        self.patch_size = ac.dit_patch_size
        self.in_channels = ac.in_channels
        self.t_embedder1 = TimestepEmbedder(ac.hidden_size, ac.frequency_embedding_size)
        self.x_embedder = BottleneckPatchEmbed(
            ac.dit_patch_size, ac.in_channels, ac.bottleneck_dim, ac.hidden_size,
        )
        self.final_layer2 = FinalLayer(ac.hidden_size, ac.dit_patch_size, ac.in_channels)
        self.tms_token_id = ac.tms_token_id

        self.image_token_id = ac.image_token_id
        self.video_token_id = ac.video_token_id
        self.vision_start_token_id = ac.vision_start_token_id

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.embed_tokens

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pixel_values = pixel_values.to(self.visual.patch_embed.proj.weight.dtype)
        visual_out = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(visual_out[..., : self.hidden_size], split_sizes)
        deepstack_image_embeds = list(visual_out[..., self.hidden_size :].chunk(
            len(self.visual.deepstack_visual_indexes), dim=-1
        ))
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        special_image_mask = input_ids == self.image_token_id
        n_image_tokens = special_image_mask.sum()
        special_image_mask_3d = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            total_feat = sum(f.numel() for f in (image_features if isinstance(image_features, (list, tuple)) else [image_features]))
            if inputs_embeds[special_image_mask_3d].numel() != total_feat:
                raise ValueError(
                    f"Image features and tokens mismatch: tokens={n_image_tokens}, features={total_feat}"
                )
        return special_image_mask, None

    def _run_decoder_sdpa(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        token_types: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Two-pass SDPA attention: causal on AR tokens, full on all tokens."""
        text_model = self.language_model
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        position_embeddings = text_model.rotary_emb(position_ids)
        cos, sin = position_embeddings

        is_gen = token_types[0].bool()
        idx_ar = torch.nonzero(~is_gen, as_tuple=False).squeeze(-1)

        hidden_states = inputs_embeds
        batch_size, total_seq_len, _ = hidden_states.shape
        head_dim = text_model.layers[0].self_attn.head_dim
        dtype = hidden_states.dtype
        min_val = torch.finfo(dtype).min

        for layer_idx, decoder_layer in enumerate(text_model.layers):
            residual = hidden_states
            normed = decoder_layer.input_layernorm(hidden_states)
            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, head_dim)
            attn = decoder_layer.self_attn

            qkv, _ = attn.qkv_proj(normed)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            q = attn.q_norm(q.view(hidden_shape)).transpose(1, 2)
            k = attn.k_norm(k.view(hidden_shape)).transpose(1, 2)
            v = v.view(hidden_shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            num_heads = q.shape[1]
            num_kv_heads = k.shape[1]
            if num_kv_heads < num_heads:
                repeat_factor = num_heads // num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)

            softmax_scale = attn.scaling

            if idx_ar.numel() > 0:
                q_ar = q[:, :, idx_ar].contiguous()
                k_ar = k[:, :, idx_ar].contiguous()
                v_ar = v[:, :, idx_ar].contiguous()
                out_ar = F.scaled_dot_product_attention(
                    q_ar, k_ar, v_ar, is_causal=True, scale=softmax_scale,
                )
            else:
                out_ar = None

            out_full = F.scaled_dot_product_attention(
                q, k, v, scale=softmax_scale,
            )

            if out_ar is not None:
                out_full = out_full.clone()
                out_full[:, :, idx_ar] = out_ar

            attn_output = out_full.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            attn_out, _ = attn.o_proj(attn_output)
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = residual + decoder_layer.mlp(
                decoder_layer.post_attention_layernorm(hidden_states)
            )

            if (deepstack_visual_embeds is not None
                    and visual_pos_masks is not None
                    and layer_idx < len(deepstack_visual_embeds)):
                hidden_states = text_model._deepstack_process(
                    hidden_states, visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        return text_model.norm(hidden_states)

    def _forward_generation(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        token_types: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        use_flash_attn: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for one denoising step; returns {'x_pred': patches}."""
        precomputed_image_embeds = kwargs.pop("precomputed_image_embeds", None)
        precomputed_deepstack = kwargs.pop("precomputed_deepstack_image_embeds", None)

        inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        deepstack_image_embeds = None

        if pixel_values is not None:
            if precomputed_image_embeds is not None and precomputed_deepstack is not None:
                image_embeds = precomputed_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                deepstack_image_embeds = [
                    e.to(inputs_embeds.device, inputs_embeds.dtype)
                    for e in precomputed_deepstack
                ]
            else:
                image_embeds_list, deepstack_image_embeds = self.get_image_features(
                    pixel_values, image_grid_thw
                )
                image_embeds = torch.cat(image_embeds_list, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
            )

        visual_pos_masks = None
        if image_mask is not None:
            visual_pos_masks = image_mask[..., 0] if image_mask.dim() == 3 else image_mask

        if isinstance(timestep, list):
            timestep = torch.cat(timestep, dim=0)
        timestep = timestep.to(inputs_embeds.device)
        t_emb = self.t_embedder1(timestep)
        tms_mask = (input_ids == self.tms_token_id)
        tms_mask_3d = tms_mask.unsqueeze(-1).expand_as(inputs_embeds)
        t_emb_expanded = t_emb.unsqueeze(1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(tms_mask_3d, t_emb_expanded, inputs_embeds)

        if isinstance(vinputs, list):
            vinputs = torch.cat(vinputs, dim=0)
        vinputs = vinputs.to(inputs_embeds.device)
        vinputs_embedded = self.x_embedder(vinputs).to(inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)

        batch_size, total_seq_len, _ = inputs_embeds.shape

        if visual_pos_masks is not None:
            vinputs_seq_len = vinputs_embedded.shape[1]
            if visual_pos_masks.shape[0] != batch_size:
                visual_pos_masks = visual_pos_masks.expand(batch_size, -1)
            pad = torch.zeros(
                visual_pos_masks.shape[0], vinputs_seq_len,
                dtype=visual_pos_masks.dtype, device=visual_pos_masks.device,
            )
            visual_pos_masks = torch.cat([visual_pos_masks, pad], dim=1)

        if isinstance(token_types, list):
            token_types = torch.cat(token_types, dim=0)
        token_types = token_types.to(inputs_embeds.device)
        if token_types.dim() == 1:
            token_types = token_types.unsqueeze(0)
        elif token_types.dim() == 2 and token_types.shape[-1] == 1:
            token_types = token_types.squeeze(-1).unsqueeze(0)
        if token_types.shape[0] == 1 and batch_size > 1:
            token_types = token_types.expand(batch_size, -1)

        if use_flash_attn:
            hidden_states = self._run_decoder_sdpa(
                inputs_embeds, position_ids, token_types,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_image_embeds,
            )
        else:
            dtype = inputs_embeds.dtype
            min_val = torch.finfo(dtype).min
            attn_masks = []
            for b in range(batch_size):
                causal = torch.full(
                    (total_seq_len, total_seq_len), min_val,
                    device=inputs_embeds.device, dtype=dtype,
                )
                causal = torch.triu(causal, diagonal=1)
                gen_positions = token_types[b].bool()
                causal[gen_positions, :] = 0
                attn_masks.append(causal)
            attention_mask_4d = torch.stack(attn_masks, dim=0).unsqueeze(1)

            hidden_states = self.language_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask_4d,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_image_embeds,
            )

        x_pred = self.final_layer2(hidden_states)
        out = {"x_pred": x_pred}
        if pixel_values is not None:
            # vision embeds cached across denoising steps
            out["cond_image_embeds"] = image_embeds
            out["cond_deepstack_image_embeds"] = deepstack_image_embeds
        return out

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        vinputs: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        token_types: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        use_flash_attn: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Main forward entry point. Dispatches to _forward_generation for diffusion."""
        if vinputs is not None:
            return self._forward_generation(
                input_ids=input_ids,
                position_ids=position_ids,
                vinputs=vinputs,
                timestep=timestep,
                token_types=token_types,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_flash_attn=use_flash_attn,
                **kwargs,
            )
        raise ValueError("HiDreamForCausalLM requires vinputs for generation mode")


EntryClass = HiDreamForCausalLM
