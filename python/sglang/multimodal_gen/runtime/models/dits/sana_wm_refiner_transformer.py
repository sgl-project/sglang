# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.sana_wm_refiner import (
    SanaWMRefinerArchConfig,
    SanaWMRefinerConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import (
    is_blocks_or_transformer_blocks,
)
from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2AdaLayerNormSingle,
    LTX2Attention,
    LTX2AudioVideoRotaryPosEmbed,
    LTX2FeedForward,
    LTX2TextProjection,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
)


def pack_latents(
    latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    """Pack a 5D latent (B, C, T, H, W) into a 3D token sequence (B, L, in_dim)."""
    B, _, T, H, W = latents.shape
    pT = T // patch_size_t
    pH = H // patch_size
    pW = W // patch_size
    latents = latents.reshape(B, -1, pT, patch_size_t, pH, patch_size, pW, patch_size)
    return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


def unpack_latents(
    tokens: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    """Inverse of `pack_latents`: (B, L, out_dim) -> (B, C, T, H, W)."""
    B = tokens.size(0)
    tokens = tokens.reshape(
        B,
        num_frames // patch_size_t,
        height // patch_size,
        width // patch_size,
        -1,
        patch_size_t,
        patch_size,
        patch_size,
    )
    return (
        tokens.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    )


def _apply_refiner_rope(
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if rotary_emb[0].ndim == 3:
        return apply_interleaved_rotary_emb(hidden_states, rotary_emb).to(
            hidden_states.dtype
        )
    if rotary_emb[0].ndim == 4:
        return apply_split_rotary_emb(hidden_states, rotary_emb)
    raise ValueError(f"Unexpected refiner RoPE rank: {rotary_emb[0].ndim}")


class SanaWMRefinerSelfAttention(LTX2Attention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.capture_kv_before_rope = False
        self.capture_kv_after_rope = False
        self.kv_prefix: dict[str, Any] | None = None
        self.captured_kv_before_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self.captured_kv_after_rope: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_kv_capture(self, mode: str, enabled: bool) -> None:
        if mode == "pre_rope":
            self.capture_kv_before_rope = enabled
            if enabled:
                self.captured_kv_before_rope = None
            return
        if mode == "post_rope":
            self.capture_kv_after_rope = enabled
            if enabled:
                self.captured_kv_after_rope = None
            return
        raise ValueError(f"Unsupported KV capture mode: {mode}")

    def take_captured_kv(self, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "pre_rope":
            captured = self.captured_kv_before_rope
            self.captured_kv_before_rope = None
        elif mode == "post_rope":
            captured = self.captured_kv_after_rope
            self.captured_kv_after_rope = None
        else:
            raise ValueError(f"Unsupported KV capture mode: {mode}")
        if captured is None:
            raise RuntimeError(f"Missing captured KV for mode {mode}")
        return captured


def refiner_self_attention(
    attn: SanaWMRefinerSelfAttention,
    hidden_states: torch.Tensor,
    video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    n_context_tokens: int,
) -> torch.Tensor:
    """Run sink/current attention and the optional streaming KV-cache hooks."""
    has_streaming_hooks = (
        attn.capture_kv_before_rope
        or attn.capture_kv_after_rope
        or attn.kv_prefix is not None
    )
    sequence_length = hidden_states.shape[1]
    if not has_streaming_hooks and (
        n_context_tokens <= 0 or n_context_tokens >= sequence_length
    ):
        return attn(
            hidden_states,
            pe=video_rotary_emb,
            skip_sequence_parallel_override=True,
        )

    gate_logits = None
    if attn.to_gate_logits is not None:
        gate_logits, _ = attn.to_gate_logits(hidden_states)

    query, _ = attn.to_q(hidden_states)
    key, _ = attn.to_k(hidden_states)
    value, _ = attn.to_v(hidden_states)
    if attn.qk_norm:
        assert attn.q_norm is not None and attn.k_norm is not None
        query_dtype, key_dtype = query.dtype, key.dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            query = attn.q_norm(query).to(query_dtype)
            key = attn.k_norm(key).to(key_dtype)

    if attn.capture_kv_before_rope:
        attn.captured_kv_before_rope = (
            key.detach().clone(),
            value.detach().clone(),
        )

    query = _apply_refiner_rope(query, video_rotary_emb)
    key = _apply_refiner_rope(key, video_rotary_emb)
    if attn.capture_kv_after_rope:
        attn.captured_kv_after_rope = (
            key.detach().clone(),
            value.detach().clone(),
        )

    prefix_length = 0
    prefix = attn.kv_prefix
    if isinstance(prefix, dict) and prefix.get("mode") == "rf_shifted_sink":
        prefix_keys = []
        prefix_values = []
        sink_key = prefix.get("sink_k_pre")
        sink_value = prefix.get("sink_v")
        if sink_key is not None and sink_value is not None and sink_key.shape[1] > 0:
            sink_rope = prefix.get("sink_pe")
            if sink_rope is None:
                raise ValueError("rf_shifted_sink prefix requires sink_pe")
            prefix_keys.append(_apply_refiner_rope(sink_key.to(key.dtype), sink_rope))
            prefix_values.append(sink_value.to(value.dtype))
        history_key = prefix.get("history_k")
        history_value = prefix.get("history_v")
        if (
            history_key is not None
            and history_value is not None
            and history_key.shape[1] > 0
        ):
            prefix_keys.append(history_key.to(key.dtype))
            prefix_values.append(history_value.to(value.dtype))
        if prefix_keys:
            prefix_length = sum(prefix_key.shape[1] for prefix_key in prefix_keys)
            key = torch.cat((*prefix_keys, key), dim=1)
            value = torch.cat((*prefix_values, value), dim=1)

    query = query.view(*query.shape[:2], attn.local_heads, attn.dim_head)
    key = key.view(*key.shape[:2], attn.local_heads, attn.dim_head)
    value = value.view(*value.shape[:2], attn.local_heads, attn.dim_head)
    if n_context_tokens <= 0 or n_context_tokens >= sequence_length:
        output = attn.attn(
            query,
            key,
            value,
            skip_sequence_parallel_override=True,
        )
    else:
        context = attn.attn(
            query[:, :n_context_tokens],
            key[:, prefix_length : prefix_length + n_context_tokens],
            value[:, prefix_length : prefix_length + n_context_tokens],
            skip_sequence_parallel_override=True,
        )
        current = attn.attn(
            query[:, n_context_tokens:],
            key,
            value,
            skip_sequence_parallel_override=True,
        )
        output = torch.cat((context, current), dim=1)

    if gate_logits is not None:
        output = output * (2.0 * torch.sigmoid(gate_logits).unsqueeze(-1))
    output, _ = attn.to_out[0](output.flatten(2))
    return attn.to_out[1](output)


class SanaWMRefinerBlock(nn.Module):
    """Video-only LTX-2 transformer block with released-checkpoint key layout."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = False,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)

        self.norm1 = nn.RMSNorm(self.dim, eps=norm_eps, elementwise_affine=False)
        self.attn1 = SanaWMRefinerSelfAttention(
            query_dim=self.dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            apply_gated_attention=apply_gated_attention,
            prefix=f"{prefix}.attn1",
            quant_config=quant_config,
        )

        self.norm2 = nn.RMSNorm(self.dim, eps=norm_eps, elementwise_affine=False)
        self.attn2 = LTX2Attention(
            query_dim=self.dim,
            context_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            use_local_attention=True,
            apply_gated_attention=apply_gated_attention,
            prefix=f"{prefix}.attn2",
            quant_config=quant_config,
        )

        self.norm3 = nn.RMSNorm(self.dim, eps=norm_eps, elementwise_affine=False)
        self.ff = LTX2FeedForward(self.dim, dim_out=self.dim, quant_config=quant_config)

        self.scale_shift_table = nn.Parameter(torch.randn(6, self.dim) / self.dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor] = None,
        n_context_tokens: int = 0,
    ) -> torch.Tensor:
        B = hidden_states.size(0)
        T = temb.size(1)
        D = self.dim
        ada = self.scale_shift_table[None, None].to(
            device=temb.device, dtype=temb.dtype
        ) + temb.reshape(B, T, 6, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada.unbind(
            dim=2
        )

        normed = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        attn_out = refiner_self_attention(
            self.attn1,
            normed,
            video_rotary_emb,
            n_context_tokens=n_context_tokens,
        )
        hidden_states = hidden_states + attn_out * gate_msa

        normed = self.norm2(hidden_states)
        ca_out = self.attn2(
            normed,
            context=encoder_hidden_states,
            mask=encoder_attention_mask,
            pe=None,
        )
        hidden_states = hidden_states + ca_out

        normed = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + self.ff(normed) * gate_mlp
        return hidden_states


class SanaWMLTX2VideoRefiner(CachableDiT, LayerwiseOffloadableModuleMixin):
    """SANA-WM stage-2 LTX-2 video-only refiner.

    Loads Diffusers-format refiner weights from `<model_path>/refiner/transformer/`.
    Audio and cross-modal params in that checkpoint are intentionally unused.
    """

    _fsdp_shard_conditions = [is_blocks_or_transformer_blocks]
    _compile_conditions = [is_blocks_or_transformer_blocks]
    param_names_mapping = SanaWMRefinerArchConfig().param_names_mapping
    reverse_param_names_mapping: dict = {}
    lora_param_names_mapping: dict = {}
    layer_names = ["transformer_blocks"]

    @staticmethod
    def is_expected_unloaded_checkpoint_key(name: str) -> bool:
        if name.startswith(("audio_", "av_cross_attn_")):
            return True
        parts = name.split(".", 3)
        if len(parts) < 3 or parts[0] != "transformer_blocks":
            return False
        branch = parts[2]
        return branch.startswith("audio_") or branch in {
            "video_a2v_cross_attn_scale_shift_table",
            "video_to_audio_attn",
        }

    def __init__(
        self,
        config: SanaWMRefinerConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config, hf_config=hf_config)
        arch = self.config

        self.in_channels = int(arch.in_channels)
        self.out_channels = int(arch.out_channels)
        self.patch_size = int(arch.patch_size)
        self.patch_size_t = int(arch.patch_size_t)
        self.hidden_size = int(arch.hidden_size)
        self.num_attention_heads = int(arch.num_attention_heads)
        self.num_channels_latents = int(arch.num_channels_latents)
        self.attention_head_dim = int(arch.attention_head_dim)
        self.timestep_scale_multiplier = float(arch.timestep_scale_multiplier)
        self.rope_type = str(arch.rope_type)

        in_dim = (
            self.in_channels * self.patch_size_t * self.patch_size * self.patch_size
        )
        out_dim = (
            self.out_channels * self.patch_size_t * self.patch_size * self.patch_size
        )

        self.proj_in = ColumnParallelLinear(
            in_dim,
            self.hidden_size,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
        )

        self.time_embed = LTX2AdaLayerNormSingle(
            self.hidden_size, embedding_coefficient=6
        )
        self.caption_projection = LTX2TextProjection(
            in_features=int(arch.caption_channels),
            hidden_size=self.hidden_size,
            out_features=self.hidden_size,
            act_fn="gelu_tanh",
        )

        self.transformer_blocks = nn.ModuleList(
            [
                SanaWMRefinerBlock(
                    dim=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    cross_attention_dim=int(arch.cross_attention_dim),
                    qk_norm=bool(arch.qk_norm),
                    norm_eps=float(arch.norm_eps),
                    apply_gated_attention=bool(arch.apply_gated_attention),
                    prefix=f"transformer_blocks.{i}",
                    quant_config=quant_config,
                )
                for i in range(int(arch.num_layers))
            ]
        )

        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.hidden_size) / self.hidden_size**0.5
        )
        self.norm_out = nn.LayerNorm(
            self.hidden_size, eps=float(arch.norm_eps), elementwise_affine=False
        )
        self.proj_out = ColumnParallelLinear(
            self.hidden_size,
            out_dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
        )

        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.hidden_size,
            patch_size=self.patch_size,
            patch_size_t=self.patch_size_t,
            base_num_frames=int(arch.base_num_frames),
            base_height=int(arch.base_height),
            base_width=int(arch.base_width),
            sampling_rate=int(arch.sampling_rate),
            hop_length=int(arch.hop_length),
            scale_factors=tuple(arch.scale_factors),
            causal_offset=int(arch.causal_offset),
            modality="video",
            rope_type=self.rope_type,
            num_attention_heads=self.num_attention_heads,
        )

    def _scale_timestep_for_adaln(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep * self.timestep_scale_multiplier

    def set_streaming_kv_prefixes(
        self, prefixes: Sequence[dict[str, Any] | None] | None
    ) -> None:
        if prefixes is None:
            self.clear_streaming_kv_prefixes()
            return
        if len(prefixes) != len(self.transformer_blocks):
            raise ValueError(
                "Expected one KV prefix per refiner block, got "
                f"{len(prefixes)} for {len(self.transformer_blocks)} blocks."
            )
        for block, prefix in zip(self.transformer_blocks, prefixes, strict=True):
            block.attn1.kv_prefix = prefix

    def clear_streaming_kv_prefixes(self) -> None:
        for block in self.transformer_blocks:
            block.attn1.kv_prefix = None

    def set_streaming_kv_capture(self, mode: str, enabled: bool) -> None:
        for block in self.transformer_blocks:
            block.attn1.set_kv_capture(mode, enabled)

    def take_streaming_kv(self, mode: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [block.attn1.take_captured_kv(mode) for block in self.transformer_blocks]

    def forward_tokens(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor] = None,
        n_context_tokens: int = 0,
    ) -> torch.Tensor:
        """Forward packed latent tokens with caller-provided absolute-position RoPE."""
        batch_size = hidden_states.size(0)
        hidden_states, _ = self.proj_in(hidden_states)

        if timestep.ndim == 3:
            if timestep.shape[-1] != 1:
                raise ValueError(
                    "A rank-3 refiner timestep must have a singleton last dimension."
                )
            timestep = timestep.squeeze(-1)
        temb, embedded_timestep = self.time_embed(
            self._scale_timestep_for_adaln(timestep).flatten(),
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.size(-1)
        )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, self.hidden_size
        )
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                video_rotary_emb=video_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                n_context_tokens=n_context_tokens,
            )

        scale_shift_values = self.scale_shift_table[None, None].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        ) + embedded_timestep[:, :, None].to(hidden_states.dtype)
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
        hidden_states, _ = self.proj_out(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states_image=None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 24.0,
        n_context_tokens: int = 0,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        input_is_packed = hidden_states.dim() == 3
        if not input_is_packed:
            if hidden_states.dim() != 5:
                raise ValueError(
                    "Refiner hidden_states must be packed 3D tokens or a 5D latent."
                )
            B_, _, T_, H_, W_ = hidden_states.shape
            if num_frames is None:
                num_frames = T_
            if height is None:
                height = H_
            if width is None:
                width = W_
            hidden_states = pack_latents(
                hidden_states,
                patch_size=self.patch_size,
                patch_size_t=self.patch_size_t,
            )
        else:
            if num_frames is None or height is None or width is None:
                raise ValueError(
                    "num_frames/height/width are required when hidden_states is pre-packed."
                )
        batch_size = hidden_states.size(0)

        video_coords = self.rope.prepare_video_coords(
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            device=hidden_states.device,
            fps=fps,
        )
        video_rotary_emb = self.rope(
            video_coords,
            device=hidden_states.device,
            out_dtype=hidden_states.dtype,
        )

        hidden_states = self.forward_tokens(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            video_rotary_emb=video_rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
            n_context_tokens=n_context_tokens,
        )
        if input_is_packed:
            return hidden_states
        return unpack_latents(
            hidden_states,
            num_frames=num_frames,
            height=height,
            width=width,
            patch_size=self.patch_size,
            patch_size_t=self.patch_size_t,
        )


EntryClass = SanaWMLTX2VideoRefiner
