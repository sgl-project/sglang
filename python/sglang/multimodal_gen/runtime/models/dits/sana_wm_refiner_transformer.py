# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.sana_wm_refiner import (
    SanaWMRefinerArchConfig,
    SanaWMRefinerConfig,
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


def _slice_rope(
    rope: tuple[torch.Tensor, torch.Tensor], start: int, end: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice along token axis for either interleaved (rank-3) or split (rank-4)."""
    cos, sin = rope
    end_ = end if end is not None else cos.shape[-2 if cos.ndim == 4 else 1]
    if cos.ndim == 3:
        return cos[:, start:end_], sin[:, start:end_]
    if cos.ndim == 4:
        return cos[:, :, start:end_, :], sin[:, :, start:end_, :]
    raise ValueError(f"Unexpected RoPE rank: {cos.ndim}")


def _streaming_self_attention(
    attn: LTX2Attention,
    hidden_states: torch.Tensor,
    video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    n_context_tokens: int,
) -> torch.Tensor:
    """Streaming SLA: context attends to context only, current attends to context+current.

    Mirrors NVlabs `inference_sana_wm.py::_streaming_self_attention`.
    """
    seq_len = hidden_states.shape[1]
    if n_context_tokens <= 0 or n_context_tokens >= seq_len:
        return attn(hidden_states, context=None, pe=video_rotary_emb)

    ctx_rope = _slice_rope(video_rotary_emb, 0, n_context_tokens)
    out_ctx = attn(
        hidden_states[:, :n_context_tokens],
        context=None,
        pe=ctx_rope,
    )

    cur_rope = _slice_rope(video_rotary_emb, n_context_tokens, seq_len)
    out_cur = attn(
        hidden_states[:, n_context_tokens:],
        context=hidden_states,
        pe=cur_rope,
        k_pe=video_rotary_emb,
    )
    return torch.cat([out_ctx, out_cur], dim=1)


class SanaWMRefinerBlock(nn.Module):
    """Video-only LTX-2 transformer block.

    Diffusers-compatible layout: `norm1 -> attn1 (self) -> norm2 -> attn2 (cross) -> norm3 -> ff`,
    each modulated via per-block `scale_shift_table` + token-wise `temb`.
    """

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
        self.attn1 = LTX2Attention(
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
        attn_out = _streaming_self_attention(
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
    Audio params present in the checkpoint are silently dropped by the loader's
    `strict=False` state_dict load.
    """

    _fsdp_shard_conditions = SanaWMRefinerArchConfig()._fsdp_shard_conditions
    _compile_conditions = SanaWMRefinerArchConfig()._compile_conditions
    _supported_attention_backends = (
        SanaWMRefinerArchConfig()._supported_attention_backends
    )
    param_names_mapping = SanaWMRefinerArchConfig().param_names_mapping
    reverse_param_names_mapping: dict = {}
    lora_param_names_mapping: dict = {}

    def __init__(
        self,
        config: SanaWMRefinerConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config, hf_config=hf_config)
        arch = config.arch_config

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

        # LTX2AudioVideoRotaryPosEmbed expects `dim` to be the *total* hidden
        # size (num_heads * head_dim), not the per-head dim. It internally
        # reshapes cos/sin to (B, T, num_heads, head_dim/2). Passing
        # `attention_head_dim` here would size the RoPE to head_dim/num_heads
        # and produce a (1, num_heads, L, 2) cos/sin that won't match
        # LTX2Attention's q/k. See LTX2Transformer3DAVModel.__init__ in
        # ltx_2.py for the canonical convention (`dim=self.hidden_size`).
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

        self.layer_names = ["transformer_blocks"]

    def _scale_timestep_for_adaln(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep * self.timestep_scale_multiplier

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
        # Accept either packed (B, L, in_dim) or raw 5D (B, C, T, H, W).
        if hidden_states.dim() == 5:
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
            packed_input = True
        else:
            if num_frames is None or height is None or width is None:
                raise ValueError(
                    "num_frames/height/width are required when hidden_states is pre-packed."
                )
            packed_input = False

        B = hidden_states.size(0)

        video_coords = self.rope.prepare_video_coords(
            batch_size=B,
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

        hidden_states, _ = self.proj_in(hidden_states)

        scaled_t = self._scale_timestep_for_adaln(timestep)
        temb, embedded_timestep = self.time_embed(
            scaled_t.flatten(), hidden_dtype=hidden_states.dtype
        )
        if timestep.dim() >= 2:
            temb = temb.view(B, -1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(
                B, -1, embedded_timestep.size(-1)
            )
        else:
            temb = temb.view(B, 1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(B, 1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(B, -1, self.hidden_size)

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

        if packed_input:
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
