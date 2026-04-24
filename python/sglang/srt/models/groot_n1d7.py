# Copyright 2026 SGLang Team
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
"""NVIDIA GR00T-N1.7 for SGLang.

One-file port of Isaac-GR00T's `gr00t_n1d7` stack:
- Embodiment-conditioned MLP primitives
  (from Isaac-GR00T/gr00t/model/modules/embodiment_conditioned_mlp.py)
- DiT + AlternateVLDiT + SelfAttentionTransformer
  (from Isaac-GR00T/gr00t/model/modules/dit.py)
- Gr00tN1d7ActionHead with flow-matching Euler loop
- Top-level Gr00tN1d7 model wrapper with load_weights

Flat single-file layout matches SGLang's alpamayo_r1.py convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn

from sglang.srt.layers.attention.masked_flash_attn import MaskedFlashAttention

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# ==============================================================================
# Embodiment-conditioned MLP primitives
# Ported from Isaac-GR00T/gr00t/model/modules/embodiment_conditioned_mlp.py
# Parameter names ( `W`, `b`, `layer1`, `layer2`, `W1`, `W2`, `W3`,
# `pos_encoding` ) are preserved exactly so upstream checkpoints load cleanly.
# Training-only `expand_action_dimension` helpers are dropped.
# ==============================================================================


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Produces a sinusoidal encoding of shape (B, T, embedding_dim) from
    timesteps of shape (B, T)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    """Linear layer with category-specific weights and biases."""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with category-specific weights per embodiment."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Action encoder with multi-embodiment support and sinusoidal timestep
    positional encoding."""

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = actions.shape

        if timesteps.dim() != 1 or timesteps.shape[0] != B:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )
        timesteps = timesteps.unsqueeze(1).expand(-1, T)

        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x


# ==============================================================================
# DiT stack (TimestepEncoder, AdaLayerNorm, BasicTransformerBlock, DiT,
# AlternateVLDiT, SelfAttentionTransformer).
#
# Ported from Isaac-GR00T/gr00t/model/modules/dit.py.  Unlike upstream, we do
# NOT inherit from diffusers ModelMixin/ConfigMixin — those pull in
# diffusers.models.modeling_utils which transitively imports a torchao path
# (`torchao.dtypes.floatx.float8_layout`) that is broken on recent torchao
# builds.  We don't need pretrained-from-HF semantics at SGLang inference
# time, so a plain nn.Module is enough.  Parameter names are preserved
# exactly so upstream checkpoints load cleanly via load_state_dict.
# ==============================================================================


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int, compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(proj)


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and num_positional_embeddings is None:
            raise ValueError(
                "If `positional_embeddings` type is defined, "
                "`num_positional_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.attn1 = MaskedFlashAttention(
            query_dim=dim,
            kv_dim=cross_attention_dim if cross_attention_dim is not None else dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            attention_bias=attention_bias,
            out_bias=attention_out_bias,
        )

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        forward_batch: Optional["ForwardBatch"] = None,
    ) -> torch.Tensor:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=(
                encoder_attention_mask
                if encoder_hidden_states is not None
                else attention_mask
            ),
            forward_batch=forward_batch,
        )
        if self.final_dropout is not None:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype: torch.dtype = torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        # Preserve every kwarg as an attribute — upstream relies on
        # register_to_config to do this automatically; we do it manually.
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_bias = attention_bias
        self.activation_fn = activation_fn
        self.num_embeds_ada_norm = num_embeds_ada_norm
        self.upcast_attention = upcast_attention
        self.norm_type = norm_type
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        self.max_num_positional_embeddings = max_num_positional_embeddings
        self.compute_dtype = compute_dtype
        self.final_dropout = final_dropout
        self.positional_embeddings = positional_embeddings
        self.interleave_self_attention = interleave_self_attention
        self.cross_attention_dim = cross_attention_dim

        self.inner_dim = num_attention_heads * attention_head_dim

        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )

        blocks = []
        for idx in range(num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = (
                cross_attention_dim if not use_self_attn else None
            )
            blocks.append(
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        forward_batch: Optional["ForwardBatch"] = None,
    ):
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                    forward_batch=forward_batch,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                    forward_batch=forward_batch,
                )
            all_hidden_states.append(hidden_states)

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)


class AlternateVLDiT(DiT):
    """DiT that alternates cross-attention between image and non-image VL
    tokens every `attend_text_every_n_blocks` cross-attention layers."""

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional["ForwardBatch"] = None,
    ):
        assert image_mask is not None, "Image mask is required"
        assert (
            self.interleave_self_attention
        ), "Interleave self attention must be enabled"

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                    forward_batch=forward_batch,
                )
            else:
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    curr_enc_attn_mask = non_image_attention_mask
                else:
                    curr_enc_attn_mask = image_attention_mask
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_enc_attn_mask,
                    temb=temb,
                    forward_batch=forward_batch,
                )
            all_hidden_states.append(hidden_states)

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype: torch.dtype = torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_bias = attention_bias
        self.activation_fn = activation_fn
        self.num_embeds_ada_norm = num_embeds_ada_norm
        self.upcast_attention = upcast_attention
        self.max_num_positional_embeddings = max_num_positional_embeddings
        self.compute_dtype = compute_dtype
        self.final_dropout = final_dropout
        self.positional_embeddings = positional_embeddings
        self.interleave_self_attention = interleave_self_attention

        self.inner_dim = num_attention_heads * attention_head_dim

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_hidden_states: bool = False,
        forward_batch: Optional["ForwardBatch"] = None,
    ):
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, forward_batch=forward_batch)
            all_hidden_states.append(hidden_states)
        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states


# ==============================================================================
# Gr00tN1d7ActionHead — flow-matching action decoder
#
# Ported from Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py
# (Gr00tN1d7ActionHead + get_action / get_action_with_features).
# Training-only paths are dropped; only the inference Euler loop is kept.
# No RTC (real-time control) support in the first release.
# ==============================================================================


from sglang.srt.configs.groot_n1d7 import (  # noqa: E402 (keep above imports close)
    Gr00tN1d7Config,
)


class Gr00tN1d7ActionHead(nn.Module):
    """Flow-matching action head.

    Submodule names mirror upstream exactly so upstream checkpoint tensors
    (under prefix `action_head.*`) load via plain `load_state_dict`:
        state_encoder, action_encoder, action_decoder, model, vlln,
        vl_self_attention, position_embedding.
    """

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.num_timestep_buckets = config.num_timestep_buckets

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim)
            if config.use_vlln
            else nn.Identity()
        )

        vlsa_cfg = getattr(config, "vl_self_attention_cfg", None)
        if (
            vlsa_cfg
            and vlsa_cfg.get("num_layers", 0) > 0
            and config.use_vl_self_attention
        ):
            self.vl_self_attention = SelfAttentionTransformer(**vlsa_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(
                config.max_seq_len, self.input_embedding_dim
            )

    def _process_vl(
        self,
        vl_embeds: torch.Tensor,
        forward_batch: Optional["ForwardBatch"] = None,
    ) -> torch.Tensor:
        vl = self.vlln(vl_embeds)
        if isinstance(self.vl_self_attention, SelfAttentionTransformer):
            vl = self.vl_self_attention(vl, forward_batch=forward_batch)
        else:
            vl = self.vl_self_attention(vl)
        return vl

    @torch.no_grad()
    def get_action(
        self,
        *,
        vl_embeds: torch.Tensor,  # [B, S, backbone_embedding_dim]
        vl_attn_mask: torch.Tensor,  # [B, S] bool
        image_mask: torch.Tensor,  # [B, S] bool
        state: torch.Tensor,  # [B, state_history_length, max_state_dim]
        embodiment_id: torch.Tensor,  # [B] long
        forward_batch: Optional["ForwardBatch"] = None,
    ) -> torch.Tensor:
        """Runs 4 Euler integration steps of the flow-matching ODE to decode
        an action trajectory.  Returns [B, action_horizon, action_dim]."""
        vl = self._process_vl(vl_embeds, forward_batch=forward_batch)

        B = vl.shape[0]
        assert state.shape[1] == self.config.state_history_length, (
            f"state history mismatch: {state.shape[1]} vs "
            f"{self.config.state_history_length}"
        )
        state_flat = state.reshape(B, 1, -1)
        state_features = self.state_encoder(state_flat, embodiment_id)

        dt = 1.0 / self.num_inference_timesteps
        device, dtype = vl.device, vl.dtype
        x = torch.randn(
            B, self.action_horizon, self.action_dim, device=device, dtype=dtype
        )

        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_discrete = int(t_cont * self.num_timestep_buckets)
            ts = torch.full((B,), t_discrete, device=device, dtype=torch.long)

            action_features = self.action_encoder(x, ts, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], device=device)
                action_features = action_features + self.position_embedding(
                    pos_ids
                ).unsqueeze(0)

            sa = torch.cat((state_features, action_features), dim=1)
            if self.config.use_alternate_vl_dit:
                mo = self.model(
                    hidden_states=sa,
                    encoder_hidden_states=vl,
                    timestep=ts,
                    image_mask=image_mask,
                    backbone_attention_mask=vl_attn_mask,
                    forward_batch=forward_batch,
                )
            else:
                mo = self.model(
                    hidden_states=sa,
                    encoder_hidden_states=vl,
                    timestep=ts,
                    forward_batch=forward_batch,
                )
            pred = self.action_decoder(mo, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            x = x + dt * pred_velocity

        return x


# ==============================================================================
# Gr00tN1d7 — SGLang-side top-level model wrapper
#
# Composes a Qwen3-VL backbone (truncated to `select_layer = 16` LM layers,
# matching Isaac-GR00T's qwen3_backbone.py) with Gr00tN1d7ActionHead.  The
# forward path wires VL layer-16 extraction into the action head and threads
# proprio_state / embodiment_id through ForwardBatch.
# ==============================================================================


import logging  # noqa: E402
from typing import Iterable, List, Optional, Tuple  # noqa: E402

logger = logging.getLogger(__name__)


# Keys we must never shadow from text_config — they have different
# semantics on the parent (e.g. `torch_dtype` on Qwen3VLConfig applies to
# the whole VLM).
_SKIP_TEXT_CONFIG_PROMOTE = {
    "torch_dtype",
    "dtype",
    "architectures",
    "model_type",
    "auto_map",
    "transformers_version",
    "_name_or_path",
    "text_config",
    "vision_config",
}


def _flatten_text_config_onto_parent(parent_cfg) -> None:
    """Promote every attribute of `parent_cfg.text_config` onto the
    parent config, unless the parent already carries a non-empty value for
    that attribute or the attribute is in `_SKIP_TEXT_CONFIG_PROMOTE`.

    sglang's Qwen3-VL backbone init (and its Qwen2/3Model ancestors) read
    model-shape fields directly on the top-level config, but transformers
    v4.49+ stores them only under `text_config`.  Mirrors alpamayo's
    `_load_alpamayo_config` flattening step.

    Also seeds sglang-only flags (`encoder_only`, `language_only`) that
    the Qwen3VL backbone's `load_weights` reads without a default; these
    are always `False` for GR00T (we use the full VLM).
    """
    tc = getattr(parent_cfg, "text_config", None)
    if tc is None:
        return
    try:
        # Prefer the explicit serialization dict so we only copy user-facing
        # fields, not internal pydantic/transformers plumbing.
        tc_items = tc.to_dict().items() if hasattr(tc, "to_dict") else vars(tc).items()
    except Exception:
        tc_items = vars(tc).items()
    for key, value in tc_items:
        if key in _SKIP_TEXT_CONFIG_PROMOTE:
            continue
        if key.startswith("_"):
            continue
        if getattr(parent_cfg, key, None) not in (None, [], {}):
            continue
        setattr(parent_cfg, key, value)
    # sglang-only flags with no safe default inside qwen3_vl.load_weights.
    for flag in ("encoder_only", "language_only"):
        if not hasattr(parent_cfg, flag):
            setattr(parent_cfg, flag, False)


def _split_groot_weights(weights: Iterable[Tuple[str, torch.Tensor]]) -> Tuple[
    List[Tuple[str, torch.Tensor]],
    List[Tuple[str, torch.Tensor]],
    List[str],
]:
    """Split a flat checkpoint-name → tensor stream into three buckets:

    - backbone_weights: tensors that belong to the Qwen3-VL backbone, with
      the leading `backbone.model.` prefix stripped so they match
      sglang.srt.models.qwen3_vl.Qwen3VLForConditionalGeneration's expected
      naming (which itself remaps ``model.language_model.*`` →
      ``language_model.model.*`` via its WeightsMapper).
    - head_weights: tensors under `action_head.*`, prefix stripped, ready to
      pass into Gr00tN1d7ActionHead.load_state_dict.
    - unrouted: names that did not match either prefix.  Callers decide how
      to handle them (usually log + ignore).
    """
    backbone_weights: List[Tuple[str, torch.Tensor]] = []
    head_weights: List[Tuple[str, torch.Tensor]] = []
    unrouted: List[str] = []

    for name, tensor in weights:
        if name.startswith("backbone.model."):
            backbone_weights.append((name[len("backbone.model.") :], tensor))
        elif name.startswith("action_head."):
            head_weights.append((name[len("action_head.") :], tensor))
        else:
            unrouted.append(name)

    return backbone_weights, head_weights, unrouted


class Gr00tN1d7(nn.Module):
    """NVIDIA GR00T-N1.7 for SGLang.

    Backbone: Qwen3-VL (Cosmos-Reason2-2B) truncated to `config.select_layer`
    language-model layers — matches Isaac-GR00T's behaviour of discarding
    layers after layer 16 at load time.  The output of the last remaining
    LM layer is the layer-16 hidden state the action head consumes.

    Action head: `Gr00tN1d7ActionHead` (DiT flow-matching decoder).

    load_weights routes checkpoint tensors:
      `backbone.model.*` → self.backbone.load_weights (VLM's own mapper
        remaps `model.language_model.*` → `language_model.model.*`)
      `action_head.*`    → self.action_head.load_state_dict
    """

    def __init__(
        self,
        config: Gr00tN1d7Config,
        quant_config=None,
    ):
        super().__init__()
        self.config = config

        # Import Qwen3VL lazily so unit tests that only exercise load_weights
        # routing don't pull in sglang's distributed-init prerequisites.
        # Build a Qwen3-VL-shaped base config pointing at Cosmos-Reason2-2B.
        from transformers import AutoConfig

        from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

        vlm_hf_config = AutoConfig.from_pretrained(
            config.model_name, trust_remote_code=True
        )

        # sglang's `Qwen3VLForConditionalGeneration.__init__` (and the
        # Qwen3Model / Qwen2Model ancestors) read model-shape fields
        # (`vocab_size`, `hidden_size`, `rope_scaling`, `attention_bias`, ...)
        # directly on the top-level config, but transformers v4.49+ stores
        # these only under `text_config`.  Promote every text_config attr
        # not already on the top-level config.  Mirrors alpamayo's
        # `_load_alpamayo_config` flattening step.
        _flatten_text_config_onto_parent(vlm_hf_config)

        self.backbone = Qwen3VLForConditionalGeneration(
            vlm_hf_config,
            quant_config=quant_config,
        )

        # Truncate language_model layers to select_layer (matches Isaac-GR00T
        # qwen3_backbone.py lines 87-88).  sglang's Qwen3LLMModel stores the
        # transformer blocks at `self.backbone.model.layers`.
        lm = getattr(self.backbone, "model", None)
        if lm is not None and hasattr(lm, "layers"):
            target = config.select_layer
            while len(lm.layers) > target:
                lm.layers.pop(-1)

        self.action_head = Gr00tN1d7ActionHead(config)

        # Expose the truncated backbone's
        # post-norm hidden states to `self.action_head`.  sglang's
        # `LogitsProcessorOutput.hidden_states` is only populated when a
        # request sets `capture_hidden_mode` (EAGLE's path); GR00T needs the
        # layer-16 output unconditionally on every forward, so we attach a
        # forward hook on `Qwen3LLMModel`.  Its forward returns either a
        # post-norm tensor `[total_tokens, hidden_size]` or
        # `(hidden_states, aux_hidden_states)` when aux is being captured —
        # we keep only `hidden_states`.
        self._layer16_cache: Optional[torch.Tensor] = None
        if lm is not None:
            lm.register_forward_hook(self._capture_layer16)

    def _capture_layer16(self, _module, _inputs, output):
        if isinstance(output, tuple):
            self._layer16_cache = output[0]
        else:
            self._layer16_cache = output

    # --- Backbone method proxies --------------------------------------
    # sglang's multimodal plumbing (`scheduler.prepare_inputs`,
    # `mm_utils.embed_mm_inputs`, vision-encoder dispatch) looks up these
    # methods on the top-level model class, not on `self.backbone`.
    # Delegating to the Qwen3-VL backbone lets GR00T inherit its image_pad
    # expansion, image/video feature extraction, and helpers without
    # duplicating the code.

    def pad_input_ids(self, input_ids, mm_inputs):
        return self.backbone.pad_input_ids(input_ids, mm_inputs)

    def get_image_feature(self, items):
        return self.backbone.get_image_feature(items)

    def get_video_feature(self, items):
        return self.backbone.get_video_feature(items)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @property
    def start_layer(self) -> int:
        return self.backbone.start_layer

    @property
    def end_layer(self) -> int:
        return self.backbone.end_layer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        **kwargs,
    ):
        """Run the truncated Qwen3-VL backbone, then route the captured
        layer-16 hidden state into the GR00T action head.

        Design (plan Task 4.3 Option B): `Qwen3LLMModel.forward` is hooked
        in `__init__` so we get the post-norm hidden states without needing
        to set `capture_hidden_mode` per-request.  We still run the backbone
        through `self.backbone.__call__` so the standard
        `LogitsProcessorOutput` (next_token_logits + optional
        customized_info) is produced for sglang's scheduler — GR00T's logits
        are semantically meaningless (lm_head is sized for the untruncated
        model) but present so the sampler pipeline stays happy.

        GR00T reuses the shared VLA contract (same as alpamayo):
          - Input:  `forward_batch.history_trajs[i]` — dict stashed by
            Gr00tN1d7Processor.  Keys:
              "proprio_state_tensor": (state_history_length, 132) fp tensor
              "embodiment_id":        int
          - Output: `customized_info["pred_traj"]` — list of per-request
            `[action_horizon=40, action_dim=132]` nested lists (None for
            requests that did not carry history_traj).  The existing
            customized_info → meta_info → SglExt.pred_traj transport
            delivers it to the HTTP response.
        """
        # Build image_mask BEFORE calling self.backbone(...): the backbone's
        # mm-embed routine clamps `input_ids` in-place to `[0, vocab_size-1]`
        # (sglang `mm_utils.embed_mm_inputs_with_split`) and consumes
        # `forward_batch.mm_inputs`, so anything we'd derive after the call
        # has already lost the per-item pad_value markers.  sglang's
        # `MultiModalityDataPaddingPatternMultimodalTokens` writes a hashed
        # pad_value into image-token positions; we recover the mask via
        # `isin(input_ids, [item.pad_value for item in image_items])`.
        from sglang.srt.managers.schedule_batch import Modality

        image_pad_values: List[int] = []
        for mm_in in getattr(forward_batch, "mm_inputs", None) or []:
            if mm_in is None:
                continue
            for item in mm_in.mm_items:
                if item.modality == Modality.IMAGE and item.pad_value is not None:
                    image_pad_values.append(int(item.pad_value))
        if image_pad_values:
            pv = torch.tensor(
                image_pad_values, device=input_ids.device, dtype=input_ids.dtype
            )
            image_mask = torch.isin(input_ids, pv)
        else:
            image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        backbone_attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        ret = self.backbone(input_ids, positions, forward_batch, **kwargs)

        backbone_hidden = self._layer16_cache
        if backbone_hidden is None:
            return ret

        # The action head only runs once per request — during prefill (extend),
        # which is when `input_ids` carries the prompt + images.  On decode
        # steps `input_ids` is just the newly-generated token, no image tokens,
        # so AlternateVLDiT's image-attn cross-block would see an all-False
        # mask and crash MaskedFlashAttention.  pred_traj reaches the response
        # via `customized_info` set during prefill.
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and forward_mode.is_decode():
            return ret

        history_trajs = getattr(forward_batch, "history_trajs", None)
        if not history_trajs or not any(
            isinstance(ht, dict)
            and ht.get("proprio_state_tensor") is not None
            and ht.get("embodiment_id") is not None
            for ht in history_trajs
        ):
            return ret

        vl_embeds = (
            backbone_hidden.unsqueeze(0)
            if backbone_hidden.dim() == 2
            else backbone_hidden
        )
        attn = (
            backbone_attention_mask.unsqueeze(0)
            if backbone_attention_mask.dim() == 1
            else backbone_attention_mask
        )
        img = image_mask.unsqueeze(0) if image_mask.dim() == 1 else image_mask

        # Current scope: drive the action head once for the active request.
        # Multi-request batching is future work (shape-compatible, just
        # needs per-req slicing of backbone_hidden).
        first_ht = next(
            ht
            for ht in history_trajs
            if isinstance(ht, dict)
            and ht.get("proprio_state_tensor") is not None
            and ht.get("embodiment_id") is not None
        )
        state = first_ht["proprio_state_tensor"]
        embodiment_id = first_ht["embodiment_id"]
        if state.dim() == 2:
            state = state.unsqueeze(0)  # (1, state_history_length, max_state_dim)
        device = vl_embeds.device
        state = state.to(device=device, dtype=vl_embeds.dtype)
        embod_tensor = torch.tensor(
            [int(embodiment_id)], dtype=torch.long, device=device
        )

        pred_action = self.action_head.get_action(
            vl_embeds=vl_embeds,
            vl_attn_mask=attn,
            image_mask=img,
            state=state,
            embodiment_id=embod_tensor,
            forward_batch=forward_batch,
        )  # (1, action_horizon, action_dim)

        # Build per-request list aligned with forward_batch.history_trajs
        # ordering — None where the request did not carry a history_traj.
        action_np = pred_action.detach().float().cpu().tolist()
        per_req: List = []
        idx = 0
        for ht in history_trajs:
            if (
                isinstance(ht, dict)
                and ht.get("proprio_state_tensor") is not None
                and ht.get("embodiment_id") is not None
            ):
                # Only the first active slot actually ran; later active
                # slots fall back to the same tensor until batched
                # inference lands.
                per_req.append(action_np[0] if idx == 0 else None)
                idx += 1
            else:
                per_req.append(None)

        ret.customized_info = {
            **(ret.customized_info or {}),
            "pred_traj": per_req,
        }
        return ret

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ):
        backbone_weights, head_weights, unrouted = _split_groot_weights(weights)

        if unrouted:
            logger.warning(
                "Gr00tN1d7.load_weights: %d unrouted tensors, ignoring. "
                "First few: %r",
                len(unrouted),
                unrouted[:5],
            )

        self.backbone.load_weights(iter(backbone_weights))
        logger.info("Gr00tN1d7: loaded %d backbone tensors", len(backbone_weights))

        head_state = {name: tensor for name, tensor in head_weights}
        missing, unexpected = self.action_head.load_state_dict(head_state, strict=False)
        # Ignore non-persistent buffers (e.g. `freqs` on sinusoidal encoders
        # that are recomputed at every forward pass).
        offending = [m for m in missing if not m.endswith("freqs")]
        if offending or unexpected:
            raise RuntimeError(
                f"Gr00tN1d7 action_head load failed: "
                f"missing={offending[:8]}, unexpected={unexpected[:8]}"
            )
        logger.info("Gr00tN1d7: loaded %d action_head tensors", len(head_weights))


EntryClass = [Gr00tN1d7]
