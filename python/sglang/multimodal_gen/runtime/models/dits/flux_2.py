# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
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

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNormContinuous

from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm
from sglang.multimodal_gen.configs.models.dits.flux import FluxConfig
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, apply_qk_norm
from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    NDRotaryEmbedding,
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)  # pylint: disable=invalid-name


def _get_qkv_projections(
    attn: "Flux2Attention", hidden_states, encoder_hidden_states=None
):
    query, _ = attn.to_q(hidden_states)
    key, _ = attn.to_k(hidden_states)
    value, _ = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query, _ = attn.add_q_proj(encoder_hidden_states)
        encoder_key, _ = attn.add_k_proj(encoder_hidden_states)
        encoder_value, _ = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


class Flux2SwiGLU(nn.Module):
    """
    Flux 2 uses a SwiGLU-style activation in the transformer feedforward sub-blocks, but with the linear projection
    layer fused into the first linear layer of the FF sub-block. Thus, this module has no trainable parameters.
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        x = self.gate_fn(x1) * x2
        return x


class Flux2FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: float = 3.0,
        inner_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # Flux2SwiGLU will reduce the dimension by half
        self.linear_in = ColumnParallelLinear(
            dim, inner_dim * 2, bias=bias, gather_output=True
        )
        self.act_fn = Flux2SwiGLU()
        self.linear_out = ColumnParallelLinear(
            inner_dim, dim_out, bias=bias, gather_output=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_in(x)
        x = self.act_fn(x)
        x, _ = self.linear_out(x)
        return x


class Flux2Attention(torch.nn.Module, AttentionModuleMixin):
    def __init__(
        self,
        query_dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * num_heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else num_heads

        self.use_bias = bias
        self.dropout = dropout

        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.to_q = ColumnParallelLinear(
            query_dim, self.inner_dim, bias=bias, gather_output=True
        )
        self.to_k = ColumnParallelLinear(
            query_dim, self.inner_dim, bias=bias, gather_output=True
        )
        self.to_v = ColumnParallelLinear(
            query_dim, self.inner_dim, bias=bias, gather_output=True
        )

        # QK Norm
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(
            ColumnParallelLinear(
                self.inner_dim, self.out_dim, bias=out_bias, gather_output=True
            )
        )
        self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            self.add_q_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias,
                gather_output=True,
            )
            self.add_k_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias,
                gather_output=True,
            )
            self.add_v_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias,
                gather_output=True,
            )
            self.to_add_out = ColumnParallelLinear(
                self.inner_dim, query_dim, bias=out_bias, gather_output=True
            )

        self.attn = USPAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(self, hidden_states, encoder_hidden_states)
        )

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        if (
            query.is_cuda
            and (self.norm_q.variance_epsilon == self.norm_k.variance_epsilon)
            and can_use_fused_inplace_qknorm(self.head_dim, query.dtype)
        ):
            query, key = apply_qk_norm(
                q=query,
                k=key,
                q_norm=self.norm_q,
                k_norm=self.norm_k,
                head_dim=self.head_dim,
                allow_inplace=True,
            )
        else:
            query = self.norm_q(query)
            key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.heads, -1))

            if (
                encoder_query.is_cuda
                and (
                    self.norm_added_q.variance_epsilon
                    == self.norm_added_k.variance_epsilon
                )
                and can_use_fused_inplace_qknorm(self.head_dim, encoder_query.dtype)
            ):
                encoder_query, encoder_key = apply_qk_norm(
                    q=encoder_query,
                    k=encoder_key,
                    q_norm=self.norm_added_q,
                    k_norm=self.norm_added_k,
                    head_dim=self.head_dim,
                    allow_inplace=True,
                )
            else:
                encoder_query = self.norm_added_q(encoder_query)
                encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            cos_sin_cache = torch.cat(
                [
                    cos.to(dtype=torch.float32).contiguous(),
                    sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )

        hidden_states = self.attn(query, key, value)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states)

        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class Flux2ParallelSelfAttention(torch.nn.Module, AttentionModuleMixin):
    """
    Flux 2 parallel self-attention for the Flux 2 single-stream transformer blocks.

    This implements a parallel transformer block, where the attention QKV projections are fused to the feedforward (FF)
    input projections, and the attention output projections are fused to the FF output projections. See the [ViT-22B
    paper](https://arxiv.org/abs/2302.05442) for a visual depiction of this type of transformer block.
    """

    # Does not support QKV fusion as the QKV projections are always fused
    _supports_qkv_fusion = False

    def __init__(
        self,
        query_dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * num_heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else num_heads

        self.use_bias = bias
        self.dropout = dropout

        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(query_dim * self.mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        # Fused QKV projections + MLP input projection
        self.to_qkv_mlp_proj = ColumnParallelLinear(
            self.query_dim,
            self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=bias,
            gather_output=True,
        )
        self.mlp_act_fn = Flux2SwiGLU()

        # QK Norm
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # Fused attention output projection + MLP output projection
        self.to_out = ColumnParallelLinear(
            self.inner_dim + self.mlp_hidden_dim,
            self.out_dim,
            bias=out_bias,
            gather_output=True,
        )

        self.attn = USPAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Parallel in (QKV + MLP in) projection
        hidden_states, _ = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states,
            [3 * self.inner_dim, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )

        # Handle the attention logic
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            cos_sin_cache = torch.cat(
                [
                    cos.to(dtype=torch.float32).contiguous(),
                    sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Handle the feedforward (FF) logic
        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)

        # Concatenate and parallel output projection
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states, _ = self.to_out(hidden_states)

        return hidden_states


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        # Note that the MLP in/out linear layers are fused with the attention QKV/out projections, respectively; this
        # is often called a "parallel" transformer block. See the [ViT-22B paper](https://arxiv.org/abs/2302.05442)
        # for a visual depiction of this type of transformer block.
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            num_heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_mod_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        split_hidden_states: bool = False,
        text_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If encoder_hidden_states is None, hidden_states is assumed to have encoder_hidden_states already
        # concatenated
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            freqs_cis=freqs_cis,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :text_seq_len],
                hidden_states[:, text_seq_len:],
            )
            return encoder_hidden_states, hidden_states
        else:
            return hidden_states


class Flux2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Flux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            num_heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_params_img: Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...
        ],
        temb_mod_params_txt: Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...
        ],
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}

        # Modulation parameters shape: [1, 1, self.dim]
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = (
            temb_mod_params_img
        )
        (c_shift_msa, c_scale_msa, c_gate_msa), (
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = temb_mod_params_txt

        # Img stream
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        # Conditioning txt stream
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            1 + c_scale_msa
        ) * norm_encoder_hidden_states + c_shift_msa

        # Attention on concatenated img + txt stream
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            freqs_cis=freqs_cis,
            **joint_attention_kwargs,
        )

        attn_output, context_attn_output = attention_outputs

        # Process attention outputs for the image stream (`hidden_states`).
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        # Process attention outputs for the text stream (`encoder_hidden_states`).
        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
        )

        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(
                in_channels=in_channels,
                time_embed_dim=embedding_dim,
                sample_proj_bias=bias,
            )
        else:
            self.guidance_embedder = None

    def forward(
        self, timestep: torch.Tensor, guidance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(timestep.dtype)
        )  # (N, D)

        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(
                guidance_proj.to(guidance.dtype)
            )  # (N, D)
            time_guidance_emb = timesteps_emb + guidance_emb
            return time_guidance_emb
        else:
            return timesteps_emb


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets

        self.linear = ColumnParallelLinear(
            dim, dim * 3 * self.mod_param_sets, bias=bias, gather_output=True
        )
        self.act_fn = nn.SiLU()

    def forward(
        self, temb: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        mod = self.act_fn(temb)
        mod, _ = self.linear(mod)

        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        # Return tuple of 3-tuples of modulation params shift/scale/gate
        return tuple(
            mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets)
        )


class Flux2PosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.rope = NDRotaryEmbedding(
            rope_dim_list=axes_dim,
            rope_theta=theta,
            use_real=False,
            repeat_interleave_real=False,
            dtype=(
                torch.float32
                if current_platform.is_mps() or current_platform.is_musa()
                else torch.float64
            ),
        )

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = ids.float()
        # TODO: potential error: flux use n_axes = ids.shape[-1]
        # see: https://github.com/huggingface/diffusers/blob/17c0e79dbdf53fb6705e9c09cc1a854b84c39249/src/diffusers/models/transformers/transformer_flux.py#L509
        freqs_cos, freqs_sin = self.rope.forward_uncached(pos=pos)
        return freqs_cos.contiguous().float(), freqs_sin.contiguous().float()


class Flux2Transformer2DModel(CachableDiT, OffloadableDiTMixin):
    """
    The Transformer model introduced in Flux 2.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    """

    param_names_mapping = FluxConfig().arch_config.param_names_mapping

    def __init__(self, config: FluxConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)
        patch_size: int = config.patch_size
        in_channels: int = config.in_channels
        out_channels: Optional[int] = config.out_channels
        num_layers: int = config.num_layers
        num_single_layers: int = config.num_single_layers
        attention_head_dim: int = config.attention_head_dim
        num_attention_heads: int = config.num_attention_heads
        joint_attention_dim: int = config.joint_attention_dim
        timestep_guidance_channels: int = config.timestep_guidance_channels
        mlp_ratio: float = config.mlp_ratio
        axes_dims_rope: Tuple[int, ...] = config.axes_dims_rope
        rope_theta: int = config.rope_theta
        eps: float = config.eps
        guidance_embeds: bool = getattr(config, "guidance_embeds", True)
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.guidance_embeds = guidance_embeds

        # 1. Sinusoidal positional embedding for RoPE on image and text tokens
        self.rotary_emb = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        # 2. Combined timestep + guidance embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )

        # 3. Modulation (double stream and single stream blocks share modulation parameters, resp.)
        # Two sets of shift/scale/gate modulation parameters for the double stream attn and FF sub-blocks
        self.double_stream_modulation_img = Flux2Modulation(
            self.inner_dim, mod_param_sets=2, bias=False
        )
        self.double_stream_modulation_txt = Flux2Modulation(
            self.inner_dim, mod_param_sets=2, bias=False
        )
        # Only one set of modulation parameters as the attn and FF sub-blocks are run in parallel for single stream
        self.single_stream_modulation = Flux2Modulation(
            self.inner_dim, mod_param_sets=1, bias=False
        )

        # 4. Input projections
        self.x_embedder = ColumnParallelLinear(
            in_channels, self.inner_dim, bias=False, gather_output=True
        )
        self.context_embedder = ColumnParallelLinear(
            joint_attention_dim, self.inner_dim, bias=False, gather_output=True
        )

        # 5. Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=eps,
            bias=False,
        )
        self.proj_out = ColumnParallelLinear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=False,
            gather_output=True,
        )

        self.layer_names = ["transformer_blocks", "single_transformer_blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        guidance: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        """
        # 0. Handle input arguments
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. Calculate timestep embedding and modulation parameters
        timestep = timestep.to(hidden_states.dtype)
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype)

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        # 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
        hidden_states, _ = self.x_embedder(hidden_states)
        encoder_hidden_states, _ = self.context_embedder(encoder_hidden_states)

        # 3. Calculate RoPE embeddings from image and text tokens
        # NOTE: the below logic means that we can't support batched inference with images of different resolutions or
        # text prompts of different lengths. Is this a use case we want to support?
        # 4. Double Stream Transformer Blocks
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                freqs_cis=freqs_cis,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        # Concatenate text and image streams for single-block inference
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 5. Single Stream Transformer Blocks
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                freqs_cis=freqs_cis,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        # Remove text tokens from concatenated stream
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # 6. Output layers
        hidden_states = self.norm_out(hidden_states, temb)
        output, _ = self.proj_out(hidden_states)

        return output


EntryClass = Flux2Transformer2DModel
