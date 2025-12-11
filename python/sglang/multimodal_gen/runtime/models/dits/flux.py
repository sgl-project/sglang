# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.attention import AttentionModuleMixin, FeedForward
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from torch.nn import LayerNorm as LayerNorm

from sglang.multimodal_gen.configs.models.dits.flux import FluxConfig
from sglang.multimodal_gen.runtime.layers.attention import USPAttention

# from sglang.multimodal_gen.runtime.layers.layernorm import LayerNorm as LayerNorm
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    NDRotaryEmbedding,
    _apply_rotary_emb,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.utils import (
    delete_projection_layers,
    fuse_linear_projections,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)  # pylint: disable=invalid-name


def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query, _ = attn.to_q(hidden_states)
    key, _ = attn.to_k(hidden_states)
    value, _ = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query, _ = attn.add_q_proj(encoder_hidden_states)
        encoder_key, _ = attn.add_k_proj(encoder_hidden_states)
        encoder_value, _ = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(
    attn: "FluxAttention", hidden_states, encoder_hidden_states=None
):
    qkv, _ = attn.to_qkv(hidden_states)
    query, key, value = qkv.chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        added_qkv, _ = attn.to_added_qkv(encoder_hidden_states)
        encoder_query, encoder_key, encoder_value = added_qkv.chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(
    attn: "FluxAttention", hidden_states, encoder_hidden_states=None
):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class FluxAttention(torch.nn.Module, AttentionModuleMixin):
    _supports_qkv_fusion = True

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
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * num_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else num_heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = RMSNorm(dim_head, eps=eps)

        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_q = ReplicatedLinear(query_dim, self.inner_dim, bias=bias)
        self.to_k = ReplicatedLinear(query_dim, self.inner_dim, bias=bias)
        self.to_v = ReplicatedLinear(query_dim, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = torch.nn.ModuleList([])
            self.to_out.append(
                ReplicatedLinear(self.inner_dim, self.out_dim, bias=out_bias)
            )
            if dropout != 0.0:
                self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            self.add_q_proj = ReplicatedLinear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_k_proj = ReplicatedLinear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_v_proj = ReplicatedLinear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.to_add_out = ReplicatedLinear(self.inner_dim, query_dim, bias=out_bias)

        self.attn = USPAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.SAGE_ATTN,
            },
        )

        self.fused_projections = False

    @torch.no_grad()
    def fuse_projections(self):
        if self.fused_projections:
            return

        self.to_qkv = fuse_linear_projections(
            self.to_q, self.to_k, self.to_v, self.use_bias, ReplicatedLinear
        )
        delete_projection_layers(self, ["to_q", "to_k", "to_v"])

        if self.added_kv_proj_dim is not None:
            self.to_added_qkv = fuse_linear_projections(
                self.add_q_proj,
                self.add_k_proj,
                self.add_v_proj,
                self.added_proj_bias,
                ReplicatedLinear,
            )
            delete_projection_layers(self, ["add_q_proj", "add_k_proj", "add_v_proj"])

        self.fused_projections = True

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        freqs_cis=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(self, x, encoder_hidden_states)
        )

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))
        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            bsz, seq_len, _, _ = query.shape
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            query = _apply_rotary_emb(
                query, cos, sin, is_neox_style=False, interleaved=False
            )
            key = _apply_rotary_emb(
                key, cos, sin, is_neox_style=False, interleaved=False
            )

        x = self.attn(query, key, value)
        x = x.flatten(2, 3)
        x = x.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, x = x.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    x.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            x, _ = self.to_out[0](x)
            if len(self.to_out) == 2:
                x = self.to_out[1](x)
            encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states)

            return x, encoder_hidden_states
        else:
            return x


class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = ReplicatedLinear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(dim + self.mlp_hidden_dim, dim)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            num_heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        proj_hidden_states, _ = self.proj_mlp(norm_hidden_states)
        mlp_hidden_states = self.act_mlp(proj_hidden_states)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            x=norm_hidden_states,
            freqs_cis=freqs_cis,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        proj_out, _ = self.proj_out(hidden_states)
        hidden_states = gate * proj_out
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states


class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            num_heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
        )

        self.norm2 = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = MLP(
            input_dim=dim, mlp_hidden_dim=dim * 4, output_dim=dim, act_type="gelu"
        )
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff_context = MLP(
            input_dim=dim, mlp_hidden_dim=dim * 4, output_dim=dim, act_type="gelu"
        )

        self.ff_context = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention.
        attention_outputs = self.attn(
            x=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            freqs_cis=freqs_cis,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output
        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.rope = NDRotaryEmbedding(
            rope_dim_list=axes_dim,
            rope_theta=theta,
            use_real=False,
            repeat_interleave_real=False,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
        )

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = ids.float()
        # freqs_cos, freqs_sin = self.rope.forward(positions=pos)
        freqs_cos, freqs_sin = self.rope.forward_uncached(pos=pos)
        return freqs_cos.contiguous().float(), freqs_sin.contiguous().float()


class FluxTransformer2DModel(CachableDiT):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/
    """

    def __init__(self, config: FluxConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        self.config = config.arch_config

        self.out_channels = (
            getattr(self.config, "out_channels", None) or self.config.in_channels
        )
        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )

        self.rotary_emb = FluxPosEmbed(theta=10000, axes_dim=self.config.axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if self.config.guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
        )

        self.context_embedder = ReplicatedLinear(
            self.config.joint_attention_dim, self.inner_dim
        )
        self.x_embedder = ReplicatedLinear(self.config.in_channels, self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for _ in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = ReplicatedLinear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.out_channels,
            bias=True,
        )

    def fuse_qkv_projections(self):
        for block in list(self.transformer_blocks) + list(
            self.single_transformer_blocks
        ):
            if hasattr(block.attn, "fuse_projections") and getattr(
                block.attn, "_supports_qkv_fusion", True
            ):
                block.attn.fuse_projections()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        guidance: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            guidance (`torch.Tensor`):
                Guidance embeddings.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        """
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
        hidden_states, _ = self.x_embedder(hidden_states)

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        encoder_hidden_states, _ = self.context_embedder(encoder_hidden_states)

        if (
            joint_attention_kwargs is not None
            and "ip_adapter_image_embeds" in joint_attention_kwargs
        ):
            ip_adapter_image_embeds = joint_attention_kwargs.pop(
                "ip_adapter_image_embeds"
            )
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                freqs_cis=freqs_cis,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                freqs_cis=freqs_cis,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = self.norm_out(hidden_states, temb)

        output, _ = self.proj_out(hidden_states)

        return output


EntryClass = FluxTransformer2DModel
