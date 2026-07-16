# Copyright 2025 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
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
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.glmimage import GlmImageDitConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import (
    ScaleResidualLayerNormScaleShift,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.mlp import FeedForward
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import Timesteps
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_is_cuda = current_platform.is_cuda()


class GlmImageLayerKVCache:
    """KV cache for GlmImage model."""

    def __init__(self):
        self.k_cache = None
        self.v_cache = None
        self.mode: Optional[str] = None  # "write", "read", "skip"

    def store(self, k: torch.Tensor, v: torch.Tensor):
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=1)
            self.v_cache = torch.cat([self.v_cache, v], dim=1)

    def get(self):
        return self.k_cache, self.v_cache

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self.mode = None


class GlmImageKVCache:
    """Container for all layers' KV caches."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.caches = [GlmImageLayerKVCache() for _ in range(num_layers)]

    def __getitem__(self, layer_idx: int) -> GlmImageLayerKVCache:
        return self.caches[layer_idx]

    def set_mode(self, mode: Optional[str]):
        if mode is not None and mode not in ["write", "read", "skip"]:
            raise ValueError(
                f"Invalid mode: {mode}, must be one of 'write', 'read', 'skip'"
            )
        for cache in self.caches:
            cache.mode = mode

    def clear(self):
        for cache in self.caches:
            cache.clear()


class GlmImageTimestepEmbedding(nn.Module):
    """
    Replacement for diffusers TimestepEmbedding using ReplicatedLinear.
    Structure: linear_1 -> act(silu) -> linear_2
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = time_embed_dim
        self.linear_1 = ReplicatedLinear(in_channels, time_embed_dim, bias=True)
        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "gelu":
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.SiLU()
        self.linear_2 = ReplicatedLinear(time_embed_dim, out_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample, _ = self.linear_1(sample)
        sample = self.act(sample)
        sample, _ = self.linear_2(sample)
        return sample


class GlmImageTextProjection(nn.Module):
    """
    Replacement for diffusers PixArtAlphaTextProjection using ReplicatedLinear.
    Structure: linear_1 -> act_1 -> linear_2
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int = None,
        act_fn: str = "silu",
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = ReplicatedLinear(in_features, hidden_size, bias=True)
        if act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        else:
            self.act_1 = nn.SiLU()
        self.linear_2 = ReplicatedLinear(hidden_size, out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class GlmImageCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        condition_dim: int,
        pooled_projection_dim: int,
        timesteps_dim: int = 256,
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=timesteps_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.condition_proj = Timesteps(
            num_channels=condition_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = GlmImageTimestepEmbedding(
            in_channels=timesteps_dim, time_embed_dim=embedding_dim
        )
        self.condition_embedder = GlmImageTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu"
        )

    def forward(
        self,
        timestep: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)

        crop_coords_proj = self.condition_proj(crop_coords.flatten()).reshape(
            crop_coords.size(0), -1
        )
        target_size_proj = self.condition_proj(target_size.flatten()).reshape(
            target_size.size(0), -1
        )

        # (B, 2 * condition_dim)
        condition_proj = torch.cat([crop_coords_proj, target_size_proj], dim=1)

        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_dtype)
        )  # (B, embedding_dim)
        condition_emb = self.condition_embedder(
            condition_proj.to(dtype=hidden_dtype)
        )  # (B, embedding_dim)

        conditioning = timesteps_emb + condition_emb
        return conditioning


class GlmImageImageProjector(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = hidden_states.shape
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size

        hidden_states = hidden_states.reshape(
            batch_size,
            channel,
            post_patch_height,
            self.patch_size,
            post_patch_width,
            self.patch_size,
        )
        hidden_states = (
            hidden_states.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        )
        hidden_states = self.proj(hidden_states)

        return hidden_states


class GlmImageAdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, dim: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.linear = ReplicatedLinear(embedding_dim, 12 * dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype
        norm_hidden_states = self.norm(hidden_states).to(dtype=dtype)
        norm_encoder_hidden_states = self.norm_context(encoder_hidden_states).to(
            dtype=dtype
        )

        emb, _ = self.linear(temb)
        (
            shift_msa,
            c_shift_msa,
            scale_msa,
            c_scale_msa,
            gate_msa,
            c_gate_msa,
            shift_mlp,
            c_shift_mlp,
            scale_mlp,
            c_scale_mlp,
            gate_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)

        hidden_states = norm_hidden_states * (
            1 + scale_msa.unsqueeze(1)
        ) + shift_msa.unsqueeze(1)
        encoder_hidden_states = norm_encoder_hidden_states * (
            1 + c_scale_msa.unsqueeze(1)
        ) + c_shift_msa.unsqueeze(1)

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        )


class GlmImageGELU(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj" if prefix else "proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate="tanh")


class GlmImageFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        inner_dim: Optional[int] = None,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList(
            [
                GlmImageGELU(
                    dim,
                    inner_dim,
                    bias=bias,
                    quant_config=quant_config,
                    prefix=f"{prefix}.net.0" if prefix else "net.0",
                ),
                nn.Dropout(0.0),
                RowParallelLinear(
                    inner_dim,
                    dim_out,
                    bias=bias,
                    input_is_parallel=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.net.2" if prefix else "net.2",
                ),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net[0](hidden_states)
        hidden_states = self.net[1](hidden_states)
        hidden_states, _ = self.net[2](hidden_states)
        return hidden_states


class GlmImageAttention(torch.nn.Module):
    def __init__(
        self,
        query_dim,
        heads,
        dim_head,
        out_dim,
        bias,
        qk_norm,
        elementwise_affine,
        eps,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.k_cache = None
        self.v_cache = None

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.dim_head = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.out_dim = out_dim if out_dim is not None else query_dim

        tp_size = get_tp_world_size()
        assert (
            self.heads % tp_size == 0
        ), f"heads ({self.heads}) must be divisible by tp_size ({tp_size})"
        self.num_local_heads = self.heads // tp_size
        self.num_local_kv_heads = self.num_local_heads

        self.to_qkv = MergedColumnParallelLinear(
            query_dim,
            [self.inner_dim, self.inner_kv_dim, self.inner_kv_dim],
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv" if prefix else "to_qkv",
        )

        # (dropout omitted)
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    self.inner_dim,
                    self.out_dim,
                    bias=True,
                    input_is_parallel=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0" if prefix else "to_out.0",
                )
            ]
        )

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(
                dim_head, eps=eps, elementwise_affine=elementwise_affine
            )
            self.norm_k = nn.LayerNorm(
                dim_head, eps=eps, elementwise_affine=elementwise_affine
            )
        else:
            raise ValueError(
                f"unknown qk_norm: {qk_norm}. Should be one of None, 'layer_norm', 'fp32_layer_norm', 'layer_norm_across_heads', 'rms_norm', 'rms_norm_across_heads', 'l2'."
            )

        self.attn = USPAttention(
            num_heads=self.num_local_heads,
            head_size=dim_head,
            num_kv_heads=self.num_local_kv_heads,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_meta: Optional[Dict[str, Any]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[GlmImageLayerKVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = encoder_hidden_states.dtype

        batch_size, text_seq_length, _ = encoder_hidden_states.shape
        _, image_seq_length, _ = hidden_states.shape
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 1. QKV projections
        qkv, _ = self.to_qkv(hidden_states)
        query, key, value = [tensor.contiguous() for tensor in qkv.chunk(3, dim=-1)]

        query = query.unflatten(2, (self.num_local_heads, -1))
        key = key.unflatten(2, (self.num_local_kv_heads, -1))
        value = value.unflatten(2, (self.num_local_kv_heads, -1))

        # 2. QK normalization
        if self.norm_q is not None:
            query = self.norm_q(query).to(dtype=dtype)
        if self.norm_k is not None:
            key = self.norm_k(key).to(dtype=dtype)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb

            if _is_cuda and cos.dim() == 2:
                q_img = query[:, text_seq_length:, :, :]
                k_img = key[:, text_seq_length:, :, :]
                cos_sin_cache = torch.cat(
                    [
                        cos.to(dtype=torch.float32).contiguous(),
                        sin.to(dtype=torch.float32).contiguous(),
                    ],
                    dim=-1,
                )
                # apply_flashinfer_rope_qk_inplace is inplace kernel and q_img/k_img are views of query/key, so we need not copy back
                apply_flashinfer_rope_qk_inplace(
                    q_img, k_img, cos_sin_cache, is_neox=True
                )
            else:
                query[:, text_seq_length:, :, :] = _apply_rotary_emb(
                    query[:, text_seq_length:, :, :], cos, sin, is_neox_style=True
                )
                key[:, text_seq_length:, :, :] = _apply_rotary_emb(
                    key[:, text_seq_length:, :, :], cos, sin, is_neox_style=True
                )

        if kv_cache is not None:
            if kv_cache.mode == "write":
                kv_cache.store(key, value)
            elif kv_cache.mode == "read":
                k_cache, v_cache = kv_cache.get()
                key = torch.cat([k_cache, key], dim=1) if k_cache is not None else key
                value = (
                    torch.cat([v_cache, value], dim=1) if v_cache is not None else value
                )
            elif kv_cache.mode == "skip":
                pass

        # 4. Attention
        if attention_mask is not None:
            text_attn_mask = attention_mask
            assert (
                text_attn_mask.dim() == 2
            ), "the shape of text_attn_mask should be (batch_size, text_seq_length)"
            assert text_attn_mask.shape == (
                batch_size,
                text_seq_length,
            ), "the shape of text_attn_mask should match the text hidden states"
            if not (
                isinstance(attention_mask_meta, dict)
                and attention_mask_meta.get("npu_varlen") is not None
            ):
                image_attn_mask = torch.ones(
                    batch_size,
                    image_seq_length,
                    dtype=text_attn_mask.dtype,
                    device=text_attn_mask.device,
                )
                attention_mask = torch.cat(
                    [text_attn_mask, image_attn_mask], dim=1
                )
        hidden_states = self.attn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            num_replicated_prefix=text_seq_length,
            attn_mask_meta=attention_mask_meta,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 5. Output projection
        hidden_states, _ = self.to_out[0](hidden_states)
        # hidden_states = self.to_out[1](hidden_states)         # (dropout omitted)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class GlmImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 2560,
        num_attention_heads: int = 64,
        attention_head_dim: int = 40,
        time_embed_dim: int = 512,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        # 1. Attention
        self.norm1 = GlmImageAdaLayerNormZero(time_embed_dim, dim)

        self.attn1 = GlmImageAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=dim,
            bias=True,
            qk_norm="layer_norm",
            elementwise_affine=False,
            eps=1e-5,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1",
            quant_config=quant_config,
        )

        # 2. Feedforward
        self.norm2 = ScaleResidualLayerNormScaleShift(
            dim, eps=1e-5, elementwise_affine=False
        )
        self.norm2_context = ScaleResidualLayerNormScaleShift(
            dim, eps=1e-5, elementwise_affine=False
        )
        self.ff = GlmImageFeedForward(
            dim=dim,
            dim_out=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.ff" if prefix else "ff",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[
            Union[
                Tuple[torch.Tensor, torch.Tensor],
                List[Tuple[torch.Tensor, torch.Tensor]],
            ]
        ] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_meta: Optional[Dict[str, Any]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        kv_cache: Optional[GlmImageLayerKVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Timestep conditioning
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, temb)

        # 2. Attention
        if attention_kwargs is None:
            attention_kwargs = {}

        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            attention_mask_meta=attention_mask_meta,
            kv_cache=kv_cache,
            **attention_kwargs,
        )

        # 3. Feedforward (fused residual + norm + scale/shift)
        norm_hidden_states, hidden_states = self.norm2(
            hidden_states,
            attn_hidden_states,
            gate_msa.unsqueeze(1),
            shift_mlp.unsqueeze(1),
            scale_mlp.unsqueeze(1),
        )
        norm_encoder_hidden_states, encoder_hidden_states = self.norm2_context(
            encoder_hidden_states,
            attn_encoder_hidden_states,
            c_gate_msa.unsqueeze(1),
            c_shift_mlp.unsqueeze(1),
            c_scale_mlp.unsqueeze(1),
        )

        ff_output = self.ff(norm_hidden_states)
        ff_output_context = self.ff(norm_encoder_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
        encoder_hidden_states = (
            encoder_hidden_states + ff_output_context * c_gate_mlp.unsqueeze(1)
        )

        return hidden_states, encoder_hidden_states


class GlmImageRotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int, theta: float = 10000.0) -> None:
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size
        device = hidden_states.device

        dim_h, dim_w = self.dim // 2, self.dim // 2
        h_inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, dim_h, 2, dtype=torch.float32, device=device)[
                    : (dim_h // 2)
                ].float()
                / dim_h
            )
        )
        w_inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, dim_w, 2, dtype=torch.float32, device=device)[
                    : (dim_w // 2)
                ].float()
                / dim_w
            )
        )
        h_seq = torch.arange(height, device=device)
        w_seq = torch.arange(width, device=device)
        freqs_h = torch.outer(h_seq, h_inv_freq)
        freqs_w = torch.outer(w_seq, w_inv_freq)

        # Create position matrices for height and width
        # [height, 1, dim//4] and [1, width, dim//4]
        freqs_h = freqs_h.unsqueeze(1)
        freqs_w = freqs_w.unsqueeze(0)
        # Broadcast freqs_h and freqs_w to [height, width, dim//4]
        freqs_h = freqs_h.expand(height, width, -1)
        freqs_w = freqs_w.expand(height, width, -1)

        # Concatenate along last dimension to get [height, width, dim//2]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        freqs = freqs.reshape(height * width, -1)  # [height * width, dim//2]
        return (freqs.cos(), freqs.sin())


class GlmImageAdaLayerNormContinuous(nn.Module):
    """
    GlmImage-only final AdaLN: LN(x) -> Linear(cond) -> chunk -> affine. Matches Megatron: **no activation** before the
    Linear on conditioning embedding.
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.linear = nn.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
            # For now, don’t replace this with sglang’s LayerNorm
            # because the model doesn’t have this parameter and it will break model loading
        elif norm_type == "rms_norm":
            self.norm = nn.RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(
        self, x: torch.Tensor, conditioning_embedding: torch.Tensor
    ) -> torch.Tensor:
        # *** NO SiLU here ***
        emb = self.linear(conditioning_embedding.to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class GlmImageTransformer2DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    r"""
    Args:
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, defaults to `40`):
            The number of channels in each head.
        num_attention_heads (`int`, defaults to `64`):
            The number of heads to use for multi-head attention.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_embed_dim (`int`, defaults to `1472`):
            Input dimension of text embeddings from the text encoder.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        condition_dim (`int`, defaults to `256`):
            The embedding dimension of the input SDXL-style resolution conditions (original_size, target_size,
            crop_coords).
        pos_embed_max_size (`int`, defaults to `128`):
            The maximum resolution of the positional embeddings, from which slices of shape `H x W` are taken and added
            to input patched latents, where `H` and `W` are the latent height and width respectively. A value of 128
            means that the maximum supported height and width for image generation is `128 * vae_scale_factor *
            patch_size => 128 * 8 * 2 => 2048`.
        sample_size (`int`, defaults to `128`):
            The base resolution of input latents. If height/width is not provided during generation, this value is used
            to determine the resolution as `sample_size * vae_scale_factor => 128 * 8 => 1024`
    """

    param_names_mapping = GlmImageDitConfig().arch_config.param_names_mapping

    def __init__(
        self,
        config: GlmImageDitConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__(config=config, hf_config=hf_config)

        self.config_data = config  # Store config
        arch_config = config.arch_config

        self.in_channels = arch_config.in_channels
        self.out_channels = arch_config.out_channels
        self.patch_size = arch_config.patch_size
        self.num_layers = arch_config.num_layers
        self.attention_head_dim = arch_config.attention_head_dim
        self.num_attention_heads = arch_config.num_attention_heads
        self.text_embed_dim = arch_config.text_embed_dim
        self.time_embed_dim = arch_config.time_embed_dim

        # GlmImage uses 2 additional SDXL-like conditions - target_size, crop_coords
        # Each of these are sincos embeddings of shape 2 * condition_dim
        pooled_projection_dim = 2 * 2 * arch_config.condition_dim
        inner_dim = arch_config.num_attention_heads * arch_config.attention_head_dim

        # 1. RoPE
        self.rotary_emb = GlmImageRotaryPosEmbed(
            arch_config.attention_head_dim, arch_config.patch_size, theta=10000.0
        )

        # 2. Patch & Text-timestep embedding
        self.image_projector = GlmImageImageProjector(
            arch_config.in_channels, inner_dim, arch_config.patch_size
        )
        self.glyph_projector = FeedForward(
            arch_config.text_embed_dim,
            inner_dim,
            inner_dim=inner_dim,
            activation_fn="gelu",
        )
        self.prior_token_embedding = nn.Embedding(
            arch_config.prior_vq_quantizer_codebook_size, inner_dim
        )
        self.prior_projector = FeedForward(
            inner_dim, inner_dim, inner_dim=inner_dim, activation_fn="linear-silu"
        )

        self.time_condition_embed = GlmImageCombinedTimestepSizeEmbeddings(
            embedding_dim=arch_config.time_embed_dim,
            condition_dim=arch_config.condition_dim,
            pooled_projection_dim=pooled_projection_dim,
            timesteps_dim=arch_config.time_embed_dim,
        )

        # 3. Transformer blocks
        self._supported_attention_backends = arch_config._supported_attention_backends
        self.transformer_blocks = nn.ModuleList(
            [
                GlmImageTransformerBlock(
                    inner_dim,
                    arch_config.num_attention_heads,
                    arch_config.attention_head_dim,
                    arch_config.time_embed_dim,
                    supported_attention_backends=self._supported_attention_backends,
                    prefix=f"transformer_blocks.{i}",
                    quant_config=quant_config,
                )
                for i in range(arch_config.num_layers)
            ]
        )
        self._attention_supports_varlen = getattr(
            self.transformer_blocks[0].attn1.attn.attn_impl,
            "supports_varlen",
            False,
        )

        # 4. Output projection
        self.norm_out = GlmImageAdaLayerNormContinuous(
            inner_dim, arch_config.time_embed_dim, elementwise_affine=False
        )
        self.proj_out = nn.Linear(
            inner_dim,
            arch_config.patch_size * arch_config.patch_size * arch_config.out_channels,
            bias=True,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        prior_token_id: torch.Tensor,
        prior_token_drop: torch.Tensor,
        timestep: torch.LongTensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_seq_lens: Optional[List[int]] = None,
        kv_caches: Optional[GlmImageKVCache] = None,
        kv_caches_mode: Optional[str] = None,
        freqs_cis: Optional[
            Union[
                Tuple[torch.Tensor, torch.Tensor],
                List[Tuple[torch.Tensor, torch.Tensor]],
            ]
        ] = None,
        ###
        guidance: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        if kv_caches is not None:
            kv_caches.set_mode(kv_caches_mode)

        batch_size, _, height, width = hidden_states.shape

        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        use_ascend_varlen = (
            current_platform.is_npu()
            and self._attention_supports_varlen
            and (attention_mask is None or text_seq_lens is not None)
        )
        if current_platform.is_npu() and batch_size > 1 and kv_caches is None:
            if use_ascend_varlen:
                logger.info_once(
                    f"GLM-Image DiT is using Ascend FA true batching (batch_size={batch_size})"
                )
            else:
                logger.warning_once(
                    "GLM-Image DiT is using the batch-1 microbatch fallback because "
                    "Ascend variable-length FA is unavailable"
                )
        if (
            current_platform.is_npu()
            and batch_size > 1
            and kv_caches is None
            and not use_ascend_varlen
        ):

            def slice_batch(value, index):
                if (
                    isinstance(value, torch.Tensor)
                    and value.dim() > 0
                    and value.shape[0] == batch_size
                ):
                    return value[index : index + 1]
                return value

            outputs = []
            for index in range(batch_size):
                sample_encoder_hidden_states = encoder_hidden_states[index : index + 1]
                sample_attention_mask = slice_batch(attention_mask, index)
                if isinstance(sample_attention_mask, torch.Tensor):
                    valid_text_length = int(sample_attention_mask.sum().item())
                    sample_encoder_hidden_states = sample_encoder_hidden_states[
                        :, :valid_text_length
                    ]
                    sample_attention_mask = None

                outputs.append(
                    self.forward(
                        hidden_states=hidden_states[index : index + 1],
                        encoder_hidden_states=sample_encoder_hidden_states,
                        prior_token_id=prior_token_id[index : index + 1],
                        prior_token_drop=prior_token_drop[index : index + 1],
                        timestep=slice_batch(timestep, index),
                        target_size=slice_batch(target_size, index),
                        crop_coords=slice_batch(crop_coords, index),
                        attention_kwargs=attention_kwargs,
                        attention_mask=sample_attention_mask,
                        text_seq_lens=(
                            [text_seq_lens[index]]
                            if text_seq_lens is not None
                            else None
                        ),
                        freqs_cis=freqs_cis,
                        guidance=slice_batch(guidance, index),
                    )
                )
            return torch.cat(outputs, dim=0)

        timestep = timestep - 1.0

        # 1. RoPE
        image_rotary_emb = freqs_cis
        if image_rotary_emb is None:
            image_rotary_emb = self.rotary_emb(hidden_states)
        # 2. Patch & Timestep embeddings
        p = self.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states = self.image_projector(hidden_states)
        encoder_hidden_states = self.glyph_projector(encoder_hidden_states)
        prior_embedding = self.prior_token_embedding(prior_token_id)
        prior_embedding = prior_embedding.masked_fill(prior_token_drop.unsqueeze(-1), 0)
        prior_hidden_states = self.prior_projector(prior_embedding)
        # SP: when latents are H-sharded, hidden_states has fewer patches than prior_hidden_states.
        # Shard prior_hidden_states along seq dim to match (prior is row-major, same as latent patches).
        if (
            get_sp_world_size() > 1
            and prior_hidden_states.shape[1] != hidden_states.shape[1]
        ):
            rank = get_sp_parallel_rank()
            sp_world_size = get_sp_world_size()
            chunk = prior_hidden_states.shape[1] // sp_world_size
            prior_hidden_states = prior_hidden_states[
                :, rank * chunk : (rank + 1) * chunk, :
            ]
        hidden_states = hidden_states + prior_hidden_states

        attention_mask_meta = None
        if attention_mask is not None and use_ascend_varlen:
            if len(text_seq_lens) != batch_size:
                raise ValueError("text_seq_lens must contain one length per sample")
            text_seq_length = encoder_hidden_states.shape[1]
            image_seq_length = (
                hidden_states.shape[1] * get_ulysses_parallel_world_size()
            )
            total_seq_length = text_seq_length + image_seq_length
            has_text_padding = any(
                length != text_seq_length for length in text_seq_lens
            )
            if batch_size > 1 and not has_text_padding:
                logger.info_once(
                    "GLM-Image DiT is using the unpadded Ascend FA fast path"
                )
            gather_indices = None
            restore_indices = None
            if has_text_padding:
                valid_lengths = torch.tensor(
                    text_seq_lens,
                    dtype=torch.long,
                    device=hidden_states.device,
                ).unsqueeze(1)
                positions = torch.arange(
                    total_seq_length, device=hidden_states.device
                ).unsqueeze(0).expand(batch_size, -1)
                image_end = valid_lengths + image_seq_length
                gather_indices = torch.where(
                    positions < valid_lengths,
                    positions,
                    torch.where(
                        positions < image_end,
                        text_seq_length + positions - valid_lengths,
                        valid_lengths + positions - image_end,
                    ),
                )
                restore_indices = torch.empty_like(gather_indices)
                restore_indices.scatter_(1, gather_indices, positions)
            attention_mask_meta = {
                "npu_varlen": {
                    "gather_indices": gather_indices,
                    "restore_indices": restore_indices,
                    "actual_seq_lengths": [
                        int(length) + image_seq_length for length in text_seq_lens
                    ],
                }
            }

        temb = self.time_condition_embed(
            timestep, target_size, crop_coords, hidden_states.dtype
        )
        temb = F.silu(temb)

        # 3. Transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                attention_mask,
                attention_mask_meta,
                attention_kwargs,
                kv_cache=kv_caches[idx] if kv_caches is not None else None,
            )

        # 4. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, -1, p, p
        )
        output = hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)

        return output.float()
        # reference: https://github.com/zRzRzRzRzRzRzR/diffusers/blob/6cfc83b4abc5b083fef56a18ec4700f48ba3aaba/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L737


EntryClass = GlmImageTransformer2DModel
