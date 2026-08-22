# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from sglang.multimodal_gen.configs.models.dits.llada_image import LLaDAImageDitConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm as SGLangRMSNorm
from sglang.multimodal_gen.runtime.layers.layernorm import (
    apply_qk_norm_with_optional_rope,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

ADALN_EMBED_DIM = 256


def apply_rmsnorm_tanh_mul_add(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    norm: SGLangRMSNorm,
) -> torch.Tensor:
    """residual + tanh(gate) * rmsnorm(x), with fp32-upcast norm semantics."""
    return residual + torch.tanh(gate) * norm(x)


SEQUENCE_MULTIPLE = 32
LLADA_IMAGE_ATTENTION_BACKENDS = {
    AttentionBackendEnum.FA,
    AttentionBackendEnum.TORCH_SDPA,
}


class LLaDAImageRMSNorm(SGLangRMSNorm):
    """RMSNorm with a synthesized unit weight for affine-free checkpoints."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__(hidden_size, eps=eps)
        self.weight.missing_param_init = "ones"


@dataclass
class _LLaDAImageSequence:
    features: list[torch.Tensor]
    position_ids: list[torch.Tensor]
    padding_masks: list[torch.Tensor]
    noise_masks: list[list[int]] | None = None


class LLaDAImageTimestepEmbedder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 1024,
        frequency_embedding_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype
    ) -> torch.Tensor:
        half_dim = self.frequency_embedding_dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half_dim, dtype=torch.float32, device=timestep.device)
            / half_dim
        )
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        if self.frequency_embedding_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return self.mlp(embedding.to(dtype=hidden_dtype))


class LLaDAImageRopeEmbedder(nn.Module):
    def __init__(
        self, theta: float, axes_dims: tuple[int, ...], axes_lens: tuple[int, ...]
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = None

    def _create_frequencies(self, device: torch.device) -> list[torch.Tensor]:
        frequencies = []
        for axis_dim, axis_len in zip(self.axes_dims, self.axes_lens):
            inverse_frequencies = 1.0 / (
                self.theta
                ** (
                    torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device)
                    / axis_dim
                )
            )
            positions = torch.arange(axis_len, dtype=torch.float32, device=device)
            angles = torch.outer(positions, inverse_frequencies)
            frequencies.append(torch.complex(torch.cos(angles), torch.sin(angles)))
        return frequencies

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        if self.freqs_cis is None or self.freqs_cis[0].device != position_ids.device:
            self.freqs_cis = self._create_frequencies(position_ids.device)

        frequencies = []
        for axis, axis_frequencies in enumerate(self.freqs_cis):
            frequencies.append(axis_frequencies[position_ids[:, axis]])
        return torch.cat(frequencies, dim=-1)


class LLaDAImageAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "LLaDAImageAttention",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
        num_replicated_suffix: int = 0,
        skip_sequence_parallel_override: bool = False,
    ) -> torch.Tensor:
        qkv, _ = attn.to_qkv(hidden_states)
        query, key, value = qkv.split(
            [attn.inner_dim, attn.inner_dim, attn.inner_dim], dim=-1
        )
        query = query.unflatten(-1, (attn.local_heads, attn.head_dim)).contiguous()
        key = key.unflatten(-1, (attn.local_heads, attn.head_dim)).contiguous()
        value = value.unflatten(-1, (attn.local_heads, attn.head_dim)).contiguous()

        if freqs_cis is not None:
            cos_sin_cache = torch.cat(
                [freqs_cis.real.float(), freqs_cis.imag.float()], dim=-1
            ).reshape(-1, attn.head_dim)
            positions = torch.arange(
                cos_sin_cache.shape[0], device=query.device, dtype=torch.long
            )
            if attn.norm_q is not None:
                query, key = apply_qk_norm_with_optional_rope(
                    q=query,
                    k=key,
                    q_norm=attn.norm_q,
                    k_norm=attn.norm_k,
                    head_dim=attn.head_dim,
                    cos_sin_cache=cos_sin_cache.contiguous(),
                    positions=positions,
                    is_neox=False,
                    allow_inplace=True,
                )
            else:
                query, key = apply_flashinfer_rope_qk_inplace(
                    query,
                    key,
                    cos_sin_cache.contiguous(),
                    head_size=attn.head_dim,
                    positions=positions,
                    is_neox=False,
                )
        elif attn.norm_q is not None:
            query, key = apply_qk_norm_with_optional_rope(
                q=query,
                k=key,
                q_norm=attn.norm_q,
                k_norm=attn.norm_k,
                head_dim=attn.head_dim,
                allow_inplace=True,
            )

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = attn.sgl_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            num_replicated_suffix=num_replicated_suffix,
            skip_sequence_parallel_override=skip_sequence_parallel_override,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states, _ = attn.to_out[0](hidden_states)
        return hidden_states


class LLaDAImageAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = LLaDAImageAttnProcessor
    _available_processors: ClassVar[list[type]] = [LLaDAImageAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(
        self,
        dim: int,
        num_heads: int,
        norm_eps: float,
        qk_norm: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by TP size ({tp_size})"
            )
        self.heads = num_heads
        self.local_heads = num_heads // tp_size
        self.head_dim = dim // num_heads
        self.inner_dim = self.local_heads * self.head_dim
        self.to_qkv = MergedColumnParallelLinear(
            dim,
            [dim, dim, dim],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )
        self.sgl_attention = USPAttention(
            num_heads=self.local_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=LLADA_IMAGE_ATTENTION_BACKENDS,
            prefix="llada_image",
        )
        self.norm_q = (
            LLaDAImageRMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None
        )
        self.norm_k = (
            LLaDAImageRMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None
        )
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    input_is_parallel=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                ),
                nn.Dropout(0.0),
            ]
        )
        self.set_processor(self._default_processor_cls())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        num_replicated_suffix: int = 0,
        skip_sequence_parallel_override: bool = False,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            attention_mask,
            freqs_cis,
            num_replicated_suffix,
            skip_sequence_parallel_override,
        )


class LLaDAImageFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_dim = int(dim / 3 * 8)
        self.w13 = MergedColumnParallelLinear(
            dim,
            [hidden_dim, hidden_dim],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.w13(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.w2(hidden_states)
        return hidden_states


def _select_per_token(
    noisy_value: torch.Tensor,
    clean_value: torch.Tensor,
    noise_mask: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    noise_mask = noise_mask.unsqueeze(-1)
    return torch.where(
        noise_mask == 1,
        noisy_value.unsqueeze(1).expand(-1, sequence_length, -1),
        clean_value.unsqueeze(1).expand(-1, sequence_length, -1),
    )


@maybe_allow_in_graph
class LLaDAImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.modulation = modulation
        self.attention = LLaDAImageAttention(
            dim,
            num_heads,
            norm_eps,
            qk_norm,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        self.feed_forward = LLaDAImageFeedForward(
            dim, quant_config=quant_config, prefix=f"{prefix}.feed_forward"
        )
        self.attention_norm1 = LLaDAImageRMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = LLaDAImageRMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = LLaDAImageRMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = LLaDAImageRMSNorm(dim, eps=norm_eps)
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        adaln_noisy: torch.Tensor | None = None,
        adaln_clean: torch.Tensor | None = None,
        num_replicated_suffix: int = 0,
        skip_sequence_parallel_override: bool = False,
    ) -> torch.Tensor:
        if self.modulation:
            sequence_length = hidden_states.shape[1]
            if noise_mask is None:
                scale_msa, gate_msa, scale_mlp, gate_mlp = (
                    self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
                )
                scale_msa = 1.0 + scale_msa
            else:
                noisy_modulation = self.adaLN_modulation(adaln_noisy)
                clean_modulation = self.adaLN_modulation(adaln_clean)
                noisy_scale_msa, noisy_gate_msa, noisy_scale_mlp, noisy_gate_mlp = (
                    noisy_modulation.chunk(4, dim=1)
                )
                clean_scale_msa, clean_gate_msa, clean_scale_mlp, clean_gate_mlp = (
                    clean_modulation.chunk(4, dim=1)
                )
                scale_msa = _select_per_token(
                    1.0 + noisy_scale_msa,
                    1.0 + clean_scale_msa,
                    noise_mask,
                    sequence_length,
                )
                scale_mlp = _select_per_token(
                    noisy_scale_mlp,
                    clean_scale_mlp,
                    noise_mask,
                    sequence_length,
                )
                gate_msa = _select_per_token(
                    noisy_gate_msa,
                    clean_gate_msa,
                    noise_mask,
                    sequence_length,
                )
                gate_mlp = _select_per_token(
                    noisy_gate_mlp,
                    clean_gate_mlp,
                    noise_mask,
                    sequence_length,
                )

            attention_output = self.attention(
                self.attention_norm1(hidden_states) * scale_msa,
                attention_mask,
                freqs_cis,
                num_replicated_suffix,
                skip_sequence_parallel_override,
            )
            hidden_states = apply_rmsnorm_tanh_mul_add(
                attention_output, gate_msa, hidden_states, self.attention_norm2
            )
            ffn_input = self.ffn_norm1(hidden_states) * (1.0 + scale_mlp)

            ffn_output = self.feed_forward(ffn_input)
            hidden_states = apply_rmsnorm_tanh_mul_add(
                ffn_output, gate_mlp, hidden_states, self.ffn_norm2
            )
        else:
            attention_output = self.attention(
                self.attention_norm1(hidden_states),
                attention_mask,
                freqs_cis,
                num_replicated_suffix,
                skip_sequence_parallel_override,
            )
            hidden_states = hidden_states + self.attention_norm2(attention_output)
            hidden_states = hidden_states + self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(hidden_states))
            )
        return hidden_states


class LLaDAImageFinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, ADALN_EMBED_DIM), dim, bias=True),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        adaln_noisy: torch.Tensor | None = None,
        adaln_clean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise_mask is None:
            scale = 1.0 + self.adaLN_modulation(adaln_input)
            scale = scale.unsqueeze(1)
        else:
            sequence_length = hidden_states.shape[1]
            noisy_scale = 1.0 + self.adaLN_modulation(adaln_noisy)
            clean_scale = 1.0 + self.adaLN_modulation(adaln_clean)
            scale = _select_per_token(
                noisy_scale, clean_scale, noise_mask, sequence_length
            )
        hidden_states = self.norm_final(hidden_states) * scale
        return self.linear(hidden_states)


class _LLaDAImageTransformer2DModel(ModelMixin, ConfigMixin, AttentionMixin):
    r"""
    The denoising transformer used by LLaDAImage for text-to-image generation and single-image editing.

    This component consumes caption features that have already passed through LLaDAImage's QueryFormer, connector, and
    projector. For editing, it additionally consumes GLM/SigVQ features and source-image latents.

    Args:
        all_patch_size (`tuple[int, ...]`, defaults to `(1,)`):
            Supported spatial patch sizes.
        all_f_patch_size (`tuple[int, ...]`, defaults to `(1,)`):
            Supported temporal patch sizes paired with `all_patch_size`.
        in_channels (`int`, defaults to `128`):
            Number of channels in the patchified Flux2 VAE latents.
        dim (`int`, defaults to `3840`):
            Transformer hidden dimension.
        n_layers (`int`, defaults to `30`):
            Number of main transformer blocks.
        n_refiner_layers (`int`, defaults to `2`):
            Number of noise, caption, and SigVQ refiner blocks.
        n_heads (`int`, defaults to `30`):
            Number of attention heads.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon used by RMS normalization layers.
        qk_norm (`bool`, defaults to `True`):
            Whether to apply RMS normalization to query and key tensors.
        cap_feat_dim (`int`, defaults to `2560`):
            Dimension of projected QueryFormer caption features.
        semantic_feat_dim (`int`, defaults to `4096`):
            Dimension of GLM/SigVQ semantic features.
        rope_theta (`float`, defaults to `256.0`):
            RoPE frequency base.
        t_scale (`float`, defaults to `1000.0`):
            Scale applied to diffusion timesteps.
        axes_dims (`tuple[int, ...]`, defaults to `(32, 48, 48)`):
            RoPE dimensions for sequence, height, and width axes.
        axes_lens (`tuple[int, ...]`, defaults to `(32768, 1024, 1024)`):
            Maximum RoPE positions for sequence, height, and width axes.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["LLaDAImageTransformerBlock"]
    _repeated_blocks: ClassVar[list[str]] = ["LLaDAImageTransformerBlock"]
    _skip_layerwise_casting_patterns: ClassVar[list[str]] = [
        "t_embedder",
        "cap_embedder",
        "semantic_embedder",
        "sigvq_embedder",
    ]

    @register_to_config
    def __init__(
        self,
        all_patch_size: tuple[int, ...] = (1,),
        all_f_patch_size: tuple[int, ...] = (1,),
        in_channels: int = 128,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        semantic_feat_dim: int = 4096,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        axes_lens: tuple[int, ...] = (32768, 1024, 1024),
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        if len(all_patch_size) != len(all_f_patch_size):
            raise ValueError(
                "`all_patch_size` and `all_f_patch_size` must have the same length."
            )
        if dim % n_heads != 0:
            raise ValueError(
                f"`dim` ({dim}) must be divisible by `n_heads` ({n_heads})."
            )
        if dim // n_heads != sum(axes_dims):
            raise ValueError(
                "The attention head dimension must equal the sum of `axes_dims`."
            )

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        self.all_x_embedder = nn.ModuleDict()
        self.all_final_layer = nn.ModuleDict()
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            patch_key = f"{patch_size}-{f_patch_size}"
            patch_dim = f_patch_size * patch_size * patch_size * in_channels
            self.all_x_embedder[patch_key] = nn.Linear(patch_dim, dim, bias=True)
            self.all_final_layer[patch_key] = LLaDAImageFinalLayer(dim, patch_dim)

        self.noise_refiner = nn.ModuleList(
            [
                LLaDAImageTransformerBlock(
                    dim,
                    n_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    quant_config=quant_config,
                    prefix=f"noise_refiner.{layer_id}",
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                LLaDAImageTransformerBlock(
                    dim,
                    n_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                    quant_config=quant_config,
                    prefix=f"context_refiner.{layer_id}",
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.sigvq_refiner = nn.ModuleList(
            [
                LLaDAImageTransformerBlock(
                    dim,
                    n_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                    quant_config=quant_config,
                    prefix=f"sigvq_refiner.{layer_id}",
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.layers = nn.ModuleList(
            [
                LLaDAImageTransformerBlock(
                    dim,
                    n_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    quant_config=quant_config,
                    prefix=f"layers.{layer_id}",
                )
                for layer_id in range(n_layers)
            ]
        )

        self.t_embedder = LLaDAImageTimestepEmbedder(min(dim, ADALN_EMBED_DIM))
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps, elementwise_affine=False),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )
        self.semantic_embedder = nn.Sequential(
            RMSNorm(semantic_feat_dim, eps=norm_eps, elementwise_affine=False),
            nn.Linear(semantic_feat_dim, dim, bias=True),
        )
        self.sigvq_embedder = nn.Sequential(
            RMSNorm(semantic_feat_dim, eps=norm_eps, elementwise_affine=False),
            nn.Linear(semantic_feat_dim, dim, bias=True),
        )

        nn.init.normal_(self.semantic_embedder[1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.semantic_embedder[1].bias)
        nn.init.normal_(self.sigvq_embedder[1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.sigvq_embedder[1].bias)

        self.x_pad_token = nn.Parameter(torch.zeros(1, dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, dim))
        self.sigvq_pad_token = nn.Parameter(torch.zeros(1, dim))
        nn.init.normal_(self.sigvq_pad_token, mean=0.0, std=0.02)

        self.rope_embedder = LLaDAImageRopeEmbedder(rope_theta, axes_dims, axes_lens)

    @staticmethod
    def _create_coordinate_grid(
        size: tuple[int, int, int],
        start: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        axes = [
            torch.arange(
                start_value, start_value + span, dtype=torch.int32, device=device
            )
            for start_value, span in zip(start, size)
        ]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def _patchify_image(
        self,
        image: torch.Tensor,
        patch_size: int,
        f_patch_size: int,
    ) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
        channels, frames, height, width = image.shape
        frame_tokens = frames // f_patch_size
        height_tokens = height // patch_size
        width_tokens = width // patch_size
        image = image.view(
            channels,
            frame_tokens,
            f_patch_size,
            height_tokens,
            patch_size,
            width_tokens,
            patch_size,
        )
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
            frame_tokens * height_tokens * width_tokens,
            f_patch_size * patch_size * patch_size * channels,
        )
        return (
            image,
            (frames, height, width),
            (frame_tokens, height_tokens, width_tokens),
        )

    def _pad_with_ids(
        self,
        features: torch.Tensor,
        position_grid_size: tuple[int, int, int],
        position_start: tuple[int, int, int],
        noise_value: int | None = None,
        sequence_multiple: int = SEQUENCE_MULTIPLE,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list[int] | None]:
        original_length = len(features)
        padding_length = (-original_length) % sequence_multiple
        padded_length = original_length + padding_length
        device = features.device

        position_ids = self._create_coordinate_grid(
            position_grid_size,
            position_start,
            device,
        ).flatten(0, 2)
        if padding_length > 0:
            padding_position_ids = (
                self._create_coordinate_grid(
                    (1, 1, 1),
                    (0, 0, 0),
                    device,
                )
                .flatten(0, 2)
                .repeat(padding_length, 1)
            )
            position_ids = torch.cat([position_ids, padding_position_ids], dim=0)
            features = torch.cat(
                [features, features[-1:].repeat(padding_length, 1)], dim=0
            )
            padding_mask = torch.cat(
                [
                    torch.zeros(original_length, dtype=torch.bool, device=device),
                    torch.ones(padding_length, dtype=torch.bool, device=device),
                ]
            )
        else:
            padding_mask = torch.zeros(original_length, dtype=torch.bool, device=device)

        noise_mask = [noise_value] * padded_length if noise_value is not None else None
        return features, position_ids, padding_mask, padded_length, noise_mask

    @staticmethod
    def _batch_sequences(
        features: list[torch.Tensor],
        frequencies: list[torch.Tensor],
        inner_padding_masks: list[torch.Tensor],
        pad_token: torch.Tensor,
        noise_masks: list[list[int]] | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, list[int], torch.Tensor | None
    ]:
        sequence_lengths = [len(item) for item in features]
        max_sequence_length = max(sequence_lengths)
        features = torch.cat(features, dim=0)
        inner_padding_mask = torch.cat(inner_padding_masks).unsqueeze(-1)
        features = torch.where(
            inner_padding_mask.to(device=features.device),
            pad_token.to(device=features.device, dtype=features.dtype),
            features,
        )
        features = list(features.split(sequence_lengths, dim=0))

        features = pad_sequence(features, batch_first=True, padding_value=0.0)
        frequencies = pad_sequence(frequencies, batch_first=True, padding_value=0.0)[
            :, : features.shape[1]
        ]

        attention_mask = None
        if not all(length == max_sequence_length for length in sequence_lengths):
            attention_mask = torch.zeros(
                (len(sequence_lengths), max_sequence_length),
                dtype=torch.bool,
                device=features.device,
            )
            for batch_index, sequence_length in enumerate(sequence_lengths):
                attention_mask[batch_index, :sequence_length] = True

        noise_mask = None
        if noise_masks is not None:
            noise_mask = pad_sequence(
                [
                    torch.tensor(mask, dtype=torch.long, device=features.device)
                    for mask in noise_masks
                ],
                batch_first=True,
                padding_value=0,
            )[:, : features.shape[1]]

        return features, frequencies, attention_mask, sequence_lengths, noise_mask

    def _unpatchify(
        self,
        hidden_states: list[torch.Tensor],
        sizes: list[tuple[int, int, int] | list[tuple[int, int, int]]],
        patch_size: int,
        f_patch_size: int,
        image_offsets: list[tuple[int, int]] | None = None,
    ) -> list[torch.Tensor]:
        outputs = []
        sequence_multiple = SEQUENCE_MULTIPLE // get_sp_world_size()
        for batch_index, batch_hidden_states in enumerate(hidden_states):
            if image_offsets is None:
                batch_sizes = [sizes[batch_index]]
                image_hidden_states = batch_hidden_states
            else:
                batch_sizes = sizes[batch_index]
                start, end = image_offsets[batch_index]
                image_hidden_states = batch_hidden_states[start:end]

            current_offset = 0
            output = None
            for frames, height, width in batch_sizes:
                original_length = (
                    (frames // f_patch_size)
                    * (height // patch_size)
                    * (width // patch_size)
                )
                padding_length = (-original_length) % sequence_multiple
                output = (
                    image_hidden_states[
                        current_offset : current_offset + original_length
                    ]
                    .view(
                        frames // f_patch_size,
                        height // patch_size,
                        width // patch_size,
                        f_patch_size,
                        patch_size,
                        patch_size,
                        self.out_channels,
                    )
                    .permute(6, 0, 3, 1, 4, 2, 5)
                    .reshape(self.out_channels, frames, height, width)
                )
                current_offset += original_length + padding_length
            outputs.append(output)
        return outputs

    def _prepare_t2i_sequences(
        self,
        x: list[torch.Tensor],
        cap_feats: list[torch.Tensor] | None,
        glm_features: list[torch.Tensor] | None,
        patch_size: int,
        f_patch_size: int,
    ) -> tuple[
        _LLaDAImageSequence,
        _LLaDAImageSequence | None,
        _LLaDAImageSequence | None,
        list[tuple[int, int, int]],
    ]:
        image_sequence = _LLaDAImageSequence([], [], [])
        cap_sequence = (
            _LLaDAImageSequence([], [], []) if cap_feats is not None else None
        )
        glm_sequence = (
            _LLaDAImageSequence([], [], []) if glm_features is not None else None
        )
        image_sizes = []

        for batch_index, latent in enumerate(x):
            position_cursor = 1
            if cap_sequence is not None:
                padded_features, position_ids, padding_mask, sequence_length, _ = (
                    self._pad_with_ids(
                        cap_feats[batch_index],
                        (len(cap_feats[batch_index]), 1, 1),
                        (position_cursor, 0, 0),
                    )
                )
                cap_sequence.features.append(padded_features)
                cap_sequence.position_ids.append(position_ids)
                cap_sequence.padding_masks.append(padding_mask)
                position_cursor += sequence_length

            if glm_sequence is not None:
                padded_features, position_ids, padding_mask, sequence_length, _ = (
                    self._pad_with_ids(
                        glm_features[batch_index],
                        (len(glm_features[batch_index]), 1, 1),
                        (position_cursor, 0, 0),
                    )
                )
                glm_sequence.features.append(padded_features)
                glm_sequence.position_ids.append(position_ids)
                glm_sequence.padding_masks.append(padding_mask)
                position_cursor += sequence_length

            patches, image_size, token_grid_size = self._patchify_image(
                latent, patch_size, f_patch_size
            )
            image_height_start = 0
            if get_sp_world_size() > 1:
                image_height_start = get_sp_parallel_rank() * token_grid_size[1]
            padded_features, position_ids, padding_mask, _, _ = self._pad_with_ids(
                patches,
                token_grid_size,
                (position_cursor, image_height_start, 0),
                sequence_multiple=SEQUENCE_MULTIPLE // get_sp_world_size(),
            )
            image_sequence.features.append(padded_features)
            image_sequence.position_ids.append(position_ids)
            image_sequence.padding_masks.append(padding_mask)
            image_sizes.append(image_size)

        return image_sequence, cap_sequence, glm_sequence, image_sizes

    def _prepare_editing_sequences(
        self,
        x: list[torch.Tensor],
        cap_feats: list[torch.Tensor],
        glm_cap_feats: list[torch.Tensor],
        source_latents: list[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ) -> tuple[
        _LLaDAImageSequence,
        _LLaDAImageSequence,
        _LLaDAImageSequence,
        list[list[tuple[int, int, int]]],
        list[tuple[int, int]],
    ]:
        image_sequence = _LLaDAImageSequence([], [], [], [])
        cap_sequence = _LLaDAImageSequence([], [], [], [])
        sigvq_sequence = _LLaDAImageSequence([], [], [], [])
        image_sizes = []
        image_offsets = []

        for batch_index, latent in enumerate(x):
            cap_end_positions = []
            position_cursor = 1
            batch_cap_features = []
            batch_cap_positions = []
            batch_cap_padding = []
            batch_cap_noise = []
            for noise_value in (0, 1):
                padded_features, position_ids, padding_mask, _, noise_mask = (
                    self._pad_with_ids(
                        cap_feats[batch_index],
                        (len(cap_feats[batch_index]), 1, 1),
                        (position_cursor, 0, 0),
                        noise_value,
                    )
                )
                batch_cap_features.append(padded_features)
                batch_cap_positions.append(position_ids)
                batch_cap_padding.append(padding_mask)
                batch_cap_noise.extend(noise_mask)
                position_cursor += len(cap_feats[batch_index])
                cap_end_positions.append(position_cursor)
                position_cursor += 2

            batch_image_features = []
            batch_image_sizes = []
            batch_image_positions = []
            batch_image_padding = []
            batch_image_noise = []
            for image, position_start, noise_value in zip(
                (source_latents[batch_index], latent),
                cap_end_positions,
                (0, 1),
            ):
                patches, image_size, token_grid_size = self._patchify_image(
                    image, patch_size, f_patch_size
                )
                image_height_start = 0
                if get_sp_world_size() > 1:
                    image_height_start = get_sp_parallel_rank() * token_grid_size[1]
                padded_features, position_ids, padding_mask, _, noise_mask = (
                    self._pad_with_ids(
                        patches,
                        token_grid_size,
                        (position_start, image_height_start, 0),
                        noise_value,
                        sequence_multiple=SEQUENCE_MULTIPLE // get_sp_world_size(),
                    )
                )
                batch_image_features.append(padded_features)
                batch_image_sizes.append(image_size)
                batch_image_positions.append(position_ids)
                batch_image_padding.append(padding_mask)
                batch_image_noise.extend(noise_mask)

            batch_cap_features = torch.cat(batch_cap_features, dim=0)
            batch_image_features = torch.cat(batch_image_features, dim=0)
            cap_sequence.features.append(batch_cap_features)
            cap_sequence.position_ids.append(torch.cat(batch_cap_positions, dim=0))
            cap_sequence.padding_masks.append(torch.cat(batch_cap_padding, dim=0))
            cap_sequence.noise_masks.append(batch_cap_noise)
            image_sequence.features.append(batch_image_features)
            image_sequence.position_ids.append(torch.cat(batch_image_positions, dim=0))
            image_sequence.padding_masks.append(torch.cat(batch_image_padding, dim=0))
            image_sequence.noise_masks.append(batch_image_noise)
            image_sizes.append(batch_image_sizes)
            image_offsets.append(
                (
                    len(batch_cap_features),
                    len(batch_cap_features) + len(batch_image_features),
                )
            )

            padded_features, position_ids, padding_mask, _, noise_mask = (
                self._pad_with_ids(
                    glm_cap_feats[batch_index],
                    (len(glm_cap_feats[batch_index]), 1, 1),
                    (
                        len(batch_cap_features)
                        + len(batch_image_features) * get_sp_world_size()
                        + 1,
                        0,
                        0,
                    ),
                    0,
                )
            )
            sigvq_sequence.features.append(padded_features)
            sigvq_sequence.position_ids.append(position_ids)
            sigvq_sequence.padding_masks.append(padding_mask)
            sigvq_sequence.noise_masks.append(noise_mask)

        return image_sequence, cap_sequence, sigvq_sequence, image_sizes, image_offsets

    @staticmethod
    def _merge_padded_sequences(
        feature_groups: tuple[torch.Tensor, ...],
        frequency_groups: tuple[torch.Tensor, ...],
        length_groups: tuple[list[int], ...],
        noise_mask_groups: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, list[int], torch.Tensor | None
    ]:
        batch_size = feature_groups[0].shape[0]
        merged_features = []
        merged_frequencies = []
        merged_noise_masks = [] if noise_mask_groups is not None else None

        for batch_index in range(batch_size):
            device = feature_groups[0].device
            merged_features.append(
                torch.cat(
                    [
                        features[batch_index, : lengths[batch_index]].to(device)
                        for features, lengths in zip(feature_groups, length_groups)
                    ],
                    dim=0,
                )
            )
            merged_frequencies.append(
                torch.cat(
                    [
                        frequencies[batch_index, : lengths[batch_index]].to(device)
                        for frequencies, lengths in zip(frequency_groups, length_groups)
                    ],
                    dim=0,
                )
            )
            if merged_noise_masks is not None:
                merged_noise_masks.append(
                    torch.cat(
                        [
                            noise_masks[batch_index, : lengths[batch_index]].to(device)
                            for noise_masks, lengths in zip(
                                noise_mask_groups, length_groups
                            )
                        ],
                        dim=0,
                    )
                )

        merged_lengths = [len(features) for features in merged_features]
        merged_features = pad_sequence(
            merged_features, batch_first=True, padding_value=0.0
        )
        merged_frequencies = pad_sequence(
            merged_frequencies, batch_first=True, padding_value=0.0
        )

        attention_mask = None
        max_length = max(merged_lengths)
        if not all(length == max_length for length in merged_lengths):
            attention_mask = torch.zeros(
                (batch_size, max_length),
                dtype=torch.bool,
                device=merged_features.device,
            )
            for batch_index, sequence_length in enumerate(merged_lengths):
                attention_mask[batch_index, :sequence_length] = True

        noise_mask = None
        if merged_noise_masks is not None:
            noise_mask = pad_sequence(
                merged_noise_masks, batch_first=True, padding_value=0
            )[:, : merged_features.shape[1]]

        return (
            merged_features,
            merged_frequencies,
            attention_mask,
            merged_lengths,
            noise_mask,
        )

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor] | None,
        glm_cap_feats: list[torch.Tensor] | None = None,
        source_latents: list[torch.Tensor] | None = None,
        patch_size: int = 1,
        f_patch_size: int = 1,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[list[torch.Tensor]]:
        r"""
        Args:
            x (`list[torch.Tensor]`):
                Target latents. Each tensor has shape `(channels, frames, height, width)`.
            t (`torch.Tensor`):
                Denoising timestep for each batch item.
            cap_feats (`list[torch.Tensor]`, *optional*):
                Projected QueryFormer features, each with shape `(sequence_length, cap_feat_dim)`.
            glm_cap_feats (`list[torch.Tensor]`, *optional*):
                GLM/SigVQ features, each with shape `(sequence_length, semantic_feat_dim)`.
            source_latents (`list[torch.Tensor]`, *optional*):
                Source-image latents for editing. When provided, `cap_feats` and `glm_cap_feats` are required.
            patch_size (`int`, defaults to `1`):
                Spatial patch size.
            f_patch_size (`int`, defaults to `1`):
                Temporal patch size.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`].

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`:
                The denoised target latents.
        """
        patch_key = f"{patch_size}-{f_patch_size}"
        if patch_key not in self.all_x_embedder:
            raise ValueError(
                f"Unsupported patch sizes: patch_size={patch_size}, f_patch_size={f_patch_size}."
            )
        if source_latents is None and cap_feats is None and glm_cap_feats is None:
            raise ValueError(
                "Text-to-image inference requires `cap_feats` or `glm_cap_feats`."
            )
        if source_latents is not None and (cap_feats is None or glm_cap_feats is None):
            raise ValueError(
                "Editing requires `cap_feats`, `glm_cap_feats`, and `source_latents`."
            )

        batch_size = len(x)
        is_editing = source_latents is not None
        adaln_input = None
        noisy_embedding = None
        clean_embedding = None
        image_offsets = None
        num_replicated_suffix = 0

        if is_editing:
            if t.shape[0] == 1:
                t = t.repeat(batch_size)
            dual_timestep = torch.cat([t, torch.zeros_like(t)], dim=0)
            dual_embedding = self.t_embedder(
                dual_timestep.abs() * self.t_scale, x[0].dtype
            )
            noisy_embedding = dual_embedding[:batch_size]
            clean_embedding = dual_embedding[batch_size:]
            image_sequence, cap_sequence, sigvq_sequence, image_sizes, image_offsets = (
                self._prepare_editing_sequences(
                    x,
                    cap_feats,
                    glm_cap_feats,
                    source_latents,
                    patch_size,
                    f_patch_size,
                )
            )
        else:
            adaln_input = self.t_embedder(t * self.t_scale, x[0].dtype)
            glm_features = (
                [
                    self.semantic_embedder(batch_features)
                    for batch_features in glm_cap_feats
                ]
                if glm_cap_feats is not None
                else None
            )
            image_sequence, cap_sequence, glm_sequence, image_sizes = (
                self._prepare_t2i_sequences(
                    x,
                    cap_feats,
                    glm_features,
                    patch_size,
                    f_patch_size,
                )
            )

        image_lengths = [len(features) for features in image_sequence.features]
        image_features = self.all_x_embedder[patch_key](
            torch.cat(image_sequence.features, dim=0)
        )
        image_frequencies = list(
            self.rope_embedder(torch.cat(image_sequence.position_ids, dim=0)).split(
                [len(position_ids) for position_ids in image_sequence.position_ids],
                dim=0,
            )
        )
        (
            image_features,
            image_frequencies,
            image_attention_mask,
            image_lengths,
            image_noise_mask,
        ) = self._batch_sequences(
            list(image_features.split(image_lengths, dim=0)),
            image_frequencies,
            image_sequence.padding_masks,
            self.x_pad_token,
            image_sequence.noise_masks,
        )

        for layer in self.noise_refiner:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if is_editing:
                    image_features = self._gradient_checkpointing_func(
                        layer,
                        image_features,
                        image_attention_mask,
                        image_frequencies,
                        None,
                        image_noise_mask,
                        noisy_embedding,
                        clean_embedding,
                    )
                else:
                    image_features = self._gradient_checkpointing_func(
                        layer,
                        image_features,
                        image_attention_mask,
                        image_frequencies,
                        adaln_input,
                    )
            elif is_editing:
                image_features = layer(
                    image_features,
                    image_attention_mask,
                    image_frequencies,
                    noise_mask=image_noise_mask,
                    adaln_noisy=noisy_embedding,
                    adaln_clean=clean_embedding,
                )
            else:
                image_features = layer(
                    image_features,
                    image_attention_mask,
                    image_frequencies,
                    adaln_input,
                )

        if is_editing:
            cap_lengths = [len(features) for features in cap_sequence.features]
            cap_features = self.cap_embedder(torch.cat(cap_sequence.features, dim=0))
            cap_frequencies = list(
                self.rope_embedder(torch.cat(cap_sequence.position_ids, dim=0)).split(
                    [len(position_ids) for position_ids in cap_sequence.position_ids],
                    dim=0,
                )
            )
            (
                cap_features,
                cap_frequencies,
                cap_attention_mask,
                cap_lengths,
                cap_noise_mask,
            ) = self._batch_sequences(
                list(cap_features.split(cap_lengths, dim=0)),
                cap_frequencies,
                cap_sequence.padding_masks,
                self.cap_pad_token,
                cap_sequence.noise_masks,
            )

            for layer in self.context_refiner:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    cap_features = self._gradient_checkpointing_func(
                        layer,
                        cap_features,
                        cap_attention_mask,
                        cap_frequencies,
                        None,
                        None,
                        None,
                        None,
                        0,
                        True,
                    )
                else:
                    cap_features = layer(
                        cap_features,
                        cap_attention_mask,
                        cap_frequencies,
                        skip_sequence_parallel_override=True,
                    )

            sigvq_lengths = [len(features) for features in sigvq_sequence.features]
            sigvq_features = self.sigvq_embedder(
                torch.cat(sigvq_sequence.features, dim=0)
            )
            sigvq_frequencies = list(
                self.rope_embedder(torch.cat(sigvq_sequence.position_ids, dim=0)).split(
                    [len(position_ids) for position_ids in sigvq_sequence.position_ids],
                    dim=0,
                )
            )
            (
                sigvq_features,
                sigvq_frequencies,
                sigvq_attention_mask,
                sigvq_lengths,
                sigvq_noise_mask,
            ) = self._batch_sequences(
                list(sigvq_features.split(sigvq_lengths, dim=0)),
                sigvq_frequencies,
                sigvq_sequence.padding_masks,
                self.sigvq_pad_token,
                sigvq_sequence.noise_masks,
            )

            if any(sigvq_lengths):
                for layer in self.sigvq_refiner:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        sigvq_features = self._gradient_checkpointing_func(
                            layer,
                            sigvq_features,
                            sigvq_attention_mask,
                            sigvq_frequencies,
                            None,
                            None,
                            None,
                            None,
                            0,
                            True,
                        )
                    else:
                        sigvq_features = layer(
                            sigvq_features,
                            sigvq_attention_mask,
                            sigvq_frequencies,
                            skip_sequence_parallel_override=True,
                        )

            if get_sp_world_size() > 1:
                if batch_size != 1:
                    raise NotImplementedError(
                        "LLaDA-Image sequence parallelism currently supports batch size 1."
                    )
                feature_groups = (image_features, cap_features, sigvq_features)
                frequency_groups = (
                    image_frequencies,
                    cap_frequencies,
                    sigvq_frequencies,
                )
                length_groups = (image_lengths, cap_lengths, sigvq_lengths)
                noise_mask_groups = (
                    image_noise_mask,
                    cap_noise_mask,
                    sigvq_noise_mask,
                )
                num_replicated_suffix = cap_lengths[0] + sigvq_lengths[0]
                image_offsets = [(0, image_lengths[0])]
            else:
                feature_groups = (cap_features, image_features, sigvq_features)
                frequency_groups = (
                    cap_frequencies,
                    image_frequencies,
                    sigvq_frequencies,
                )
                length_groups = (cap_lengths, image_lengths, sigvq_lengths)
                noise_mask_groups = (
                    cap_noise_mask,
                    image_noise_mask,
                    sigvq_noise_mask,
                )

            (
                unified_features,
                unified_frequencies,
                unified_attention_mask,
                _,
                unified_noise_mask,
            ) = self._merge_padded_sequences(
                feature_groups,
                frequency_groups,
                length_groups,
                noise_mask_groups,
            )
        else:
            condition_feature_groups = []
            condition_frequency_groups = []
            condition_length_groups = []

            if cap_sequence is not None:
                cap_lengths = [len(features) for features in cap_sequence.features]
                cap_features = self.cap_embedder(
                    torch.cat(cap_sequence.features, dim=0)
                )
                cap_padding_mask = (
                    torch.cat(cap_sequence.padding_masks)
                    .unsqueeze(-1)
                    .to(cap_features.device)
                )
                cap_features = torch.where(
                    cap_padding_mask,
                    self.cap_pad_token.to(
                        device=cap_features.device, dtype=cap_features.dtype
                    ),
                    cap_features,
                )
                cap_features = pad_sequence(
                    list(cap_features.split(cap_lengths, dim=0)),
                    batch_first=True,
                    padding_value=0.0,
                )
                cap_frequencies = list(
                    self.rope_embedder(
                        torch.cat(cap_sequence.position_ids, dim=0)
                    ).split(
                        [
                            len(position_ids)
                            for position_ids in cap_sequence.position_ids
                        ],
                        dim=0,
                    )
                )
                cap_frequencies = pad_sequence(
                    cap_frequencies, batch_first=True, padding_value=0.0
                )
                condition_feature_groups.append(cap_features)
                condition_frequency_groups.append(cap_frequencies)
                condition_length_groups.append(cap_lengths)

            if glm_sequence is not None:
                glm_lengths = [len(features) for features in glm_sequence.features]
                glm_features = torch.cat(glm_sequence.features, dim=0)
                glm_padding_mask = (
                    torch.cat(glm_sequence.padding_masks)
                    .unsqueeze(-1)
                    .to(glm_features.device)
                )
                glm_features = torch.where(
                    glm_padding_mask,
                    self.cap_pad_token.to(
                        device=glm_features.device, dtype=glm_features.dtype
                    ),
                    glm_features,
                )
                glm_features = pad_sequence(
                    list(glm_features.split(glm_lengths, dim=0)),
                    batch_first=True,
                    padding_value=0.0,
                )
                glm_frequencies = list(
                    self.rope_embedder(
                        torch.cat(glm_sequence.position_ids, dim=0)
                    ).split(
                        [
                            len(position_ids)
                            for position_ids in glm_sequence.position_ids
                        ],
                        dim=0,
                    )
                )
                glm_frequencies = pad_sequence(
                    glm_frequencies, batch_first=True, padding_value=0.0
                )
                condition_feature_groups.append(glm_features)
                condition_frequency_groups.append(glm_frequencies)
                condition_length_groups.append(glm_lengths)

            (
                condition_features,
                condition_frequencies,
                condition_attention_mask,
                condition_lengths,
                _,
            ) = self._merge_padded_sequences(
                tuple(condition_feature_groups),
                tuple(condition_frequency_groups),
                tuple(condition_length_groups),
            )

            for layer in self.context_refiner:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    condition_features = self._gradient_checkpointing_func(
                        layer,
                        condition_features,
                        condition_attention_mask,
                        condition_frequencies,
                        None,
                        None,
                        None,
                        None,
                        0,
                        True,
                    )
                else:
                    condition_features = layer(
                        condition_features,
                        condition_attention_mask,
                        condition_frequencies,
                        skip_sequence_parallel_override=True,
                    )

            (
                unified_features,
                unified_frequencies,
                unified_attention_mask,
                _,
                unified_noise_mask,
            ) = self._merge_padded_sequences(
                (image_features, condition_features),
                (image_frequencies, condition_frequencies),
                (image_lengths, condition_lengths),
            )
            if get_sp_world_size() > 1:
                if batch_size != 1:
                    raise NotImplementedError(
                        "LLaDA-Image sequence parallelism currently supports batch size 1."
                    )
                num_replicated_suffix = condition_lengths[0]

        for layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if is_editing:
                    unified_features = self._gradient_checkpointing_func(
                        layer,
                        unified_features,
                        unified_attention_mask,
                        unified_frequencies,
                        None,
                        unified_noise_mask,
                        noisy_embedding,
                        clean_embedding,
                        num_replicated_suffix,
                    )
                else:
                    unified_features = self._gradient_checkpointing_func(
                        layer,
                        unified_features,
                        unified_attention_mask,
                        unified_frequencies,
                        adaln_input,
                        None,
                        None,
                        None,
                        num_replicated_suffix,
                    )
            elif is_editing:
                unified_features = layer(
                    unified_features,
                    unified_attention_mask,
                    unified_frequencies,
                    noise_mask=unified_noise_mask,
                    adaln_noisy=noisy_embedding,
                    adaln_clean=clean_embedding,
                    num_replicated_suffix=num_replicated_suffix,
                )
            else:
                unified_features = layer(
                    unified_features,
                    unified_attention_mask,
                    unified_frequencies,
                    adaln_input,
                    num_replicated_suffix=num_replicated_suffix,
                )

        if is_editing:
            unified_features = self.all_final_layer[patch_key](
                unified_features,
                noise_mask=unified_noise_mask,
                adaln_noisy=noisy_embedding,
                adaln_clean=clean_embedding,
            )
        else:
            unified_features = self.all_final_layer[patch_key](
                unified_features,
                adaln_input=adaln_input,
            )

        output = self._unpatchify(
            list(unified_features.unbind(dim=0)),
            image_sizes,
            patch_size,
            f_patch_size,
            image_offsets,
        )
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


@dataclass
class LLaDAImageQueryFormerOutput(BaseOutput):
    query_embeds: torch.Tensor


class LLaDAImageQueryAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "LLaDAImageQueryAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = F.linear(
            hidden_states,
            attn.in_proj_weight[: attn.inner_dim],
            attn.in_proj_bias[: attn.inner_dim],
        )
        key = F.linear(
            encoder_hidden_states,
            attn.in_proj_weight[attn.inner_dim : 2 * attn.inner_dim],
            attn.in_proj_bias[attn.inner_dim : 2 * attn.inner_dim],
        )
        value = F.linear(
            encoder_hidden_states,
            attn.in_proj_weight[2 * attn.inner_dim :],
            attn.in_proj_bias[2 * attn.inner_dim :],
        )

        query = query.unflatten(-1, (attn.heads, attn.head_dim))
        key = key.unflatten(-1, (attn.heads, attn.head_dim))
        value = value.unflatten(-1, (attn.heads, attn.head_dim))

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=attn.dropout if attn.training else 0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.out_proj(hidden_states)


class LLaDAImageQueryAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = LLaDAImageQueryAttnProcessor
    _available_processors: ClassVar[list[type]] = [LLaDAImageQueryAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.inner_dim = hidden_size
        self.heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.in_proj_weight = nn.Parameter(torch.zeros(3 * hidden_size, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.set_processor(self._default_processor_cls())

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.processor(
            self, hidden_states, encoder_hidden_states, attention_mask
        )


@maybe_allow_in_graph
class LLaDAImageQueryFormerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float,
        norm_eps: float,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.norm_k = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.cross_attn = LLaDAImageQueryAttention(hidden_size, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.mlp.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(
        self,
        query_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_embeds = self.norm_q(query_embeds)
        encoder_hidden_states = self.norm_k(encoder_hidden_states)
        attention_output = self.cross_attn(
            query_embeds, encoder_hidden_states, attention_mask
        )
        query_embeds = query_embeds + self.dropout(attention_output)
        query_embeds = self.norm1(query_embeds)
        mlp_output = self.mlp.fc2(
            F.gelu(self.mlp.fc1(query_embeds), approximate="tanh")
        )
        return query_embeds + self.dropout(mlp_output)


class LLaDAImageQueryFormerModel(ModelMixin, ConfigMixin, AttentionMixin):
    r"""
    QueryFormer used by LLaDA-Image to derive learnable image-generation queries from LLaDA token embeddings.

    This model is independent from the LLaDA text encoder. It returns refined query embeddings. The pipeline appends
    them to the text embeddings and invokes the text encoder backbone.

    Args:
        num_queries (`int`, defaults to `256`):
            Number of learnable query tokens.
        hidden_size (`int`, defaults to `2048`):
            Query and LLaDA token embedding dimension.
        num_hidden_layers (`int`, defaults to `1`):
            Number of QueryFormer blocks.
        num_attention_heads (`int`, defaults to `16`):
            Number of cross-attention heads.
        intermediate_size (`int`, defaults to `8192`):
            Hidden dimension of the QueryFormer MLP.
        dropout (`float`, defaults to `0.0`):
            Dropout probability.
        norm_eps (`float`, defaults to `1e-6`):
            Epsilon used by parameter-free layer normalization.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["LLaDAImageQueryFormerBlock"]
    _repeated_blocks: ClassVar[list[str]] = ["LLaDAImageQueryFormerBlock"]
    _skip_layerwise_casting_patterns: ClassVar[list[str]] = ["norm"]

    @register_to_config
    def __init__(
        self,
        num_queries: int = 256,
        hidden_size: int = 2048,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 16,
        intermediate_size: int = 8192,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({hidden_size}) must be divisible by `num_attention_heads` ({num_attention_heads})."
            )

        self.meta_queries = nn.Parameter(torch.zeros(num_queries, hidden_size))
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(hidden_size))
        self.query_blocks = nn.ModuleList(
            [
                LLaDAImageQueryFormerBlock(
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    dropout,
                    norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
    ) -> LLaDAImageQueryFormerOutput | tuple[torch.Tensor]:
        r"""
        Args:
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                LLaDA input token embeddings.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Mask whose nonzero entries identify valid text tokens.
            return_dict (`bool`, defaults to `True`):
                Whether to return [`LLaDAImageQueryFormerOutput`] instead of a tuple.

        Returns:
            [`LLaDAImageQueryFormerOutput`] or `tuple`:
                The refined query embeddings.
        """
        batch_size = inputs_embeds.shape[0]
        query_embeds = self.meta_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = attention_mask.bool()

        for query_block in self.query_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                query_embeds = self._gradient_checkpointing_func(
                    query_block,
                    query_embeds,
                    inputs_embeds,
                    attention_mask,
                )
            else:
                query_embeds = query_block(query_embeds, inputs_embeds, attention_mask)

        if not return_dict:
            return (query_embeds,)
        return LLaDAImageQueryFormerOutput(query_embeds=query_embeds)


@dataclass
class LLaDAImageTextProjectionOutput(BaseOutput):
    hidden_states: torch.Tensor


class LLaDAImageTextProjectionAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "LLaDAImageTextProjectionAttention",
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        query = attn.q_proj(hidden_states).unflatten(-1, (attn.heads, attn.head_dim))
        key = attn.k_proj(hidden_states).unflatten(-1, (attn.heads, attn.head_dim))
        value = attn.v_proj(hidden_states).unflatten(-1, (attn.heads, attn.head_dim))

        query = attn.q_norm(query)
        key = attn.k_norm(key)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=attn.dropout if attn.training else 0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.out_proj(hidden_states)


class LLaDAImageTextProjectionAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = LLaDAImageTextProjectionAttnProcessor
    _available_processors: ClassVar[list[type]] = [
        LLaDAImageTextProjectionAttnProcessor
    ]
    _supports_qkv_fusion = False

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        norm_eps: float,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout = attention_dropout

        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=False)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=False)
        self.set_processor(self._default_processor_cls())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)


class LLaDAImageTextProjectionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        return self.fc2(hidden_states)


@maybe_allow_in_graph
class LLaDAImageTextProjectionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        norm_eps: float,
    ):
        super().__init__()
        self.self_attn = LLaDAImageTextProjectionAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout,
            norm_eps,
        )
        self.layer_norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.mlp = LLaDAImageTextProjectionMLP(hidden_size, intermediate_size)
        self.layer_norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class LLaDAImageTextProjectionModel(ModelMixin, ConfigMixin, AttentionMixin):
    r"""
    Connector and output projection used to map LLaDA hidden states to the LLaDA-Image denoiser context dimension.

    Args:
        hidden_size (`int`, defaults to `2048`):
            Input and connector hidden dimension.
        intermediate_size (`int`, defaults to `8960`):
            Connector MLP hidden dimension.
        num_hidden_layers (`int`, defaults to `6`):
            Number of connector layers.
        num_attention_heads (`int`, defaults to `32`):
            Number of connector self-attention heads.
        projection_dim (`int`, defaults to `2560`):
            Output dimension expected by the denoising transformer.
        attention_dropout (`float`, defaults to `0.0`):
            Attention dropout probability.
        norm_eps (`float`, defaults to `1e-6`):
            Epsilon used by parameter-free RMS normalization.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["LLaDAImageTextProjectionBlock"]
    _repeated_blocks: ClassVar[list[str]] = ["LLaDAImageTextProjectionBlock"]
    _skip_layerwise_casting_patterns: ClassVar[list[str]] = [
        "layer_norm",
        "q_norm",
        "k_norm",
    ]

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 8960,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 32,
        projection_dim: int = 2560,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({hidden_size}) must be divisible by `num_attention_heads` ({num_attention_heads})."
            )

        self.layers = nn.ModuleList(
            [
                LLaDAImageTextProjectionBlock(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    attention_dropout,
                    norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.projector = nn.Linear(hidden_size, projection_dim, bias=True)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> LLaDAImageTextProjectionOutput | tuple[torch.Tensor]:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Hidden states produced by the LLaDA text backbone.
            return_dict (`bool`, defaults to `True`):
                Whether to return [`LLaDAImageTextProjectionOutput`] instead of a tuple.

        Returns:
            [`LLaDAImageTextProjectionOutput`] or `tuple`:
                Hidden states projected to the denoiser caption dimension.
        """
        for layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(layer, hidden_states)
            else:
                hidden_states = layer(hidden_states)

        hidden_states = self.projector(hidden_states)
        if not return_dict:
            return (hidden_states,)
        return LLaDAImageTextProjectionOutput(hidden_states=hidden_states)


@dataclass
class LLaDAImageSigVQOutput(BaseOutput):
    semantic_features: torch.Tensor
    token_ids: torch.Tensor


class LLaDAImageSigVQAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self, attn: "LLaDAImageSigVQAttention", hidden_states: torch.Tensor
    ) -> torch.Tensor:
        query, key, value = attn.qkv(hidden_states).chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, attn.head_dim))
        key = key.unflatten(-1, (attn.heads, attn.head_dim))
        value = value.unflatten(-1, (attn.heads, attn.head_dim))

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=attn.dropout if attn.training else 0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.proj(hidden_states)


class LLaDAImageSigVQAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = LLaDAImageSigVQAttnProcessor
    _available_processors: ClassVar[list[type]] = [LLaDAImageSigVQAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_bias: bool,
        attention_dropout: float,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout = attention_dropout
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=attention_bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.set_processor(self._default_processor_cls())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)


class LLaDAImageSigVQMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))


@maybe_allow_in_graph
class LLaDAImageSigVQVisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_bias: bool,
        attention_dropout: float,
        norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = LLaDAImageSigVQAttention(
            hidden_size,
            num_attention_heads,
            attention_bias,
            attention_dropout,
        )
        self.mlp = LLaDAImageSigVQMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class LLaDAImageSigVQPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = pixel_values.shape
        grid_height = height // self.patch_size
        grid_width = width // self.patch_size
        patches = pixel_values.reshape(
            batch_size,
            channels,
            grid_height,
            self.patch_size,
            grid_width,
            self.patch_size,
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            batch_size * grid_height * grid_width,
            channels,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(patches).flatten(1)
        return hidden_states.reshape(batch_size, grid_height * grid_width, -1)


class LLaDAImageSigVQEmbeddings(nn.Module):
    def __init__(self, image_size: int, patch_size: int, hidden_size: int):
        super().__init__()
        num_positions = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(num_positions, hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, grid_height: int, grid_width: int
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        position_embedding = self.position_embedding.weight
        hidden_size = position_embedding.shape[1]
        original_size = int(position_embedding.shape[0] ** 0.5)
        position_embedding = position_embedding.reshape(
            original_size, original_size, hidden_size
        )
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).float()

        height_coordinates = torch.arange(
            grid_height, device=hidden_states.device, dtype=torch.float32
        )
        width_coordinates = torch.arange(
            grid_width, device=hidden_states.device, dtype=torch.float32
        )
        height_coordinates, width_coordinates = torch.meshgrid(
            height_coordinates,
            width_coordinates,
            indexing="ij",
        )
        normalized_width = ((width_coordinates.flatten() + 0.5) / grid_width) * 2 - 1
        normalized_height = ((height_coordinates.flatten() + 0.5) / grid_height) * 2 - 1
        grid = torch.stack((normalized_width, normalized_height), dim=-1)
        grid = grid.reshape(1, grid_height * grid_width, 1, 2).expand(
            batch_size, -1, -1, -1
        )

        position_embedding = F.grid_sample(
            position_embedding.expand(batch_size, -1, -1, -1),
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        position_embedding = (
            position_embedding.squeeze(-1).transpose(1, 2).to(hidden_states.dtype)
        )
        return hidden_states + position_embedding


class LLaDAImageSigVQQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        hidden_states = F.normalize(
            hidden_states.reshape(-1, hidden_states.shape[-1]), p=2, dim=-1
        )
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        distances = (
            torch.sum(hidden_states**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.matmul(hidden_states, embedding.t())
        )
        return torch.argmin(distances, dim=1)


class LLaDAImageSigVQModel(ModelMixin, ConfigMixin, AttentionMixin):
    r"""
    Minimal GLM SigVQ image encoder used by LLaDA-Image editing.

    The model contains only the GLM vision encoder, VQ quantizer, and prior token projection used during inference.
    Input images must already be RGB tensors normalized to `[-1, 1]`, have one common size, and be divisible by
    `patch_size`.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["LLaDAImageSigVQVisionBlock"]
    _repeated_blocks: ClassVar[list[str]] = ["LLaDAImageSigVQVisionBlock"]
    _skip_layerwise_casting_patterns: ClassVar[list[str]] = [
        "patch_embed",
        "position_embedding",
        "norm",
        "quantize",
    ]

    @register_to_config
    def __init__(
        self,
        image_size: int = 2048,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1536,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 16,
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-6,
        codebook_size: int = 16384,
        codebook_embed_dim: int = 2048,
        semantic_embed_dim: int = 4096,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({hidden_size}) must be divisible by `num_attention_heads` ({num_attention_heads})."
            )

        self.visual = nn.Module()
        self.visual.patch_embed = LLaDAImageSigVQPatchEmbed(
            in_channels, hidden_size, patch_size
        )
        self.visual.embeddings = LLaDAImageSigVQEmbeddings(
            image_size, patch_size, hidden_size
        )
        self.visual.blocks = nn.ModuleList(
            [
                LLaDAImageSigVQVisionBlock(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    attention_bias,
                    attention_dropout,
                    norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.vqmodel = nn.Module()
        self.vqmodel.quant_conv = nn.Conv2d(
            hidden_size, codebook_embed_dim, kernel_size=1
        )
        self.vqmodel.quantize = LLaDAImageSigVQQuantizer(
            codebook_size, codebook_embed_dim
        )

        self.prior_token_embedding = nn.Embedding(codebook_size, semantic_embed_dim)
        self.prior_projector = FeedForward(
            semantic_embed_dim,
            semantic_embed_dim,
            inner_dim=semantic_embed_dim,
            activation_fn="linear-silu",
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> LLaDAImageSigVQOutput | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            pixel_values (`torch.Tensor` of shape `(batch_size, 3, height, width)`, *optional*):
                RGB images normalized to `[-1, 1]`. Mutually exclusive with `token_ids`.
            token_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Precomputed VQ codebook IDs. Mutually exclusive with `pixel_values`.
            return_dict (`bool`, defaults to `True`):
                Whether to return [`LLaDAImageSigVQOutput`] instead of a tuple.

        Returns:
            [`LLaDAImageSigVQOutput`] or `tuple`:
                The projected semantic features and their discrete token IDs.
        """
        if (pixel_values is None) == (token_ids is None):
            raise ValueError("Provide exactly one of `pixel_values` or `token_ids`.")

        if pixel_values is not None:
            if pixel_values.ndim != 4:
                raise ValueError(
                    f"`pixel_values` must have 4 dimensions, got shape {tuple(pixel_values.shape)}."
                )
            height, width = pixel_values.shape[-2:]
            if (
                height % self.config.patch_size != 0
                or width % self.config.patch_size != 0
            ):
                raise ValueError(
                    f"Image height and width must be divisible by {self.config.patch_size}, got {height}x{width}."
                )

            grid_height = height // self.config.patch_size
            grid_width = width // self.config.patch_size
            hidden_states = self.visual.patch_embed(pixel_values)
            hidden_states = self.visual.embeddings(
                hidden_states, grid_height, grid_width
            )

            for block in self.visual.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        block, hidden_states
                    )
                else:
                    hidden_states = block(hidden_states)

            hidden_states = hidden_states.transpose(1, 2).reshape(
                pixel_values.shape[0],
                self.config.hidden_size,
                grid_height,
                grid_width,
            )
            hidden_states = self.vqmodel.quant_conv(hidden_states)
            token_ids = self.vqmodel.quantize(hidden_states).reshape(
                pixel_values.shape[0], -1
            )
        elif token_ids.ndim != 2:
            raise ValueError(
                f"`token_ids` must have 2 dimensions, got shape {tuple(token_ids.shape)}."
            )

        semantic_features = self.prior_projector(self.prior_token_embedding(token_ids))

        if not return_dict:
            return semantic_features, token_ids
        return LLaDAImageSigVQOutput(
            semantic_features=semantic_features, token_ids=token_ids
        )


class LLaDAImageTransformer2DModel(_LLaDAImageTransformer2DModel):
    """SGLang diffusion adapter that preserves the converted checkpoint layout."""

    _fsdp_shard_conditions: ClassVar[list] = [
        lambda name, module: isinstance(module, LLaDAImageTransformerBlock)
    ]
    _compile_conditions: ClassVar[list] = []
    param_names_mapping: ClassVar[dict] = (
        LLaDAImageDitConfig().arch_config.param_names_mapping
    )
    reverse_param_names_mapping: ClassVar[dict] = {}

    def __init__(self, config, hf_config: dict, quant_config=None):
        init_kwargs = {
            key: value for key, value in hf_config.items() if not key.startswith("_")
        }
        super().__init__(quant_config=quant_config, **init_kwargs)
        self.sgl_config = config
        self.hidden_size = int(self.config.dim)
        self.num_attention_heads = int(self.config.n_heads)
        self.num_channels_latents = int(self.config.in_channels)

    def post_load_weights(self) -> None:
        """Run model-specific post-load fixups (none are required)."""
        return

    @staticmethod
    def _as_feature_list(
        features: torch.Tensor | list[torch.Tensor],
        attention_mask: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        if isinstance(features, list):
            return features
        if features.ndim == 2:
            return [features]
        if features.ndim != 3:
            raise ValueError(
                "LLaDA-Image condition features must have two or three dimensions"
            )
        if attention_mask is None:
            return list(features.unbind(dim=0))
        if isinstance(attention_mask, list):
            attention_mask = torch.stack(attention_mask)
        return [sample[mask.bool()] for sample, mask in zip(features, attention_mask)]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        encoder_attention_mask: torch.Tensor | list[torch.Tensor] | None = None,
        source_latents: list[torch.Tensor] | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        del guidance, kwargs
        if hidden_states.ndim != 4:
            raise ValueError(
                "LLaDA-Image latents must have shape [batch, channels, height, width]"
            )

        model_dtype = next(self.parameters()).dtype
        latent_list = [latent.unsqueeze(1).to(model_dtype) for latent in hidden_states]
        cap_feats = self._as_feature_list(encoder_hidden_states, encoder_attention_mask)
        glm_cap_feats = (
            None
            if encoder_hidden_states_image is None
            or (
                isinstance(encoder_hidden_states_image, list)
                and not encoder_hidden_states_image
            )
            else self._as_feature_list(encoder_hidden_states_image)
        )
        output = (
            super()
            .forward(
                x=latent_list,
                t=(timestep / 1000.0).to(model_dtype),
                cap_feats=cap_feats,
                glm_cap_feats=glm_cap_feats,
                source_latents=source_latents,
            )
            .sample
        )
        return -torch.stack(output, dim=0).squeeze(2).float()


EntryClass = [
    LLaDAImageTransformer2DModel,
    LLaDAImageQueryFormerModel,
    LLaDAImageTextProjectionModel,
    LLaDAImageSigVQModel,
]
