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
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from sglang.multimodal_gen.configs.models.dits.llada_image import (
    SEQUENCE_MULTIPLE,
    LLaDAImageDitConfig,
    editing_rope_rows,
)
from sglang.multimodal_gen.runtime.distributed import (
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


LLADA_IMAGE_ATTENTION_BACKENDS = {
    AttentionBackendEnum.FA,
    AttentionBackendEnum.TORCH_SDPA,
}


class LLaDAImageRMSNorm(SGLangRMSNorm):
    """RMSNorm with a synthesized unit weight for affine-free checkpoints."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__(hidden_size, eps=eps, cast_x_before_out_mul=True)
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


class LLaDAImageAttention(nn.Module):
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
                )
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.to_qkv(hidden_states)
        query, key, value = (
            tensor.unflatten(-1, (self.local_heads, self.head_dim)).contiguous()
            for tensor in qkv.split([self.inner_dim] * 3, dim=-1)
        )
        cos_sin_cache = torch.cat(
            [freqs_cis.real.float(), freqs_cis.imag.float()], dim=-1
        ).reshape(-1, self.head_dim)
        positions = torch.arange(
            cos_sin_cache.shape[0], device=query.device, dtype=torch.long
        )
        if self.norm_q is not None:
            query, key = apply_qk_norm_with_optional_rope(
                q=query,
                k=key,
                q_norm=self.norm_q,
                k_norm=self.norm_k,
                head_dim=self.head_dim,
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
                head_size=self.head_dim,
                positions=positions,
                is_neox=False,
            )
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
        hidden_states = self.sgl_attention(query, key, value, attn_mask=attention_mask)
        hidden_states, _ = self.to_out[0](hidden_states.flatten(2, 3))
        return hidden_states


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


def _padding_attention_mask(
    lengths: list[int], device: torch.device
) -> torch.Tensor | None:
    max_length = max(lengths)
    if all(length == max_length for length in lengths):
        return None
    attention_mask = torch.zeros(
        (len(lengths), max_length), dtype=torch.bool, device=device
    )
    for batch_index, length in enumerate(lengths):
        attention_mask[batch_index, :length] = True
    return attention_mask


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
    ) -> torch.Tensor:
        if self.modulation:
            if noise_mask is None:
                modulation = (
                    self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
                )
            else:
                modulation = [
                    _select_per_token(noisy, clean, noise_mask, hidden_states.shape[1])
                    for noisy, clean in zip(
                        self.adaLN_modulation(adaln_noisy).chunk(4, dim=1),
                        self.adaLN_modulation(adaln_clean).chunk(4, dim=1),
                    )
                ]
            scale_msa, gate_msa, scale_mlp, gate_mlp = modulation

            attention_output = self.attention(
                self.attention_norm1(hidden_states) * (1.0 + scale_msa),
                attention_mask,
                freqs_cis,
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
            scale = self.adaLN_modulation(adaln_input).unsqueeze(1)
        else:
            scale = _select_per_token(
                self.adaLN_modulation(adaln_noisy),
                self.adaLN_modulation(adaln_clean),
                noise_mask,
                hidden_states.shape[1],
            )
        hidden_states = self.norm_final(hidden_states) * (1.0 + scale)
        return self.linear(hidden_states)


class _LLaDAImageTransformer2DModel(ModelMixin, ConfigMixin):
    """Denoiser for text-to-image generation and single-image editing."""

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
        self.out_channels = in_channels
        self.t_scale = t_scale

        self.all_x_embedder = nn.ModuleDict()
        self.all_final_layer = nn.ModuleDict()
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            patch_key = f"{patch_size}-{f_patch_size}"
            patch_dim = f_patch_size * patch_size * patch_size * in_channels
            self.all_x_embedder[patch_key] = nn.Linear(patch_dim, dim, bias=True)
            self.all_final_layer[patch_key] = LLaDAImageFinalLayer(dim, patch_dim)

        def blocks(name: str, count: int, modulation: bool) -> nn.ModuleList:
            return nn.ModuleList(
                LLaDAImageTransformerBlock(
                    dim,
                    n_heads,
                    norm_eps,
                    qk_norm,
                    modulation=modulation,
                    quant_config=quant_config,
                    prefix=f"{name}.{layer_id}",
                )
                for layer_id in range(count)
            )

        def feature_embedder(feature_dim: int) -> nn.Sequential:
            return nn.Sequential(
                RMSNorm(feature_dim, eps=norm_eps, elementwise_affine=False),
                nn.Linear(feature_dim, dim, bias=True),
            )

        self.noise_refiner = blocks("noise_refiner", n_refiner_layers, True)
        self.context_refiner = blocks("context_refiner", n_refiner_layers, False)
        self.sigvq_refiner = blocks("sigvq_refiner", n_refiner_layers, False)
        self.layers = blocks("layers", n_layers, True)

        self.t_embedder = LLaDAImageTimestepEmbedder(min(dim, ADALN_EMBED_DIM))
        self.cap_embedder = feature_embedder(cap_feat_dim)
        # Checkpoint weights that neither generation nor editing reads.
        self.semantic_embedder = feature_embedder(semantic_feat_dim)
        self.sigvq_embedder = feature_embedder(semantic_feat_dim)

        self.x_pad_token = nn.Parameter(torch.zeros(1, dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, dim))
        self.sigvq_pad_token = nn.Parameter(torch.zeros(1, dim))

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list[int] | None]:
        original_length = len(features)
        padding_length = (-original_length) % SEQUENCE_MULTIPLE
        padded_length = original_length + padding_length
        position_ids = self._create_coordinate_grid(
            position_grid_size, position_start, features.device
        ).flatten(0, 2)
        padding_mask = torch.zeros(
            padded_length, dtype=torch.bool, device=features.device
        )
        if padding_length > 0:
            # Padding rows sit at position zero and repeat the last feature.
            position_ids = torch.cat(
                [position_ids, position_ids.new_zeros((padding_length, 3))], dim=0
            )
            features = torch.cat(
                [features, features[-1:].repeat(padding_length, 1)], dim=0
            )
            padding_mask[original_length:] = True

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
        attention_mask = _padding_attention_mask(sequence_lengths, features.device)

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
                padding_length = (-original_length) % SEQUENCE_MULTIPLE
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
        cap_feats: list[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ) -> tuple[_LLaDAImageSequence, _LLaDAImageSequence, list[tuple[int, int, int]]]:
        image_sequence = _LLaDAImageSequence([], [], [])
        cap_sequence = _LLaDAImageSequence([], [], [])
        image_sizes = []

        for latent, batch_cap_feats in zip(x, cap_feats):
            padded_features, position_ids, padding_mask, cap_length, _ = (
                self._pad_with_ids(
                    batch_cap_feats, (len(batch_cap_feats), 1, 1), (1, 0, 0)
                )
            )
            cap_sequence.features.append(padded_features)
            cap_sequence.position_ids.append(position_ids)
            cap_sequence.padding_masks.append(padding_mask)

            patches, image_size, token_grid_size = self._patchify_image(
                latent, patch_size, f_patch_size
            )
            padded_features, position_ids, padding_mask, _, _ = self._pad_with_ids(
                patches, token_grid_size, (1 + cap_length, 0, 0)
            )
            image_sequence.features.append(padded_features)
            image_sequence.position_ids.append(position_ids)
            image_sequence.padding_masks.append(padding_mask)
            image_sizes.append(image_size)

        return image_sequence, cap_sequence, image_sizes

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
            image_token_counts = []
            for image, position_start, noise_value in zip(
                (source_latents[batch_index], latent),
                cap_end_positions,
                (0, 1),
            ):
                patches, image_size, token_grid_size = self._patchify_image(
                    image, patch_size, f_patch_size
                )
                image_token_counts.append(len(patches))
                padded_features, position_ids, padding_mask, _, noise_mask = (
                    self._pad_with_ids(
                        patches, token_grid_size, (position_start, 0, 0), noise_value
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

            rope_rows = editing_rope_rows(
                len(cap_feats[batch_index]),
                image_token_counts,
                len(glm_cap_feats[batch_index]),
            )
            if rope_rows > self.rope_embedder.axes_lens[0]:
                # An out-of-range RoPE gather would poison the CUDA context.
                raise ValueError(
                    f"LLaDA-Image editing needs {rope_rows} sequence positions "
                    "for this image size and prompt, above the model limit of "
                    f"{self.rope_embedder.axes_lens[0]}. Use a smaller image "
                    "size or a shorter prompt."
                )
            padded_features, position_ids, padding_mask, _, noise_mask = (
                self._pad_with_ids(
                    glm_cap_feats[batch_index],
                    (len(glm_cap_feats[batch_index]), 1, 1),
                    (
                        len(batch_cap_features) + len(batch_image_features) + 1,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        def merge(groups: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
            return [
                torch.cat(
                    [
                        group[batch_index, : lengths[batch_index]]
                        for group, lengths in zip(groups, length_groups)
                    ],
                    dim=0,
                )
                for batch_index in range(feature_groups[0].shape[0])
            ]

        merged_features = merge(feature_groups)
        attention_mask = _padding_attention_mask(
            [len(features) for features in merged_features],
            feature_groups[0].device,
        )
        noise_mask = None
        if noise_mask_groups is not None:
            noise_mask = pad_sequence(
                merge(noise_mask_groups), batch_first=True, padding_value=0
            )
        return (
            pad_sequence(merged_features, batch_first=True, padding_value=0.0),
            pad_sequence(merge(frequency_groups), batch_first=True, padding_value=0.0),
            attention_mask,
            noise_mask,
        )

    def _embed_sequence(
        self,
        sequence: _LLaDAImageSequence,
        embedder: nn.Module,
        pad_token: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, list[int], torch.Tensor | None
    ]:
        lengths = [len(features) for features in sequence.features]
        features = embedder(torch.cat(sequence.features, dim=0))
        frequencies = self.rope_embedder(torch.cat(sequence.position_ids, dim=0))
        return self._batch_sequences(
            list(features.split(lengths, dim=0)),
            list(frequencies.split(lengths, dim=0)),
            sequence.padding_masks,
            pad_token,
            sequence.noise_masks,
        )

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        glm_cap_feats: list[torch.Tensor] | None = None,
        source_latents: list[torch.Tensor] | None = None,
        patch_size: int = 1,
        f_patch_size: int = 1,
    ) -> list[torch.Tensor]:
        patch_key = f"{patch_size}-{f_patch_size}"
        batch_size = len(x)
        is_editing = source_latents is not None
        image_offsets = None
        if is_editing:
            if t.shape[0] == 1:
                t = t.repeat(batch_size)
            dual_timestep = torch.cat([t, torch.zeros_like(t)], dim=0)
            dual_embedding = self.t_embedder(
                dual_timestep.abs() * self.t_scale, x[0].dtype
            )
            adaln = {
                "adaln_noisy": dual_embedding[:batch_size],
                "adaln_clean": dual_embedding[batch_size:],
            }
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
            adaln = {"adaln_input": self.t_embedder(t * self.t_scale, x[0].dtype)}
            image_sequence, cap_sequence, image_sizes = self._prepare_t2i_sequences(
                x, cap_feats, patch_size, f_patch_size
            )

        image_features, image_frequencies, image_mask, image_lengths, image_noise = (
            self._embed_sequence(
                image_sequence, self.all_x_embedder[patch_key], self.x_pad_token
            )
        )
        for layer in self.noise_refiner:
            image_features = layer(
                image_features,
                image_mask,
                image_frequencies,
                noise_mask=image_noise,
                **adaln,
            )

        cap_features, cap_frequencies, cap_mask, cap_lengths, cap_noise = (
            self._embed_sequence(cap_sequence, self.cap_embedder, self.cap_pad_token)
        )
        for layer in self.context_refiner:
            cap_features = layer(cap_features, cap_mask, cap_frequencies)

        if is_editing:
            (
                sigvq_features,
                sigvq_frequencies,
                sigvq_mask,
                sigvq_lengths,
                sigvq_noise,
            ) = self._embed_sequence(
                sigvq_sequence, self.sigvq_embedder, self.sigvq_pad_token
            )
            if any(sigvq_lengths):
                for layer in self.sigvq_refiner:
                    sigvq_features = layer(
                        sigvq_features, sigvq_mask, sigvq_frequencies
                    )
            unified_features, unified_frequencies, unified_mask, unified_noise = (
                self._merge_padded_sequences(
                    (cap_features, image_features, sigvq_features),
                    (cap_frequencies, image_frequencies, sigvq_frequencies),
                    (cap_lengths, image_lengths, sigvq_lengths),
                    (cap_noise, image_noise, sigvq_noise),
                )
            )
        else:
            unified_features, unified_frequencies, unified_mask, unified_noise = (
                self._merge_padded_sequences(
                    (image_features, cap_features),
                    (image_frequencies, cap_frequencies),
                    (image_lengths, cap_lengths),
                )
            )

        for layer in self.layers:
            unified_features = layer(
                unified_features,
                unified_mask,
                unified_frequencies,
                noise_mask=unified_noise,
                **adaln,
            )
        unified_features = self.all_final_layer[patch_key](
            unified_features, noise_mask=unified_noise, **adaln
        )
        return self._unpatchify(
            list(unified_features.unbind(dim=0)),
            image_sizes,
            patch_size,
            f_patch_size,
            image_offsets,
        )


class LLaDAImageQueryAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.in_proj_weight = nn.Parameter(torch.zeros(3 * hidden_size, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.in_proj_weight.chunk(3)
        biases = self.in_proj_bias.chunk(3)
        query, key, value = (
            F.linear(inputs, weight, bias).unflatten(-1, (self.heads, self.head_dim))
            for inputs, weight, bias in zip(
                (hidden_states, encoder_hidden_states, encoder_hidden_states),
                weights,
                biases,
            )
        )
        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask[:, None, None, :]
        )
        return self.out_proj(hidden_states.flatten(2, 3))


class LLaDAImageQueryFormerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.norm_k = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.cross_attn = LLaDAImageQueryAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.mlp.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(
        self,
        query_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_embeds = self.norm_q(query_embeds)
        encoder_hidden_states = self.norm_k(encoder_hidden_states)
        query_embeds = query_embeds + self.cross_attn(
            query_embeds, encoder_hidden_states, attention_mask
        )
        query_embeds = self.norm1(query_embeds)
        return query_embeds + self.mlp.fc2(
            F.gelu(self.mlp.fc1(query_embeds), approximate="tanh")
        )


class LLaDAImageQueryFormerModel(ModelMixin, ConfigMixin):
    """Derives the image-generation query tokens from LLaDA token embeddings."""

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
        self.meta_queries = nn.Parameter(torch.zeros(num_queries, hidden_size))
        self.query_blocks = nn.ModuleList(
            LLaDAImageQueryFormerBlock(
                hidden_size, num_attention_heads, intermediate_size, norm_eps
            )
            for _ in range(num_hidden_layers)
        )

    def forward(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        query_embeds = self.meta_queries.unsqueeze(0).expand(
            inputs_embeds.shape[0], -1, -1
        )
        for query_block in self.query_blocks:
            query_embeds = query_block(
                query_embeds, inputs_embeds, attention_mask.bool()
            )
        return query_embeds


class LLaDAImageTextProjectionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, norm_eps: float):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=False)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        heads = (self.heads, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).unflatten(-1, heads))
        key = self.k_norm(self.k_proj(hidden_states).unflatten(-1, heads))
        value = self.v_proj(hidden_states).unflatten(-1, heads)
        hidden_states = dispatch_attention_fn(query, key, value)
        return self.out_proj(hidden_states.flatten(2, 3))


class LLaDAImageMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, approximate: str):
        super().__init__()
        self.approximate = approximate
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate=self.approximate))


class LLaDAImageTextProjectionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.self_attn = LLaDAImageTextProjectionAttention(
            hidden_size, num_attention_heads, norm_eps
        )
        self.layer_norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.mlp = LLaDAImageMLP(hidden_size, intermediate_size, approximate="tanh")
        self.layer_norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class LLaDAImageTextProjectionModel(ModelMixin, ConfigMixin):
    """Maps LLaDA hidden states to the denoiser caption dimension."""

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
        self.layers = nn.ModuleList(
            LLaDAImageTextProjectionBlock(
                hidden_size, intermediate_size, num_attention_heads, norm_eps
            )
            for _ in range(num_hidden_layers)
        )
        self.projector = nn.Linear(hidden_size, projection_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.projector(hidden_states)


class LLaDAImageSigVQAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, bias: bool):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query, key, value = (
            tensor.unflatten(-1, (self.heads, self.head_dim))
            for tensor in self.qkv(hidden_states).chunk(3, dim=-1)
        )
        hidden_states = dispatch_attention_fn(query, key, value)
        return self.proj(hidden_states.flatten(2, 3))


class LLaDAImageSigVQVisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_bias: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = LLaDAImageSigVQAttention(
            hidden_size, num_attention_heads, attention_bias
        )
        self.mlp = LLaDAImageMLP(hidden_size, intermediate_size, approximate="none")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class LLaDAImageSigVQPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
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


class LLaDAImageSigVQModel(ModelMixin, ConfigMixin):
    """GLM SigVQ image encoder, quantizer, and prior projection for editing."""

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
        self.visual = nn.Module()
        self.visual.patch_embed = LLaDAImageSigVQPatchEmbed(
            in_channels, hidden_size, patch_size
        )
        self.visual.embeddings = LLaDAImageSigVQEmbeddings(
            image_size, patch_size, hidden_size
        )
        self.visual.blocks = nn.ModuleList(
            LLaDAImageSigVQVisionBlock(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                attention_bias,
                norm_eps,
            )
            for _ in range(num_hidden_layers)
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

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Map RGB images in [-1, 1] to projected semantic features."""
        batch_size, _, height, width = pixel_values.shape
        grid_height = height // self.config.patch_size
        grid_width = width // self.config.patch_size
        hidden_states = self.visual.patch_embed(pixel_values)
        hidden_states = self.visual.embeddings(hidden_states, grid_height, grid_width)
        for block in self.visual.blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, self.config.hidden_size, grid_height, grid_width
        )
        hidden_states = self.vqmodel.quant_conv(hidden_states)
        token_ids = self.vqmodel.quantize(hidden_states).reshape(batch_size, -1)
        return self.prior_projector(self.prior_token_embedding(token_ids))


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
        if quant_config is not None or "quantization_config" in hf_config:
            raise ValueError(
                "LLaDA-Image serves only the BF16 checkpoints, such as "
                "inclusionAI/LLaDA-Image, and this transformer is quantized"
            )
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: list[torch.Tensor],
        timestep: torch.Tensor,
        encoder_hidden_states_image: list[torch.Tensor] | None = None,
        source_latents: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        model_dtype = next(self.parameters()).dtype
        output = super().forward(
            x=[latent.unsqueeze(1).to(model_dtype) for latent in hidden_states],
            t=(timestep / 1000.0).to(model_dtype),
            cap_feats=encoder_hidden_states,
            # Text-to-image passes no SigVQ features.
            glm_cap_feats=encoder_hidden_states_image or None,
            source_latents=source_latents,
        )
        return -torch.stack(output, dim=0).squeeze(2).float()


EntryClass = [
    LLaDAImageTransformer2DModel,
    LLaDAImageQueryFormerModel,
    LLaDAImageTextProjectionModel,
    LLaDAImageSigVQModel,
]
