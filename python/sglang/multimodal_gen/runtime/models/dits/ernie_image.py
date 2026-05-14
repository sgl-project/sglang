# Copyright 2026 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from sglang.multimodal_gen.configs.models.dits.ernie_image import (
    ErnieImageDitConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, apply_qk_norm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.managers.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT


def _rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)  # codespell:ignore nd
    return out.float()


class EmbedND3(nn.Module):
    """3D rotary positional embedding for (temporal/batch_idx, height, width)."""

    def __init__(self, dim: int, theta: int, axes_dim: Tuple[int, int, int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat(
            [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)],
            dim=-1,
        )
        emb = emb.unsqueeze(1).permute(2, 0, 1, 3)
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)


class ErnieImageSelfAttention(nn.Module):
    """Self-attention with separate Q/K/V projections and QK LayerNorm.

    Module name hierarchy matches diffusers Attention naming convention:
      self_attention.to_q, self_attention.to_k, self_attention.to_v,
      self_attention.to_out.0, self_attention.norm_q, self_attention.norm_k.

    Supports tensor parallelism: Q/K/V projections use ColumnParallelLinear
    (output dim sharded by heads), output projection uses RowParallelLinear
    (input dim sharded, all-reduce after matmul).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        tp_size = get_tp_world_size()
        self.num_local_heads = num_heads // tp_size
        assert (
            num_heads % tp_size == 0
        ), f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"

        self.to_q = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.to_q",
        )
        self.to_k = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.to_v",
        )
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    prefix=f"{prefix}.to_out.0",
                ),
            ]
        )

        self.qk_layernorm = qk_layernorm
        if qk_layernorm:
            self.norm_q = RMSNorm(head_dim, eps=eps)
            self.norm_k = RMSNorm(head_dim, eps=eps)

        self.attn = USPAttention(
            num_heads=self.num_local_heads,
            head_size=head_dim,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        B, S, H = x.shape

        q, _ = self.to_q(x)
        k, _ = self.to_k(x)
        v, _ = self.to_v(x)

        q = q.view(B, S, self.num_local_heads, self.head_dim)
        k = k.view(B, S, self.num_local_heads, self.head_dim)
        v = v.view(B, S, self.num_local_heads, self.head_dim)

        if self.qk_layernorm:
            q, k = apply_qk_norm(
                q,
                k,
                self.norm_q,
                self.norm_k,
                self.head_dim,
            )

        q = _apply_rotary_bshd(q, rotary_pos_emb)
        k = _apply_rotary_bshd(k, rotary_pos_emb)

        attn_out = self.attn(q, k, v)
        attn_out = attn_out.reshape(B, S, self.num_local_heads * self.head_dim)
        out, _ = self.to_out[0](attn_out)
        return out


class ErnieImageMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [ffn_hidden_size, ffn_hidden_size],
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.linear_fc2 = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            prefix=f"{prefix}.linear_fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = up * F.gelu(gate)
        x, _ = self.linear_fc2(x)
        return x


class ErnieImageSharedAdaLNBlock(nn.Module):
    """Single-stream transformer block with externally-computed Shared AdaLN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        ffn_hidden_size: int,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.adaLN_sa_ln = RMSNorm(hidden_size, eps=eps)
        self.self_attention = ErnieImageSelfAttention(
            hidden_size,
            num_heads,
            head_dim,
            eps,
            qk_layernorm,
            prefix=f"{prefix}.self_attention",
        )
        self.adaLN_mlp_ln = RMSNorm(hidden_size, eps=eps)
        self.mlp = ErnieImageMLP(hidden_size, ffn_hidden_size, prefix=f"{prefix}.mlp")

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        shift_msa: torch.Tensor,
        scale_msa: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x = self.adaLN_sa_ln(x) * (1 + scale_msa) + shift_msa
        x = residual + gate_msa * self.self_attention(x, rotary_pos_emb)

        residual = x
        x = self.adaLN_mlp_ln(x) * (1 + scale_mlp) + shift_mlp
        x = residual + gate_mlp * self.mlp(x)

        return x


def _apply_rotary_bshd(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    freqs = freqs.permute(1, 0, 2, 3)
    rot_dim = freqs.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    cos_ = torch.cos(freqs).to(x.dtype)
    sin_ = torch.sin(freqs).to(x.dtype)

    x1, x2 = x_rot.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)

    x_rot = x_rot * cos_ + x_rotated * sin_
    return torch.cat((x_rot, x_pass), dim=-1)


class ErnieImageTransformer2DModel(CachableDiT, OffloadableDiTMixin):
    """ErnieImage DiT: Single-stream transformer with Shared AdaLN."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ErnieImageSharedAdaLNBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    _fsdp_shard_conditions = ErnieImageDitConfig().arch_config._fsdp_shard_conditions
    _compile_conditions = []
    param_names_mapping = ErnieImageDitConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = {}

    def __init__(
        self,
        config: ErnieImageDitConfig,
        hf_config: dict[str, Any],
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.out_channels
        self.head_dim = arch.attention_head_dim
        self.num_layers = arch.num_layers
        self.patch_size = arch.patch_size
        self.out_channels = arch.out_channels
        self.inner_dim = self.hidden_size

        tp_size = get_tp_world_size()

        self.x_embedder = nn.ModuleDict(
            {
                "proj": nn.Conv2d(
                    arch.in_channels,
                    self.inner_dim,
                    kernel_size=arch.patch_size,
                    stride=arch.patch_size,
                    bias=True,
                ),
            }
        )

        if arch.text_in_dim != self.inner_dim:
            self.text_proj = nn.Linear(arch.text_in_dim, self.inner_dim, bias=False)
        else:
            self.text_proj = None

        self.time_proj = Timesteps(
            self.inner_dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=self.inner_dim,
            time_embed_dim=self.inner_dim,
        )

        self.pos_embed = EmbedND3(
            dim=self.head_dim,
            theta=arch.rope_theta,
            axes_dim=arch.rope_axes_dim,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.inner_dim, 6 * self.inner_dim),
        )

        self.layers = nn.ModuleList(
            [
                ErnieImageSharedAdaLNBlock(
                    hidden_size=self.inner_dim,
                    num_heads=self.num_attention_heads,
                    head_dim=self.head_dim,
                    ffn_hidden_size=arch.ffn_hidden_size,
                    eps=arch.eps,
                    qk_layernorm=arch.qk_layernorm,
                    prefix=f"layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        self.final_norm = nn.ModuleDict(
            {
                "norm": nn.LayerNorm(
                    self.inner_dim, elementwise_affine=False, eps=arch.eps
                ),
                "linear": nn.Linear(self.inner_dim, self.inner_dim * 2),
            }
        )

        self.final_linear = ColumnParallelLinear(
            self.inner_dim,
            arch.patch_size * arch.patch_size * self.out_channels,
            bias=True,
            gather_output=True,
            prefix="final_linear",
        )

        self.layer_names = ["layers"]

        self.__post_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, C, H, W] latent images (patchified, 128 channels)
            encoder_hidden_states: [B, T, text_dim] or list of text embeddings
            timestep: [B] timestep values
        Returns:
            output: [B, C, H, W] predicted noise / denoised output
        """
        device, dtype = hidden_states.device, hidden_states.dtype
        B, C, H, W = hidden_states.shape
        p = self.patch_size
        Hp, Wp = H // p, W // p
        N_img = Hp * Wp

        img_tokens = self.x_embedder["proj"](hidden_states)  # [B, D, Hp, Wp]
        img_tokens = img_tokens.reshape(B, self.inner_dim, N_img).transpose(
            1, 2
        )  # [B, N_img, D]

        if isinstance(encoder_hidden_states, (list, tuple)):
            encoder_hidden_states = encoder_hidden_states[0]
        text_tokens = encoder_hidden_states  # [B, T, text_dim]
        if self.text_proj is not None and text_tokens.numel() > 0:
            text_tokens = self.text_proj(text_tokens)
        Tmax = text_tokens.shape[1]

        x = torch.cat([img_tokens, text_tokens], dim=1)  # [B, S, D]

        grid_yx = torch.stack(
            torch.meshgrid(
                torch.arange(Hp, device=device, dtype=torch.float32),
                torch.arange(Wp, device=device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)

        image_ids = torch.cat(
            [
                torch.full((B, N_img, 1), Tmax, device=device, dtype=torch.float32),
                grid_yx.view(1, N_img, 2).expand(B, -1, -1),
            ],
            dim=-1,
        )

        if Tmax > 0:
            text_ids = torch.cat(
                [
                    torch.arange(Tmax, device=device, dtype=torch.float32)
                    .view(1, Tmax, 1)
                    .expand(B, -1, -1),
                    torch.zeros((B, Tmax, 2), device=device),
                ],
                dim=-1,
            )
        else:
            text_ids = torch.zeros((B, 0, 3), device=device)

        all_ids = torch.cat([image_ids, text_ids], dim=1)
        rotary_pos_emb = self.pos_embed(all_ids)

        t_emb = self.time_proj(timestep.to(dtype))
        c = self.time_embedding(t_emb.to(dtype=dtype))

        mod_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            t.unsqueeze(1) for t in mod_params.chunk(6, dim=-1)
        )

        for layer in self.layers:
            x = layer(
                x,
                rotary_pos_emb,
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            )

        scale, shift = self.final_norm["linear"](c).chunk(2, dim=-1)
        x = self.final_norm["norm"](x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        patches, _ = self.final_linear(x[:, :N_img, :])

        output = patches.view(B, Hp, Wp, p, p, self.out_channels)
        output = output.permute(0, 5, 1, 3, 2, 4).contiguous()
        output = output.view(B, self.out_channels, H, W)

        return output


EntryClass = ErnieImageTransformer2DModel
