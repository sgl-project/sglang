# Copyright 2026 Black Forest Labs. All rights reserved.
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
# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 video VAE: Swin3D encoder/decoder with NATTEN neighborhood attention.

Adapted from the FLUX Action reference implementation
(https://github.com/black-forest-labs/flux-action,
``flux_action/models/video_vae.py``). The module tree matches
``video_vae.safetensors`` so the checkpoint loads without renaming.

Latents are ``(B, 96, 1 + (T - 1) // 4, H // 32, W // 32)`` and normalized by
the running statistics stored in the checkpoint. Neighborhood attention runs
on NATTEN (https://natten.org) when installed, else on a FlexAttention fallback
(``flux3_neighborhood_attention``).
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models.vaes.flux3_video import (
    Flux3VideoVAEArchConfig,
    Flux3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.vaes.flux3_neighborhood_attention import (
    natten_available,
    neighborhood_attention,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_norm_layer = partial(nn.LayerNorm, eps=1e-5)

# NATTEN backend, probed once per token rank (2D/3D): the fused CUTLASS
# kernels only exist for some architectures; flex-attention runs everywhere.
_PERSISTENT_KERNEL_BACKENDS = ("blackwell", "hopper")
_natten_backends: dict[int, str] = {}


def _natten_attention_kwargs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: list[int],
    is_causal: list[bool] | None = None,
) -> dict:
    backend = _natten_backends.get(q.ndim)
    if backend is None:
        backend = envs.SGLANG_DIFFUSION_FLUX3_NATTEN_BACKEND
        if not backend:
            from natten import backends as nb

            if nb.can_run_cutlass_blackwell_fna(q, k, v):
                backend = "blackwell-fna"
            elif nb.can_run_cutlass_hopper_fna(q, k, v):
                backend = "hopper-fna"
            elif nb.can_run_cutlass_fna(q, k, v):
                backend = "cutlass-fna"
            else:
                backend = "flex-fna"
        logger.info("FLUX 3 video VAE natten backend (%dD): %s", q.ndim - 3, backend)
        _natten_backends[q.ndim] = backend
    # A window covering every non-causal axis is dense attention; NATTEN then
    # dispatches to its "*-fmha" kernels.
    na_dim = q.ndim - 3
    causal = is_causal or [False] * na_dim
    if all(
        kk == s and not c
        for kk, s, c in zip(kernel_size, q.shape[1 : 1 + na_dim], causal)
    ):
        backend = backend.replace("-fna", "-fmha")
    kwargs = {"backend": backend}
    if backend.split("-")[0] in _PERSISTENT_KERNEL_BACKENDS:
        kwargs["run_persistent_kernel"] = True
    return kwargs


class DistributedRunningStats(nn.Module):
    """Latent mean / variance of the trained VAE; normalizes ``(B, C, ...)``."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))
        self.register_buffer("initialized", torch.tensor(False))

    def _shape(self, x: torch.Tensor) -> tuple:
        return (1, -1) + (1,) * (x.dim() - 2)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        s = self._shape(x)
        return (x - self.running_mean.view(s)) / self.running_var.sqrt().view(s)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        s = self._shape(x)
        return x * self.running_var.sqrt().view(s) + self.running_mean.view(s)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = _norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w, _ = x.shape
        if h % 2 == 1 or w % 2 == 1:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        b, d, h, w, c = x.shape
        x = x.reshape(b, d, h // 2, 2, w // 2, 2, c)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).flatten(4)
        return self.reduction(self.norm(x))


class TemporalMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = _norm_layer(2 * dim)
        self.reduction = nn.Linear(2 * dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 == 1:
            x = torch.concat([x[:, :1], x], dim=1)
        b, d, h, w, c = x.shape
        x = x.reshape(b, d // 2, 2, h, w, c)
        skip = x.mean(2)
        x = x.permute(0, 1, 3, 4, 2, 5).reshape(b, d // 2, h, w, 2 * c)
        return self.reduction(self.norm(x)) + skip


class PatchExpansion(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = _norm_layer(dim)
        self.expansion = nn.Linear(dim, 4 * out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w, _ = x.shape
        x = self.expansion(self.norm(x))
        x = x.view(b, d, h, w, 2, 2, self.out_dim)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return x.view(b, d, h * 2, w * 2, self.out_dim)


class TemporalExpansion(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = _norm_layer(dim)
        self.expansion = nn.Linear(dim, 2 * out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w, _ = x.shape
        x = self.expansion(self.norm(x)) + torch.concat([x, x], -1)
        x = x.view(b, d, h, w, 2, self.out_dim)
        x = x.permute(0, 1, 4, 2, 3, 5).contiguous()
        x = x.view(b, d * 2, h, w, self.out_dim)
        return x[:, 1:]


def _temporal_core_halo(
    core_start: int, core_end: int, length: int, kernel: int, causal: bool
) -> tuple[int, int]:
    """Frames ``[halo_start, halo_end)`` a looped block must attend over to
    reproduce frames ``[core_start, core_end)`` of the full-sequence output.

    NATTEN gives frame ``i`` the keys ``[start(i), start(i) + kernel)`` with
    ``start(i) = clamp(i - kernel // 2, 0, length - kernel)`` (windows shift
    inward at the edges), or ``[max(0, i - kernel + 1), i]`` when causal.
    """
    if causal:
        return max(0, core_start - kernel + 1), core_end

    def window_start(index: int) -> int:
        return min(max(index - kernel // 2, 0), length - kernel)

    return window_start(core_start), window_start(core_end - 1) + kernel


class RotaryPositionEmbedding3D(nn.Module):
    def __init__(self, head_dim: int, base: float = 256.0):
        super().__init__()
        assert head_dim % 8 == 0, "head dimension must be divisible by 8"
        self.head_dim = head_dim
        self.chunk_dim = head_dim // 4
        axis_inv_freq = 1.0 / (
            base ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim)
        )
        # (t, h, w, unused) axes; the checkpoint stores this buffer.
        inv_freq = torch.stack(
            [
                axis_inv_freq,
                axis_inv_freq,
                axis_inv_freq,
                torch.zeros(self.chunk_dim // 2),
            ]
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        temporal_offset: int | torch.Tensor = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``(B, T, H, W, heads, head_dim)``; ``temporal_offset`` is the time of frame 0."""
        _, t, h, w, _, _ = q.shape
        device, dtype = q.device, q.dtype
        grids = torch.meshgrid(
            torch.arange(t, device=device, dtype=torch.float32) + temporal_offset,
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        pos = torch.stack(grids + (torch.zeros_like(grids[0]),), dim=-1)
        freqs = torch.einsum("...a,af->...af", pos, self.inv_freq.float())
        freqs = freqs.reshape(1, t, h, w, 1, -1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class Natten3D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: list[int],
        num_heads: int,
        causal: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.window_size = list(window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.qk_norm = qk_norm
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPositionEmbedding3D(self.head_dim)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=False)

    def _qkv(self, x: torch.Tensor):
        b, t, h, w, _ = x.shape
        q, k, v = (
            self.qkv(x).reshape(b, t, h, w, 3, self.num_heads, self.head_dim).unbind(4)
        )
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = x.shape
        q, k, v = self._qkv(x)
        q, k = self.rope(q, k)
        return self.proj(self._attend(q, k, v).reshape(b, t, h, w, c))

    def forward_region(
        self,
        x: torch.Tensor,
        *,
        temporal_offset: torch.Tensor,
        output_start: int,
        output_end: int,
    ) -> torch.Tensor:
        """Attention over a temporal halo of a longer sequence; returns frames ``[output_start, output_end)``."""
        b, t, h, w, c = x.shape
        q, k, v = self._qkv(x)
        q, k = self.rope(q, k, temporal_offset=temporal_offset)
        out = self._attend(q, k, v)[:, output_start:output_end]
        return self.proj(out.reshape(b, output_end - output_start, h, w, c))

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        if q.shape[1] == 1:  # a single frame uses the 2-D kernel
            q2, k2, v2 = q.squeeze(1), k.squeeze(1), v.squeeze(1)
            kernel = self.window_size[1:]
            if not natten_available():
                out = neighborhood_attention(q2, k2, v2, kernel_size=kernel)
                return out.unsqueeze(1)
            from natten.functional import na2d

            kwargs = _natten_attention_kwargs(q2, k2, v2, kernel_size=kernel)
            return na2d(
                q2, k2, v2, kernel_size=kernel, attention_kwargs=kwargs
            ).unsqueeze(1)
        causal = [self.causal, False, False]
        if not natten_available():
            return neighborhood_attention(
                q, k, v, kernel_size=self.window_size, is_causal=causal
            )
        from natten.functional import na3d

        kwargs = _natten_attention_kwargs(
            q, k, v, kernel_size=self.window_size, is_causal=causal
        )
        return na3d(
            q,
            k,
            v,
            is_causal=causal,
            kernel_size=self.window_size,
            attention_kwargs=kwargs,
        )


class GLUMLP(nn.Module):
    def __init__(self, dim: int, align_to: int = 64):
        super().__init__()
        hidden_dim = align_to * ((int(dim * 8 / 3) + align_to - 1) // align_to)
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class SwinTransformerBlock(nn.Module):
    """Pre-norm neighborhood-attention block.

    With ``max_t`` set the block runs "looped": attention and MLP are applied
    to temporal windows of at most ``max_t`` frames (each window attends over
    its halo), writing the output into the input in place. The result equals
    the full forward frame for frame at a fraction of the activation memory.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list[int],
        causal: bool = False,
        qk_norm: bool = False,
        max_t: int | None = None,
    ):
        super().__init__()
        if max_t is not None and max_t < window_size[0]:
            raise ValueError(
                f"max_t={max_t} must be at least the temporal kernel {window_size[0]}"
            )
        self.norm1 = _norm_layer(dim)
        self.attn = Natten3D(
            dim, window_size, num_heads, causal=causal, qk_norm=qk_norm
        )
        self.norm2 = _norm_layer(dim)
        self.mlp = GLUMLP(dim)
        self.max_t = max_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_t is None or x.shape[1] <= self.max_t:
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))
        return self._forward_looped(x)

    def _forward_looped(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("looped decoding writes in place; run it under no_grad")
        length = x.shape[1]
        kernel = self.attn.window_size[0]
        # A halo reaches at most kernel - 1 < max_t frames back into the
        # previous window, so each window's output is written one iteration late.
        pending: tuple[int, int, torch.Tensor] | None = None
        for core_start in range(0, length, self.max_t):
            core_end = min(core_start + self.max_t, length)
            halo_start, halo_end = _temporal_core_halo(
                core_start, core_end, length, kernel, self.attn.causal
            )
            normalized_halo = self.norm1(x[:, halo_start:halo_end])
            if pending is not None:
                x[:, pending[0] : pending[1]].copy_(pending[2])
            out = x[:, core_start:core_end] + self.attn.forward_region(
                normalized_halo,
                temporal_offset=torch.tensor(float(halo_start), device=x.device),
                output_start=core_start - halo_start,
                output_end=core_end - halo_start,
            )
            pending = (core_start, core_end, out + self.mlp(self.norm2(out)))
        assert pending is not None
        x[:, pending[0] : pending[1]].copy_(pending[2])
        return x


class PatchEmbed3d(nn.Module):
    def __init__(
        self, patch_size, in_channels: int = 3, embed_dim: int = 96, norm: bool = True
    ):
        super().__init__()
        self.patch = tuple(patch_size)
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=self.patch, stride=self.patch
        )
        self.norm = _norm_layer(embed_dim) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.shape
        pad = [(p - s % p) % p for s, p in zip((t, h, w), self.patch)]
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
        x = self.proj(x).permute(0, 2, 3, 4, 1)
        return self.norm(x)


class EncoderSwin3D(nn.Module):
    def __init__(
        self,
        z_ch: int,
        patch_size,
        embed_dim: int,
        depths,
        temporal,
        num_heads,
        window_size,
        causal: bool = False,
        qk_norm: bool = False,
        patch_norm: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(embed_dim * 2 ** (len(depths) - 1), z_ch)
        self.patch_embed = PatchEmbed3d(
            patch_size=patch_size, embed_dim=embed_dim, norm=patch_norm
        )
        layers: list[nn.Module] = []
        for i_stage in range(len(depths)):
            dim = embed_dim * 2**i_stage
            layers.append(
                nn.Sequential(
                    *[
                        SwinTransformerBlock(
                            dim,
                            num_heads[i_stage],
                            window_size,
                            causal=causal,
                            qk_norm=qk_norm,
                        )
                        for _ in range(depths[i_stage])
                    ]
                )
            )
            downsampled = False
            if i_stage < len(depths) - 1:
                layers.append(PatchMerging(dim, 2 * dim))
                downsampled = True
            if temporal[i_stage]:
                heads = num_heads[i_stage + 1] if downsampled else num_heads[i_stage]
                dim = 2 * dim if downsampled else dim
                layers.append(
                    SwinTransformerBlock(
                        dim, heads, window_size, causal=causal, qk_norm=qk_norm
                    )
                )
                layers.append(TemporalMerging(dim, dim))
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.features(x)
        x = self.proj(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class DecoderSwin3D(nn.Module):
    def __init__(
        self,
        z_ch: int,
        patch_size,
        embed_dim: int,
        depths,
        temporal,
        num_heads,
        window_size,
        causal: bool = False,
        qk_norm: bool = False,
        max_t: int | None = None,
    ):
        super().__init__()
        self.ps = list(patch_size)
        self.proj_in = nn.Linear(z_ch, embed_dim * 2 ** (len(depths) - 1))
        self.proj_out = nn.Linear(embed_dim, math.prod(patch_size) * 3)

        def block(dim: int, heads: int) -> SwinTransformerBlock:
            return SwinTransformerBlock(
                dim, heads, window_size, causal=causal, qk_norm=qk_norm, max_t=max_t
            )

        layers: list[nn.Module] = []
        for i_stage in reversed(range(len(depths))):
            dim = embed_dim * 2**i_stage
            layers.append(
                nn.Sequential(
                    *[block(dim, num_heads[i_stage]) for _ in range(depths[i_stage])]
                )
            )
            if temporal[i_stage]:
                layers.append(TemporalExpansion(dim, dim))
                layers.append(block(dim, num_heads[i_stage]))
            if i_stage > 0:
                layers.append(PatchExpansion(dim, dim // 2))
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x.permute(0, 2, 3, 4, 1).contiguous())
        x = self.proj_out(self.features(x))
        b, t, h, w, _ = x.shape
        x = x.view(b, t, h, w, self.ps[0], self.ps[1], self.ps[2], 3)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(b, t * self.ps[0], h * self.ps[1], w * self.ps[2], 3)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class ViTNorm(nn.Module):
    def __init__(self, arch: Flux3VideoVAEArchConfig, *, load_decoder: bool = True):
        super().__init__()
        self.z_dim = arch.z_dim
        common = dict(
            patch_size=arch.patch_size,
            window_size=arch.window_size,
            embed_dim=arch.embed_dim,
            num_heads=arch.num_heads,
            temporal=arch.temporal,
            qk_norm=arch.qk_norm,
        )
        self.encoder = EncoderSwin3D(
            z_ch=2 * arch.z_dim,
            depths=arch.enc_depths,
            causal=arch.enc_causal,
            patch_norm=arch.patch_norm,
            **common,
        )
        self.decoder = (
            DecoderSwin3D(
                z_ch=arch.z_dim,
                depths=arch.dec_depths,
                causal=arch.dec_causal,
                max_t=arch.decoder_max_t,
                **common,
            )
            if load_decoder
            else None
        )
        self.z_normalizer = DistributedRunningStats(arch.z_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x).chunk(2, dim=-4)
        return self.z_normalizer.normalize(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("this video VAE was built without its decoder")
        return self.decoder(self.z_normalizer.denormalize(z))


class Flux3VideoVAE(nn.Module, LayerwiseOffloadableModuleMixin):
    """Frozen FLUX 3 video VAE: ``encode`` / ``encode_frame`` / ``decode``."""

    layerwise_offload_dit_group_enabled = False
    layer_names = ["model.encoder.features", "model.decoder.features"]

    def __init__(self, config: Flux3VideoVAEConfig, **kwargs):
        super().__init__()
        arch: Flux3VideoVAEArchConfig = config.arch_config
        if tuple(arch.temporal) != (False, False, True, True) or tuple(
            arch.patch_size
        ) != (1, 4, 4):
            raise ValueError(
                "compression ratios assume temporal=(F, F, T, T) and patch (1, 4, 4)"
            )
        self.config = config
        self.arch = arch
        self.model = ViTNorm(arch, load_decoder=config.load_decoder)

    @property
    def temporal_compression_ratio(self) -> int:
        return self.arch.temporal_compression_ratio

    @property
    def spatial_compression_ratio(self) -> int:
        return self.arch.spatial_compression_ratio

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, 3, T, H, W)`` in ``[-1, 1]`` with ``T = 1 (mod 4)`` -> normalized latents.

        Clips longer than one chunk are encoded in ``chunk_size_frames`` chunks
        overlapping by one frame (the repeated first latent of each later chunk
        is dropped); shorter clips are padded by repeating the last frame.
        """
        num_frames = video.shape[2]
        if (num_frames - 1) % self.temporal_compression_ratio:
            raise ValueError(
                f"video VAE encode expects T = 1 (mod 4) frames, got {num_frames}"
            )
        chunk = self.arch.chunk_size_frames
        stride = chunk - 1
        padded = chunk + max(0, -(-(num_frames - chunk) // stride)) * stride
        if padded > num_frames:
            tail = video[:, :, -1:].expand(-1, -1, padded - num_frames, -1, -1)
            video = torch.cat([video, tail], dim=2)
        pieces = []
        for start in range(0, padded - chunk + 1, stride):
            z = self.model.encode(video[:, :, start : start + chunk])
            pieces.append(z if start == 0 else z[:, :, 1:])
        latent = torch.cat(pieces, dim=2)
        return latent[:, :, : 1 + (num_frames - 1) // self.temporal_compression_ratio]

    @torch.no_grad()
    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` -> ``(B, 96, 1, H // 32, W // 32)``: the frame alone, no chunk padding."""
        return self.model.encode(frame[:, :, None])

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``(B, 96, T_lat, h, w)`` -> pixels ``(B, 3, 4 * T_lat - 3, 32 h, 32 w)`` in ``[-1, 1]``."""
        return self.model.decode(latents).clamp(-1, 1)

    def load_checkpoint(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Strict load of ``video_vae.safetensors`` (decoder tensors skipped when not built)."""
        if self.model.decoder is None:
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("model.decoder.")
            }
        self.load_state_dict(state_dict, strict=True, assign=True)


EntryClass = Flux3VideoVAE
