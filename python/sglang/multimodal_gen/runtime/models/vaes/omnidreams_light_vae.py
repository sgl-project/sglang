# SPDX-License-Identifier: Apache-2.0
"""LightVAE (75%-pruned Wan 2.1) streaming causal encoder for OmniDreams.

Ported from FlashDreams ``recipes/wan/autoencoder/vae.py`` (the Wan 2.1
``is_residual=False`` encoder path only: ``Encoder3d`` + the quant ``conv1`` +
the streaming ``CausalConv3d``/``Resample`` building blocks), trimmed to the
OmniDreams single-view config (``base_dim=96``, ``z_dim=16``, ``patch_size=1``,
``dim_mult=(1,2,4,4)``, ``temperal_downsample=(False,True,True)``, 8x spatial).

Swaps the Wan VAE *encode* (first-frame image + HD-map clip) for the LightX2V
LightVAE encoder (``pruning_rate=0.75``). Same 16-ch Wan 2.1 latent space, so
the latent mean/std are unchanged; the decode path is unaffected. Gated behind
``OmniDreamsPipelineConfig.use_light_vae_encoder`` (default off).

The public entry point is :class:`LightVAEEncoder`, a drop-in for the Wan VAE on
the encode side: ``encode(x) -> dist`` (``.mode()`` returns the RAW latent mean,
matching ``AutoencoderKLWan.encode``), plus ``latents_mean`` / ``latents_std``
attributes so the shared ``_vae_encode_normalized`` normalizes uniformly.

Checkpoint: ``lightvaew2_1.pth`` (lightx2v/Autoencoders), original-Wan key
naming (``encoder.downsamples.<i>.residual.<j>`` + ``conv1``); the decoder
weights it also contains are dropped (``strict=False``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits.omnidreams_cuda_graph import set_or_copy
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

CACHE_T = 2
TEMPORAL_WINDOW = 4


@dataclass
class WanVAEEncCache:
    """Streaming encoder cache; per-block left-context keyed by ``id(module)``."""

    enc_state: Dict[int, torch.Tensor] = field(default_factory=dict)


class CausalConv3d(nn.Conv3d):
    """3D conv with causal time padding and a streaming left-context slot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.padding, tuple)
        ph, pw = self.padding[1], self.padding[2]
        self._spatial_pad = (pw, pw, ph, ph)
        self._has_spatial_pad = ph > 0 or pw > 0
        self._time_pad = 2 * self.padding[0]
        self.padding = (0, 0, 0)

    def forward(
        self, x: torch.Tensor, prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        time_pad = self._time_pad
        if prev is not None and time_pad > 0:
            x = torch.cat([prev, x], dim=2)
            time_pad = max(0, time_pad - prev.shape[2])
        if time_pad or self._has_spatial_pad:
            x = F.pad(x, (*self._spatial_pad, time_pad, 0), mode="constant")
        return super().forward(x)

    def cache_step(
        self, x: torch.Tensor, state: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        key = id(self)
        prev = state.get(key)
        out = self.forward(x, prev)
        new_tail = x[:, :, -CACHE_T:]
        if new_tail.shape[2] < CACHE_T and prev is not None:
            new_tail = torch.cat([prev[:, :, -1:], new_tail], dim=2)
        set_or_copy(state, key, new_tail)
        return out


class RMS_norm(nn.Module):
    """RMS-norm with a learnable per-channel scale (no bias; Wan VAE)."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True):
        super().__init__()
        broadcast = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcast) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma + self.bias


def _bt_flatten(x: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def _bt_unflatten(x: torch.Tensor, b: int) -> torch.Tensor:
    bt, c, h, w = x.shape
    t = bt // b
    return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class Resample(nn.Module):
    """Spatial 2x resample, optionally with temporal downsample (encode path)."""

    def __init__(self, dim: int, mode: str):
        assert mode in (
            "downsample2d",
            "downsample3d",
        ), f"encode-only Resample mode must be downsample2d/3d; got {mode}"
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(dim, dim, 3, stride=(2, 2)),
        )
        if mode == "downsample3d":
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return _bt_unflatten(self.resample(_bt_flatten(x)), b)

    def forward(self, x: torch.Tensor, state: Dict[int, torch.Tensor]) -> torch.Tensor:
        if self.mode == "downsample3d":
            return self._downsample3d_step(self._spatial(x), state)
        return self._spatial(x)

    def _downsample3d_step(
        self, x: torch.Tensor, state: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        key = id(self)
        prev = state.get(key)
        new_tail = x[:, :, -1:]
        if prev is not None:
            x = self.time_conv(torch.cat([prev, x], dim=2))
        set_or_copy(state, key, new_tail)
        return x


class ResidualBlock(nn.Module):
    """Two-conv residual block with RMS-norm + SiLU."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut: nn.Module = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor, state: Dict[int, torch.Tensor]) -> torch.Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            x = (
                layer.cache_step(x, state)
                if isinstance(layer, CausalConv3d)
                else layer(x)
            )
        return x + h


class AttentionBlock(nn.Module):
    """Single-head self-attention; stateless across streaming chunks."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(
        self, x: torch.Tensor, state: Optional[Dict[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        b, c, t, h, w = x.shape
        identity = x
        x = _bt_flatten(x)
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, h * w)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        return _bt_unflatten(x, b) + identity


class Encoder3d(nn.Module):
    """Wan 2.1 streaming causal encoder body (``is_residual=False``)."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 32,  # 2 * latent z_dim (mu/logvar) for the head
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        temperal_downsample=(False, True, True),
        dropout: float = 0.0,
        pruning_rate: float = 0.75,
        in_channels: int = 3,
    ):
        super().__init__()
        dims = [int(dim * u * (1 - pruning_rate)) for u in (1,) + tuple(dim_mult)]
        scale = 1.0

        self.conv1 = CausalConv3d(in_channels, dims[0], 3, padding=1)

        downsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            is_last_stage = i == len(dim_mult) - 1
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if not is_last_stage:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, state: Dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.conv1.cache_step(x, state)
        for layer in self.downsamples:
            x = layer(x, state)
        for layer in self.middle:
            x = layer(x, state)
        norm, act, conv = self.head
        assert isinstance(conv, CausalConv3d)
        return conv.cache_step(act(norm(x)), state)

    @torch.no_grad()
    def normalize_state_for_body(self, state: Dict[int, torch.Tensor]) -> None:
        """Pad each CausalConv3d seed state (T=1) up to ``CACHE_T`` frames.

        Bit-equivalent to the zero-prepad ``CausalConv3d.forward`` does when
        ``prev`` is shorter than ``time_pad``; keeps body-chunk state shapes
        identical across AR steps.
        """
        for module in self.modules():
            if not isinstance(module, CausalConv3d):
                continue
            key = id(module)
            if key not in state:
                continue
            prev = state[key]
            if prev.shape[2] >= CACHE_T:
                continue
            pad = CACHE_T - prev.shape[2]
            b, c, _, h, w = prev.shape
            zeros = prev.new_zeros(b, c, pad, h, w)
            state[key] = torch.cat([zeros, prev], dim=2)


@dataclass
class _LatentDist:
    """Minimal distribution adapter so callers can ``.mode()`` the raw latent."""

    mu: torch.Tensor

    def mode(self) -> torch.Tensor:
        return self.mu

    def sample(self, generator=None) -> torch.Tensor:  # deterministic (distilled)
        return self.mu


class LightVAEEncoder(nn.Module):
    """SGLang-facing LightVAE encode wrapper (drop-in for the Wan VAE encode).

    ``encode(x)`` returns a :class:`_LatentDist` whose ``.mode()`` is the RAW
    latent mean (same convention as ``AutoencoderKLWan.encode``); the shared
    ``_vae_encode_normalized`` applies ``(z - latents_mean) / latents_std``.
    Single-pass per call (a fresh streaming cache), matching the OmniDreams
    encode use (one first-frame image or one HD-map clip at a time).
    """

    use_feature_cache = True

    def __init__(
        self,
        checkpoint_path: str,
        latents_mean: list[float],
        latents_std: list[float],
        dtype: torch.dtype = torch.float32,
        base_dim: int = 96,
        z_dim: int = 16,
        pruning_rate: float = 0.75,
        temperal_downsample: tuple[bool, bool, bool] = (False, True, True),
    ) -> None:
        super().__init__()
        self.latents_mean = list(latents_mean)
        self.latents_std = list(latents_std)
        self._patch_size = 1
        with torch.device("meta"):
            self.encoder = Encoder3d(
                dim=base_dim,
                z_dim=z_dim * 2,
                pruning_rate=pruning_rate,
                temperal_downsample=temperal_downsample,
                in_channels=3,
            )
            self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(sd, Mapping) and "state_dict" in sd:
            sd = sd["state_dict"]
        # The checkpoint also carries decoder.* + conv2.* (decode side); drop
        # them. ``conv1`` here is the WanVAE quant conv (z_dim*2 -> z_dim*2).
        self.load_state_dict(sd, strict=False, assign=True)
        self.eval().requires_grad_(False)
        self.to(dtype=dtype)

        # Per-channel latent normalization buffers (consumed by callers that
        # need the raw mean / inv_std). ``persistent=False`` keeps them out of
        # state_dict (strict=False load ignores them anyway).
        self.register_buffer(
            "mean",
            torch.tensor(self.latents_mean, dtype=torch.float32).reshape(-1),
            persistent=False,
        )
        self.register_buffer(
            "inv_std",
            (1.0 / torch.tensor(self.latents_std, dtype=torch.float32)).reshape(-1),
            persistent=False,
        )

    def enable_tiling(self, *args, **kwargs) -> None:  # encode/decode-stage no-op
        return None

    def initialize_ar_encode_cache(self) -> WanVAEEncCache:
        """Allocate a fresh per-rollout VAE encode cache.

        For autoregressive per-chunk HD-map encoding, the causal conv left-
        context must persist across chunks (chunk 0 seeds, chunk 1+ continues
        from the accumulated state) — matching the FlashDreams streaming
        contract (one ``WanVAECache`` per rollout, fed to every AR step).
        Re-creating the cache each call instead re-seeds every chunk and
        leaves a ``len_t*tc-1``-frame tail that breaks ``time_conv``
        (kernel=3) at the deepest downsample stage.
        """
        return WanVAEEncCache()

    @torch.inference_mode()
    def encode(
        self,
        x: torch.Tensor,
        cache: WanVAEEncCache | None = None,
        is_first_chunk: bool = True,
    ) -> _LatentDist:
        """``[B, 3, T, H, W]`` pixels -> raw latent mean ``[B, z_dim, Tl, H/8, W/8]``.

        For autoregressive per-chunk HD-map encoding, pass a *persistent*
        ``cache`` (from :meth:`initialize_ar_encode_cache`) and set
        ``is_first_chunk=True`` only on chunk 0. The causal conv left-context
        then flows across chunks (FlashDreams streaming contract), avoiding
        the ``time_conv`` (kernel=3) underflow that a per-call fresh cache
        triggers on later chunks' short tails. When ``cache`` is ``None`` a
        fresh one-shot cache is allocated (single-call path).
        """
        x = x.to(self.conv1.weight.dtype)
        if cache is None:
            cache = WanVAEEncCache()
        state = cache.enc_state
        outs: list[torch.Tensor] = []
        if is_first_chunk:
            # 1-frame causal seed, then 4-frame body chunks + a short tail.
            outs.append(self.encoder(x[:, :, :1], state))
            x = x[:, :, 1:]
            self.encoder.normalize_state_for_body(state)
        else:
            # Continuation chunk: no re-seed; T must be a multiple of the
            # 4-frame window so no short tail reaches the deepest downsample.
            assert x.shape[2] % TEMPORAL_WINDOW == 0, (
                f"LightVAE continuation chunk requires T % {TEMPORAL_WINDOW} "
                f"== 0; got T={x.shape[2]}"
            )
        t = x.shape[2]
        body = (t // TEMPORAL_WINDOW) * TEMPORAL_WINDOW
        for i in range(0, body, TEMPORAL_WINDOW):
            outs.append(self.encoder(x[:, :, i : i + TEMPORAL_WINDOW], state))
        if body < t:
            outs.append(self.encoder(x[:, :, body:], state))
        mu, _log_var = self.conv1(torch.cat(outs, dim=2)).chunk(2, dim=1)
        return _LatentDist(mu)
