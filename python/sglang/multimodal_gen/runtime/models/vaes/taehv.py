# SPDX-License-Identifier: Apache-2.0
"""LightTAE (TAEHV) tiny streaming video decoder for OmniDreams.

Ported from FlashDreams ``recipes/taehv/impl.py`` + ``checkpoint.py`` +
``recipes/taehv/__init__.py`` (``TeahvVAEDecoder``), trimmed to the OmniDreams
single-view ``wan21`` variant (ReLU, patch_size=1, latent_channels=16) and the
SGLang single-pass decode contract.

Swaps the Wan 2.1 VAE *decode* for the LightX2V TAEHV tiny decoder, trading
quality (FVD ~24.8 -> ~45.4, paper Table 5) for a large decode speedup. Encode
is unaffected (still the Wan VAE, unless LightVAE is also enabled). Gated
behind ``OmniDreamsPipelineConfig.use_light_tae`` (default off).

The public entry point is :class:`LightTAEDecoder`, which exposes a
``decode(latents) -> video`` matching the contract the SGLang ``DecodingStage``
expects (input ``[B, C, F, H, W]``, output ``[B, C, F_out, H, W]`` in ``[-1, 1]``).
Because TAEHV applies its OWN per-channel latent mean/std internally, the
LightTAE decode path must NOT pre-apply the Wan ``scale_and_shift`` (see
``OmniDreamsLightTAEDecodingStage``).

Checkpoint: ``lighttaew2_1.pth`` (lightx2v/Autoencoders). Channels
``(256, 128, 64, 64)``; the legacy flat ``decoder.<i>.*`` keys are remapped to
``decoder.blocks.<i>.*`` and the oversize stride-2 ``TGrow`` weight at idx 7 is
clipped to the stride-1 slice the live model expects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits.omnidreams_cuda_graph import (
    CUDAGraphWrapper,
    set_or_copy,
)

# --------------------------------------------------------------------------- #
# Checkpoint state-dict transforms (ported from recipes/taehv/checkpoint.py)   #
# --------------------------------------------------------------------------- #
StateDictTransform = Callable[[Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]


def legacy_to_blocks_keys(
    sd: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Re-key legacy ``decoder.<i>.*`` weights to ``decoder.blocks.<i>.*``.

    The current :class:`Decoder` wraps its ``Sequential`` in a ``blocks``
    attribute; older checkpoints flatten to ``decoder.<idx>.*``. Keys already
    under ``decoder.blocks.`` (and keys outside ``decoder.``) pass through.
    """
    return {
        (
            k.replace("decoder.", "decoder.blocks.", 1)
            if k.startswith("decoder.") and not k.startswith("decoder.blocks.")
            else k
        ): v
        for k, v in sd.items()
    }


def truncate_oversize_tgrow_weights(
    *,
    channels: tuple[int, int, int, int],
    decoder_time_upscale: tuple[bool, bool] = (True, True),
) -> StateDictTransform:
    """Clip oversize ``TGrow`` ``conv.weight`` to the model's stride.

    Some shipped TAEHV checkpoints store the stride=2 ``TGrow`` weight even when
    the target model is built with stride=1 at that position; keep only the
    last-timestep slice (matching the model's expected ``conv.weight.shape[0]``).
    The Sequential indices (7, 13, 19) match :class:`Decoder`'s body.
    """
    expected_channels: dict[int, int] = {
        7: channels[0] * 1,
        13: channels[1] * (2 if decoder_time_upscale[0] else 1),
        19: channels[2] * (2 if decoder_time_upscale[1] else 1),
    }

    def transform(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(sd)
        for idx, expected in expected_channels.items():
            key = f"decoder.blocks.{idx}.conv.weight"
            if key in out and out[key].shape[0] > expected:
                out[key] = out[key][-expected:]
        return out

    return transform


def compose(*transforms: StateDictTransform) -> StateDictTransform:
    """Compose transforms left-to-right: ``compose(f, g)(sd) == g(f(sd))``."""

    def composed(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = dict(sd)
        for t in transforms:
            out = t(out)
        return out

    return composed


_LIGHTTAE_CHANNELS: tuple[int, int, int, int] = (256, 128, 64, 64)
lighttae_state_dict_transform: StateDictTransform = compose(
    legacy_to_blocks_keys,
    truncate_oversize_tgrow_weights(channels=_LIGHTTAE_CHANNELS),
)


# --------------------------------------------------------------------------- #
# TAEHV network (ported from recipes/taehv/impl.py)                            #
# --------------------------------------------------------------------------- #
@dataclass
class TAEHVCache:
    """Streaming decoder cache; one slot per ``MemBlock`` keyed by ``id(module)``.

    Each slot holds the last input frame of the previous chunk (rolled-in left
    context). Slot storage addresses are stable after the first chunk so
    CUDA-graph replay (if enabled) can write through them in place.
    """

    dec_state: Dict[int, torch.Tensor] = field(default_factory=dict)


def _conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    """Soft saturating clamp ``tanh(x/3) * 3``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Residual block with a 1-frame temporal-left memory slot."""

    def __init__(self, n_in: int, n_out: int, act_func: nn.Module):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out),
            act_func,
            _conv(n_out, n_out),
            act_func,
            _conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.act = act_func

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

    def cache_step(
        self, x: torch.Tensor, state: Dict[int, torch.Tensor], batch: int
    ) -> torch.Tensor:
        """Apply with streaming left-context: prepend the previous chunk's last
        frame to ``x``, run the conv stack, save the new last frame.

        ``state[id(self)]`` must already exist (see
        :meth:`Decoder.initialize_state`). Zero ``prev`` makes the first chunk
        bit-equivalent to the legacy ``F.pad`` zero-pad path.
        """
        key = id(self)
        bt, c, h, w = x.shape
        t = bt // batch
        x5 = x.view(batch, t, c, h, w)
        prev = state[key]
        past = torch.cat([prev, x5[:, :-1]], dim=1)
        out = self.forward(x, past.reshape(bt, c, h, w))
        set_or_copy(state, key, x5[:, -1:])
        return out


class TGrow(nn.Module):
    """Temporal upsample by ``stride`` (channel-expand + reshape; stateless)."""

    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _NT, C, H, W = x.shape
        return self.conv(x).reshape(-1, C, H, W)


class Decoder(nn.Module):
    """TAEHV decoder body.

    Input: ``[B, T, C_z, H, W]`` latent. Output: raw frames
    ``[B, T_out, C_img * patch**2, H_out, W_out]`` (clamp / pixel-shuffle / trim
    happen in :meth:`TAEHV.decode`).
    """

    def __init__(
        self,
        n_f: tuple[int, int, int, int],
        latent_channels: int,
        image_channels: int,
        patch_size: int,
        decoder_time_upscale: tuple[bool, bool],
        decoder_space_upscale: tuple[bool, bool, bool],
        act_func: nn.Module,
    ):
        super().__init__()
        # Layer indices must match the legacy nn.Sequential so checkpoint keys
        # (``decoder.<idx>.<param>``) load unchanged.
        self.blocks = nn.Sequential(
            Clamp(),
            _conv(latent_channels, n_f[0]),
            act_func,
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            _conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            _conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            _conv(n_f[2], n_f[3], bias=False),
            act_func,
            _conv(n_f[3], image_channels * patch_size**2),
        )

    def forward(
        self, z: torch.Tensor, state: Dict[int, torch.Tensor], batch: int
    ) -> torch.Tensor:
        b, t, c, h, w = z.shape
        x = z.reshape(b * t, c, h, w)
        for blk in self.blocks:
            if isinstance(blk, MemBlock):
                x = blk.cache_step(x, state, batch)
            else:
                x = blk(x)
        bt, c_out, h_out, w_out = x.shape
        return x.reshape(b, bt // b, c_out, h_out, w_out)

    @torch.no_grad()
    def initialize_state(
        self,
        z_shape: tuple[int, int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        state: Dict[int, torch.Tensor],
    ) -> None:
        """Populate ``state`` with one zero ``[B, 1, C_i, H_i, W_i]`` per MemBlock.

        Walks ``self.blocks`` once with a synthetic zero input to derive each
        MemBlock's input shape (conv outputs discarded). Zero ``prev`` makes the
        first chunk bit-equivalent to the legacy zero-pad path.
        """
        batch, t, c_z, h_z, w_z = z_shape
        x = torch.zeros(batch * t, c_z, h_z, w_z, dtype=dtype, device=device)
        for blk in self.blocks:
            if isinstance(blk, MemBlock):
                bt_x, c_x, h_x, w_x = x.shape
                t_x = bt_x // batch
                state[id(blk)] = torch.zeros(
                    batch, 1, c_x, h_x, w_x, dtype=dtype, device=device
                )
                past = (
                    state[id(blk)]
                    .expand(-1, t_x, -1, -1, -1)
                    .reshape(bt_x, c_x, h_x, w_x)
                )
                x = blk.forward(x, past)
            else:
                x = blk(x)


class TAEHV(nn.Module):
    """TAEHV streaming decode-only network (OmniDreams ``wan21`` variant).

    Loads a TAEHV checkpoint and exposes :meth:`decode`. Encoder weights in the
    checkpoint are silently dropped (``strict=False``).
    """

    TEMPORAL_COMPRESSION_RATIO = 4
    SPATIAL_COMPRESSION_RATIO = 8

    decoder: Decoder

    def __init__(
        self,
        checkpoint_path: str | None = None,
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
        patch_size: int = 1,
        latent_channels: int = 16,
        channels: tuple[int, int, int, int] = _LIGHTTAE_CHANNELS,
        clamp_output: bool = True,
        use_cuda_graph: bool = False,
        use_compile: bool = False,
        warmup_iters: int = 2,
        state_dict_transform: StateDictTransform | None = lighttae_state_dict_transform,
    ):
        super().__init__()
        act_func = nn.ReLU(inplace=True)
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        self.channels = channels
        self.clamp_output = clamp_output
        # Frames dropped from the front of the first chunk output
        # (legacy 2 ** sum(time_upscale) - 1 formula).
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1

        # Build on meta so only the checkpoint allocates real memory.
        with torch.device("meta"):
            self.decoder = Decoder(
                n_f=channels,
                latent_channels=latent_channels,
                image_channels=self.image_channels,
                patch_size=patch_size,
                decoder_time_upscale=decoder_time_upscale,
                decoder_space_upscale=decoder_space_upscale,
                act_func=act_func,
            )

        self._use_cuda_graph = use_cuda_graph
        self._use_compile = use_compile
        self._warmup_iters = warmup_iters
        self._decoder_wrapper: CUDAGraphWrapper | None = None

        if checkpoint_path is not None:
            self.load_from_checkpoint(
                checkpoint_path, state_dict_transform=state_dict_transform
            )

    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        state_dict_transform: StateDictTransform | None = lighttae_state_dict_transform,
    ) -> None:
        """Load weights (remapped) into the meta-built decoder and wire decode."""
        if state_dict_transform is None:
            state_dict_transform = lighttae_state_dict_transform
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(sd, Mapping) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = state_dict_transform(sd)
        # assign=True: meta params become the checkpoint tensors directly;
        # strict=False: silently drop encoder-only weights.
        self.load_state_dict(sd, strict=False, assign=True)
        self.eval().requires_grad_(False)

        if self._use_compile:
            self.decoder = torch.compile(
                self.decoder, mode="max-autotune-no-cudagraphs"
            )
        self._decoder_wrapper = (
            CUDAGraphWrapper(self.decoder, warmup_iters=self._warmup_iters)
            if self._use_cuda_graph
            else None
        )

    @property
    def _decoder_call(self) -> Callable[..., torch.Tensor]:
        return (
            self._decoder_wrapper if self._decoder_wrapper is not None else self.decoder
        )

    def prepare_cache(self) -> TAEHVCache:
        if self._use_cuda_graph and self._decoder_wrapper is not None:
            self._decoder_wrapper.reset()
        return TAEHVCache()

    @torch.inference_mode()
    def decode(
        self,
        z: torch.Tensor,
        cache: Optional[TAEHVCache] = None,
        **_: object,
    ) -> torch.Tensor:
        """Streaming decode of an ``[N, T, C_z, H, W]`` latent -> ``[N, T_out, C_img, H, W]``.

        First call (empty cache) runs eagerly and trims the leading
        ``frames_to_trim`` frames; same-shape body chunks replay the captured
        graph thereafter (only when ``use_cuda_graph``).
        """
        if cache is None:
            cache = self.prepare_cache()
        state = cache.dec_state
        first_decode = not state
        if first_decode:
            b, t, c_z, h_z, w_z = z.shape
            self.decoder.initialize_state(
                (b, t, c_z, h_z, w_z), z.dtype, z.device, state
            )
        if self._use_cuda_graph and self._decoder_wrapper is not None:
            decoder = (
                self._decoder_wrapper.drain if first_decode else self._decoder_call
            )
        else:
            decoder = self.decoder

        b = z.shape[0]
        x = decoder(z, state, b)
        if self.clamp_output:
            x = x.clamp(0, 1)
        if self.patch_size > 1:
            n, t, c, h, w = x.shape
            x = F.pixel_shuffle(x.reshape(n * t, c, h, w), self.patch_size)
            x = x.reshape(n, t, x.shape[1], x.shape[2], x.shape[3])
        if first_decode:
            x = x[:, self.frames_to_trim :]
        return x


# --------------------------------------------------------------------------- #
# SGLang decode adapter (ported from TeahvVAEDecoder)                          #
# --------------------------------------------------------------------------- #
# Per-channel latent mean/std the ``lighttae`` checkpoint was trained against
# (its own scaling, distinct from the Wan 2.1 latents_mean/std). TAEHV consumes
# ``z * std + mean``; do NOT also apply the Wan scale_and_shift upstream.
_LIGHTTAE_MEAN: tuple[float, ...] = (
    -0.7571, -0.7089, -0.9113, 0.1075,
    -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632,
    -0.1922, -0.9497, 0.2503, -0.2921,
)  # fmt: skip
_LIGHTTAE_STD: tuple[float, ...] = (
    2.8184, 1.4541, 2.3275, 2.6558,
    1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
)  # fmt: skip


class LightTAEDecoder(nn.Module):
    """SGLang-facing LightTAE decode wrapper (replaces the Wan VAE decode).

    Exposes ``decode(latents)`` with the contract the :class:`DecodingStage`
    expects: input ``[B, C, F, H, W]`` (channels-first, the SGLang latent
    layout), output ``[B, C_img, F_out, H, W]`` in ``[-1, 1]``. The DiT-space
    latent is un-normalized with the LightTAE per-channel mean/std internally,
    so the decode stage MUST skip the Wan ``scale_and_shift`` (see
    ``OmniDreamsLightTAEDecodingStage``).
    """

    # Satisfies the OmniDreams denoising-stage VAE guard if ever passed there.
    use_feature_cache = True

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype = torch.bfloat16,
        channels: tuple[int, int, int, int] = _LIGHTTAE_CHANNELS,
        use_cuda_graph: bool = False,
        use_compile: bool = False,
    ) -> None:
        super().__init__()
        self.taehv = TAEHV(
            checkpoint_path=checkpoint_path,
            channels=channels,
            use_cuda_graph=use_cuda_graph,
            use_compile=use_compile,
        ).to(dtype=dtype)
        self.register_buffer(
            "mean",
            torch.tensor(_LIGHTTAE_MEAN, dtype=dtype).view(1, 1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(_LIGHTTAE_STD, dtype=dtype).view(1, 1, -1, 1, 1),
            persistent=False,
        )

    def enable_tiling(self, *args, **kwargs) -> None:  # decode-stage no-op hook
        return None

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # SGLang latent [B, C, F, H, W] -> TAEHV [B, T, C, H, W].
        z = latents.permute(0, 2, 1, 3, 4).contiguous()
        z = z.to(self.std.dtype)
        z = z * self.std + self.mean
        # [-> [0, 1]] then map to [-1, 1] for the stage's /2+0.5 de-normalize.
        x = self.taehv.decode(z, cache=self.taehv.prepare_cache())
        x = x.mul(2).sub(1)
        # [B, T, C_img, H, W] -> [B, C_img, F_out, H, W].
        return x.permute(0, 2, 1, 3, 4).contiguous()
