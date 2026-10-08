# SPDX-License-Identifier: Apache-2.0
#
# Adapters exposing the official Wan encoder API over the diffusers/transformers models the
# sglang loaders produce: image_encoder.visual takes list[torch.Tensor] of [C, T, H, W] and returns
# [B, 257, 1280]; vae.encode/decode fold in the Wan latent scaling.
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context


class WanAnimate2ImageEncoderAdapter:
    """Official Wan CLIP API over a ``CLIPVisionModel``: ``visual`` takes ``list[torch.Tensor]`` of ``[C, T, H, W]``
    in [-1, 1] and returns ``hidden_states[-2]`` as ``[B, 257, 1280]`` in the model dtype, the pre-final-LayerNorm
    slice the Wan-I2V ``postprocess_image`` consumes."""

    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        image_mean: tuple[float, float, float],
        image_std: tuple[float, float, float],
        hidden_state_index: int = -2,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.hidden_state_index = hidden_state_index
        self.device = device

    @classmethod
    def from_loaded(
        cls,
        model: nn.Module,
        image_processor: CLIPImageProcessor,
        device: torch.device | str | None = None,
    ) -> WanAnimate2ImageEncoderAdapter:
        """Build the adapter with image_size / mean / std from the checkpoint's
        ``CLIPImageProcessor`` (``crop_size`` is the post-resize crop the model sees)."""
        crop_h = int(image_processor.crop_size["height"])
        crop_w = int(image_processor.crop_size["width"])
        if crop_h != crop_w:
            raise ValueError(
                "WanAnimate2ImageEncoderAdapter expects a square CLIP crop_size; got "
                f"height={crop_h}, width={crop_w}."
            )
        return cls(
            model,
            image_size=crop_h,
            image_mean=tuple(image_processor.image_mean),
            image_std=tuple(image_processor.image_std),
            device=device,
        )

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return next(self.model.parameters()).device

    @torch.no_grad()
    def visual(self, videos: list[torch.Tensor]) -> torch.Tensor:
        # The caller holds the component through the residency manager, so the model is
        # already where it should be; inputs follow it. Runs in the model's own dtype with
        # autocast off: the fused patch-embed conv does not autocast an fp32 weight against
        # a bf16 input.
        device = self._resolve_device()
        model_dtype = next(self.model.parameters()).dtype
        size = (self.image_size, self.image_size)

        # Each video [C, T, H, W] becomes T images [T, C, image_size, image_size]; the images
        # of all videos are stacked into one batch [N, C, image_size, image_size], N = total
        # frame count.
        pixels = torch.cat(
            [
                F.interpolate(
                    video.transpose(0, 1).to(device=device, dtype=model_dtype),
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                )
                for video in videos
            ]
        )

        # [-1, 1] -> [0, 1] then Normalize(mean, std), in the model's own dtype.
        mean = torch.tensor(self.image_mean, device=device, dtype=pixels.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor(self.image_std, device=device, dtype=pixels.dtype).view(
            1, 3, 1, 1
        )
        pixels = pixels.mul(0.5).add(0.5).sub(mean).div(std)

        # Encode; return the hidden_states[-2] slice.
        with (
            torch.autocast(device_type=device.type, enabled=False),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            outputs = self.model(pixel_values=pixels, output_hidden_states=True)
        return outputs.hidden_states[self.hidden_state_index]


class WanAnimate2VaeAdapter(nn.Module):
    """Stage-local latent-scaling adapter over the pipeline's native ``AutoencoderKLWan``.
    ``encode`` takes ``list[torch.Tensor]`` (or ``[B, C, T, H, W]``) and returns ``list[torch.Tensor]`` of
    ``[16, T', H', W']`` fp32, ``(mode() - latents_mean) / latents_std``; ``decode`` inverts that and returns
    ``[B, C, T, H, W]`` fp32 in [-1, 1], which is why ``Wan_Animate_2_14B_Config.get_decode_scale_and_shift``
    returns ``(1.0, None)``."""

    # DecodingStage._can_use_parallel_decode may read this; decode is single-process.
    use_parallel_decode: bool = False

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        if not isinstance(vae, nn.Module):
            raise TypeError(
                "WanAnimate2VaeAdapter expects an nn.Module AutoencoderKLWan; got "
                f"{type(vae).__name__!r}."
            )
        self.vae = vae

        # Per-channel latent scaling (length z_dim), which the sglang AutoencoderKLWan
        # copies from its config onto the module. Plain (unregistered) tensors, reshaped
        # to broadcast over [B, C, T, H, W].
        self._latents_mean = torch.as_tensor(
            vae.latents_mean, dtype=torch.float32
        ).view(1, -1, 1, 1, 1)
        self._latents_std = torch.as_tensor(vae.latents_std, dtype=torch.float32).view(
            1, -1, 1, 1, 1
        )

    @staticmethod
    def _to_list_of_tensors(
        items: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        """Per-item tensors from either a list or a batched ``[B, ...]`` tensor; the stages
        pass lists, the shared decode path passes a batch."""
        if isinstance(items, (list, tuple)):
            return list(items)
        return [items[i] for i in range(items.shape[0])]

    def _mean_std_like(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._latents_mean.to(device=input.device, dtype=input.dtype),
            self._latents_std.to(device=input.device, dtype=input.dtype),
        )

    @torch.no_grad()
    def encode(
        self,
        videos: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        *args: Any,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for sample in self._to_list_of_tensors(videos):
            # Run fp32 with autocast off: the fused causal-conv path has no dtype guard, so
            # an outer autocast(bf16) would cast the conv input against the fp32 bias.
            # The sglang AutoencoderKLWan returns the DiagonalGaussianDistribution itself,
            # not a diffusers AutoencoderKLOutput.
            with torch.autocast(device_type=sample.device.type, enabled=False):
                posterior = self.vae.encode(sample.unsqueeze(0).float())

            # Deterministic mu (not a sampled draw).
            mu = posterior.mode().float()
            mean, std = self._mean_std_like(mu)
            z = (mu - mean) / std
            out.append(z.squeeze(0))
        return out

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        out: list[torch.Tensor] = []
        for sample in self._to_list_of_tensors(latents):
            z = sample.unsqueeze(0).float()
            mean, std = self._mean_std_like(z)
            z = z * std + mean

            # fp32, autocast OFF: see ``encode`` above. decode returns a plain tensor.
            with torch.autocast(device_type=z.device.type, enabled=False):
                decoded = self.vae.decode(z)
            out.append(decoded.float().clamp_(-1.0, 1.0).squeeze(0))

        # Re-stack to the batched [B, C, T, H, W] tensor the DenoisingStage expects.
        return torch.stack(out, dim=0)

    def forward(
        self,
        z: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.decode(z, *args, **kwargs)
