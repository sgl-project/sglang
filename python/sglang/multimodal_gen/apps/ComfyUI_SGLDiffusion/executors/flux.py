"""Flux adapter for the ComfyUI DiT-forward contract."""

import torch

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor


def _flux_guidance_scale(guidance) -> float:
    if guidance is None:
        return 3.5
    if torch.is_tensor(guidance):
        return float(guidance.detach().reshape(-1)[0].item())
    return float(guidance)


def _pad_to_patch_size(latents: torch.Tensor, patch_size: int) -> torch.Tensor:
    # Equivalent to comfy.ldm.common_dit.pad_to_patch_size, inlined so this
    # module keeps loading without ComfyUI installed (see executors/__init__.py).
    pad_h = (patch_size - latents.shape[-2] % patch_size) % patch_size
    pad_w = (patch_size - latents.shape[-1] % patch_size) % patch_size
    return torch.nn.functional.pad(latents, (0, pad_w, 0, pad_h), mode="circular")


class FluxAdapter(ComfyUIModelAdapter):
    model_types = ("flux",)
    pipeline_class_name = "FluxPipeline"
    patch_size = 2

    def pack(
        self, x, timestep, context, y=None, guidance=None, **kwargs
    ) -> PackedForward:
        packed, padded_height, padded_width = self._pack_latents(x)
        t5_seq = int(context.shape[-2]) if context.ndim >= 2 else int(context.shape[0])
        clip_batch = int(y.shape[0]) if y is not None else 1
        return PackedForward(
            latents=packed,
            timesteps=timestep * 1000.0,
            prompt_embeds=[y, context],
            prompt_seq_lens=[[clip_batch], [t5_seq]],
            pooled_embeds=[y],
            height=padded_height * 8,
            width=padded_width * 8,
            guidance_scale=_flux_guidance_scale(guidance),
            unpack_ctx={
                "height": padded_height,
                "width": padded_width,
                "channels": x.shape[1],
            },
        )

    def unpack(self, noise_pred, packed, x):
        ctx = packed.unpack_ctx
        latents = self._unpack_latents(
            noise_pred, ctx["height"], ctx["width"], ctx["channels"]
        )
        return latents[:, :, : x.shape[-2], : x.shape[-1]].to(x.device)

    @staticmethod
    def _unpack_latents(latents, height, width, channels):
        batch_size = latents.shape[0]
        latents = latents.view(batch_size, height // 2, width // 2, channels, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels, height, width)

    @classmethod
    def _pack_latents(cls, latents):
        latents = _pad_to_patch_size(latents, cls.patch_size)
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )
        return latents, height, width


class FluxExecutor(SGLDiffusionExecutor):
    adapter_cls = FluxAdapter
