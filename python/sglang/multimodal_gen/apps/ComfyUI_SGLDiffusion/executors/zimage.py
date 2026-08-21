"""Z-Image (lumina2) adapter for the ComfyUI DiT-forward contract."""

import torch

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor


class ZImageAdapter(ComfyUIModelAdapter):
    model_types = ("lumina2",)
    pipeline_class_name = "ZImagePipeline"

    def pack(self, x, timestep, context, **kwargs) -> PackedForward:
        context = context.squeeze(0)
        return PackedForward(
            latents=x.unsqueeze(2),
            timesteps=timestep * 1000.0,
            prompt_embeds=[context],
            prompt_seq_lens=[[int(context.shape[0])]],
            height=x.shape[-2] * 8,
            width=x.shape[-1] * 8,
        )

    def unpack(self, noise_pred, packed, x):
        # SGLD returns 5D [B, C, T, H, W]; ComfyUI samples 4D [B, C, H, W].
        if noise_pred.ndim == 5:
            noise_pred = noise_pred.squeeze(2)
        return noise_pred.to(x.device)


class ZImageExecutor(SGLDiffusionExecutor):
    adapter_cls = ZImageAdapter
