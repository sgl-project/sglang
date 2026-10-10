# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 / FLUX.2 Klein adapter for the ComfyUI DiT-forward contract.

ComfyUI's FLUX.2 latent is already 128-channel and patch size 1, so the
adapter only flattens it to tokens. It also builds the position ids here so a
ComfyUI batch of more than one row (positive and negative stacked) works: the
worker's own latent preparation assumes a single row.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    _prepare_image_ids,
    _prepare_latent_ids,
    flux2_pack_latents,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.flux2 import (
    FLUX2_PIPELINE,
    flux2_pipeline_name,
)

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor

# FLUX.2 VAE: 8x spatial compression, then 2x2 patchify folded into channels.
_PIXELS_PER_LATENT = 16
_DEFAULT_GUIDANCE = 4.0


def _guidance_scale(guidance) -> float:
    if guidance is None:
        return _DEFAULT_GUIDANCE
    if torch.is_tensor(guidance):
        return float(guidance.detach().reshape(-1)[0].item())
    return float(guidance)


def _pack_reference_latents(ref_latents, batch_size: int) -> torch.Tensor:
    packed = [flux2_pack_latents(ref) for ref in ref_latents]
    packed = [p.expand(batch_size, -1, -1) if p.shape[0] == 1 else p for p in packed]
    return torch.cat(packed, dim=1)


class Flux2Adapter(ComfyUIModelAdapter):
    model_types = ("flux2",)
    pipeline_class_name = FLUX2_PIPELINE

    @classmethod
    def pipeline_class_for(cls, model_config) -> str:
        unet_config = model_config.unet_config
        return flux2_pipeline_name(
            {
                "joint_dim": unet_config["context_in_dim"],
                "guidance_embeds": unet_config["guidance_embed"],
            }
        )

    def pack(
        self,
        x,
        timestep,
        context,
        y=None,
        guidance=None,
        ref_latents=None,
        attention_mask=None,
        **kwargs,
    ) -> PackedForward:
        if attention_mask is not None:
            raise NotImplementedError(
                "FLUX.2 attention masks are not supported in SGLang integrated mode"
            )
        batch_size, _, height, width = x.shape
        # The worker joins these with device tensors, so build them on x's device.
        extra_req = {"latent_ids": _prepare_latent_ids(x).to(x.device)}
        if ref_latents:
            extra_req["image_latent"] = _pack_reference_latents(ref_latents, batch_size)
            extra_req["condition_image_latent_ids"] = _prepare_image_ids(
                [ref[:1] for ref in ref_latents]
            ).to(x.device)
        return PackedForward(
            latents=flux2_pack_latents(x),
            # Rows share one sigma. ComfyUI hands FLUX 0..1; the DiT wants 0..1000.
            timesteps=timestep.reshape(-1)[:1] * 1000.0,
            prompt_embeds=[context],
            prompt_seq_lens=[[int(context.shape[1])] * batch_size],
            height=height * _PIXELS_PER_LATENT,
            width=width * _PIXELS_PER_LATENT,
            guidance_scale=_guidance_scale(guidance),
            extra_req=extra_req,
            unpack_ctx={"height": height, "width": width},
        )

    def unpack(self, noise_pred, packed, x):
        ctx = packed.unpack_ctx
        batch_size, _, channels = noise_pred.shape
        out = noise_pred.permute(0, 2, 1).reshape(
            batch_size, channels, ctx["height"], ctx["width"]
        )
        return out.to(device=x.device, dtype=x.dtype)

    def drop_cached_fields(self, packed: PackedForward) -> None:
        super().drop_cached_fields(packed)
        # The worker caches these with image_latent, so they come back as a pair.
        packed.extra_req.pop("condition_image_latent_ids", None)


class Flux2Executor(SGLDiffusionExecutor):
    adapter_cls = Flux2Adapter
