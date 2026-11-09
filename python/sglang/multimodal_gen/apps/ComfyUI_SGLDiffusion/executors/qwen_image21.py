# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 adapter for the ComfyUI DiT-forward contract.

ComfyUI calls ``diffusion_model(x, timestep, context, ref_latents=...,
image_slots=..., transformer_options=...)`` with ``x`` as ``[B, 64, H, W]``
latents (no patchify).
"""

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor

# Must match COMFYUI_COND_EXTRA_KEY in the worker's qwen_image21 stages; not
# imported so the ComfyUI process does not load the denoising stack.
COND_EXTRA_KEY = "qwen21_cond"
# Spatial downscale of the Qwen-Image 2.1 VAE.
_LATENT_SCALE = 16


class QwenImage21Adapter(ComfyUIModelAdapter):
    model_types = ("qwen_image21",)
    pipeline_class_name = "QwenImage21Pipeline"
    applied_conditioning = ("ref_latents",)
    cached_extra_keys = ComfyUIModelAdapter.cached_extra_keys + (COND_EXTRA_KEY,)

    def pack(
        self,
        x,
        timestep,
        context,
        ref_latents=None,
        image_slots=None,
        transformer_options=None,
        **kwargs,
    ) -> PackedForward:
        sigmas = (transformer_options or {}).get("sample_sigmas")
        final_sigma = None
        if sigmas is not None and (sigmas > 0).any():
            final_sigma = float(sigmas[sigmas > 0].min())
        payload = {
            "image_slots": list(image_slots or []),
            "ref_latents": list(ref_latents or []),
            # The worker frees this cond's prefix K/V cache after this sigma.
            "final_sigma": final_sigma,
        }
        return PackedForward(
            latents=x,
            timesteps=timestep.reshape(-1).float() * 1000.0,
            prompt_embeds=[context],
            height=int(x.shape[-2]) * _LATENT_SCALE,
            width=int(x.shape[-1]) * _LATENT_SCALE,
            extra_req={COND_EXTRA_KEY: payload},
        )

    def unpack(self, noise_pred, packed, x):
        # The worker returns the native token layout [B, H*W, 64].
        return noise_pred.transpose(1, 2).reshape(x.shape).to(x.device)

    def fill_req(self, req, packed: PackedForward) -> None:
        super().fill_req(req, packed)
        if COND_EXTRA_KEY in packed.extra_req:
            req.extra = {**req.extra, COND_EXTRA_KEY: packed.extra_req[COND_EXTRA_KEY]}


class QwenImage21Executor(SGLDiffusionExecutor):
    adapter_cls = QwenImage21Adapter

    def reset_prefix_cache(self, enabled):
        """Called by ComfyUI's QwenImage21 on pre_run / cleanup; the cache lives on the worker."""
