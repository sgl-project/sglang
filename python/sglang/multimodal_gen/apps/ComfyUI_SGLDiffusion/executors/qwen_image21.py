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


def _dit_patches(transformer_options) -> bool:
    opts = transformer_options or {}
    patches = opts.get("patches") or {}
    return bool(
        (opts.get("patches_replace") or {}).get("dit")
        or patches.get("post_input")
        or patches.get("single_block")
        or patches.get("attn1_patch")
    )


class QwenImage21Adapter(ComfyUIModelAdapter):
    model_types = ("qwen_image21",)
    pipeline_class_name = "QwenImage21Pipeline"

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
        if _dit_patches(transformer_options):
            raise NotImplementedError(
                "Qwen-Image 2.1 SGLD integrated mode does not run ComfyUI DiT "
                "patches (model patches, Fun-Control, attention hooks); use UNETLoader"
            )
        payload = {
            "image_slots": list(image_slots or []),
            "ref_latents": [r for r in (ref_latents or []) if r is not None],
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
        payload = packed.extra_req.get(COND_EXTRA_KEY)
        if payload is not None:
            req.extra = {**(req.extra or {}), COND_EXTRA_KEY: payload}

    def drop_cached_fields(self, packed: PackedForward) -> None:
        super().drop_cached_fields(packed)
        packed.extra_req.pop(COND_EXTRA_KEY, None)


class QwenImage21Executor(SGLDiffusionExecutor):
    adapter_cls = QwenImage21Adapter

    def __init__(self, generator, model_path, model, config):
        super().__init__(generator, model_path, model, config)
        self.current_patcher = None

    @classmethod
    def validate_sgld_options(cls, sgld_options: dict | None) -> None:
        options = sgld_options or {}
        if options.get("enable_cfg_parallel"):
            raise ValueError(
                "enable_cfg_parallel does not apply to Qwen-Image 2.1 in ComfyUI "
                "integrated mode: ComfyUI runs CFG itself, so every CFG rank would "
                "recompute the same DiT call. Use sp_degree=2 or tp_size=2."
            )

    def reset_prefix_cache(self, enabled):
        """Called by ComfyUI's QwenImage21 on pre_run / cleanup; the cache lives on the worker."""
