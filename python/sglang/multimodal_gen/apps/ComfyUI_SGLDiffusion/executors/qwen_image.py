"""Qwen-Image adapters for the ComfyUI DiT-forward contract."""

import comfy.ldm.common_dit

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor


class QwenImageAdapter(ComfyUIModelAdapter):
    model_types = ("qwen_image",)
    pipeline_class_name = "QwenImagePipeline"
    patch_size = 2

    def pack(self, x, timestep, context, **kwargs) -> PackedForward:
        latents, orig_shape = self._pack_latents(x)
        return PackedForward(
            latents=latents,
            timesteps=timestep * 1000.0,
            prompt_embeds=[context],
            height=orig_shape[-2] * 8,
            width=orig_shape[-1] * 8,
            unpack_ctx={"num_embeds": latents.shape[1], "orig_shape": orig_shape, "x": x},
        )

    def unpack(self, noise_pred, packed, x):
        ctx = packed.unpack_ctx
        return self._unpack_latents(
            noise_pred, ctx["num_embeds"], ctx["orig_shape"], ctx["x"]
        )

    def _pack_latents(self, x):
        latents = comfy.ldm.common_dit.pad_to_patch_size(
            x, (1, self.patch_size, self.patch_size)
        )
        orig_shape = latents.shape
        latents = latents.view(
            orig_shape[0],
            orig_shape[1],
            orig_shape[-3],
            orig_shape[-2] // 2,
            2,
            orig_shape[-1] // 2,
            2,
        )
        latents = latents.permute(0, 2, 3, 5, 1, 4, 6)
        latents = latents.reshape(
            orig_shape[0],
            orig_shape[-3] * (orig_shape[-2] // 2) * (orig_shape[-1] // 2),
            orig_shape[1] * 4,
        )
        return latents, orig_shape

    @staticmethod
    def _unpack_latents(latents, num_embeds, orig_shape, x):
        latents = latents[:, :num_embeds].view(
            orig_shape[0],
            orig_shape[-3],
            orig_shape[-2] // 2,
            orig_shape[-1] // 2,
            orig_shape[1],
            2,
            2,
        )
        latents = latents.permute(0, 4, 1, 2, 5, 3, 6)
        return latents.reshape(orig_shape)[:, :, :, : x.shape[-2], : x.shape[-1]]


class QwenImageEditAdapter(QwenImageAdapter):
    model_types = ("qwen_image_edit",)
    pipeline_class_name = "QwenImageEditPlusPipeline"

    def pack(
        self,
        x,
        timestep,
        context,
        attention_mask=None,
        ref_latents=None,
        additional_t_cond=None,
        transformer_options=None,
        **kwargs,
    ) -> PackedForward:
        packed = super().pack(x, timestep, context, **kwargs)
        if ref_latents:
            pack_ref, orig_ref_shape = self._pack_latents(ref_latents[0])
            packed.extra_req["image_latent"] = pack_ref
            packed.extra_req["vae_image_sizes"] = [
                (orig_ref_shape[-1], orig_ref_shape[-2])
            ]
        return packed


class QwenImageExecutor(SGLDiffusionExecutor):
    adapter_cls = QwenImageAdapter


class QwenImageEditExecutor(SGLDiffusionExecutor):
    adapter_cls = QwenImageEditAdapter
