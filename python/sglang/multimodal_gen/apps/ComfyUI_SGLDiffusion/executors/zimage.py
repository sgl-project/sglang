"""Z-Image (lumina2) adapter for the ComfyUI DiT-forward contract."""

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor


class ZImageAdapter(ComfyUIModelAdapter):
    model_types = ("lumina2",)
    pipeline_class_name = "ZImagePipeline"
    supports_native_batching = True

    def pack(
        self, x, timestep, context, siglip_feats=None, ref_contexts=None, **kwargs
    ) -> PackedForward:
        # The worker has no path for Z-Image Omni's reference conditioning.
        if siglip_feats or ref_contexts:
            raise ValueError(
                "ZImageAdapter does not support Z-Image Omni reference "
                "conditioning (siglip_feats / ref_contexts); running this "
                "workflow would silently drop the reference images."
            )
        batch = int(x.shape[0])
        if batch == 1:
            context = context.squeeze(0)
        # ComfyUI gives every row of a batch the same context length.
        return PackedForward(
            latents=x.unsqueeze(2),
            timesteps=timestep * 1000.0,
            prompt_embeds=[context],
            prompt_seq_lens=[[int(context.shape[-2])] * batch],
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
