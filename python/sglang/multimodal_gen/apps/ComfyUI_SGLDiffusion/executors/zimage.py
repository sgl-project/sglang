"""Z-Image (lumina2) adapter for the ComfyUI DiT-forward contract."""

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor


class ZImageAdapter(ComfyUIModelAdapter):
    model_types = ("lumina2",)
    pipeline_class_name = "ZImagePipeline"

    def pack(
        self, x, timestep, context, siglip_feats=None, ref_contexts=None, **kwargs
    ) -> PackedForward:
        # ComfyUI's Lumina2 always sends num_tokens (derived from the text
        # context) and an attention_mask that native ComfyUI itself discards
        # in patchify_and_embed, so dropping those two matches native
        # behavior exactly. siglip_feats/ref_contexts (Z-Image Omni's
        # multi-reference inputs) are different: the worker's ZImagePipeline
        # has no code path for them, so forwarding would silently run as a
        # plain text-to-image call instead of the requested reference edit.
        if siglip_feats or ref_contexts:
            raise ValueError(
                "ZImageAdapter does not support Z-Image Omni reference "
                "conditioning (siglip_feats / ref_contexts); running this "
                "workflow would silently drop the reference images."
            )
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
