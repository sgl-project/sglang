"""Flux adapter for the ComfyUI DiT-forward contract."""

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor


class FluxAdapter(ComfyUIModelAdapter):
    model_types = ("flux",)
    pipeline_class_name = "FluxPipeline"

    def pack(self, x, timestep, context, y=None, guidance=None, **kwargs) -> PackedForward:
        packed = self._pack_latents(x)
        t5_seq = int(context.shape[-2]) if context.ndim >= 2 else int(context.shape[0])
        clip_batch = int(y.shape[0]) if y is not None else 1
        return PackedForward(
            latents=packed,
            timesteps=timestep * 1000.0,
            prompt_embeds=[y, context],
            prompt_seq_lens=[[clip_batch], [t5_seq]],
            pooled_embeds=[y],
            height=x.shape[-2] * 8,
            width=x.shape[-1] * 8,
            guidance_scale=3.5,
            unpack_ctx={"height": x.shape[-2], "width": x.shape[-1], "channels": x.shape[1]},
        )

    def unpack(self, noise_pred, packed, x):
        ctx = packed.unpack_ctx
        return self._unpack_latents(
            noise_pred, ctx["height"], ctx["width"], ctx["channels"]
        ).to(x.device)

    @staticmethod
    def _unpack_latents(latents, height, width, channels):
        batch_size = latents.shape[0]
        latents = latents.view(batch_size, height // 2, width // 2, channels, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels, height, width)

    @staticmethod
    def _pack_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )


class FluxExecutor(SGLDiffusionExecutor):
    adapter_cls = FluxAdapter
