# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    DecodingStage,
    _ensure_tensor_decode_output,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


class RealtimeVAEState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.image_latent: torch.Tensor | None = None

    def dispose(self):
        super().dispose()
        self.image_latent = None


class RealtimeImageVAEEncodingStage(ImageVAEEncodingStage):
    """Reuse the first chunk's conditioning image latent across a realtime session."""

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        state = None
        if batch.session is not None:
            state = batch.session.get_or_create_state(RealtimeVAEState)
            if batch.block_idx == 0:
                state.image_latent = None
            elif state.image_latent is not None:
                batch.image_latent = state.image_latent
                return batch

        if batch.condition_image is None:
            if state is not None and state.image_latent is not None:
                batch.image_latent = state.image_latent
            return batch

        batch = super().forward(batch, server_args)

        if state is not None and batch.image_latent is not None:
            state.image_latent = batch.image_latent
        return batch


class CausalVaeDecodingStage(DecodingStage):
    """Decode realtime chunks with a persistent causal VAE cache when available."""

    @torch.no_grad()
    def decode_causal(
        self, latents: torch.Tensor, server_args: ServerArgs
    ) -> torch.Tensor:
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        self.vae = self.vae.to(device=get_local_torch_device(), dtype=vae_dtype)
        latents = latents.to(get_local_torch_device())
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        latents = self.scale_and_shift(latents, server_args)
        latents = server_args.pipeline_config.preprocess_decoding(
            latents, server_args, vae=self.vae
        )

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass

            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)

            decode_fn = getattr(self.vae, "causal_decode", None)
            if decode_fn is None:
                decode_output = self.vae.decode(latents)
            else:
                decode_output = decode_fn(latents)
            image = _ensure_tensor_decode_output(decode_output)

        return (image / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        if batch.session is None:
            return super().forward(batch, server_args)

        self.load_model()

        reset_causal_state = getattr(self.vae, "reset_causal_decode_state", None)
        if batch.block_idx == 0 and callable(reset_causal_state):
            reset_causal_state()

        frames = self.decode_causal(batch.latents, server_args)
        frames = server_args.pipeline_config.post_decoding(frames, server_args)

        return OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=None,
            metrics=batch.metrics,
            noise_pred=None,
        )
