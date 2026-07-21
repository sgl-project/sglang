# SPDX-License-Identifier: Apache-2.0

import os

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    unpatchify as wan_unpatchify,
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
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


class RealtimeVAEState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.image_latent: torch.Tensor | None = None

    def dispose(self):
        super().dispose()
        self.image_latent = None


class RealtimeVAEDecodeState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.reset_causal_decode_state = None
        self.taehv_streaming_decoder = None
        self.taehv_output_queue: list[torch.Tensor] = []

    def dispose(self):
        reset_causal_decode_state = self.reset_causal_decode_state
        self.reset_causal_decode_state = None
        if callable(reset_causal_decode_state):
            reset_causal_decode_state()
        self.taehv_streaming_decoder = None
        self.taehv_output_queue.clear()


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

    TAEHV_CHECKPOINT_ENV = "SGLANG_REALTIME_TAEHV_CHECKPOINT_PATH"

    def _taehv_checkpoint_path(self) -> str | None:
        if not hasattr(self, "_cached_taehv_checkpoint_path"):
            value = os.environ.get(self.TAEHV_CHECKPOINT_ENV)
            self._cached_taehv_checkpoint_path = (
                value.strip() if value and value.strip() else None
            )
        return self._cached_taehv_checkpoint_path

    @staticmethod
    def _supports_wan_decoder_cache(vae) -> bool:
        return all(
            hasattr(vae, attr)
            for attr in (
                "clear_cache",
                "post_quant_conv",
                "decoder",
                "_feat_map",
                "_conv_idx",
            )
        )

    def _get_causal_decode_reset_fn(self):
        reset_causal_state = getattr(self.vae, "reset_causal_decode_state", None)
        if callable(reset_causal_state):
            return reset_causal_state
        if self._supports_wan_decoder_cache(self.vae):
            return self.vae.clear_cache
        return None

    def _decode_wan_with_persistent_cache(
        self,
        latents: torch.Tensor,
        *,
        first_chunk: bool,
    ) -> torch.Tensor:
        x = self.vae.post_quant_conv(latents)
        decoded_frames = []
        for frame_idx in range(x.shape[2]):
            self.vae._conv_idx = [0]
            decoded = self.vae.decoder(
                x[:, :, frame_idx : frame_idx + 1],
                feat_cache=self.vae._feat_map,
                feat_idx=self.vae._conv_idx,
                first_chunk=first_chunk and frame_idx == 0,
            )
            decoded_frames.append(decoded)

        image = torch.cat(decoded_frames, dim=2)
        if getattr(self.vae.config, "patch_size", None) is not None:
            image = wan_unpatchify(image, patch_size=self.vae.config.patch_size)
        return image.clamp(-1.0, 1.0)

    @torch.no_grad()
    def decode_causal(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
        *,
        first_chunk: bool,
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
            if callable(decode_fn):
                decode_output = decode_fn(latents)
                image = _ensure_tensor_decode_output(decode_output)
            elif self._supports_wan_decoder_cache(self.vae):
                image = self._decode_wan_with_persistent_cache(
                    latents,
                    first_chunk=first_chunk,
                )
            else:
                decode_output = self.vae.decode(latents)
                image = _ensure_tensor_decode_output(decode_output)

        return (image / 2 + 0.5).clamp(0, 1)

    def _get_or_create_taehv_streaming_decoder(
        self,
        decode_state: RealtimeVAEDecodeState,
        server_args: ServerArgs,
    ):
        if decode_state.taehv_streaming_decoder is not None:
            return decode_state.taehv_streaming_decoder

        if not hasattr(self, "_taehv_model"):
            checkpoint_path = self._taehv_checkpoint_path()
            if checkpoint_path is None:
                raise RuntimeError("TAEHV checkpoint path is not configured")

            try:
                from taehv import TAEHV
            except ImportError as exc:
                raise RuntimeError(
                    "TAEHV realtime decode requires the `taehv` package. "
                    "Install it and set SGLANG_REALTIME_TAEHV_CHECKPOINT_PATH "
                    "to a TAEHV checkpoint."
                ) from exc

            vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
            self._taehv_model = TAEHV(checkpoint_path=checkpoint_path).to(
                get_local_torch_device(), vae_dtype
            )
            self._taehv_model.eval()

        try:
            from taehv import StreamingTAEHV
        except ImportError as exc:
            raise RuntimeError(
                "TAEHV realtime decode requires the `taehv` package. "
                "Install it and set SGLANG_REALTIME_TAEHV_CHECKPOINT_PATH "
                "to a TAEHV checkpoint."
            ) from exc

        decoder = StreamingTAEHV(self._taehv_model)
        decode_state.taehv_streaming_decoder = decoder
        return decoder

    @torch.no_grad()
    def decode_taehv_streaming(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
        decode_state: RealtimeVAEDecodeState,
        *,
        first_chunk: bool,
    ) -> torch.Tensor:
        decoder = self._get_or_create_taehv_streaming_decoder(
            decode_state, server_args
        )
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        latents = latents.to(get_local_torch_device(), dtype=vae_dtype)

        # SGLang/Wan uses NCTHW. TAEHV uses NTCHW and consumes model latents
        # directly, without the full-VAE scale/shift path.
        latents_ntchw = latents.permute(0, 2, 1, 3, 4).contiguous()
        target_frames = max(
            1,
            latents_ntchw.shape[1] * int(decoder.taehv.t_upscale)
            - (int(decoder.taehv.frames_to_trim) if first_chunk else 0),
        )

        if first_chunk:
            decoder.reset()
            decode_state.taehv_output_queue.clear()

        produced: list[torch.Tensor] = []
        for latent_t in latents_ntchw.split(1, dim=1):
            frame = decoder.decode(latent_t)
            if frame is not None:
                produced.append(frame)
            while True:
                frame = decoder.decode()
                if frame is None:
                    break
                produced.append(frame)

        decode_state.taehv_output_queue.extend(produced)
        if not decode_state.taehv_output_queue:
            raise RuntimeError("TAEHV produced no frames for realtime chunk")

        all_frames = torch.cat(decode_state.taehv_output_queue, dim=1)
        take = min(target_frames, all_frames.shape[1])
        if take == 0:
            raise RuntimeError("TAEHV produced no frames for realtime chunk")

        frames_ntchw = all_frames[:, :take].contiguous()
        remaining = all_frames[:, take:].contiguous()
        decode_state.taehv_output_queue = (
            [remaining] if remaining.shape[1] > 0 else []
        )
        return frames_ntchw.permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        if batch.session is None:
            return super().forward(batch, server_args)

        self.load_model()

        reset_causal_state = self._get_causal_decode_reset_fn()
        decode_state = batch.session.get_or_create_state(RealtimeVAEDecodeState)
        decode_state.reset_causal_decode_state = reset_causal_state
        if batch.block_idx == 0 and callable(reset_causal_state):
            reset_causal_state()

        if self._taehv_checkpoint_path() is not None:
            frames = self.decode_taehv_streaming(
                batch.latents,
                server_args,
                decode_state,
                first_chunk=batch.block_idx == 0,
            )
        else:
            frames = self.decode_causal(
                batch.latents,
                server_args,
                first_chunk=batch.block_idx == 0,
            )
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
