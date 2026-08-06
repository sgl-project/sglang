# SPDX-License-Identifier: Apache-2.0

import time

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
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    log_realtime_trace_for_batch,
    realtime_trace_span,
    tensor_trace_metadata,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


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
        self.taehv_checkpoint_path: str | None = None
        self.taehv_dtype: torch.dtype | None = None
        self.taehv_streaming_decoder = None

    def dispose(self):
        reset_causal_decode_state = self.reset_causal_decode_state
        self.reset_causal_decode_state = None
        if callable(reset_causal_decode_state):
            reset_causal_decode_state()
        self.reset_taehv_decoder()
        self.taehv_streaming_decoder = None
        self.taehv_checkpoint_path = None
        self.taehv_dtype = None

    def reset_taehv_decoder(self):
        reset = getattr(self.taehv_streaming_decoder, "reset", None)
        if callable(reset):
            reset()


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
                log_realtime_trace_for_batch(
                    logger,
                    batch,
                    "server.vae_encode_complete",
                    component="vae_encoder_cache",
                    chunk_index=batch.block_idx,
                    first_chunk=False,
                    cache_hit=True,
                    duration_ms=0.0,
                    **tensor_trace_metadata(batch.image_latent, prefix="image_latent"),
                )
                return batch

        if batch.condition_image is None:
            if state is not None and state.image_latent is not None:
                batch.image_latent = state.image_latent
                log_realtime_trace_for_batch(
                    logger,
                    batch,
                    "server.vae_encode_complete",
                    component="vae_encoder_cache",
                    chunk_index=batch.block_idx,
                    first_chunk=batch.block_idx == 0,
                    cache_hit=True,
                    duration_ms=0.0,
                    **tensor_trace_metadata(batch.image_latent, prefix="image_latent"),
                )
            return batch

        with realtime_trace_span(
            logger,
            batch,
            "server.vae_encode_complete",
            component="vae_encoder",
            input_tensor=batch.condition_image,
            chunk_index=batch.block_idx,
            first_chunk=batch.block_idx == 0,
        ) as trace_span:
            batch = super().forward(batch, server_args)
            trace_span.add_fields(
                **tensor_trace_metadata(batch.image_latent, prefix="image_latent"),
            )

        if state is not None and batch.image_latent is not None:
            state.image_latent = batch.image_latent
        return batch


class CausalVaeDecodingStage(DecodingStage):
    """Decode realtime chunks with a persistent causal VAE cache when available."""

    def __init__(self, vae, pipeline=None, component_name: str = "vae") -> None:
        super().__init__(vae=vae, pipeline=pipeline, component_name=component_name)
        self._taehv_models = {}

        vae_config = getattr(self.server_args.pipeline_config, "vae_config", None)
        checkpoint_path = self._taehv_checkpoint_path(vae_config)
        if checkpoint_path is not None:
            vae_dtype = PRECISION_TO_TYPE[
                self.server_args.pipeline_config.vae_precision
            ]
            self._get_or_load_taehv_model(checkpoint_path, vae_dtype)

    @staticmethod
    def _taehv_checkpoint_path(vae_config) -> str | None:
        path = getattr(vae_config, "taehv_checkpoint_path", None)
        if isinstance(path, str):
            path = path.strip()
        return path or None

    def _load_taehv_model(
        self,
        checkpoint_path: str,
        vae_dtype: torch.dtype,
    ):
        try:
            from taehv import TAEHV
        except ImportError as exc:
            raise RuntimeError(
                "TAEHV realtime decoder was requested, but the `taehv` package "
                "is not installed. Install it or unset "
                "`--vae-config.taehv-checkpoint-path`."
            ) from exc

        taehv = TAEHV(checkpoint_path=checkpoint_path).eval()
        taehv = taehv.to(device=get_local_torch_device(), dtype=vae_dtype)
        return taehv.requires_grad_(False)

    def _get_or_load_taehv_model(
        self,
        checkpoint_path: str,
        vae_dtype: torch.dtype,
    ):
        models = getattr(self, "_taehv_models", None)
        if models is None:
            models = {}
            self._taehv_models = models

        key = (checkpoint_path, vae_dtype)
        model = models.get(key)
        if model is None:
            started_at = time.perf_counter()
            logger.info(
                "Preloading TAEHV decoder weights: checkpoint=%s device=%s dtype=%s",
                checkpoint_path,
                get_local_torch_device(),
                vae_dtype,
            )
            model = self._load_taehv_model(checkpoint_path, vae_dtype)
            models[key] = model
            logger.info(
                "TAEHV decoder weights ready in %.3fs: checkpoint=%s",
                time.perf_counter() - started_at,
                checkpoint_path,
            )
        return model

    @staticmethod
    def _create_streaming_taehv_decoder(taehv_model):
        from taehv import StreamingTAEHV

        return StreamingTAEHV(taehv_model).eval()

    def _get_or_create_streaming_taehv_decoder(
        self,
        decode_state: RealtimeVAEDecodeState,
        checkpoint_path: str,
        vae_dtype: torch.dtype,
    ):
        if (
            decode_state.taehv_streaming_decoder is not None
            and decode_state.taehv_checkpoint_path == checkpoint_path
            and decode_state.taehv_dtype == vae_dtype
        ):
            return decode_state.taehv_streaming_decoder

        decode_state.reset_taehv_decoder()
        taehv_model = self._get_or_load_taehv_model(checkpoint_path, vae_dtype)
        decoder = self._create_streaming_taehv_decoder(taehv_model)
        decode_state.taehv_streaming_decoder = decoder
        decode_state.taehv_checkpoint_path = checkpoint_path
        decode_state.taehv_dtype = vae_dtype
        return decoder

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

    def _decode_with_streaming_taehv(
        self,
        latents: torch.Tensor,
        decode_state: RealtimeVAEDecodeState,
        checkpoint_path: str,
        vae_dtype: torch.dtype,
        *,
        first_chunk: bool,
    ) -> torch.Tensor:
        decoder = self._get_or_create_streaming_taehv_decoder(
            decode_state,
            checkpoint_path,
            vae_dtype,
        )
        if first_chunk:
            decode_state.reset_taehv_decoder()

        taehv_latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        decoded_frames = []
        frame = decoder.decode(taehv_latents)
        while frame is not None:
            decoded_frames.append(frame)
            frame = decoder.decode()

        if not decoded_frames:
            expected_height = int(
                latents.shape[-2]
                * getattr(getattr(decoder, "taehv", None), "patch_size", 1)
                * 8
            )
            expected_width = int(
                latents.shape[-1]
                * getattr(getattr(decoder, "taehv", None), "patch_size", 1)
                * 8
            )
            return latents.new_empty(
                (latents.shape[0], 3, 0, expected_height, expected_width)
            )

        frames = torch.cat(decoded_frames, dim=1)
        return frames.permute(0, 2, 1, 3, 4).contiguous().clamp(0, 1)

    @torch.no_grad()
    def decode_causal(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
        *,
        first_chunk: bool,
        decode_state: RealtimeVAEDecodeState | None = None,
    ) -> torch.Tensor:
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        self.vae = self.vae.to(device=get_local_torch_device(), dtype=vae_dtype)
        latents = latents.to(get_local_torch_device())
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        taehv_checkpoint_path = self._taehv_checkpoint_path(
            getattr(server_args.pipeline_config, "vae_config", None)
        )
        if taehv_checkpoint_path is not None:
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            if decode_state is None:
                decode_state = RealtimeVAEDecodeState()
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=vae_dtype,
                enabled=vae_autocast_enabled,
            ):
                return self._decode_with_streaming_taehv(
                    latents,
                    decode_state,
                    taehv_checkpoint_path,
                    vae_dtype,
                    first_chunk=first_chunk,
                )

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

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        if batch.session is None:
            return super().forward(batch, server_args)

        decode_state = batch.session.get_or_create_state(RealtimeVAEDecodeState)
        vae_config = getattr(server_args.pipeline_config, "vae_config", None)
        taehv_checkpoint_path = self._taehv_checkpoint_path(vae_config)
        with realtime_trace_span(
            logger,
            batch,
            "server.vae_decoder_load_complete",
            component="vae_decoder_load",
            measure_cuda=False,
            chunk_index=batch.block_idx,
            first_chunk=batch.block_idx == 0,
        ):
            self.load_model()
            if taehv_checkpoint_path is not None:
                vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
                self._get_or_create_streaming_taehv_decoder(
                    decode_state,
                    taehv_checkpoint_path,
                    vae_dtype,
                )

        reset_causal_state = self._get_causal_decode_reset_fn()
        decode_state.reset_causal_decode_state = reset_causal_state
        if batch.block_idx == 0 and callable(reset_causal_state):
            reset_causal_state()

        with realtime_trace_span(
            logger,
            batch,
            "server.vae_decode_complete",
            component="vae_decoder",
            input_tensor=batch.latents,
            chunk_index=batch.block_idx,
            first_chunk=batch.block_idx == 0,
            decoder_backend="taehv"
            if taehv_checkpoint_path is not None
            else "causal_vae",
            taehv_checkpoint_path=taehv_checkpoint_path,
            vae_precision=server_args.pipeline_config.vae_precision,
            vae_tiling=server_args.pipeline_config.vae_tiling,
            use_parallel_decode=bool(
                getattr(vae_config, "use_parallel_decode", False)
            ),
            parallel_decode_mode=getattr(vae_config, "parallel_decode_mode", None),
        ) as trace_span:
            frames = self.decode_causal(
                batch.latents,
                server_args,
                first_chunk=batch.block_idx == 0,
                decode_state=decode_state,
            )
            trace_span.add_fields(**tensor_trace_metadata(frames, prefix="frames"))
        with realtime_trace_span(
            logger,
            batch,
            "server.post_decode_complete",
            component="post_decode",
            input_tensor=frames,
            measure_cuda=False,
            chunk_index=batch.block_idx,
            first_chunk=batch.block_idx == 0,
        ) as trace_span:
            frames = server_args.pipeline_config.post_decoding(frames, server_args)
            trace_span.add_fields(
                **tensor_trace_metadata(frames, prefix="post_decoded_frames")
            )

        return OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=None,
            metrics=batch.metrics,
            noise_pred=None,
        )
