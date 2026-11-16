from collections.abc import Callable

import torch
from diffusers.video_processor import VideoProcessor

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_replica_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.encode_while_decoding import (
    EncodeWhileDecoding,
    can_encode_while_decoding,
    streamed_video_output_path,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    align_tensor_to_module_dtype,
    autocast_context,
    autocast_enabled,
    resolve_decode_precision,
    resolve_precision,
    temporary_module_dtype,
)

logger = init_logger(__name__)


class LTX2AVDecodingStage(DecodingStage):
    """
    LTX-2 specific decoding stage that handles both video and audio decoding.
    """

    def __init__(self, vae, audio_vae, vocoder, pipeline=None, diffusion_decoder=None):
        super().__init__(vae, pipeline)
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        # Replaces the convolutional decoder; latents and denormalization are
        # identical either way.
        self.diffusion_decoder = diffusion_decoder
        self.video_processor = VideoProcessor(vae_scale_factor=32)

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        vae_dtype = resolve_decode_precision(server_args, "vae")
        audio_vae_dtype = resolve_precision(
            server_args, "audio_vae", precision_attr="audio_vae_precision"
        )
        uses = [ComponentUse(stage_name, "vae", target_dtype=vae_dtype)]
        if self.diffusion_decoder is not None:
            uses.append(
                ComponentUse(
                    stage_name,
                    "diffusion_decoder",
                    target_dtype=vae_dtype,
                    allow_prefetch=False,
                )
            )
        uses.extend(
            [
                ComponentUse(stage_name, "audio_vae", target_dtype=audio_vae_dtype),
                ComponentUse(stage_name, "vocoder"),
            ]
        )
        return uses

    @staticmethod
    def _ltx2_should_externally_denorm_video_latents(server_args: ServerArgs) -> bool:
        arch_config = server_args.pipeline_config.vae_config.arch_config
        return str(arch_config.video_decoder_variant) != "ltx_2_3"

    def _decode_with_diffusion_decoder(
        self, decoder, latents, batch, server_args: ServerArgs
    ):
        """Decode with the LTX-2.5 diffusion decoder.

        It is a diffusion model in its own right, so it needs a generator; the
        request's seed keeps a decode reproducible.
        """
        # Untiled, every stage attends over the whole volume
        decoder.use_tiling = bool(server_args.pipeline_config.diffusion_decoder_tiling)
        # Splitting the tiles is a collective over the decode-parallel group
        decoder.use_parallel_tiling = (
            bool(server_args.pipeline_config.diffusion_decoder_parallel_tiling)
            and self.parallelism_type == StageParallelismType.REPLICATED
        )
        generator = torch.Generator(device=latents.device).manual_seed(int(batch.seed))
        return decoder(latents, generator=generator)

    def _prepare_video_latents(self, batch: Req, module, server_args: ServerArgs):
        latents = batch.latents.to(get_local_torch_device())
        if self._ltx2_should_externally_denorm_video_latents(server_args):
            std = module.latents_std.view(1, -1, 1, 1, 1).to(latents)
            mean = module.latents_mean.view(1, -1, 1, 1, 1).to(latents)
            latents = latents * std + mean
        return server_args.pipeline_config.preprocess_decoding(
            latents, server_args, vae=module
        )

    def _postprocess_video(self, video: torch.Tensor):
        if not video.is_cuda:
            return self.video_processor.postprocess_video(video, output_type="np")
        batch_size, channels, num_frames, height, width = video.shape
        try:
            host = torch.empty(
                (batch_size, num_frames, height, width, channels),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
        except RuntimeError:
            return self.video_processor.postprocess_video(video.cpu(), output_type="np")

        # Bound each FP32 conversion to 64 MiB (or one frame), independently
        # of clip length and batch size. Normalize in the source dtype first.
        chunk_frames = max(1, (64 * 1024 * 1024) // (channels * height * width * 4))
        stream = torch.cuda.current_stream(video.device)
        try:
            for batch_idx in range(batch_size):
                for start in range(0, num_frames, chunk_frames):
                    end = min(start + chunk_frames, num_frames)
                    frames = self.video_processor.postprocess_video(
                        video[batch_idx : batch_idx + 1, :, start:end], output_type="pt"
                    )
                    frames = (
                        frames[0]
                        .permute(0, 2, 3, 1)
                        .to(dtype=torch.float32, memory_format=torch.contiguous_format)
                    )
                    host[batch_idx, start:end].copy_(frames, non_blocking=True)
        except torch.cuda.OutOfMemoryError:
            stream.synchronize()
            host = None
            return self.video_processor.postprocess_video(video.cpu(), output_type="np")
        stream.synchronize()
        return host.numpy()

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self.load_model()
        output_batch = OutputBatch(
            output=None,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=None,
            metrics=batch.metrics,
            rollout_trajectory_data=batch.rollout_trajectory_data,
        )
        audio_first = (
            not batch.use_diffusion_decoder
            and not batch.return_trajectory_decoded
            and isinstance(self.vae, ParallelTiledVAE)
            and self.vae.supports_decode_on_frames
            and can_encode_while_decoding(
                batch, server_args, int(batch.latents.shape[0])
            )
        )
        save_path = (
            streamed_video_output_path(batch, server_args, int(batch.latents.shape[0]))
            if audio_first
            else None
        )
        # All ranks use the same order. Raw-frame and unsupported requests keep
        # the original video-before-audio order.
        if audio_first:
            self._decode_audio(batch, server_args, output_batch)
        stream = None
        if save_path is not None:
            # ffmpeg needs the audio input before it can consume the first video frames
            stream = EncodeWhileDecoding(
                save_path,
                batch,
                finish_frames=lambda frames: (frames / 2 + 0.5).clamp(0, 1).float(),
                audio=output_batch.audio,
                audio_sample_rate=output_batch.audio_sample_rate,
            )
        try:
            video = self._decode_video(batch, server_args, on_frames=stream)
            output_paths = stream.finish() if stream is not None else None
        except BaseException:
            if stream is not None:
                stream.abort()
            raise
        if output_paths is not None:
            output_batch.output_file_paths = output_paths
        elif (
            batch.extra.get("retain_decoded_frames", False)
            or not model_parallel_is_initialized()
            or get_replica_group().rank_in_group == 0
        ):
            output_batch.output = self._postprocess_video(video)
        if not audio_first:
            self._decode_audio(batch, server_args, output_batch)
        return output_batch

    def _decode_video(
        self,
        batch: Req,
        server_args: ServerArgs,
        on_frames: Callable[[torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        vae_dtype = resolve_decode_precision(server_args, "vae")
        vae_autocast_enabled = autocast_enabled(vae_dtype, server_args.disable_autocast)

        if batch.use_diffusion_decoder:
            if self.diffusion_decoder is None:
                raise ValueError(
                    "use_diffusion_decoder was requested, but the decoder is not "
                    "loaded. Start the server with --load-diffusion-decoder."
                )
            with self.use_declared_component(
                component_name="diffusion_decoder", module=self.diffusion_decoder
            ) as decoder:
                assert decoder is not None
                decoder.eval()
                latents = self._prepare_video_latents(batch, decoder, server_args)
                with autocast_context(
                    dtype=vae_dtype,
                    disable_autocast=server_args.disable_autocast,
                    enabled=vae_autocast_enabled,
                ):
                    if not vae_autocast_enabled:
                        latents = latents.to(vae_dtype)
                    decode_output = self._decode_with_diffusion_decoder(
                        decoder, latents, batch, server_args
                    )
        else:
            with self.use_declared_component(
                component_name="vae", module=self.vae
            ) as vae:
                assert vae is not None
                self.vae = vae
                self.vae.eval()
                latents = self._prepare_video_latents(batch, self.vae, server_args)
                with autocast_context(
                    dtype=vae_dtype,
                    disable_autocast=server_args.disable_autocast,
                    enabled=vae_autocast_enabled,
                ):
                    try:
                        if server_args.pipeline_config.vae_tiling:
                            self.vae.enable_tiling()
                    except Exception:
                        pass
                    should_cast_vae = not vae_autocast_enabled
                    if not vae_autocast_enabled:
                        latents = latents.to(vae_dtype)
                    with temporary_module_dtype(
                        self.vae, vae_dtype, enabled=should_cast_vae
                    ) as vae:
                        decode_kwargs = (
                            {} if on_frames is None else {"on_frames": on_frames}
                        )
                        decode_output = vae.decode(latents, **decode_kwargs)

        if isinstance(decode_output, tuple):
            video = decode_output[0]
        elif isinstance(decode_output, torch.Tensor):
            video = decode_output
        else:
            video = decode_output.sample
        return video

    def _decode_audio(
        self, batch: Req, server_args: ServerArgs, output_batch: OutputBatch
    ):
        try:
            audio_latents = batch.audio_latents
        except AttributeError:
            audio_latents = None
        if audio_latents is not None:
            # Ensure device/dtype
            device = get_local_torch_device()
            with self.use_declared_component(
                component_name="audio_vae",
                module=self.audio_vae,
            ) as audio_vae:
                assert audio_vae is not None
                self.audio_vae = audio_vae
                self.audio_vae.eval()
                audio_vae_dtype = resolve_precision(
                    server_args,
                    "audio_vae",
                    precision_attr="audio_vae_precision",
                )
                dtype = audio_vae_dtype
                audio_latents = audio_latents.to(device, dtype=dtype)
                try:
                    latents_std = self.audio_vae.latents_std
                except AttributeError:
                    latents_std = None
                if isinstance(latents_std, torch.Tensor) and torch.all(
                    latents_std == 0
                ):
                    logger.warning(
                        "audio_vae.latents_std is all zeros; audio denorm may be incorrect."
                    )
                try:
                    latents_mean = self.audio_vae.latents_mean
                except AttributeError:
                    latents_mean = None
                if isinstance(latents_mean, torch.Tensor) and isinstance(
                    latents_std, torch.Tensor
                ):
                    latents_mean = latents_mean.to(device=device, dtype=dtype)
                    latents_std = latents_std.to(device=device, dtype=dtype)
                    if audio_latents.ndim == 4:
                        latents_mean = latents_mean.view(
                            1, audio_latents.shape[1], 1, audio_latents.shape[3]
                        )
                        latents_std = latents_std.view(
                            1, audio_latents.shape[1], 1, audio_latents.shape[3]
                        )
                    audio_latents = audio_latents * latents_std + latents_mean

                audio_vae_autocast_enabled = autocast_enabled(
                    audio_vae_dtype, server_args.disable_autocast
                )
                should_cast_audio_vae = not audio_vae_autocast_enabled
                with (
                    torch.no_grad(),
                    autocast_context(
                        dtype=audio_vae_dtype,
                        disable_autocast=server_args.disable_autocast,
                        enabled=audio_vae_autocast_enabled,
                    ),
                ):
                    # Decode latents to spectrogram
                    with temporary_module_dtype(
                        self.audio_vae,
                        audio_vae_dtype,
                        enabled=should_cast_audio_vae,
                    ) as audio_vae:
                        spectrogram = audio_vae.decode(
                            audio_latents, return_dict=False
                        )[0]

            with self.use_declared_component(
                component_name="vocoder",
                module=self.vocoder,
            ) as vocoder:
                assert vocoder is not None
                self.vocoder = vocoder
                self.vocoder.eval()
                if hasattr(self.vocoder, "conv_in") and hasattr(
                    self.vocoder.conv_in, "in_channels"
                ):
                    expected_in = int(self.vocoder.conv_in.in_channels)
                    actual_in = int(spectrogram.shape[1]) * int(spectrogram.shape[3])
                    if actual_in != expected_in:
                        raise ValueError(
                            f"Vocoder expects channels*mel_bins={expected_in}, got {actual_in} from spectrogram shape {tuple(spectrogram.shape)}"
                        )
                # Decode spectrogram to waveform
                spectrogram = align_tensor_to_module_dtype(spectrogram, self.vocoder)
                with torch.no_grad():
                    waveform = self.vocoder(spectrogram)
            output_batch.audio = waveform.cpu().float()
            try:
                pipeline_audio_cfg = server_args.pipeline_config.audio_vae_config
            except AttributeError:
                pipeline_audio_cfg = None
            try:
                pipeline_audio_arch = pipeline_audio_cfg.arch_config  # type: ignore[union-attr]
            except AttributeError:
                pipeline_audio_arch = None
            try:
                pipeline_audio_sr = pipeline_audio_arch.sample_rate  # type: ignore[union-attr]
            except AttributeError:
                pipeline_audio_sr = None

            try:
                vocoder_sr = self.vocoder.sample_rate
            except AttributeError:
                vocoder_sr = None
            try:
                audio_vae_sr = self.audio_vae.sample_rate
            except AttributeError:
                audio_vae_sr = None
            output_batch.audio_sample_rate = (
                vocoder_sr or audio_vae_sr or pipeline_audio_sr
            )
