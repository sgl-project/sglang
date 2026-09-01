import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
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
        # Add video processor for postprocessing
        from diffusers.video_processor import VideoProcessor

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
        # Untiled, every stage attends over the whole volume -- minutes at a
        # full-length 121-frame grid.
        decoder.use_tiling = bool(server_args.pipeline_config.diffusion_decoder_tiling)
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

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self.load_model()

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
                        decode_output = vae.decode(latents)

        if isinstance(decode_output, tuple):
            video = decode_output[0]
        elif isinstance(decode_output, torch.Tensor):
            video = decode_output
        else:
            video = decode_output.sample
        video = self.video_processor.postprocess_video(video, output_type="np")

        output_batch = OutputBatch(
            output=video,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=None,
            metrics=batch.metrics,
            rollout_trajectory_data=batch.rollout_trajectory_data,
        )

        # 2. Decode Audio
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

        return output_batch
