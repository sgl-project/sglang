import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    align_tensor_to_module_dtype,
    autocast_enabled,
    resolve_precision,
    temporary_module_dtype,
)

logger = init_logger(__name__)


def _ltx2_should_externally_denorm_video_latents(server_args: ServerArgs) -> bool:
    arch_config = server_args.pipeline_config.vae_config.arch_config
    return str(getattr(arch_config, "video_decoder_variant", "ltx_2")) != "ltx_2_3"


def _decode_video(stage, batch: Req, server_args: ServerArgs):
    """Run the video VAE. ``stage`` owns the ``vae`` component."""
    vae_dtype = resolve_precision(server_args, "vae", precision_attr="vae_precision")
    vae_autocast_enabled = autocast_enabled(vae_dtype, server_args.disable_autocast)

    with stage.use_declared_component(component_name="vae", module=stage.vae) as vae:
        assert vae is not None
        stage.vae = vae
        stage.vae.eval()
        latents = batch.latents.to(get_local_torch_device())
        if _ltx2_should_externally_denorm_video_latents(server_args):
            std = stage.vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
            mean = stage.vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
            latents = latents * std + mean
        latents = server_args.pipeline_config.preprocess_decoding(
            latents, server_args, vae=stage.vae
        )

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                if server_args.pipeline_config.vae_tiling:
                    stage.vae.enable_tiling()
            except Exception:
                pass
            should_cast_vae = not vae_autocast_enabled
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            with temporary_module_dtype(
                stage.vae, vae_dtype, enabled=should_cast_vae
            ) as vae:
                decode_output = vae.decode(latents)
            if isinstance(decode_output, tuple):
                video = decode_output[0]
            elif hasattr(decode_output, "sample"):
                video = decode_output.sample
            else:
                video = decode_output

    return stage.video_processor.postprocess_video(video, output_type="np")


def _decode_audio_into(
    stage, batch: Req, server_args: ServerArgs, output_batch: OutputBatch
) -> None:
    """Run the audio VAE and vocoder, writing into *output_batch*.

    ``stage`` owns the ``audio_vae`` and ``vocoder`` components.  Does nothing
    when the request carries no audio latents.
    """
    try:
        audio_latents = batch.audio_latents
    except AttributeError:
        audio_latents = None
    if audio_latents is None:
        return

    device = get_local_torch_device()
    with stage.use_declared_component(
        component_name="audio_vae",
        module=stage.audio_vae,
    ) as audio_vae:
        assert audio_vae is not None
        stage.audio_vae = audio_vae
        stage.audio_vae.eval()
        audio_vae_dtype = resolve_precision(
            server_args,
            "audio_vae",
            precision_attr="audio_vae_precision",
        )
        dtype = audio_vae_dtype
        audio_latents = audio_latents.to(device, dtype=dtype)
        try:
            latents_std = stage.audio_vae.latents_std
        except AttributeError:
            latents_std = None
        if isinstance(latents_std, torch.Tensor) and torch.all(latents_std == 0):
            logger.warning(
                "audio_vae.latents_std is all zeros; audio denorm may be incorrect."
            )
        try:
            latents_mean = stage.audio_vae.latents_mean
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
            torch.autocast(
                device_type=current_platform.device_type,
                dtype=audio_vae_dtype,
                enabled=audio_vae_autocast_enabled,
            ),
        ):
            # Decode latents to spectrogram
            with temporary_module_dtype(
                stage.audio_vae,
                audio_vae_dtype,
                enabled=should_cast_audio_vae,
            ) as audio_vae:
                spectrogram = audio_vae.decode(audio_latents, return_dict=False)[0]

    with stage.use_declared_component(
        component_name="vocoder",
        module=stage.vocoder,
    ) as vocoder:
        assert vocoder is not None
        stage.vocoder = vocoder
        stage.vocoder.eval()
        if hasattr(stage.vocoder, "conv_in") and hasattr(
            stage.vocoder.conv_in, "in_channels"
        ):
            expected_in = int(stage.vocoder.conv_in.in_channels)
            actual_in = int(spectrogram.shape[1]) * int(spectrogram.shape[3])
            if actual_in != expected_in:
                raise ValueError(
                    f"Vocoder expects channels*mel_bins={expected_in}, got {actual_in} from spectrogram shape {tuple(spectrogram.shape)}"
                )
        # Decode spectrogram to waveform
        spectrogram = align_tensor_to_module_dtype(spectrogram, stage.vocoder)
        with torch.no_grad():
            waveform = stage.vocoder(spectrogram)

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
        vocoder_sr = stage.vocoder.sample_rate
    except AttributeError:
        vocoder_sr = None
    try:
        audio_vae_sr = stage.audio_vae.sample_rate
    except AttributeError:
        audio_vae_sr = None
    output_batch.audio_sample_rate = vocoder_sr or audio_vae_sr or pipeline_audio_sr


class LTX2VideoDecodingStage(DecodingStage):
    """Decode LTX-2 video latents to pixels.

    Split out from the combined AV stage so a deployment can put the video VAE
    on its own pool and scale it independently of the audio VAE, which is far
    cheaper and would otherwise be forced to the same replica count.
    """

    def __init__(self, vae, pipeline=None):
        super().__init__(vae, pipeline)
        from diffusers.video_processor import VideoProcessor

        self.video_processor = VideoProcessor(vae_scale_factor=32)

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )
        return [ComponentUse(stage_name, "vae", target_dtype=vae_dtype)]

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self.load_model()
        return OutputBatch(
            output=_decode_video(self, batch, server_args),
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=None,
            metrics=batch.metrics,
            rollout_trajectory_data=batch.rollout_trajectory_data,
        )


class LTX2AudioDecodingStage(DecodingStage):
    """Decode LTX-2 audio latents to a waveform via the audio VAE and vocoder."""

    def __init__(self, audio_vae, vocoder, pipeline=None):
        super().__init__(audio_vae, pipeline, component_name="audio_vae")
        self.vocoder = vocoder

    # The base class stores the stage's primary component as ``vae``; expose
    # it under the name the shared decode helper uses.
    @property
    def audio_vae(self):
        return self.vae

    @audio_vae.setter
    def audio_vae(self, module) -> None:
        self.vae = module

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        audio_vae_dtype = resolve_precision(
            server_args, "audio_vae", precision_attr="audio_vae_precision"
        )
        return [
            ComponentUse(stage_name, "audio_vae", target_dtype=audio_vae_dtype),
            ComponentUse(stage_name, "vocoder"),
        ]

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self.load_model()
        output_batch = OutputBatch(metrics=batch.metrics)
        _decode_audio_into(self, batch, server_args, output_batch)
        return output_batch


class LTX2AVDecodingStage(LTX2VideoDecodingStage):
    """
    LTX-2 specific decoding stage that handles both video and audio decoding.
    """

    def __init__(self, vae, audio_vae, vocoder, pipeline=None):
        super().__init__(vae, pipeline)
        self.audio_vae = audio_vae
        self.vocoder = vocoder

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        audio_vae_dtype = resolve_precision(
            server_args, "audio_vae", precision_attr="audio_vae_precision"
        )
        return [
            *super().component_uses(server_args, stage_name),
            ComponentUse(stage_name, "audio_vae", target_dtype=audio_vae_dtype),
            ComponentUse(stage_name, "vocoder"),
        ]

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        output_batch = super().forward(batch, server_args)
        _decode_audio_into(self, batch, server_args, output_batch)
        return output_batch
