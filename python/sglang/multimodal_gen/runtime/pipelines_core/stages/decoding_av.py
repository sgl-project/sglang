import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class LTX2AVDecodingStage(DecodingStage):
    """
    LTX-2 specific decoding stage that handles both video and audio decoding.
    """

    def __init__(self, vae, audio_vae, vocoder, pipeline=None):
        super().__init__(vae, pipeline)
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        # Add video processor for postprocessing
        from diffusers.video_processor import VideoProcessor

        self.video_processor = VideoProcessor(vae_scale_factor=32)

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self.load_model()

        self.vae = self.vae.to(get_local_torch_device())
        self.vae.eval()
        latents = batch.latents.to(get_local_torch_device())

        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
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
            decode_output = self.vae.decode(latents)
            if isinstance(decode_output, tuple):
                video = decode_output[0]
            elif hasattr(decode_output, "sample"):
                video = decode_output.sample
            else:
                video = decode_output

        video = self.video_processor.postprocess_video(video, output_type="np")

        output_batch = OutputBatch(
            output=video,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=None,
            timings=batch.timings,
        )

        # 2. Decode Audio
        try:
            audio_latents = batch.audio_latents
        except AttributeError:
            audio_latents = None
        if audio_latents is not None:
            # Ensure device/dtype
            device = get_local_torch_device()
            self.audio_vae = self.audio_vae.to(device)
            self.vocoder = self.vocoder.to(device)
            self.audio_vae.eval()
            self.vocoder.eval()
            try:
                dtype = self.audio_vae.dtype
            except AttributeError:
                dtype = None
            if dtype is None:
                try:
                    dtype = next(self.audio_vae.parameters()).dtype
                except StopIteration:
                    dtype = torch.float32
            audio_latents = audio_latents.to(device, dtype=dtype)
            try:
                latents_std = self.audio_vae.latents_std
            except AttributeError:
                latents_std = None
            if isinstance(latents_std, torch.Tensor) and torch.all(latents_std == 0):
                logger.warning(
                    "audio_vae.latents_std is all zeros; audio denorm may be incorrect."
                )

            with torch.no_grad():
                # Decode latents to spectrogram
                spectrogram = self.audio_vae.decode(audio_latents, return_dict=False)[0]
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

        self.offload_model()
        return output_batch
