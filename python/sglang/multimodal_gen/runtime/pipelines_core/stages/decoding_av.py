import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    DecodingStage,
    _ensure_tensor_decode_output,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE
from sglang.multimodal_gen.runtime.platforms import current_platform

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

    def _decode_video(self, batch, server_args):
        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        latents = batch.latents.to(get_local_torch_device())

        # Unpack and Denormalize
        # Note: We pass None for audio_latents here as we handle it separately
        latents, _ = server_args.pipeline_config._unpad_and_unpack_latents(
            latents, None, batch, self.vae, self.audio_vae
        )

        # Decode latents
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

            # LTX-2 VAE decode expects 5D tensor
            # The decode method in diffusers LTX-2 VAE takes timestep argument but it can be optional/None
            # if timestep_conditioning is False.
            # In Diffusers pipeline:
            # if not self.vae.config.timestep_conditioning: timestep = None
            # else: ... (logic for decode_timestep)
            # We assume timestep_conditioning is False or handled by default for now,
            # or we pass None explicitly.
            timestep = None
            decode_output = self.vae.decode(latents, timestep, return_dict=False)[0]
            image = _ensure_tensor_decode_output(decode_output)

        # Postprocess video
        if hasattr(self, "video_processor"):
            image = self.video_processor.postprocess_video(image, output_type="np")

        return image

    def _decode_audio(self, batch, server_args):
        audio_latents = getattr(batch, "audio_latents", None)
        if audio_latents is None:
            return None

        # Ensure device/dtype
        device = get_local_torch_device()
        self.audio_vae = self.audio_vae.to(device)
        self.vocoder = self.vocoder.to(device)
        dtype = getattr(self.audio_vae, "dtype", None)
        if dtype is None:
            try:
                dtype = next(self.audio_vae.parameters()).dtype
            except StopIteration:
                dtype = torch.float32

        audio_latents = audio_latents.to(device, dtype=dtype)

        # Unpack and Denormalize Audio
        # We reuse the helper but only care about audio output
        # We need to pass valid latents for shape calculation
        latents_dummy = batch.latents.to(device)
        _, audio_latents = server_args.pipeline_config._unpad_and_unpack_latents(
            latents_dummy, audio_latents, batch, self.vae, self.audio_vae
        )

        # To match diffusers, we need to cast to audio_vae.dtype
        audio_latents = audio_latents.to(self.audio_vae.dtype)

        with torch.no_grad():
            # Decode latents to spectrogram
            spectrogram = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            # Decode spectrogram to waveform
            waveform = self.vocoder(spectrogram)

        return waveform.cpu().float()

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        # load vae if not already loaded
        self.load_model()
        self.vae = self.vae.to(get_local_torch_device())

        # Decode Video
        frames = self._decode_video(batch, server_args)

        # Decode Audio
        audio = self._decode_audio(batch, server_args)

        # Update batch
        output_batch = OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=None,
            timings=batch.timings,
            audio=audio,
        )

        self.offload_model()

        return output_batch
