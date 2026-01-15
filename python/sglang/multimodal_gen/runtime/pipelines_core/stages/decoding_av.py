import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

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

        # 1. Unpack latents (Already done in Denoising Stage)
        # latents = batch.latents

        # 3. Decode Video using VAE
        output_batch = super().forward(batch, server_args)

        # 4. Postprocess video to ensure correct format
        # This matches the diffusers implementation
        if hasattr(self, "video_processor") and hasattr(
            self.video_processor, "postprocess_video"
        ):
            output_batch.output = self.video_processor.postprocess_video(
                output_batch.output, output_type="np"
            )

        # 2. Decode Audio
        audio_latents = getattr(batch, "audio_latents", None)
        if audio_latents is not None:
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

            # Denormalization also done in Denoising Stage

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

        return output_batch
