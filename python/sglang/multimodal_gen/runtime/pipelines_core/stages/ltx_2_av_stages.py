import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = init_logger(__name__)


class LTX2AVLatentPreparationStage(LatentPreparationStage):
    """
    LTX-2 specific latent preparation stage that handles both video and audio latents.
    """

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. Prepare Video Latents using base class logic
        # This sets batch.latents and batch.raw_latent_shape
        batch = super().forward(batch, server_args)

        # 2. Prepare Audio Latents
        device = get_local_torch_device()
        dtype = batch.prompt_embeds[0].dtype
        generator = batch.generator
        config = server_args.pipeline_config

        # Calculate audio latent dimensions
        # LTX-2 Audio VAE produces latents at ~25Hz (16kHz / 160 hop / 4 downsample = 25Hz)
        # Assuming video is also 25fps, audio_frames ~= video_frames
        # TODO: If frame_rate is variable, we need to access it here.
        # For now, assume 1:1 mapping as per LTX-2 defaults.
        audio_latent_frames = batch.num_frames

        # Shape: [B, C, T, F] where F is mel_bins
        shape = (
            batch.batch_size,
            config.audio_latent_channels,
            audio_latent_frames,
            config.audio_latent_mel_bins,
        )

        # Generate random noise
        audio_latents = randn_tensor(
            shape, generator=generator, device=device, dtype=dtype
        )

        # Scale initial noise
        if hasattr(self.scheduler, "init_noise_sigma"):
            audio_latents = audio_latents * self.scheduler.init_noise_sigma

        # Store in batch
        # Note: We dynamically add these fields to Req
        batch.audio_latents = audio_latents
        batch.raw_audio_latent_shape = shape

        if batch.debug:
            logger.debug(f"{batch.raw_audio_latent_shape=}")

        return batch


class LTX2AVDenoisingStage(DenoisingStage):
    """
    LTX-2 specific denoising stage that handles joint video and audio generation.
    """

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Get inputs
        latents = batch.latents
        audio_latents = getattr(batch, "audio_latents", None)
        
        if audio_latents is None:
            # Fallback or error? For now, assume audio is required for LTX-2
            # But if user didn't request audio, maybe we should skip?
            # LTX-2 is inherently AV.
            raise ValueError("Audio latents not found in batch for LTX2AVDenoisingStage")

        # Prepare extra kwargs for the model (guidance, etc.)
        # extra_step_kwargs = self.prepare_extra_step_kwargs(batch, server_args)
        
        # Denoising loop
        # num_warmup_steps = len(self.scheduler.timesteps) - self.scheduler.num_inference_steps * self.scheduler.order
        
        # Ensure latents are on correct device/dtype
        latents = latents.to(self.device, dtype=self.transformer.dtype)
        audio_latents = audio_latents.to(self.device, dtype=self.transformer.dtype)
        
        with self.progress_bar(total=self.scheduler.num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                # 1. Predict noise/velocity
                
                # Expand latents for CFG
                do_cfg = self.do_classifier_free_guidance(batch)
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                audio_latent_model_input = torch.cat([audio_latents] * 2) if do_cfg else audio_latents
                
                # Scale latents (FlowMatch usually doesn't, but check scheduler)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # Audio latents scaling? Assuming same as video or none.
                
                # Prepare kwargs
                # Note: prompt_embeds should be [neg, pos] if CFG is on.
                # LatentPreparationStage/TextEncodingStage should have handled this.
                # But we need to verify if batch.prompt_embeds is a list or tensor.
                encoder_hidden_states = batch.prompt_embeds
                if isinstance(encoder_hidden_states, list):
                    # If list of tensors, concat them? Or pick one?
                    # Usually it's a single tensor [B*2, L, D] after processing.
                    # If it's still a list [pos_embeds], we need to handle neg_embeds.
                    if do_cfg:
                        # We need to concat neg and pos
                        # batch.negative_prompt_embeds should be available
                        pos = batch.prompt_embeds[0]
                        neg = batch.negative_prompt_embeds[0]
                        encoder_hidden_states = torch.cat([neg, pos])
                    else:
                        encoder_hidden_states = batch.prompt_embeds[0]

                encoder_attention_mask = batch.prompt_attention_mask
                if isinstance(encoder_attention_mask, list):
                     if do_cfg:
                        pos_mask = batch.prompt_attention_mask[0]
                        neg_mask = batch.negative_attention_mask[0]
                        encoder_attention_mask = torch.cat([neg_mask, pos_mask])
                     else:
                        encoder_attention_mask = batch.prompt_attention_mask[0]

                kwargs = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                    "audio_hidden_states": audio_latent_model_input,
                    "return_dict": False,
                }
                
                # Run model
                # LTX-2 DiT forward returns (video_pred, audio_pred)
                noise_pred = self.transformer(
                    latent_model_input,
                    timestep=t,
                    **kwargs
                )
                
                if isinstance(noise_pred, tuple):
                    video_pred, audio_pred = noise_pred
                else:
                    video_pred = noise_pred
                    audio_pred = torch.zeros_like(audio_latents)
                
                # Perform CFG
                if do_cfg:
                    video_pred_uncond, video_pred_text = video_pred.chunk(2)
                    audio_pred_uncond, audio_pred_text = audio_pred.chunk(2)
                    
                    guidance_scale = batch.guidance_scale
                    
                    video_pred = video_pred_uncond + guidance_scale * (video_pred_text - video_pred_uncond)
                    audio_pred = audio_pred_uncond + guidance_scale * (audio_pred_text - audio_pred_uncond)
                
                # Compute previous sample (Step)
                # Manual FlowMatch Euler step
                sigma = self.scheduler.sigmas[i]
                sigma_next = self.scheduler.sigmas[i + 1]
                dt = sigma_next - sigma
                
                # Update latents
                latents = latents + dt * video_pred
                audio_latents = audio_latents + dt * audio_pred
                
                # Update progress bar
                progress_bar.update()
                    
        # Save final result
        batch.latents = latents
        batch.audio_latents = audio_latents
        
        return batch

    def do_classifier_free_guidance(self, batch: Req) -> bool:
        return batch.guidance_scale > 1.0


from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.models.upsamplers.ltx_2_upsampler import upsample_video

class LTX2UpsamplingStage(PipelineStage):
    def __init__(self, upsampler, video_encoder_stats):
        super().__init__()
        self.upsampler = upsampler
        self.video_encoder_stats = video_encoder_stats

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Upsample video latents
        if batch.latents is not None:
            # Ensure latents are on correct device
            batch.latents = batch.latents.to(self.device)
            
            # upsample_video expects [B, C, F, H, W]
            # batch.latents is [B, C, F, H, W]
            batch.latents = upsample_video(batch.latents, self.video_encoder_stats, self.upsampler)
            
            # Update dimensions
            batch.height = batch.height * 2
            batch.width = batch.width * 2
            
            # Update raw_latent_shape
            batch.raw_latent_shape = batch.latents.shape
            
        return batch

class LTX2RefinementStage(LTX2AVDenoisingStage):
    def __init__(self, transformer, scheduler, distilled_sigmas):
        super().__init__(transformer, scheduler)
        self.distilled_sigmas = torch.tensor(distilled_sigmas)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. Add noise to latents
        noise_scale = self.distilled_sigmas[0].to(batch.latents.device)
        noise = torch.randn_like(batch.latents)
        batch.latents = batch.latents + noise * noise_scale
        
        # 2. Run denoising loop with distilled_sigmas
        # Save original sigmas
        original_sigmas = self.scheduler.sigmas
        original_timesteps = self.scheduler.timesteps
        original_num_inference_steps = self.scheduler.num_inference_steps
        
        # Set distilled sigmas
        self.scheduler.sigmas = self.distilled_sigmas.to(self.scheduler.sigmas.device)
        # Approximation for timesteps (FlowMatch doesn't strictly use them for step calculation if we use sigmas directly)
        self.scheduler.timesteps = self.scheduler.sigmas * 1000 
        self.scheduler.num_inference_steps = len(self.distilled_sigmas) - 1
        
        # Call parent forward
        try:
            batch = super().forward(batch, server_args)
        finally:
            # Restore original sigmas
            self.scheduler.sigmas = original_sigmas
            self.scheduler.timesteps = original_timesteps
            self.scheduler.num_inference_steps = original_num_inference_steps
        
        return batch

    def do_classifier_free_guidance(self, batch: Req) -> bool:
        return False # Stage 2 uses simple denoising (no CFG)

class LTX2AVDecodingStage(DecodingStage):
    """
    LTX-2 specific decoding stage that handles both video and audio decoding.
    """
    def __init__(self, vae, audio_vae, vocoder, pipeline=None):
        super().__init__(vae, pipeline)
        self.audio_vae = audio_vae
        self.vocoder = vocoder

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        # 1. Decode Video (using parent)
        output_batch = super().forward(batch, server_args)
        
        # 2. Decode Audio
        audio_latents = getattr(batch, "audio_latents", None)
        if audio_latents is not None:
            # Ensure models are on correct device
            device = self.device
            dtype = self.vae.dtype # Use video VAE dtype as reference?
            
            # audio_vae and vocoder might be on CPU if not moved explicitly?
            # Assuming they are on device.
            
            with torch.no_grad():
                # Decode spectrogram
                # audio_latents: [B, C, T, F]
                spectrogram = self.audio_vae(audio_latents)
                
                # Vocode to waveform
                # spectrogram: [B, C, T, F] -> [B, C, F, T] for vocoder?
                # LTX-2 Vocoder expects [B, C, F, T] (channels, mel_bins, time)?
                # Let's check vocoder.py forward:
                # x = x.transpose(2, 3) # (batch, channels, time, mel_bins) -> (batch, channels, mel_bins, time)
                # So it expects [B, C, T, F] input!
                
                waveform = self.vocoder(spectrogram)
                # waveform: [B, out_channels, time]
            
            # Dynamically attach to output_batch
            output_batch.audio_output = waveform
            
        return output_batch