import math

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

        # 2. Prepare Audio Latents (optional)
        if not getattr(batch, "generate_audio", True):
            batch.audio_latents = None
            batch.raw_audio_latent_shape = None
            return batch

        device = get_local_torch_device()
        if isinstance(batch.prompt_embeds, list) and batch.prompt_embeds:
            dtype = batch.prompt_embeds[0].dtype
        elif isinstance(batch.prompt_embeds, torch.Tensor):
            dtype = batch.prompt_embeds.dtype
        else:
            dtype = torch.float16
        generator = batch.generator
        config = server_args.pipeline_config

        # Calculate audio latent dimensions.
        # The Audio VAE latent time axis is ~25 Hz (16k / 160 hop / 4 downsample).
        # We compute latent frames from video duration, accounting for causal decode cropping.
        fps = getattr(batch, "fps", None) or 24
        duration_s = float(batch.num_frames) / float(fps)
        mel_hz = 100.0  # 16k / 160
        mel_frames_target = int(math.ceil(duration_s * mel_hz))
        down = int(getattr(config, "audio_latent_downsample_factor", 4))
        is_causal = True
        causal_crop = (down - 1) if is_causal else 0
        audio_latent_frames = int(math.ceil((mel_frames_target + causal_crop) / down))

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
            raise ValueError(
                "Audio latents not found in batch. If you want video-only, set `generate_audio=False`."
            )

        # Prepare extra kwargs for the model (guidance, etc.)
        # extra_step_kwargs = self.prepare_extra_step_kwargs(batch, server_args)
        
        # Denoising loop
        # num_warmup_steps = len(self.scheduler.timesteps) - self.scheduler.num_inference_steps * self.scheduler.order
        
        device = getattr(self.transformer, "device", None) or get_local_torch_device()
        target_dtype = getattr(self.transformer, "dtype", latents.dtype)

        # Ensure latents are on correct device/dtype
        latents = latents.to(device, dtype=target_dtype)
        audio_latents = audio_latents.to(device, dtype=target_dtype)

        extra_step_kwargs = getattr(batch, "extra_step_kwargs", None) or {}
        
        with self.progress_bar(total=self.scheduler.num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                # 1. Predict noise/velocity
                
                # Expand latents for CFG
                do_cfg = self.do_classifier_free_guidance(batch)
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                audio_latent_model_input = torch.cat([audio_latents] * 2) if do_cfg else audio_latents
                
                # Scale latents (FlowMatch usually doesn't, but check scheduler)
                t_device = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_device)
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
                        pos_mask = encoder_attention_mask[0]
                        neg_mask = (batch.negative_attention_mask or [None])[0]
                        encoder_attention_mask = (
                            torch.cat([neg_mask, pos_mask]) if neg_mask is not None else pos_mask
                        )
                    else:
                        encoder_attention_mask = encoder_attention_mask[0]

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
                    timestep=t_device,
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
                
                # Compute the previous noisy sample.
                latents = self.scheduler.step(
                    model_output=video_pred,
                    timestep=t_device,
                    sample=latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                audio_latents = self.scheduler.step(
                    model_output=audio_pred,
                    timestep=t_device,
                    sample=audio_latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                
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
        if audio_latents is not None and getattr(batch, "generate_audio", True):
            with torch.no_grad():
                spectrogram = self.audio_vae(audio_latents)
                waveform = self.vocoder(spectrogram)

            # Pack audio alongside per-sample video tensor so entrypoints can save both.
            if isinstance(output_batch.output, torch.Tensor):
                videos = output_batch.output
                output_batch.output = [(videos[i], waveform[i]) for i in range(videos.shape[0])]
        
        return output_batch
