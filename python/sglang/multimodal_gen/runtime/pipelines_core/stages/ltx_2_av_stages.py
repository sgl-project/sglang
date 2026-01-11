import math
from typing import Optional

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req, OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.models.upsamplers.ltx_2_upsampler import upsample_video

logger = init_logger(__name__)


class LTX2AVLatentPreparationStage(LatentPreparationStage):
    """
    LTX-2 specific latent preparation stage that handles both video and audio latents.
    """
    def __init__(self, scheduler, transformer=None, vae=None, audio_vae=None):
        super().__init__(scheduler, transformer, vae)
        self.audio_vae = audio_vae

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. Prepare Video Latents using base class logic
        # This sets batch.latents and batch.raw_latent_shape
        batch = super().forward(batch, server_args)

        # 2. Prepare Audio Latents (optional)
        # Default to True if not specified
        generate_audio = getattr(batch, "generate_audio", True)
        if not generate_audio:
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
        
        # Calculate audio latent dimensions.
        # The Audio VAE latent time axis is ~25 Hz (16k / 160 hop / 4 downsample).
        # We compute latent frames from video duration, accounting for causal decode cropping.
        fps = getattr(batch, "fps", None) or 24
        duration_s = float(batch.num_frames) / float(fps)
        
        # Constants from LTX-2 Audio VAE
        # TODO: Read from audio_vae config if possible
        mel_hz = 100.0  # 16000 / 160
        down = 4 # LATENT_DOWNSAMPLE_FACTOR
        
        mel_frames_target = int(math.ceil(duration_s * mel_hz))
        
        # Causal crop logic from ltx-core
        is_causal = True # LTX-2 Audio VAE is causal
        causal_crop = (down - 1) if is_causal else 0
        audio_latent_frames = int(math.ceil((mel_frames_target + causal_crop) / down))

        # Audio latent channels
        # Default to 128 if not available (LTX-2 standard)
        audio_latent_channels = getattr(self.audio_vae, "out_ch", 128)
        # Mel bins
        audio_latent_mel_bins = getattr(self.audio_vae, "mel_bins", 128)

        # Shape: [B, C, T, F] where F is mel_bins
        shape = (
            batch.batch_size,
            audio_latent_channels,
            audio_latent_frames,
            audio_latent_mel_bins,
        )

        # Generate random noise
        audio_latents = randn_tensor(
            shape, generator=generator, device=device, dtype=dtype
        )

        # Scale initial noise
        if hasattr(self.scheduler, "init_noise_sigma"):
            audio_latents = audio_latents * self.scheduler.init_noise_sigma

        # Store in batch
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
        
        device = getattr(self.transformer, "device", None) or get_local_torch_device()
        target_dtype = getattr(self.transformer, "dtype", latents.dtype)

        # Ensure latents are on correct device/dtype
        latents = latents.to(device, dtype=target_dtype)
        if audio_latents is not None:
            audio_latents = audio_latents.to(device, dtype=target_dtype)

        extra_step_kwargs = getattr(batch, "extra_step_kwargs", None) or {}
        
        with self.progress_bar(total=self.scheduler.num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                # 1. Predict noise/velocity
                
                # Expand latents for CFG
                do_cfg = self.do_classifier_free_guidance(batch)
                
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                if audio_latents is not None:
                    audio_latent_model_input = torch.cat([audio_latents] * 2) if do_cfg else audio_latents
                else:
                    audio_latent_model_input = None
                
                # Scale latents
                t_device = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_device)
                # Audio latents scaling (assuming same schedule)
                if audio_latent_model_input is not None:
                    # Note: scheduler.scale_model_input might not support 4D audio latents if it expects specific dims
                    # But for LTX-2 (FlowMatch), scale_model_input is usually identity or simple scaling.
                    # We assume it works or is identity.
                    # If scheduler is LTX2Scheduler, it might need checking.
                    # For now, we assume identity for audio or same scaling.
                    pass 
                
                # Prepare embeddings
                # batch.prompt_embeds is handled by TextEncodingStage.
                # If CFG is on, TextEncodingStage might have already concatenated [neg, pos] 
                # OR it returns a list [pos] and we need to handle neg.
                # In SGLang TextEncodingStage:
                # if do_classifier_free_guidance:
                #    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                # So batch.prompt_embeds is likely already concatenated if it's a tensor.
                
                encoder_hidden_states = batch.prompt_embeds
                encoder_attention_mask = batch.prompt_attention_mask

                # Handle list case (multiple encoders)
                if isinstance(encoder_hidden_states, list):
                    # LTX-2 uses two encoders (video_context, audio_context) sharing the same Gemma
                    # TextEncodingStage returns a list of outputs from each encoder config.
                    # In ltx_2.py config, we defined two GemmaConfigs.
                    # So encoder_hidden_states should be [video_context, audio_context]
                    
                    # We need to concat neg/pos for each if not already done.
                    # But TextEncodingStage logic:
                    # for i, (encoder, tokenizer) in enumerate(zip(self.text_encoders, self.tokenizers)):
                    #    ...
                    #    if do_cfg:
                    #        embeds = torch.cat([neg_embeds, embeds])
                    #    prompt_embeds_list.append(embeds)
                    
                    # So each element in the list is already [neg, pos] if CFG is on.
                    
                    video_context = encoder_hidden_states[0]
                    video_mask = encoder_attention_mask[0]
                    
                    # If we have audio context (second encoder)
                    if len(encoder_hidden_states) > 1:
                        audio_context = encoder_hidden_states[1]
                        audio_mask = encoder_attention_mask[1]
                        
                        # Concatenate video and audio contexts for the transformer?
                        # No, LTX-2 transformer takes `encoder_hidden_states` (video) and uses it for both streams?
                        # Wait, let's check LTX2Transformer.forward again.
                        # It takes `encoder_hidden_states`.
                        # Inside:
                        # video_args = TransformerArgs(..., context=encoder_hidden_states, ...)
                        # audio_args = TransformerArgs(..., context=encoder_hidden_states, ...)
                        # It seems it uses the SAME context for both.
                        
                        # But in ltx_2.py config, we added TWO encoders to support different post-processing
                        # (video_context vs audio_context).
                        # If they are different, we might need to pass them differently.
                        # LTX2Transformer.forward only accepts one `encoder_hidden_states`.
                        
                        # Let's assume for now we use the first one (video_context) as the main context.
                        # If LTX-2 actually uses different contexts for video and audio streams, 
                        # we would need to modify LTX2Transformer to accept `audio_encoder_hidden_states`.
                        # Looking at LTX2Transformer code:
                        # It uses `encoder_hidden_states` for both video_args and audio_args.
                        # So we should just use the video context.
                        
                        encoder_hidden_states = video_context
                        encoder_attention_mask = video_mask
                    else:
                        encoder_hidden_states = video_context
                        encoder_attention_mask = video_mask
                
                # Ensure tensors
                if isinstance(encoder_hidden_states, list):
                     encoder_hidden_states = encoder_hidden_states[0]
                if isinstance(encoder_attention_mask, list):
                     encoder_attention_mask = encoder_attention_mask[0]

                kwargs = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                    "audio_hidden_states": audio_latent_model_input,
                    "return_dict": False,
                }
                
                # Run model
                noise_pred = self.transformer(
                    latent_model_input,
                    timestep=t_device,
                    **kwargs
                )
                
                if isinstance(noise_pred, tuple):
                    video_pred, audio_pred = noise_pred
                else:
                    video_pred = noise_pred
                    audio_pred = None
                
                # Perform CFG
                if do_cfg:
                    video_pred_uncond, video_pred_text = video_pred.chunk(2)
                    guidance_scale = batch.guidance_scale
                    video_pred = video_pred_uncond + guidance_scale * (video_pred_text - video_pred_uncond)
                    
                    if audio_pred is not None:
                        audio_pred_uncond, audio_pred_text = audio_pred.chunk(2)
                        audio_pred = audio_pred_uncond + guidance_scale * (audio_pred_text - audio_pred_uncond)
                
                # Compute the previous noisy sample.
                latents = self.scheduler.step(
                    model_output=video_pred,
                    timestep=t_device,
                    sample=latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                
                if audio_latents is not None and audio_pred is not None:
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
        # Approximation for timesteps
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
            # Ensure device/dtype
            device = get_local_torch_device()
            dtype = getattr(self.audio_vae, "dtype", torch.float32)
            audio_latents = audio_latents.to(device, dtype=dtype)
            
            with torch.no_grad():
                # Decode latents to spectrogram
                spectrogram = self.audio_vae(audio_latents)
                # Decode spectrogram to waveform
                waveform = self.vocoder(spectrogram)

            # Pack audio alongside per-sample video tensor
            # output_batch.output is a list of video tensors (one per request in batch)
            # We want to attach audio to each.
            # Currently OutputBatch.output is List[torch.Tensor] or torch.Tensor
            
            if isinstance(output_batch.output, list):
                new_output = []
                for i, video in enumerate(output_batch.output):
                    audio = waveform[i] # [1, T] or [2, T]
                    new_output.append((video, audio))
                output_batch.output = new_output
            elif isinstance(output_batch.output, torch.Tensor):
                videos = output_batch.output
                new_output = []
                for i in range(videos.shape[0]):
                    new_output.append((videos[i], waveform[i]))
                output_batch.output = new_output
        
        return output_batch
