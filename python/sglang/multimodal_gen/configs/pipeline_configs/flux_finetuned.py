"""
Pipeline configuration for Flux fine-tuned/distilled models.

This module provides specialized handling for Flux fine-tuned models from HuggingFace,
such as fal/FLUX.2-Tiny-AutoEncoder and other community fine-tuned variants.

Key differences from standard Flux2PipelineConfig:
- Handles custom VAE architectures loaded via auto_map
- Supports both patchified (128 channels) and unpatchified (32 channels) latents
- Dynamically adapts scale/shift based on VAE type
- Properly handles 5D latents (batch, channels, frames, height, width) for decoding
"""

from dataclasses import dataclass

import torch

from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2PipelineConfig,
    _unpatchify_latents,
)


@dataclass
class Flux2FinetunedPipelineConfig(Flux2PipelineConfig):
    """
    Pipeline configuration for Flux fine-tuned/distilled models.
    
    This configuration automatically detects and handles custom VAE architectures
    (e.g., Flux2TinyAutoEncoder) loaded via HuggingFace's auto_map mechanism.
    
    Features:
    - Automatic VAE type detection (standard vs. distilled)
    - Proper handling of patchified/unpatchified latents
    - Support for custom scaling factors from fine-tuned models
    - 5D latents support for both single-frame and multi-frame generation
    """

    def preprocess_decoding(self, latents: torch.Tensor, server_args=None) -> torch.Tensor:
        """
        Preprocess latents before decoding.
        
        Handles both standard Flux2 VAE and fine-tuned/distilled VAEs:
        - Standard Flux2 VAE (has bn): needs unpatchify (128 channels -> 32 channels)
        - Distilled/Finetuned VAE (no bn): keeps patchified latents (128 channels)
        
        Also handles 5D latents (batch, channels, frames, height, width) by converting
        to 4D (batch, channels, height, width) for single-frame cases.
        
        Args:
            latents: Input latents tensor, can be 4D or 5D
            server_args: Server arguments containing VAE reference
            
        Returns:
            Preprocessed latents ready for VAE decoding
        """
        # Handle 5D latents (batch, channels, frames, height, width)
        # For single frame, squeeze frames dimension to 4D
        original_shape = latents.shape
        if latents.ndim == 5:
            batch_size, channels, frames, height, width = latents.shape
            if frames == 1:
                # Single frame: squeeze to 4D
                latents = latents.squeeze(2)  # (batch, channels, height, width)
            else:
                # Multi-frame: reshape to (batch * frames, channels, height, width)
                # This allows processing all frames at once
                latents = latents.permute(0, 2, 1, 3, 4).contiguous()
                latents = latents.view(batch_size * frames, channels, height, width)
        
        # Check if VAE has bn attribute to determine if it's a standard or distilled VAE
        vae = getattr(server_args, '_current_vae', None) if server_args else None
        if vae is not None and hasattr(vae, 'bn') and vae.bn is not None:
            # Standard Flux2 VAE: needs unpatchify (128 channels -> 32 channels)
            latents = _unpatchify_latents(latents)
        else:
            # Distilled/Finetuned VAE or unknown: keep patchified latents (128 channels)
            # Examples: Flux2TinyAutoEncoder, other community fine-tuned VAEs
            pass
        
        return latents

    def get_decode_scale_and_shift(self, device, dtype, vae):
        """
        Get scale and shift for decoding.
        
        Dynamically adapts based on VAE type:
        - Standard Flux2 VAE (has bn): uses BatchNorm statistics
        - Distilled/Finetuned VAE (no bn): uses scaling_factor from config
        
        Args:
            device: Target device for tensors
            dtype: Target dtype for tensors
            vae: VAE model instance
            
        Returns:
            Tuple of (scaling_factor, shift_factor)
            - scaling_factor: Tensor or scalar to divide latents by
            - shift_factor: Tensor or scalar to add to latents (None for distilled VAEs)
        """
        vae_arch_config = self.vae_config.arch_config
        
        # Check if VAE has bn attribute (standard Flux2 VAE)
        if hasattr(vae, 'bn') and vae.bn is not None:
            # Standard Flux2 VAE: use BatchNorm statistics
            latents_bn_mean = (
                vae.bn.running_mean.view(1, -1, 1, 1).to(device=device).to(device, dtype)
            )
            latents_bn_std = torch.sqrt(
                vae.bn.running_var.view(1, -1, 1, 1) + vae_arch_config.batch_norm_eps
            ).to(device, dtype)
            return 1 / latents_bn_std, latents_bn_mean
        else:
            # Distilled/Finetuned VAE: use scaling_factor from config
            # For decode, we need to divide latents by scaling_factor (symmetric to encode)
            # So we return scaling_factor directly, not 1/scaling_factor
            scaling_factor = (
                getattr(vae.config, "scaling_factor", None)
                if hasattr(vae, "config")
                else getattr(vae, "scaling_factor", None)
            ) or getattr(vae_arch_config, "scaling_factor", 0.13025)
            
            # Create scale tensor with proper shape for broadcasting
            # Shape (1, 1, 1, 1) works for both 4D and 5D latents
            scale = torch.tensor(scaling_factor, device=device, dtype=dtype).view(1, 1, 1, 1)
            return scale, None

