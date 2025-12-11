# SPDX-License-Identifier: Apache-2.0
"""
Control image encoding stage for ControlNet-enabled diffusion pipelines.

This module contains the implementation of the control image encoding stage
that loads and prepares control images for ControlNet conditioning.
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ControlEncodingStage(PipelineStage):
    """
    Stage for loading and encoding control images for ControlNet.

    This stage handles loading control images from disk or URLs, resizing them
    to match the generation dimensions, VAE encoding, and patchification.

    The control image is processed to match the latent space format expected
    by the ControlNet, following the InstantX/diffusers pattern.
    """

    def __init__(self, vae=None) -> None:
        """
        Initialize the control encoding stage.

        Args:
            vae: VAE model for encoding control images to latent space.
                 If None, control images will be passed as raw tensors.
        """
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Load and encode the control image for ControlNet conditioning.

        Args:
            batch: The current batch information containing control_image_path.
            server_args: The inference arguments.

        Returns:
            The batch with encoded control_image tensor.
        """
        # Skip if no control image specified
        if batch.control_image_path is None:
            logger.debug("No control image specified, skipping control encoding")
            return batch

        # Load control image
        control_image = self._load_control_image(batch.control_image_path)

        # Resize to match generation dimensions
        target_size = (batch.width, batch.height)
        if control_image.size != target_size:
            logger.info(
                f"Resizing control image from {control_image.size} to {target_size}"
            )
            control_image = control_image.resize(target_size, Image.LANCZOS)

        # Convert to tensor [1, 3, H, W] in range [0, 1]
        control_tensor = self._image_to_tensor(control_image, server_args)

        # If VAE is available, encode to latent space and patchify
        if self.vae is not None:
            control_tensor = self._encode_and_patchify(control_tensor, server_args)
            logger.info(
                f"Control image VAE-encoded and patchified: shape={control_tensor.shape}, "
                f"dtype={control_tensor.dtype}, device={control_tensor.device}"
            )
        else:
            logger.info(
                f"Control image loaded (no VAE encoding): shape={control_tensor.shape}, "
                f"dtype={control_tensor.dtype}, device={control_tensor.device}"
            )

        # Store in batch
        batch.control_image = control_tensor

        return batch

    def _encode_and_patchify(
        self, control_tensor: torch.Tensor, server_args: ServerArgs
    ) -> torch.Tensor:
        """
        Encode control image with VAE and patchify to match latent format.

        Args:
            control_tensor: Control image tensor [1, 3, H, W] in range [0, 1]
            server_args: Server arguments

        Returns:
            Encoded and patchified tensor [1, num_patches, channels*4]
        """
        # Move VAE to correct device if needed
        device = control_tensor.device
        vae_device = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype

        if vae_device != device:
            # Temporarily move VAE to the control tensor device
            self.vae = self.vae.to(device)

        # Add temporal dimension [1, 3, H, W] -> [1, 3, 1, H, W]
        if control_tensor.ndim == 4:
            control_tensor = control_tensor.unsqueeze(2)

        # Convert from [0, 1] to [-1, 1] range expected by VAE
        control_tensor = (control_tensor * 2.0 - 1.0).clamp(-1, 1)

        # Cast input to match VAE dtype (VAE may be in float32)
        control_tensor = control_tensor.to(dtype=vae_dtype)

        # VAE encode: [1, 3, 1, H, W] -> [1, C, 1, H/8, W/8]
        # QwenImage VAE's encode() returns DiagonalGaussianDistribution directly
        posterior = self.vae.encode(control_tensor)
        latent = posterior.sample()

        # Cast back to the target dtype for the rest of the pipeline
        dit_precision = server_args.pipeline_config.dit_precision
        target_dtype = torch.bfloat16 if dit_precision in ("bf16", "bfloat16") else torch.float16
        latent = latent.to(dtype=target_dtype)

        # Apply VAE normalization using pre-computed shift_factor and scaling_factor
        # QwenImage VAE stores these as tensors: shift_factor (latents_mean) and scaling_factor (1/latents_std)
        if hasattr(self.vae, 'shift_factor') and self.vae.shift_factor is not None:
            shift_factor = self.vae.shift_factor.to(device=device, dtype=latent.dtype)
            scaling_factor = self.vae.scaling_factor.to(device=device, dtype=latent.dtype)
            latent = (latent - shift_factor) * scaling_factor

        # Permute: [1, C, 1, H/8, W/8] -> [1, 1, C, H/8, W/8]
        latent = latent.permute(0, 2, 1, 3, 4)

        # Patchify (pack 2x2 patches into sequence)
        # [1, 1, C, H/8, W/8] -> [1, num_patches, C*4]
        latent = self._pack_latents(latent)

        # Move VAE back to CPU if offloading is enabled
        if server_args.vae_cpu_offload and vae_device.type == "cpu":
            self.vae = self.vae.to("cpu")

        return latent

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pack latents by combining 2x2 spatial patches into the channel dimension.

        Args:
            latents: Tensor of shape [B, T, C, H, W]

        Returns:
            Packed tensor of shape [B, (H/2)*(W/2), C*4]
        """
        batch_size, num_frames, num_channels, height, width = latents.shape

        # Reshape to extract 2x2 patches
        # [B, T, C, H, W] -> [B, T, C, H/2, 2, W/2, 2]
        latents = latents.view(
            batch_size, num_frames, num_channels, height // 2, 2, width // 2, 2
        )

        # Permute to group patches: [B, T, H/2, W/2, C, 2, 2]
        latents = latents.permute(0, 1, 3, 5, 2, 4, 6)

        # Flatten patches: [B, T*(H/2)*(W/2), C*4]
        latents = latents.reshape(
            batch_size,
            num_frames * (height // 2) * (width // 2),
            num_channels * 4
        )

        return latents

    def _load_control_image(
        self, image_path: Union[str, Path]
    ) -> Image.Image:
        """
        Load control image from file path or URL.

        Args:
            image_path: Path to the control image file or URL.

        Returns:
            PIL Image in RGB format.

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image cannot be loaded.
        """
        image_path = str(image_path)

        # Handle URLs
        if image_path.startswith("http://") or image_path.startswith("https://"):
            try:
                import requests
                from io import BytesIO

                logger.info(f"Downloading control image from {image_path}")
                response = requests.get(image_path, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                raise ValueError(
                    f"Failed to download control image from {image_path}: {e}"
                )
        # Handle local files
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Control image not found at path: {image_path}"
                )
            try:
                image = Image.open(image_path)
            except Exception as e:
                raise ValueError(
                    f"Failed to load control image from {image_path}: {e}"
                )

        # Ensure RGB format
        if image.mode != "RGB":
            logger.debug(f"Converting control image from {image.mode} to RGB")
            image = image.convert("RGB")

        return image

    def _image_to_tensor(
        self, image: Image.Image, server_args: ServerArgs
    ) -> torch.Tensor:
        """
        Convert PIL Image to PyTorch tensor.

        Args:
            image: PIL Image in RGB format.
            server_args: Server arguments containing device and dtype info.

        Returns:
            Tensor of shape [1, 3, H, W] in range [0, 1].
        """
        # Convert to numpy array [H, W, 3] in range [0, 255]
        image_np = np.array(image).astype(np.float32)

        # Normalize to [0, 1]
        image_np = image_np / 255.0

        # Convert to tensor and rearrange to [3, H, W]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        # Add batch dimension [1, 3, H, W]
        image_tensor = image_tensor.unsqueeze(0)

        # Move to appropriate device and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Get dtype from pipeline config (dit_precision: "bf16" or "fp16")
        dit_precision = server_args.pipeline_config.dit_precision
        dtype = torch.bfloat16 if dit_precision in ("bf16", "bfloat16") else torch.float16

        image_tensor = image_tensor.to(device=device, dtype=dtype)

        return image_tensor
