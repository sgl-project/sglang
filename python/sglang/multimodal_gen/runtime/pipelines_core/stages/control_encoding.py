# SPDX-License-Identifier: Apache-2.0
"""
Control image encoding stage for ControlNet-enabled diffusion pipelines.

This module contains the implementation of the control image encoding stage
that loads and prepares control images for ControlNet conditioning.

Supports both:
- Union ControlNet (canny, depth, pose, soft edge) - single control image
- Inpainting ControlNet - control image + mask
"""

import os
from pathlib import Path
from typing import Union

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

    Supports:
    - Union ControlNet: control_image_path only → output shape [1, seq, 64]
    - Inpainting ControlNet: control_image_path + control_mask_path → output shape [1, seq, 68]
    """

    def __init__(self, vae=None, extra_condition_channels: int = 0) -> None:
        """
        Initialize the control encoding stage.

        Args:
            vae: VAE model for encoding control images to latent space.
                 If None, control images will be passed as raw tensors.
            extra_condition_channels: Number of extra channels for inpainting mask (typically 4).
                                      If 0, no mask processing is done.
        """
        super().__init__()
        self.vae = vae
        self.extra_condition_channels = extra_condition_channels

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Load and encode the control image(s) for ControlNet conditioning.

        Supports both single control image (backwards compatible) and multiple
        control images for multi-controlnet setups.

        Args:
            batch: The current batch information containing control_image_path
                   or control_image_paths, and optionally control_mask_path.
            server_args: The inference arguments.

        Returns:
            The batch with encoded control_image or control_images tensor(s).
        """
        # Check for multiple control images first
        if batch.control_image_paths is not None and len(batch.control_image_paths) > 0:
            return self._encode_multiple_images(batch, server_args)

        # Skip if no control image specified
        if batch.control_image_path is None:
            logger.debug("No control image specified, skipping control encoding")
            return batch

        # Single image encoding (backwards compatible)
        return self._encode_single_image(batch, server_args)

    def _encode_single_image(self, batch: Req, server_args: ServerArgs) -> Req:
        """Encode a single control image (backwards compatible)."""
        # Load control image
        control_image = self._load_image(batch.control_image_path)

        # Resize to match generation dimensions
        target_size = (batch.width, batch.height)
        if control_image.size != target_size:
            logger.info(
                f"Resizing control image from {control_image.size} to {target_size}"
            )
            control_image = control_image.resize(target_size, Image.LANCZOS)

        # Check if this is an inpainting request
        is_inpainting = batch.control_mask_path is not None

        if is_inpainting:
            # Use specialized inpainting encoding
            batch.control_image = self._encode_for_inpainting(
                control_image, batch.control_mask_path, target_size, server_args
            )
        else:
            # Standard control image encoding (Union ControlNet)
            # Convert to tensor [1, 3, H, W] in range [0, 1]
            control_tensor = self._image_to_tensor(control_image, server_args)

            # If VAE is available, encode to latent space and patchify
            if self.vae is not None:
                control_latent = self._encode_and_patchify(control_tensor, server_args)
                logger.info(
                    f"Control image VAE-encoded and patchified: shape={control_latent.shape}, "
                    f"dtype={control_latent.dtype}, device={control_latent.device}"
                )
                batch.control_image = control_latent
            else:
                logger.info(
                    f"Control image loaded (no VAE encoding): shape={control_tensor.shape}, "
                    f"dtype={control_tensor.dtype}, device={control_tensor.device}"
                )
                batch.control_image = control_tensor

        return batch

    def _encode_multiple_images(self, batch: Req, server_args: ServerArgs) -> Req:
        """Encode multiple control images for multi-controlnet."""
        target_size = (batch.width, batch.height)
        control_latents = []

        logger.info(f"Encoding {len(batch.control_image_paths)} control images...")

        for i, image_path in enumerate(batch.control_image_paths):
            # Load control image
            control_image = self._load_image(image_path)

            # Resize to match generation dimensions
            if control_image.size != target_size:
                logger.info(
                    f"Resizing control image {i+1} from {control_image.size} to {target_size}"
                )
                control_image = control_image.resize(target_size, Image.LANCZOS)

            # Convert to tensor [1, 3, H, W] in range [0, 1]
            control_tensor = self._image_to_tensor(control_image, server_args)

            # If VAE is available, encode to latent space and patchify
            if self.vae is not None:
                control_latent = self._encode_and_patchify(control_tensor, server_args)
                logger.info(
                    f"Control image {i+1} VAE-encoded: shape={control_latent.shape}"
                )
            else:
                control_latent = control_tensor
                logger.info(
                    f"Control image {i+1} loaded (no VAE encoding): shape={control_latent.shape}"
                )

            control_latents.append(control_latent)

        # Store list of control images
        batch.control_images = control_latents
        logger.info(
            f"Encoded {len(control_latents)} control images for multi-controlnet"
        )

        return batch

    def _encode_for_inpainting(
        self,
        control_image: Image.Image,
        mask_path: str,
        target_size: tuple,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        """
        Encode control image and mask for inpainting ControlNet.

        This follows the diffusers QwenImageControlNetInpaintPipeline approach:
        1. Load and resize mask to match image
        2. Apply mask to image (set masked regions to -1 before VAE encoding)
        3. VAE encode the masked image
        4. Concatenate mask with image latents along channel dimension
        5. Patchify the combined tensor

        Args:
            control_image: PIL Image to use as control
            mask_path: Path or URL to the mask (white=inpaint, black=keep)
            target_size: Target (width, height)
            server_args: Server arguments

        Returns:
            Encoded tensor [1, num_patches, 68] (64 image channels + 4 mask channels)
        """
        if self.vae is None:
            raise ValueError("VAE is required for inpainting ControlNet encoding")

        # Load and resize mask
        mask_image = self._load_image(mask_path, mode="L")
        if mask_image.size != target_size:
            logger.info(f"Resizing mask from {mask_image.size} to {target_size}")
            mask_image = mask_image.resize(target_size, Image.NEAREST)

        # Convert mask to tensor [H, W] normalized to [0, 1]
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        # Invert mask: white (1.0) in input means inpaint, but we want 1.0 = keep
        # So invert: masked regions become 0.0, keep regions become 1.0
        mask_np_inverted = 1.0 - mask_np

        # Convert control image to tensor [1, 3, H, W] in range [0, 1]
        image_np = np.array(control_image).astype(np.float32) / 255.0
        image_tensor = (
            torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]
        mask_tensor = (
            torch.from_numpy(mask_np_inverted).unsqueeze(0).unsqueeze(0)
        )  # [1, 1, H, W]

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dit_precision = server_args.pipeline_config.dit_precision
        dtype = (
            torch.bfloat16 if dit_precision in ("bf16", "bfloat16") else torch.float16
        )

        image_tensor = image_tensor.to(device=device, dtype=dtype)
        mask_tensor = mask_tensor.to(device=device, dtype=dtype)

        # Apply mask to image: set masked regions (mask=0) to -1
        # This is done by: image * mask + (mask - 1)
        # Where mask=1 (keep): image * 1 + 0 = image
        # Where mask=0 (inpaint): image * 0 + (-1) = -1
        masked_image = image_tensor * mask_tensor + (mask_tensor - 1.0)

        logger.info(
            f"Inpainting: masked_image range=[{masked_image.min():.3f}, {masked_image.max():.3f}]"
        )

        # Add temporal dimension [1, 3, H, W] -> [1, 3, 1, H, W]
        masked_image = masked_image.unsqueeze(2)

        # VAE expects [-1, 1] range, our masked image is already in that range
        # (original [0,1] scaled to [0,1] for keep regions, -1 for inpaint regions)
        # We need to scale the keep regions properly: from [0,1] to [-1,1]
        # masked_image = (masked_image * 2.0 - 1.0) won't work because -1 regions
        # So we apply the scaling only to the valid regions:
        # For kept regions (mask=1): scale from [0,1] to [-1,1]
        # For masked regions: keep at -1
        mask_3d = mask_tensor.unsqueeze(2).expand_as(masked_image)
        # Scale kept regions: (image * 2 - 1), masked regions stay -1
        masked_image_scaled = torch.where(
            mask_3d > 0.5,
            masked_image * 2.0 - 1.0,  # Scale kept regions to [-1, 1]
            torch.full_like(masked_image, -1.0),  # Masked regions = -1
        )

        # Move VAE to correct device if needed
        vae_device = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        if vae_device != device:
            self.vae = self.vae.to(device)

        # Cast input to match VAE dtype
        masked_image_scaled = masked_image_scaled.to(dtype=vae_dtype)

        # VAE encode: [1, 3, 1, H, W] -> [1, C, 1, H/8, W/8]
        posterior = self.vae.encode(masked_image_scaled)
        image_latents = posterior.sample()

        # Cast back to target dtype
        image_latents = image_latents.to(dtype=dtype)

        # Apply VAE normalization
        if hasattr(self.vae, "shift_factor") and self.vae.shift_factor is not None:
            shift_factor = self.vae.shift_factor.to(
                device=device, dtype=image_latents.dtype
            )
            scaling_factor = self.vae.scaling_factor.to(
                device=device, dtype=image_latents.dtype
            )
            image_latents = (image_latents - shift_factor) * scaling_factor

        logger.info(f"Inpainting: image_latents shape={image_latents.shape}")

        # Downsample mask to latent space (H/8, W/8)
        height, width = target_size[1], target_size[0]
        latent_h, latent_w = height // 8, width // 8

        # Invert mask for ControlNet conditioning: diffusers does mask = 1 - mask
        # Input mask: white=1=inpaint, black=0=keep
        # ControlNet expects: 0=inpaint, 1=keep (inverted)
        mask_for_latent = torch.from_numpy(1.0 - mask_np).to(device=device, dtype=dtype)
        mask_for_latent = mask_for_latent.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask_latent = torch.nn.functional.interpolate(
            mask_for_latent, size=(latent_h, latent_w), mode="nearest"
        )

        # Add temporal dimension: [1, 1, H/8, W/8] -> [1, 1, 1, H/8, W/8]
        mask_latent = mask_latent.unsqueeze(2)

        logger.info(f"Inpainting: mask_latent shape={mask_latent.shape}")

        # Permute both to match: [1, C, 1, H/8, W/8] -> [1, 1, C, H/8, W/8]
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        mask_latent = mask_latent.permute(0, 2, 1, 3, 4)

        # Concatenate mask with image latents along channel dimension
        # [1, 1, 16, H/8, W/8] + [1, 1, 1, H/8, W/8] -> [1, 1, 17, H/8, W/8]
        combined = torch.cat([image_latents, mask_latent], dim=2)

        logger.info(f"Inpainting: combined shape={combined.shape}")

        # Patchify: [1, 1, 17, H/8, W/8] -> [1, num_patches, 17*4=68]
        control_latent = self._pack_latents(combined)

        logger.info(
            f"Inpainting control encoded: shape={control_latent.shape}, "
            f"dtype={control_latent.dtype}"
        )

        # Move VAE back to CPU if offloading is enabled
        if server_args.vae_cpu_offload and vae_device.type == "cpu":
            self.vae = self.vae.to("cpu")

        return control_latent

    def _encode_mask(
        self,
        mask_path: str,
        target_size: tuple,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        """
        Load and encode the inpainting mask.

        The mask is processed to match the latent space dimensions without VAE encoding.
        It's downsampled and patchified to match the control image latent shape.

        Args:
            mask_path: Path or URL to the mask image (white=inpaint, black=keep)
            target_size: Target (width, height) to resize mask to
            server_args: Server arguments

        Returns:
            Encoded mask tensor [1, num_patches, 4]
        """
        # Load mask image
        mask_image = self._load_image(mask_path, mode="L")  # Grayscale

        # Resize to match target size
        if mask_image.size != target_size:
            logger.info(f"Resizing mask from {mask_image.size} to {target_size}")
            mask_image = mask_image.resize(target_size, Image.NEAREST)

        # Convert to tensor [H, W] in range [0, 1]
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dit_precision = server_args.pipeline_config.dit_precision
        dtype = (
            torch.bfloat16 if dit_precision in ("bf16", "bfloat16") else torch.float16
        )
        mask_tensor = mask_tensor.to(device=device, dtype=dtype)

        # Downsample to latent space dimensions (H/8, W/8)
        # Using average pooling to downsample
        height, width = mask_tensor.shape
        latent_h, latent_w = height // 8, width // 8

        # Reshape for pooling: [1, 1, H, W]
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor, size=(latent_h, latent_w), mode="nearest"
        )

        # Expand to 4 channels (matching extra_condition_channels=4)
        # [1, 1, H/8, W/8] -> [1, 4, H/8, W/8]
        mask_tensor = mask_tensor.expand(-1, 4, -1, -1)

        # Add temporal dimension: [1, 4, H/8, W/8] -> [1, 4, 1, H/8, W/8]
        mask_tensor = mask_tensor.unsqueeze(2)

        # Permute to [1, 1, 4, H/8, W/8]
        mask_tensor = mask_tensor.permute(0, 2, 1, 3, 4)

        # Patchify to match control image latent
        # [1, 1, 4, H/8, W/8] -> [1, num_patches, 4*4] but we want [1, num_patches, 4]
        # So we use a different packing that doesn't multiply channels
        mask_tensor = self._pack_mask(mask_tensor)

        return mask_tensor

    def _pack_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Pack mask by combining 2x2 spatial patches, averaging values.

        Args:
            mask: Tensor of shape [B, T, C, H, W] where C=4

        Returns:
            Packed tensor of shape [B, (H/2)*(W/2), C]
        """
        batch_size, num_frames, num_channels, height, width = mask.shape

        # Reshape to extract 2x2 patches
        # [B, T, C, H, W] -> [B, T, C, H/2, 2, W/2, 2]
        mask = mask.view(
            batch_size, num_frames, num_channels, height // 2, 2, width // 2, 2
        )

        # Average over the 2x2 patch (dims 4 and 6)
        mask = mask.mean(dim=(4, 6))

        # [B, T, C, H/2, W/2] -> [B, T, H/2, W/2, C]
        mask = mask.permute(0, 1, 3, 4, 2)

        # Flatten: [B, T*(H/2)*(W/2), C]
        mask = mask.reshape(
            batch_size, num_frames * (height // 2) * (width // 2), num_channels
        )

        return mask

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
        target_dtype = (
            torch.bfloat16 if dit_precision in ("bf16", "bfloat16") else torch.float16
        )
        latent = latent.to(dtype=target_dtype)

        # Apply VAE normalization using pre-computed shift_factor and scaling_factor
        # QwenImage VAE stores these as tensors: shift_factor (latents_mean) and scaling_factor (1/latents_std)
        if hasattr(self.vae, "shift_factor") and self.vae.shift_factor is not None:
            shift_factor = self.vae.shift_factor.to(device=device, dtype=latent.dtype)
            scaling_factor = self.vae.scaling_factor.to(
                device=device, dtype=latent.dtype
            )
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
            batch_size, num_frames * (height // 2) * (width // 2), num_channels * 4
        )

        return latents

    def _load_image(
        self, image_path: Union[str, Path], mode: str = "RGB"
    ) -> Image.Image:
        """
        Load image from file path or URL.

        Args:
            image_path: Path to the image file or URL.
            mode: Image mode ("RGB" for control images, "L" for grayscale masks)

        Returns:
            PIL Image in the specified mode.

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image cannot be loaded.
        """
        image_path = str(image_path)

        # Handle URLs
        if image_path.startswith("http://") or image_path.startswith("https://"):
            try:
                from io import BytesIO

                import requests

                logger.info(f"Downloading image from {image_path}")
                response = requests.get(image_path, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                raise ValueError(f"Failed to download image from {image_path}: {e}")
        # Handle local files
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at path: {image_path}")
            try:
                image = Image.open(image_path)
            except Exception as e:
                raise ValueError(f"Failed to load image from {image_path}: {e}")

        # Convert to specified mode
        if image.mode != mode:
            logger.debug(f"Converting image from {image.mode} to {mode}")
            image = image.convert(mode)

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
        dtype = (
            torch.bfloat16 if dit_precision in ("bf16", "bfloat16") else torch.float16
        )

        image_tensor = image_tensor.to(device=device, dtype=dtype)

        return image_tensor
