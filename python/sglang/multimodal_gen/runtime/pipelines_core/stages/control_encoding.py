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
    to match the generation dimensions, and converting them to tensors.

    Note: This stage does NOT perform control-specific preprocessing (e.g., canny
    edge detection, depth estimation). Users are expected to provide pre-processed
    control images, following the diffusers pattern.
    """

    def __init__(self) -> None:
        """
        Initialize the control encoding stage.
        """
        super().__init__()

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

        # Store in batch
        batch.control_image = control_tensor
        logger.info(
            f"Control image loaded and encoded: shape={control_tensor.shape}, "
            f"dtype={control_tensor.dtype}, device={control_tensor.device}"
        )

        return batch

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
        dtype = torch.bfloat16 if server_args.dtype == "bfloat16" else torch.float16

        image_tensor = image_tensor.to(device=device, dtype=dtype)

        return image_tensor
