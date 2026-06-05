import base64
import io
import os
import shutil
import time
import uuid

import folder_paths
import numpy as np
import torch
from comfy_api.input import VideoInput
from PIL import Image


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numpy_image(image: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI image tensor to uint8 numpy array (H, W, C)."""
    if image.dim() == 4:
        image = image[0]
    if image.dim() == 3 and image.shape[0] in (1, 3, 4):
        image = image.permute(1, 2, 0)
    elif image.dim() == 2:
        image = image.unsqueeze(-1)
    np_img = image.detach().cpu().numpy()
    np_img = np.clip(np_img, 0.0, 1.0)
    np_img = (np_img * 255).astype(np.uint8)
    if np_img.shape[-1] == 1:
        np_img = np.repeat(np_img, 3, axis=-1)
    return np_img


def _to_hwc_tensor(image: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI image tensor to HWC format (normalized [0, 1])."""
    img = image.clone()
    if img.dim() == 4:
        img = img[0]
    if img.dim() == 3 and img.shape[0] in (1, 3, 4):
        img = img.permute(1, 2, 0)
    elif img.dim() == 2:
        img = img.unsqueeze(-1)

    img = torch.clamp(img, 0.0, 1.0)
    if img.shape[-1] == 1:
        img = img.repeat(1, 1, 3)

    return img


def is_empty_image(image: torch.Tensor, tolerance: float = 1e-6) -> bool:
    """
    Check if the input image is an empty/solid color image (like ComfyUI's empty image).
    Args:
        image: Input tensor image in ComfyUI format (BCHW, CHW, HWC, etc.)
        tolerance: Tolerance for floating point comparison (default: 1e-6)

    Returns:
        True if the image is empty (all pixels have same color), False otherwise
    """
    if image is None:
        return True

    # Convert to HWC format
    img_hwc = _to_hwc_tensor(image)

    # Get the first pixel's RGB values
    first_pixel = img_hwc[0, 0, :]

    h, w, c = img_hwc.shape
    pixels = img_hwc.reshape(-1, c)

    diff = torch.abs(pixels - first_pixel)
    max_diff = torch.max(diff)

    return max_diff.item() <= tolerance


def get_image_path(image: torch.Tensor) -> str:
    """
    Save tensor image to ComfyUI temp directory as PNG and return the path.
    """
    temp_dir = folder_paths.get_temp_directory()

    # Build file name
    ts = time.strftime("%Y%m%d-%H%M%S")
    unique = uuid.uuid4().hex[:8]
    file_name = f"sgl_output_{ts}_{unique}.png"
    file_path = os.path.join(temp_dir, file_name)

    # Save image
    np_img = _to_numpy_image(image)
    img = Image.fromarray(np_img)
    img.save(file_path, format="PNG")

    return file_path


def convert_b64_to_tensor_image(b64_image: str) -> torch.Tensor:
    """
    Convert base64 encoded image to ComfyUI IMAGE format (torch.Tensor).

    Args:
        b64_image: Base64 encoded image string

    Returns:
        torch.Tensor with shape [batch_size, height, width, channels] (BHWC format),
        values normalized to [0, 1] range, RGB format (3 channels)
    """
    # Decode base64
    image_bytes = base64.b64decode(b64_image)

    # Open image and convert to RGB
    pil_image = Image.open(io.BytesIO(image_bytes))
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(pil_image).astype(np.float32) / 255.0

    # Add batch dimension: [height, width, channels] -> [1, height, width, channels]
    image_array = image_array[np.newaxis, ...]

    # Convert to torch.Tensor
    tensor_image = torch.from_numpy(image_array)

    return tensor_image


class SGLDVideoInput(VideoInput):
    def __init__(self, video_path: str, height: int, width: int):
        super().__init__()

        self.video_path = video_path
        self.height = height
        self.width = width

    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        return self.width, self.height

    def get_components(self):
        """
        Returns the components of the video input.
        This is required by the VideoInput abstract base class.
        """
        return [self.video_path]

    def save_to(self, path: str, format=None, codec=None, metadata=None):
        """
        Abstract method to save the video input to a file.
        """
        save_path = path
        # Copy video file from video_path to save_path
        if os.path.exists(self.video_path):
            # Ensure destination directory exists
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            shutil.copy2(self.video_path, save_path)


def convert_video_to_comfy_video(
    video_path: str, height: int, width: int
) -> VideoInput:
    """
    Convert video to ComfyUI VIDEO format (VideoInput).
    """
    video_input = SGLDVideoInput(video_path, height, width)
    return video_input
