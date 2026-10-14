import base64
import io
import os
import shutil
import subprocess
import time
import uuid

import folder_paths
import numpy as np
import requests
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
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Video not found at '{self.video_path}'. This file must be "
                "readable from the ComfyUI process; a path on a remote "
                "SGLang Diffusion server is not."
            )
        save_path = path
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        shutil.copy2(self.video_path, save_path)

    def as_trimmed(
        self,
        start_time: float | None = None,
        duration: float | None = None,
        strict_duration: bool = False,
    ):
        """
        Required by VideoInput since ComfyUI #12107.

        Trims via ffmpeg stream-copy (fast, no re-encode); `strict_duration`
        is accepted for interface compatibility but not enforced to frame
        accuracy, since this node never needs sub-frame precision.
        """
        start_time = start_time or 0.0
        if duration is not None and duration < 0:
            return None

        temp_dir = folder_paths.get_temp_directory()
        _ensure_dir(temp_dir)
        ext = os.path.splitext(self.video_path)[1] or ".mp4"
        dest = os.path.join(temp_dir, f"sgl_video_trim_{uuid.uuid4().hex[:8]}{ext}")

        cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", self.video_path]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += ["-c", "copy", dest]
        subprocess.run(cmd, capture_output=True, check=True)

        return SGLDVideoInput(dest, self.height, self.width)


def resolve_video_path(file_path: str | None, url: str | None) -> str:
    """
    Resolve the generated video to a path this process can use.

    The server sets `file_path` to None once a cloud upload succeeds (the
    local file is removed); fetch `url` into the ComfyUI temp directory in
    that case. Otherwise `file_path` is returned as-is; a server on another
    host without cloud storage returns a path this process can't read, and
    `SGLDVideoInput.save_to` raises clearly when that path turns out to be
    unreadable.
    """
    if file_path:
        return file_path
    if not url:
        raise RuntimeError(
            "Generated video has no local file_path and no url to fetch it from."
        )

    temp_dir = folder_paths.get_temp_directory()
    _ensure_dir(temp_dir)
    ext = os.path.splitext(url.split("?", 1)[0])[1] or ".mp4"
    dest = os.path.join(temp_dir, f"sgl_video_{uuid.uuid4().hex[:8]}{ext}")

    with requests.get(url, timeout=120, stream=True) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return dest


def convert_video_to_comfy_video(
    video_path: str, height: int, width: int
) -> VideoInput:
    """
    Convert video to ComfyUI VIDEO format (VideoInput).
    """
    video_input = SGLDVideoInput(video_path, height, width)
    return video_input
