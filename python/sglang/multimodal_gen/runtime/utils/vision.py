# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from io import BytesIO
from typing import BinaryIO
from urllib.parse import unquote, urlparse
from urllib.request import DataHandler, Request

import imageio
import numpy as np
import PIL.Image
import PIL.ImageOps
import requests
import torch
from packaging import version

from sglang.kernels import get_kernel
from sglang.multimodal_gen.runtime.utils.image_io import temporary_media_directory
from sglang.srt.environ import envs
from sglang.srt.multimodal.media_io import get_image_bytes as srt_get_image_bytes
from sglang.srt.multimodal.media_io import media_url_timeout, open_remote_media

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


def pil_to_numpy(images: list[PIL.Image.Image] | PIL.Image.Image) -> np.ndarray:
    r"""
    Convert a PIL image or a list of PIL images to NumPy arrays.

    Args:
        images (`PIL.Image.Image` or `List[PIL.Image.Image]`):
            The PIL image or list of images to convert to NumPy format.

    Returns:
        `np.ndarray`:
            A NumPy array representation of the images.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images_arr: np.ndarray = np.stack(images, axis=0)

    return images_arr


def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    r"""
    Convert a NumPy image to a PyTorch tensor.

    Args:
        images (`np.ndarray`):
            The NumPy image array to convert to PyTorch format.

    Returns:
        `torch.Tensor`:
            A PyTorch tensor representation of the images.
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images


def normalize(images: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    r"""
    Normalize an image array to [-1,1].

    Args:
        images (`np.ndarray` or `torch.Tensor`):
            The image array to normalize.

    Returns:
        `np.ndarray` or `torch.Tensor`:
            The normalized image array.
    """
    return 2.0 * images - 1.0


def preprocess_image(
    image: PIL.Image.Image,
    *,
    device: torch.device,
    size: tuple[int, int] | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    """Pillow Lanczos resize, crop, and [-1, 1] normalization to [1, C, H, W].

    Size is (width, height); crop is (top, left, height, width).
    Non-RGB inputs and small images retain the CPU recipe.
    """
    width, height = size or image.size
    if (
        device.type != "cuda"
        or torch.version.hip is not None
        or envs.SGLANG_FORCE_CPU_IMAGE_PREPROCESSING.get()
        or image.mode != "RGB"
        or max(image.width * image.height, width * height) < 256 * 256
    ):
        if size is not None:
            image = image.resize(size, PIL.Image.Resampling.LANCZOS)
        if crop is not None:
            top, left, height, width = crop
            image = image.crop((left, top, left + width, top + height))
        return normalize(numpy_to_pt(pil_to_numpy(image))).to(device)

    # upload compact pixels; the lookup table preserves the CPU FP32 rounding
    pixels = torch.from_numpy(np.array(image)).to(device, non_blocking=True)
    resize_rgb = get_kernel("mm.lanczos_resize")
    if crop == (0, 0, height, width):
        crop = None
    pixels = resize_rgb(
        pixels,
        height,
        width,
        crop=crop,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    return pixels.unsqueeze(0).permute(0, 3, 1, 2)


def call_with_sanitized_media_errors(
    read: Callable, *args, invalid_input: str | None = None
):
    try:
        return read(*args)
    except requests.RequestException as exc:
        # Requests errors can retain both the original URL and the prepared request.
        category = next(
            (
                cls
                for cls in (
                    requests.Timeout,
                    requests.TooManyRedirects,
                    requests.HTTPError,
                    requests.ConnectionError,
                )
                if isinstance(exc, cls)
            ),
            (
                requests.exceptions.InvalidURL
                if isinstance(exc, ValueError)
                else requests.RequestException
            ),
        )
        status = exc.response.status_code if exc.response is not None else None
        detail = f" (HTTP {status})" if type(status) is int else ""
        error = category(f"Media download failed: {category.__name__}{detail}.")
    except ValueError:
        if invalid_input is None:
            raise
        error = ValueError(invalid_input)
    # Raise outside the handler so the outgoing error does not retain the HTTP error.
    raise error from None


# adapted from diffusers.utils import load_image
def load_image(
    image: str | bytes | PIL.Image.Image,
    convert_method: Callable[[PIL.Image.Image], PIL.Image.Image] | None = None,
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], *optional*):
            A conversion method to apply to the image after loading it. When set to `None` the image will be converted
            "RGB".
    """
    if isinstance(image, (str, bytes)):
        if isinstance(image, str) and os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            # in-memory loading path
            data = call_with_sanitized_media_errors(
                srt_get_image_bytes,
                image,
                invalid_input="Invalid or disallowed image source.",
            )
            image = PIL.Image.open(BytesIO(data))
    elif not isinstance(image, PIL.Image.Image):
        raise ValueError(
            "Incorrect format used for the image. Should be bytes, a URL, a local path, base64/data URL, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image


_VIDEO_DOWNLOAD_TIMEOUT_S = 60


def download_video_to_stream(video: str, output: BinaryIO) -> None:
    """Fetch a remote video into a caller-owned stream under the media URL policy.

    ``open_remote_media`` checks every redirect hop, pins each connection to
    the checked address, and bounds the body by the configured size cap; the
    body is streamed instead of buffered in memory.
    """
    with open_remote_media(video, media_url_timeout(_VIDEO_DOWNLOAD_TIMEOUT_S)) as (
        _,
        chunks,
    ):
        for chunk in chunks:
            output.write(chunk)


@contextmanager
def open_video_source(video: str) -> Iterator[BinaryIO]:
    """A seekable stream over a client-supplied video, for PyAV to open.

    FFmpeg never sees the client string, so it cannot pick a protocol
    (``file:``, ``concat:``, ...) from it. Open the stream with
    ``options={"protocol_whitelist": "data"}`` so a demuxer cannot fetch the
    URLs a playlist names. Remote clips are fully staged under the download cap.
    """
    with ExitStack() as files:
        try:
            scheme = urlparse(video).scheme.lower()
            if scheme in ("http", "https"):
                # An explicitly configured scratch volume must not fall back to another disk.
                root = os.environ.get("TMPDIR") or None
                if root is not None and not os.path.isabs(root):
                    raise ValueError("TMPDIR must be absolute")
                stream = files.enter_context(tempfile.TemporaryFile(dir=root))
                download_video_to_stream(video, stream)
                stream.seek(0)
            elif scheme == "data":
                # DataHandler only reads inline bytes; it does not dispatch URLs.
                stream = files.enter_context(DataHandler().data_open(Request(video)))
            elif not scheme or len(scheme) == 1:
                stream = files.enter_context(open(video, "rb"))
            else:
                stream = None
        except (ValueError, OSError) as exc:
            # Both HTTP and filesystem exceptions may retain the original source.
            category = type(exc).__name__
        else:
            if stream is None:
                raise ValueError("Unsupported video path scheme")
            yield stream
            return
        raise ValueError(f"Could not open video source ({category}).")


# adapted from diffusers.utils import load_video
def load_video(
    video: str,
    convert_method: (
        Callable[[list[PIL.Image.Image]], list[PIL.Image.Image]] | None
    ) = None,
    max_frames: int | None = None,
) -> list[PIL.Image.Image]:
    """
    Loads `video` to a list of PIL Image.
    Args:
        video (`str`):
            A URL or Path to a video to convert to a list of PIL Image format.
        convert_method (Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]], *optional*):
            A conversion method to apply to the video after loading it. When set to `None` the images will be converted
            to "RGB".
        max_frames (`int`, *optional*):
            Stop decoding after this many frames. Decoding is sequential, so a caller that only needs a prefix of a
            long clip should pass it here instead of slicing the result; the frames returned are the same either way.
    Returns:
        `List[PIL.Image.Image]`:
            The video as a list of PIL images.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}.")
    pil_images = _read_video_source(video, max_frames)
    return convert_method(pil_images) if convert_method is not None else pil_images


@contextmanager
def local_video_path(video: str) -> Iterator[str]:
    """A local file for a client-supplied video: an http(s) URL is downloaded to
    a temp file that lives for the ``with`` body; anything else is yielded as is."""
    if not video.lower().startswith(("http://", "https://")):
        yield video
        return
    parsed_url = call_with_sanitized_media_errors(
        urlparse, video, invalid_input="Invalid video URL."
    )
    file_name = os.path.basename(unquote(parsed_url.path))
    # FFmpeg probes the content; a client-chosen suffix such as .m3u8
    # would let it read the file as a playlist of further URLs.
    suffix = ".gif" if file_name.lower().endswith(".gif") else ".mp4"
    with temporary_media_directory() as temp_dir:
        video_path = os.path.join(temp_dir, f"video{suffix}")
        with open(video_path, "wb") as output:
            call_with_sanitized_media_errors(download_video_to_stream, video, output)
        yield video_path


def _read_video_source(video: str, max_frames: int | None) -> list[PIL.Image.Image]:
    with local_video_path(video) as video_path:
        if not os.path.isfile(video_path):
            raise ValueError("Expected an existing video file or an HTTP(S) URL.")
        return _read_video_frames(video_path, max_frames)


def _read_video_frames(
    video: str, max_frames: int | None = None
) -> list[PIL.Image.Image]:
    pil_images = []
    if video.endswith(".gif"):
        gif = PIL.Image.open(video)
        try:
            while max_frames is None or len(pil_images) < max_frames:
                pil_images.append(gif.copy())
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

    else:
        try:
            imageio.plugins.ffmpeg.get_exe()
        except AttributeError:
            raise AttributeError(
                "`Unable to find an ffmpeg installation on your machine. Please install via `pip install imageio-ffmpeg"
            ) from None

        with imageio.get_reader(video) as reader:
            # Closing the reader early stops the ffmpeg subprocess, so a prefix
            # costs only the frames it needs.
            for frame in reader:
                pil_images.append(PIL.Image.fromarray(frame))
                if max_frames is not None and len(pil_images) >= max_frames:
                    break

    return pil_images


def get_default_height_width(
    image: PIL.Image.Image | np.ndarray | torch.Tensor,
    vae_scale_factor: int,
    height: int | None = None,
    width: int | None = None,
) -> tuple[int, int]:
    r"""
    Returns the height and width of the image, downscaled to the next integer multiple of `vae_scale_factor`.

    Args:
        image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
            The image input, which can be a PIL image, NumPy array, or PyTorch tensor. If it is a NumPy array, it
            should have shape `[batch, height, width]` or `[batch, height, width, channels]`. If it is a PyTorch
            tensor, it should have shape `[batch, channels, height, width]`.
        height (`Optional[int]`, *optional*, defaults to `None`):
            The height of the preprocessed image. If `None`, the height of the `image` input will be used.
        width (`Optional[int]`, *optional*, defaults to `None`):
            The width of the preprocessed image. If `None`, the width of the `image` input will be used.

    Returns:
        `Tuple[int, int]`:
            A tuple containing the height and width, both resized to the nearest integer multiple of
            `vae_scale_factor`.
    """

    if height is None:
        if isinstance(image, PIL.Image.Image):
            height = image.height
        elif isinstance(image, torch.Tensor):
            height = image.shape[2]
        else:
            height = image.shape[1]

    if width is None:
        if isinstance(image, PIL.Image.Image):
            width = image.width
        elif isinstance(image, torch.Tensor):
            width = image.shape[3]
        else:
            width = image.shape[2]

    width, height = (
        x - x % vae_scale_factor for x in (width, height)
    )  # resize to integer multiple of vae_scale_factor

    return height, width


def resize(
    image: PIL.Image.Image | np.ndarray | torch.Tensor,
    height: int,
    width: int,
    resize_mode: str = "default",  # only "default" is supported
    resample: str = "lanczos",
) -> PIL.Image.Image | np.ndarray | torch.Tensor:
    """Resize a PIL image to (width, height) with the given resample filter,
    ignoring the aspect ratio. Only ``resize_mode="default"`` is supported."""
    if resize_mode != "default" and not isinstance(image, PIL.Image.Image):
        raise ValueError(
            f"Only PIL image input is supported for resize_mode {resize_mode}"
        )
    assert isinstance(image, PIL.Image.Image)
    if resize_mode == "default":
        image = image.resize((width, height), resample=PIL_INTERPOLATION[resample])
    else:
        raise ValueError(f"resize_mode {resize_mode} is not supported")
    return image
