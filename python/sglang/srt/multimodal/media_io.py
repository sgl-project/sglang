# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Loading, decoding, and encoding image, video, and audio inputs."""

from __future__ import annotations

import errno
import ipaddress
import logging
import math
import os
import stat
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import BinaryIO, Dict, Optional, Sequence, Union
from urllib.parse import unquote, urljoin, urlparse

import numpy as np
import pybase64
import requests
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from typing_extensions import Literal

from sglang.srt.utils.device import is_cuda
from sglang.srt.utils.video_decoder import _BACKEND, VideoDecoderWrapper

logger = logging.getLogger(__name__)

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


_mm_http_session = threading.local()


_DEFAULT_MEDIA_URL_MAX_FILE_SIZE_MB = 64


_MAX_MEDIA_URL_REDIRECTS = 5


_MEDIA_URL_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


_allowed_media_domains: frozenset[str] = frozenset()


_media_url_max_file_size_bytes = _DEFAULT_MEDIA_URL_MAX_FILE_SIZE_MB * 1024 * 1024


_media_url_timeout: Optional[float] = None


_allow_local_media_files = True


def _normalize_media_domain(domain: str) -> str:
    if not isinstance(domain, str):
        raise ValueError("allowed media domains must be strings")

    domain = domain.strip().rstrip(".")
    if not domain:
        raise ValueError("allowed media domains cannot be empty")
    if "://" in domain or any(char in domain for char in "/?#@"):
        raise ValueError(
            f"Invalid allowed media domain {domain!r}: provide a hostname only"
        )

    # Brackets are URL syntax, not part of an IPv6 hostname.
    if domain.startswith("[") and domain.endswith("]"):
        domain = domain[1:-1]
    try:
        return str(ipaddress.ip_address(domain))
    except ValueError:
        if ":" in domain:
            raise ValueError(
                f"Invalid allowed media domain {domain!r}: ports are not supported"
            )

    try:
        normalized = domain.encode("idna").decode("ascii").lower()
    except UnicodeError as e:
        raise ValueError(f"Invalid allowed media domain {domain!r}") from e
    if not normalized:
        raise ValueError("allowed media domains cannot be empty")
    return normalized


def configure_media_url_security(
    allowed_media_domains: Optional[Sequence[str]] = None,
    max_file_size_mb: int = _DEFAULT_MEDIA_URL_MAX_FILE_SIZE_MB,
    timeout: Optional[float] = None,
    allow_local_files: bool = True,
) -> list[str]:
    """Configure process-wide safeguards for client-supplied media URLs.

    A serving worker hosts one engine configuration, while media loading fans
    out to worker threads. Keeping the immutable policy here makes the same
    checks apply to image, video, audio, cache, and model-specific loaders.
    """

    if max_file_size_mb < 0:
        raise ValueError("media_url_max_file_size_mb must be non-negative")
    if timeout is not None and timeout <= 0:
        raise ValueError("media_url_timeout must be positive")

    normalized_domains = sorted(
        {_normalize_media_domain(domain) for domain in allowed_media_domains or []}
    )
    global _allowed_media_domains, _media_url_max_file_size_bytes
    global _media_url_timeout, _allow_local_media_files
    _allowed_media_domains = frozenset(normalized_domains)
    _media_url_max_file_size_bytes = max_file_size_mb * 1024 * 1024
    _media_url_timeout = timeout
    _allow_local_media_files = allow_local_files
    return normalized_domains


def media_url_security_config() -> tuple:
    """The configured policy, as ``configure_media_url_security`` arguments."""
    return (
        sorted(_allowed_media_domains),
        _media_url_max_file_size_bytes // (1024 * 1024),
        _media_url_timeout,
        _allow_local_media_files,
    )


def _assert_media_url_allowed(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"Invalid media URL: {url!r}")

    hostname = _normalize_media_domain(parsed.hostname)
    if _allowed_media_domains and hostname not in _allowed_media_domains:
        raise ValueError(
            "Media URL domain is not allowed. "
            f"Allowed domains: {sorted(_allowed_media_domains)}; "
            f"input domain: {hostname}"
        )


def media_url_timeout(default: float) -> float:
    if _media_url_timeout is not None:
        return _media_url_timeout
    return float(os.getenv("REQUEST_TIMEOUT", default))


def download_remote_media(url: str, timeout: float) -> bytes:
    """Download one HTTP(S) media object under the configured URL policy.

    Redirects are followed manually so every destination is validated before
    a connection is made. The response is streamed to enforce both the total
    request deadline and the configured byte limit without first buffering an
    attacker-controlled body in memory.
    """

    if timeout <= 0:
        raise ValueError("media URL timeout must be positive")

    session = get_mm_http_session()
    deadline = time.monotonic() + timeout
    current_url = url

    for redirect_count in range(_MAX_MEDIA_URL_REDIRECTS + 1):
        # Validate the same normalized URL representation that requests sends
        # to urllib3. This avoids parser disagreements around backslashes and
        # userinfo separators.
        prepared_url = requests.Request("GET", current_url).prepare().url
        if prepared_url is None:
            raise ValueError(f"Invalid media URL: {current_url!r}")
        _assert_media_url_allowed(prepared_url)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise requests.exceptions.Timeout(
                f"Timed out while downloading media URL: {url}"
            )

        with session.get(
            prepared_url,
            allow_redirects=False,
            stream=True,
            timeout=remaining,
        ) as response:
            location = response.headers.get("Location")
            if response.status_code in _MEDIA_URL_REDIRECT_STATUS_CODES and location:
                if redirect_count == _MAX_MEDIA_URL_REDIRECTS:
                    raise requests.exceptions.TooManyRedirects(
                        f"Media URL exceeded {_MAX_MEDIA_URL_REDIRECTS} redirects: {url}"
                    )
                current_url = urljoin(response.url, location)
                continue

            response.raise_for_status()
            max_bytes = _media_url_max_file_size_bytes
            content_length = response.headers.get("Content-Length")
            if max_bytes and content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > max_bytes:
                    raise ValueError(
                        f"Remote media exceeds the {max_bytes} byte download limit"
                    )

            content = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                if time.monotonic() > deadline:
                    raise requests.exceptions.Timeout(
                        f"Timed out while downloading media URL: {url}"
                    )
                if max_bytes and len(content) + len(chunk) > max_bytes:
                    raise ValueError(
                        f"Remote media exceeds the {max_bytes} byte download limit"
                    )
                content.extend(chunk)
            return bytes(content)

    raise AssertionError("unreachable")


def get_mm_http_session() -> requests.Session:
    """Per-thread HTTP session for multimodal downloads, to pool/reuse TCP
    connections. Pid-checked so a forked worker rebuilds its own, not the parent's.
    """
    pid = os.getpid()
    session = getattr(_mm_http_session, "session", None)
    if session is None or getattr(_mm_http_session, "pid", None) != pid:
        session = requests.Session()
        _mm_http_session.session = session
        _mm_http_session.pid = pid
    return session


# Raised by the loaders below when client-supplied media cannot be fetched or
# decoded. ValueError is in the set because invalid base64 raises binascii.Error.
# Pillow raises DecompressionBombError from Image.open, before any decode, for an
# image above twice Image.MAX_IMAGE_PIXELS; it subclasses Exception, not OSError.
CLIENT_MEDIA_EXCEPTIONS = (
    ValueError,
    UnidentifiedImageError,
    Image.DecompressionBombError,
    requests.exceptions.RequestException,
)


# The client named a path that cannot exist as a file. Anything else (EMFILE,
# ENOSPC, EACCES, ...) is a server-side fault and propagates.
_CLIENT_PATH_ERRNOS = frozenset(
    {
        errno.ENOENT,
        errno.ENOTDIR,
        errno.EISDIR,
        errno.ENAMETOOLONG,
        errno.ELOOP,
        errno.ENXIO,
    }
)


def shorten_media_source(value: str) -> str:
    return value if len(value) <= 100 else value[:100] + "..."


def _open_local_media_file(path: str) -> BinaryIO:
    if not _allow_local_media_files:
        raise ValueError("Local media files are not allowed")
    try:
        # A FIFO opens at once instead of waiting for a writer; fstat rejects it.
        f = open(path, "rb", opener=lambda p, flags: os.open(p, flags | os.O_NONBLOCK))
    except OSError as e:
        if e.errno not in _CLIENT_PATH_ERRNOS:
            raise
        # str(e) repeats the full path; e.strerror does not.
        raise ValueError(
            f"Could not read media from path {shorten_media_source(path)}: {e.strerror}"
        ) from e
    st = os.fstat(f.fileno())
    max_bytes = _media_url_max_file_size_bytes
    if not stat.S_ISREG(st.st_mode):
        f.close()
        raise ValueError(
            f"Media path is not a regular file: {shorten_media_source(path)}"
        )
    if max_bytes and st.st_size > max_bytes:
        f.close()
        raise ValueError(
            f"Media file exceeds the {max_bytes} byte limit: {shorten_media_source(path)}"
        )
    return f


def check_local_media_file(path: str) -> str:
    """Raise ValueError unless ``path`` is an allowed, in-cap regular file."""
    _open_local_media_file(path).close()
    return path


def read_local_media_file(path: str) -> bytes:
    max_bytes = _media_url_max_file_size_bytes
    with _open_local_media_file(path) as f:
        # Bounded even when st_size understates the content (e.g. /proc files).
        data = f.read(max_bytes + 1) if max_bytes else f.read()
    if max_bytes and len(data) > max_bytes:
        raise ValueError(
            f"Media file exceeds the {max_bytes} byte limit: {shorten_media_source(path)}"
        )
    return data


def is_local_media_path(value: str) -> bool:
    """Probes the filesystem only when local files are allowed, so a disallowed
    path never reveals whether it exists."""
    return _allow_local_media_files and os.path.exists(value)


def _decode_bare_base64(value: str) -> bytes:
    try:
        return pybase64.b64decode(value, validate=True)
    except ValueError as e:
        if not value.startswith("/"):
            raise
        # Neither base64 nor an existing local file: say both.
        hint = (
            "no such file exists"
            if _allow_local_media_files
            else "local media files are not allowed"
        )
        raise ValueError(
            f"Media is not valid base64 ({e}) and {hint}: {shorten_media_source(value)}"
        ) from e


def load_audio(
    audio_file: Union[str, bytes], sr: Optional[int] = None, mono: bool = True
) -> np.ndarray:
    if sr is None:
        sr = 16000

    # Normalize input: resolve URL / base64 / file:// to bytes or path
    if isinstance(audio_file, bytes):
        source = audio_file
    elif isinstance(audio_file, str) and audio_file.startswith("data:"):
        source = pybase64.b64decode(audio_file.split(",")[1], validate=True)
    elif isinstance(audio_file, str) and (
        audio_file.startswith("http://") or audio_file.startswith("https://")
    ):
        timeout = media_url_timeout(5)
        source = download_remote_media(audio_file, timeout=timeout)
    elif isinstance(audio_file, str) and audio_file.startswith("file://"):
        source = check_local_media_file(unquote(urlparse(audio_file).path))
    elif isinstance(audio_file, str):
        source = check_local_media_file(audio_file)
    else:
        raise ValueError(f"Invalid audio format: {audio_file}")

    from sglang.srt.multimodal.audio_from_video import (
        decode_audio_container,
        is_audio_container,
    )

    if isinstance(source, bytes):
        header = source[:16]
    else:
        with open(source, "rb") as audio_stream:
            header = audio_stream.read(16)

    if is_audio_container(header):
        return decode_audio_container(
            source,
            target_sr=sr,
            mono=mono,
        )

    if _BACKEND == "torchcodec":
        from torchcodec.decoders import AudioDecoder

        try:
            decoder = AudioDecoder(
                source,
                sample_rate=sr,
                num_channels=1 if mono else None,
            )
            samples = decoder.get_all_samples()
            if mono:
                return samples.data.squeeze(0).numpy()
            return samples.data.T.numpy()
        except Exception as e:
            # torchcodec's bytes-buffer IO can fail on WAV files that carry
            # large trailing metadata chunks. Fall back to soundfile, which reads the PCM payload directly.
            logger.warning(
                f"torchcodec AudioDecoder failed ({e}); falling back to soundfile + torchaudio."
            )

    # Fallback: soundfile + torchaudio (ARM / no FFmpeg / torchcodec failure)
    import soundfile as sf
    import torch
    import torchaudio

    try:
        if isinstance(source, bytes):
            audio, original_sr = sf.read(BytesIO(source))
        else:
            audio, original_sr = sf.read(source)
    except sf.LibsndfileError as e:
        raise ValueError(f"Could not decode audio: {e}") from e

    if mono and len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if original_sr != sr:
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio_tensor.T
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=original_sr, new_freq=sr
        )
        if audio_tensor.shape[0] == 1:
            audio = audio_tensor.squeeze(0).numpy()
        else:
            audio = audio_tensor.T.numpy()

    return audio


@dataclass
class ImageData:
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"
    max_dynamic_patch: Optional[int] = None
    preprocess_kwargs: Optional[Dict] = None
    content_hash: Optional[str] = None
    max_long_side_pixel: Optional[int] = None


GLM_MEDIA_CONFIG_KEYS = (
    "fps",
    "max_frames",
    "max_tokens_per_frame",
    "max_image_tokens",
)


@dataclass
class VideoData:
    url: str
    preprocess_kwargs: Optional[Dict] = None
    max_long_side_pixel: Optional[int] = None


image_extension_names = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
)


GPUImageDecodeMode = Union[bool, Literal["nvjpeg_fancy"]]


def smart_to_rgb(
    image: Union[torch.Tensor, Image.Image],
) -> Union[torch.Tensor, Image.Image]:
    if not isinstance(image, Image.Image):
        return image

    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
        width, height = image.size
        edge_pixels = []

        for x in range(0, width, max(1, width // 20)):
            for y in (0, height - 1):
                pixel = image.getpixel((x, y))
                if pixel[3] > 128:
                    edge_pixels.append(pixel[:3])

        for y in range(0, height, max(1, height // 20)):
            for x in (0, width - 1):
                pixel = image.getpixel((x, y))
                if pixel[3] > 128:
                    edge_pixels.append(pixel[:3])

        if edge_pixels:
            avg_brightness = sum(sum(pixel) for pixel in edge_pixels) / (
                len(edge_pixels) * 3
            )
            background_color = (32, 32, 32) if avg_brightness > 128 else (240, 240, 240)
        else:
            background_color = (255, 255, 255)

        background = Image.new("RGB", image.size, background_color)
        background.paste(image, mask=image.getchannel("A"))
        return background

    return image.convert("RGB")


def is_jpeg_with_cuda(
    image_bytes: bytes = b"", gpu_image_decode: GPUImageDecodeMode = True
) -> bool:
    """
    Check three conditions:
    1. whether CUDA is available.
    2. whether input is recognized as JPEG.
    3. whether GPU image decode is enabled (some models such as CPM forcibly disable this).
    """
    if not is_cuda() or not gpu_image_decode:
        return False
    if image_bytes != b"":
        return image_bytes.startswith(b"\xff\xd8") and image_bytes.endswith(b"\xff\xd9")
    return False


@lru_cache(maxsize=16)
def _warn_fancy_jpeg_fallback(error: str) -> None:
    logger.warning(
        "High-fidelity GPU JPEG decode is unavailable; falling back to PIL. "
        "Install the Kimi-K3 serving image or NVIDIA nvImageCodec. Error: %s",
        error,
    )


def _load_image(
    image_bytes: bytes = b"",
    image_file: str = "",
    gpu_image_decode: GPUImageDecodeMode = True,
) -> Union[torch.Tensor, Image.Image]:
    """
    Try to decode JPEG with nvJPEG on GPU and return a torch device tensor,
    otherwise fallback to decode with PIL on CPU and return a PIL Image.
    Keep the fallback path since nvJPEG may fail on some JPEG images that are not strictly compliant with the standard, while PIL is more tolerant.
    """
    if image_file != "":
        image_bytes = get_image_bytes(image_file)
    if is_jpeg_with_cuda(image_bytes, gpu_image_decode):
        try:
            if gpu_image_decode == "nvjpeg_fancy":
                from sglang.srt.utils.nvjpeg_decoder import (
                    decode_jpeg_with_fancy_upsampling,
                )

                return decode_jpeg_with_fancy_upsampling(image_bytes)
            from torchvision.io import decode_jpeg  # lazy: ~1 s of torch._dynamo

            encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)
            image_tensor = decode_jpeg(encoded_image, device="cuda")
            return image_tensor
        except Exception as e:
            if gpu_image_decode == "nvjpeg_fancy":
                _warn_fancy_jpeg_fallback(f"{type(e).__name__}: {e}")
            else:
                logger.warning(
                    "Failed to decode JPEG on GPU, falling back to CPU. Error: %s",
                    e,
                )
    try:
        image = Image.open(BytesIO(image_bytes))
    except (OSError, SyntaxError) as e:
        raise ValueError(f"Could not decode image: {e}") from e
    return _fully_load_pil_image(image)


def _fully_load_pil_image(image: Image.Image) -> Image.Image:
    """Force PIL's lazy decode while malformed input is still request-local."""
    try:
        image.load()
    except (OSError, SyntaxError) as e:
        raise ValueError(f"Could not decode image: {e}") from e
    return image


def load_image(
    image_file: Union[Image.Image, str, ImageData, bytes],
    gpu_image_decode: GPUImageDecodeMode = True,
) -> tuple[Union[torch.Tensor, Image.Image], Optional[tuple[int, int]]]:
    """
    Load image from multiple input formats, including:
    ImageData, PIL Image, bytes, URL, file path, or base64 string.
    """
    if isinstance(image_file, ImageData):
        image_file = image_file.url

    image = None
    image_size: Optional[tuple[int, int]] = None
    if isinstance(image_file, Image.Image):
        image = _fully_load_pil_image(image_file)
        image_size = (image.width, image.height)
    elif isinstance(image_file, bytes):
        image = _load_image(image_bytes=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(image_file, str) and image_file.startswith(("http://", "https://")):
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(image_file, str) and image_file.startswith("file://"):
        image = _load_image(
            image_file=unquote(urlparse(image_file).path),
            gpu_image_decode=gpu_image_decode,
        )
    elif isinstance(image_file, str) and image_file.lower().endswith(
        image_extension_names
    ):
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(image_file, str) and image_file.startswith("data:"):
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(
        image_file, str
    ):  # Other formats, try to decode as base64 by default
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    else:
        raise ValueError(f"Invalid image: {image_file}")
    if image_size is not None and isinstance(image, Image.Image):
        image_size = (image.width, image.height)
    return image, image_size


def get_image_bytes(image_file: Union[str, bytes]) -> bytes:
    """Normalize various image inputs into raw bytes."""
    if isinstance(image_file, bytes):
        return image_file
    if image_file.startswith(("http://", "https://")):
        timeout = media_url_timeout(3)
        return download_remote_media(image_file, timeout=timeout)
    if image_file.startswith("file://"):
        return read_local_media_file(unquote(urlparse(image_file).path))
    # Raw base64 JPEG also starts with "/" ("/9j/..."), so only existing paths
    # take the path branch.
    if image_file.startswith("/") and is_local_media_path(image_file):
        return read_local_media_file(image_file)
    if isinstance(image_file, str) and image_file.startswith("data:"):
        _, encoded = image_file.split(",", 1)
        return pybase64.b64decode(encoded, validate=True)
    if isinstance(image_file, str):
        return _decode_bare_base64(image_file)
    raise NotImplementedError(f"Invalid image: {image_file}")


def _normalize_video_input(
    video_file: Union[str, bytes],
) -> Union[str, bytes, None]:
    """Normalize video input (URL, base64, file://, etc.) to a file path or bytes.

    Returns a file path or bytes suitable for a decoder, or None on failure.
    URLs and base64 are returned as bytes (no temp files needed since both
    torchcodec and VideoDecoderWrapper accept bytes natively).
    """
    if isinstance(video_file, bytes):
        return video_file
    elif isinstance(video_file, str):
        if video_file.startswith(("http://", "https://")):
            timeout = media_url_timeout(10)
            return download_remote_media(video_file, timeout=timeout)
        elif video_file.startswith("data:"):
            _, encoded = video_file.split(",", 1)
            return pybase64.b64decode(encoded, validate=True)
        elif video_file.startswith("file://"):
            return check_local_media_file(unquote(urlparse(video_file).path))
        elif is_local_media_path(video_file):
            return check_local_media_file(video_file)
        else:
            return _decode_bare_base64(video_file)
    else:
        return None


def get_video_bytes(video_file: Union[str, bytes, VideoData]) -> bytes:
    """Normalize a video input and return its encoded bytes."""
    if isinstance(video_file, VideoData):
        video_file = video_file.url

    source = _normalize_video_input(video_file)
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        return read_local_media_file(source)
    raise ValueError(f"Unsupported video input type: {type(video_file)}")


def load_video(video_file: Union[str, bytes, VideoData], use_gpu: bool = True):
    if isinstance(video_file, VideoData):
        # preprocess_kwargs is consumed by the multimodal processor, not here.
        video_file = video_file.url

    if isinstance(video_file, (list, tuple, torch.Tensor, np.ndarray)):
        return video_file

    source = _normalize_video_input(video_file)
    if source is None:
        raise ValueError(f"Unsupported video input type: {type(video_file)}")

    device = "cuda" if use_gpu else "cpu"
    try:
        return VideoDecoderWrapper(source, device=device)
    except (ImportError, MemoryError):
        raise  # missing backend / OOM is not a bad payload
    except Exception as e:
        # Broad on purpose: torchcodec raises RuntimeError, decord its own type.
        raise ValueError(f"Could not decode video: {e}") from e


def sample_video_frames(video, *, desired_fps: int, max_frames: int) -> list[int]:
    total_frames = len(video)
    assert total_frames > 0, "Video must have at least one frame"

    avg_fps = video.avg_fps
    duration = total_frames / avg_fps if avg_fps > 0 else 0
    fps = min(desired_fps, avg_fps)

    num_frames = math.floor(duration * fps)
    num_frames = min(max_frames, num_frames, total_frames)
    num_frames = max(1, num_frames)  # At least one frame
    if num_frames == total_frames:
        return list(range(total_frames))
    else:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()


def encode_video(video_path, frame_count_limit=None):
    if not os.path.exists(video_path):
        logger.error(f"Video {video_path} does not exist")
        return []

    if frame_count_limit == 0:
        return []

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    decoder = VideoDecoderWrapper(video_path)
    avg_fps = decoder.avg_fps
    total_frames = len(decoder)

    sample_fps = round(avg_fps / 1)
    if sample_fps == 0:
        sample_fps = 1
    frame_indices = [i for i in range(0, total_frames, sample_fps)]
    if frame_count_limit is not None and len(frame_indices) > frame_count_limit:
        frame_indices = uniform_sample(frame_indices, frame_count_limit)

    if not frame_indices:
        return []

    frames_data = decoder.get_frames_at(frame_indices)
    frames = [Image.fromarray(v.astype("uint8")) for v in frames_data]

    return frames
