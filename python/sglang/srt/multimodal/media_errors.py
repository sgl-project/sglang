"""Public error descriptions for multimodal loaders."""

import binascii
import errno
import os

import requests
from PIL import Image, UnidentifiedImageError

from sglang.srt.multimodal.audio_from_video import AudioContainerDecodeError
from sglang.srt.multimodal.media_io import ImageData, ImageDecodeError, VideoData


def _source_description(data):
    if isinstance(data, (ImageData, VideoData)):
        data = data.url
    if isinstance(data, str):
        for scheme in ("https", "http", "file"):
            if data.startswith(scheme + "://"):
                return f"<url scheme={scheme}>"
        if data.startswith("data:"):
            comma = data.find(",")
            return f"<data-uri length={len(data) - comma - 1 if comma >= 0 else 0}>"
        return f"<string length={len(data)}>"
    if isinstance(data, (bytes, bytearray, memoryview)):
        return f"<bytes length={len(data)}>"
    return f"<{type(data).__name__}>"


def media_load_error(error_type, modality, data, error):
    # Decoder messages can contain arbitrary fragments of the input. Only carry
    # structured diagnostics into the public exception and its cause.
    name = type(error).__name__
    detail = name
    if isinstance(error, requests.RequestException) and error.response is not None:
        status = error.response.status_code
        if isinstance(status, int) and 100 <= status <= 599:
            detail = f"{name} (HTTP {status})"
    elif isinstance(error, OSError) and error.errno in errno.errorcode:
        detail = os.strerror(error.errno)
    elif isinstance(error, binascii.Error):
        detail = "Invalid base64 media"
    elif isinstance(error, Image.DecompressionBombError):
        detail = "Image exceeds the decompression bomb limit"
    elif isinstance(error, AudioContainerDecodeError):
        detail = str(AudioContainerDecodeError())
    elif isinstance(error, ImageDecodeError):
        detail = "Could not decode image"
        if isinstance(error.__cause__, UnidentifiedImageError):
            detail += ": cannot identify image file"

    cause_type = next(
        (
            cls
            for cls in (
                requests.exceptions.InvalidURL,
                requests.HTTPError,
                requests.ConnectionError,
                requests.Timeout,
                requests.RequestException,
                Image.DecompressionBombError,
                binascii.Error,
                ValueError,
                OSError,
                MemoryError,
                ImportError,
            )
            if isinstance(error, cls)
        ),
        RuntimeError,
    )
    cause = cause_type(detail)
    if isinstance(error, ImageDecodeError) and isinstance(error.__cause__, OSError):
        cause.__cause__ = OSError("Image decoder failed")
    public = error_type(
        f"Error while loading {modality.name.lower()} data {_source_description(data)}: {detail}"
    )
    return public, cause
