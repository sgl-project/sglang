import binascii
import io
import struct
from contextlib import contextmanager
from typing import List, Optional, Tuple

import pybase64
from PIL import Image, ImageOps

from sglang.srt.entrypoints.decision.families.base import DecisionInputError
from sglang.srt.entrypoints.decision.protocol import JevImage, JevImageUpload

# The limits of the official upload handler, Intern-Decision src/service/uploads.py.
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
}
# Base64 of MAX_IMAGE_BYTES plus slack for a data URL header, checked before decoding.
_MAX_ENCODED_CHARS = 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 128


def normalize_images(images: List[JevImage]) -> List[str]:
    urls: List[str] = []
    total = normalized_total = 0
    for index, image in enumerate(images):
        loc = ("body", "images", index)
        payload, content_type = _payload(image, loc)
        try:
            data = pybase64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as e:
            raise DecisionInputError(
                "not base64 image data; paths and URLs are not loaded", loc
            ) from e
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise DecisionInputError("each image must be 1 byte to 12 MiB", loc)
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise DecisionInputError(
                "the images are limited to 32 MiB combined", ("body", "images")
            )
        png = _upright_rgb_png(data, content_type, loc)
        normalized_total += len(png)
        if normalized_total > MAX_TOTAL_BYTES:
            raise DecisionInputError(
                "the decoded images exceed 32 MiB combined", ("body", "images")
            )
        urls.append("data:image/png;base64," + pybase64.b64encode(png).decode())
    return urls


def _payload(image: JevImage, loc: Tuple) -> Tuple[str, Optional[str]]:
    if isinstance(image, JevImageUpload):
        data, declared = image.data, image.type.lower()
    else:
        data, declared = image, None
    if len(data) > _MAX_ENCODED_CHARS:
        raise DecisionInputError("each image must be 1 byte to 12 MiB", loc)
    if data.startswith("data:"):
        header, separator, data = data.partition(",")
        if not separator or not header.endswith(";base64"):
            raise DecisionInputError(
                "an image data URL must be data:image/<type>;base64,...", loc
            )
        declared = header[len("data:") :].split(";")[0].lower()
    if declared is not None and declared not in FORMATS:
        raise DecisionInputError(
            "supported image formats are JPEG, PNG, WebP, and static GIF", loc
        )
    return data, declared


def _upright_rgb_png(data: bytes, content_type: Optional[str], loc: Tuple) -> bytes:
    with _pillow_errors(loc):
        source = Image.open(io.BytesIO(data))
    with source:
        allowed = FORMATS.values() if content_type is None else [FORMATS[content_type]]
        if source.format not in allowed:
            raise DecisionInputError(
                "the image content does not match its declared format", loc
            )
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise DecisionInputError("each image is limited to 16 million pixels", loc)
        with _pillow_errors(loc):
            # Not a passive read: n_frames parses every frame descriptor and fails on
            # truncated data. Pillow defines it only on multi-frame formats.
            frames = getattr(source, "n_frames", 1)
        if frames != 1:
            raise DecisionInputError("animated images are not supported", loc)
        with _pillow_errors(loc):
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    image.info.clear()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@contextmanager
def _pillow_errors(loc: Tuple):
    try:
        yield
    # Pillow's parsers also surface malformed data as struct.error, EOFError,
    # SyntaxError, or IndexError from reading past a truncated block.
    except (
        OSError,
        ValueError,
        struct.error,
        EOFError,
        SyntaxError,
        IndexError,
        Image.DecompressionBombError,
    ) as e:
        raise DecisionInputError("invalid, incomplete, or oversized image", loc) from e
