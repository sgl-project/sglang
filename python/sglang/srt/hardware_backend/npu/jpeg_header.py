"""JPEG eligibility and encoded-header preparation for the DVPP image path.

These CPU helpers inspect headers, not decoded pixels. Odd baseline 4:2:0
dimensions are rounded within the existing 16x16 MCU grid in a private copy;
the compressed scan is unchanged and Dvpp crops the final RGB tensor.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch
from PIL import Image, JpegImagePlugin


class UnsupportedJPEG(ValueError):
    """Valid media outside the supported NPU JPEG input contract."""


class UnsupportedSOF(ValueError):
    """A valid frame outside the supported baseline 4:2:0 contract."""


def _frame_dimension_offset(
    data: bytes | bytearray | memoryview, width: int, height: int
) -> int:
    """Validate baseline 8-bit 4:2:0 headers and locate their dimension fields.

    Walk length-delimited markers up to the frame header, check its dimensions
    against Pillow's metadata, and require Y22/Cb11/Cr11 sampling. Return the
    height-field offset (width follows it), without decoding pixels. Valid but
    unsupported frame types raise UnsupportedSOF; malformed headers raise
    ValueError rather than silently entering the PIL fallback.
    """
    if data[:2] != b"\xff\xd8":
        raise ValueError("SOF normalization requires JPEG SOI")
    position = 2
    while position < len(data):
        if data[position] != 255:
            raise ValueError("invalid JPEG header marker")
        while position < len(data) and data[position] == 255:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker == 0x01:  # TEM has no segment length.
            continue
        if marker in (0, 0xD8, 0xD9, 0xDA) or 0xD0 <= marker <= 0xD7:
            raise ValueError("missing frame header before scan")
        if position + 2 > len(data):
            break
        length = (data[position] << 8) | data[position + 1]
        if length < 2 or position + length > len(data):
            raise ValueError("invalid JPEG segment length")
        if marker in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            if length < 8:
                raise ValueError("incomplete JPEG frame header")
            components = data[position + 7]
            if not components or length != 8 + 3 * components:
                raise ValueError("invalid JPEG frame component length")
            actual_height = (data[position + 3] << 8) | data[position + 4]
            actual_width = (data[position + 5] << 8) | data[position + 6]
            if (actual_width, actual_height) != (width, height):
                raise ValueError("SOF/PIL dimensions disagree")
            if marker != 0xC0:
                raise UnsupportedSOF("device decoding requires baseline SOF0")
            if data[position + 2] != 8:
                raise ValueError("baseline SOF0 requires 8-bit sample precision")
            if components != 3:
                raise UnsupportedSOF("device decoding requires three components")
            for index in (position + 9, position + 12, position + 15):
                sampling = data[index]
                if not (1 <= (sampling >> 4) <= 4 and 1 <= (sampling & 15) <= 4):
                    raise ValueError("invalid JPEG sampling factor")
            if (data[position + 9], data[position + 12], data[position + 15]) != (
                0x22,
                0x11,
                0x11,
            ):
                raise UnsupportedSOF("device decoding requires Y22/Cb11/Cr11 sampling")
            return position + 3
        position += length
    raise ValueError("JPEG has no complete frame header")


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class JPEGInfo:
    width: int
    height: int
    width_stride: int
    height_stride: int
    dimension_offset: int

    def normalized_header(self, encoded):
        """Round odd dimensions in a private copy; keep even inputs zero-copy."""
        if self.width == self.decode_width and self.height == self.decode_height:
            return encoded
        normalized = bytearray(encoded)
        struct.pack_into(
            ">HH",
            normalized,
            self.dimension_offset,
            self.decode_height,
            self.decode_width,
        )
        return normalized

    @property
    def decode_width(self):
        return align_up(self.width, 2)

    @property
    def decode_height(self):
        return align_up(self.height, 2)

    @property
    def nv12_bytes(self):
        return self.width_stride * self.height_stride * 3 // 2


class _BufferReader:
    """Let Pillow read encoded headers without copying the entire input."""

    def __init__(self, data):
        self.data = memoryview(data)
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid whence")
        position = (0, self.position, len(self.data))[whence] + offset
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        end = len(self.data) if size < 0 else self.position + size
        chunk = self.data[self.position : end].tobytes()
        self.position += len(chunk)
        return chunk


def inspect_jpeg(data: bytes | bytearray | memoryview):
    """Read device eligibility and metadata without CPU pixel decoding."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("encoded input must be a byte buffer")
    if not data:
        raise ValueError("encoded input must not be empty")
    if len(data) > 32 * 1024 * 1024:
        raise UnsupportedJPEG("encoded input exceeds the 32 MiB device-path limit")
    with Image.open(_BufferReader(data)) as im:
        if im.format != "JPEG":
            raise UnsupportedJPEG("not JPEG")
        w, h = im.size
        if im.info.get("progressive") or im.info.get("progression"):
            raise UnsupportedJPEG("progressive JPEG uses CPU fallback")
        if im.mode != "RGB" or JpegImagePlugin.get_sampling(im) != 2:
            raise UnsupportedJPEG("v1 accepts only three-component YCbCr 4:2:0 JPEG")
        # VPC color conversion accepts at most 4096 in either dimension.
        # Its even decode extent and the NV12 strides remain within that limit.
        if not (32 <= w <= 4096 and 32 <= h <= 4096):
            raise UnsupportedJPEG("JPEGD/VPC requires dimensions in [32, 4096]")
        try:
            offset = _frame_dimension_offset(data, w, h)
        except UnsupportedSOF as error:
            raise UnsupportedJPEG(str(error)) from error
        if im.getexif() or im.info.get("icc_profile") or im.info.get("exif"):
            raise UnsupportedJPEG("EXIF/ICC metadata uses CPU fallback")
    return JPEGInfo(w, h, align_up(w, 64), align_up(h, 16), offset)


def prepare_jpeg(data):
    """Validate encoded bytes/CPU uint8 input without creating device resources.

    Return the contiguous input and its metadata, or None for supported PIL
    fallback cases. Keep contiguous storage; malformed input errors propagate.
    """
    if not isinstance(data, bytes):
        if not isinstance(data, torch.Tensor):
            raise TypeError("NPU JPEG input must be bytes or a CPU uint8 tensor")
        if data.ndim != 1 or data.dtype != torch.uint8 or data.device.type != "cpu":
            raise ValueError("NPU JPEG tensor input must be one-dimensional CPU uint8")
        data = data.contiguous()
    encoded = memoryview(data if isinstance(data, bytes) else data.numpy())
    if not (encoded[:2] == b"\xff\xd8" and encoded[-2:] == b"\xff\xd9"):
        return None
    try:
        return data, inspect_jpeg(encoded)
    except UnsupportedJPEG:
        return None
