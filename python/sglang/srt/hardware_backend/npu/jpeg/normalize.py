"""Round odd SOF0 dimensions within the existing 4:2:0 MCU grid.

Only a private JPEGD input copy changes; entropy data and request bytes remain
intact. The RGB result is cropped back to the original valid dimensions.
"""

import struct


class UnsupportedSOF(ValueError):
    """A valid frame outside the supported baseline 4:2:0 contract."""


def inspect_420_sof(data: bytes, width: int, height: int) -> int:
    """Return the SOF0 height-field offset, without decoding or copying pixels."""
    if not data.startswith(b"\xff\xd8"):
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


def normalize_420_sof(data: bytes, width: int, height: int) -> bytes:
    if not (width % 2 or height % 2):
        return data
    offset = inspect_420_sof(data, width, height)
    return (
        data[:offset]
        + struct.pack(">HH", height + height % 2, width + width % 2)
        + data[offset + 4 :]
    )
