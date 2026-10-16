"""Helpers the DSV4.1 Lamport kernels need from elsewhere."""


def mxfp8_scale_bytes(tokens: int, width: int) -> int:
    """Bytes of F8_128x4 scales for `tokens` rows of `width` MXFP8 values."""
    return -(-tokens // 128) * 128 * -(-(width // 32) // 4) * 4
