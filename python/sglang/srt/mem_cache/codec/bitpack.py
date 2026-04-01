from __future__ import annotations

import logging
import os
from typing import Sequence

import torch
from torch.utils.cpp_extension import load

try:
    import numpy as np
except Exception:
    np = None

_abs_path = os.path.dirname(os.path.abspath(__file__))
_bitpack_cpp = None
_bitpack_cpp_failed = False
_bitpack_backend = None

logger = logging.getLogger(__name__)


def _get_bitpack_cpp():
    global _bitpack_cpp, _bitpack_cpp_failed, _bitpack_backend
    if _bitpack_cpp is not None:
        return _bitpack_cpp
    if _bitpack_cpp_failed:
        return None
    try:
        _bitpack_cpp = load(
            name="bitpack_cpp",
            sources=[f"{_abs_path}/bitpack_binding.cpp"],
            extra_cflags=["-O3", "-std=c++17"],
        )
        _bitpack_backend = "cpp"
        logger.info("KVTC bitpack backend: cpp")
        return _bitpack_cpp
    except Exception as e:
        _bitpack_cpp_failed = True
        _bitpack_backend = None
        logger.warning("KVTC bitpack cpp backend unavailable, fallback to numpy/python: %s", e)
        return None


def warmup_bitpack_backend() -> str:
    ext = _get_bitpack_cpp()
    if ext is not None:
        return "cpp"
    if np is not None:
        return "numpy"
    return "python"


def _twos_complement_to_uint(x: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return x & mask


def _uint_to_twos_complement(u: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    mask = (1 << bits) - 1
    u &= mask
    return u - (1 << bits) if (u & sign_bit) else u


def pack_ints_twos_complement(values: Sequence[int], bits: Sequence[int]) -> bytes:
    if len(values) != len(bits):
        raise ValueError("values and bits must have same length")

    out = bytearray()
    bitbuf = 0
    bitcount = 0
    for v, b in zip(values, bits):
        b = int(b)
        if b <= 0:
            continue
        u = _twos_complement_to_uint(int(v), b)
        bitbuf |= u << bitcount
        bitcount += b
        while bitcount >= 8:
            out.append(bitbuf & 0xFF)
            bitbuf >>= 8
            bitcount -= 8

    if bitcount:
        out.append(bitbuf & 0xFF)
    return bytes(out)


def unpack_ints_twos_complement(blob: bytes, bits: Sequence[int]) -> list[int]:
    out: list[int] = []
    bitbuf = 0
    bitcount = 0
    idx = 0
    n = len(blob)
    for b in bits:
        b = int(b)
        if b <= 0:
            out.append(0)
            continue
        while bitcount < b and idx < n:
            bitbuf |= blob[idx] << bitcount
            bitcount += 8
            idx += 1
        if bitcount < b:
            raise ValueError("Not enough bits in blob for requested unpack")
        mask = (1 << b) - 1
        u = bitbuf & mask
        bitbuf >>= b
        bitcount -= b
        out.append(_uint_to_twos_complement(u, b))
    return out


def pack_tensor_ints_twos_complement(values: torch.Tensor, bits: torch.Tensor) -> bytes:
    if values.numel() != bits.numel():
        raise ValueError("values and bits must have same numel")
    ext = _get_bitpack_cpp()
    if ext is not None:
        return ext.pack_tensor_ints_twos_complement_cpp(values, bits)
    if np is not None:
        v = values.detach().cpu().to(torch.int32).flatten().numpy().astype(np.int64)
        b = bits.detach().cpu().to(torch.int16).flatten().numpy().astype(np.int64)
        active = b > 0
        if not np.any(active):
            return b""
        u = np.zeros_like(v, dtype=np.uint64)
        masks = (np.left_shift(np.uint64(1), b[active].astype(np.uint64)) - np.uint64(1))
        u[active] = np.bitwise_and(v[active].astype(np.int64), masks.astype(np.int64)).astype(
            np.uint64
        )
        bit_offsets = np.cumsum(np.concatenate(([0], b[:-1])), dtype=np.int64)
        active_offsets = bit_offsets[active]
        active_u = u[active]
        out_len = int((int(b.sum()) + 7) // 8)
        out = np.zeros(out_len + 1, dtype=np.uint16)
        byte_idx = (active_offsets // 8).astype(np.int64)
        shift = (active_offsets % 8).astype(np.uint64)
        low = np.left_shift(active_u, shift)
        np.add.at(out, byte_idx, (low & np.uint64(0xFF)).astype(np.uint16))
        np.add.at(out, byte_idx + 1, ((low >> np.uint64(8)) & np.uint64(0xFF)).astype(np.uint16))
        return out[:out_len].astype(np.uint8, copy=False).tobytes()
    v = values.detach().cpu().to(torch.int32).flatten().tolist()
    b = bits.detach().cpu().to(torch.int16).flatten().tolist()
    return pack_ints_twos_complement(v, b)


def unpack_tensor_ints_twos_complement(blob: bytes, bits: torch.Tensor) -> torch.Tensor:
    ext = _get_bitpack_cpp()
    if ext is not None:
        return ext.unpack_tensor_ints_twos_complement_cpp(blob, bits)
    if np is not None:
        b = bits.detach().cpu().to(torch.int16).flatten().numpy().astype(np.int64)
        n = b.shape[0]
        out = np.zeros(n, dtype=np.int32)
        active = b > 0
        if np.any(active):
            bit_offsets = np.cumsum(np.concatenate(([0], b[:-1])), dtype=np.int64)
            active_offsets = bit_offsets[active]
            active_bits = b[active]
            data = np.frombuffer(blob, dtype=np.uint8)
            padded = np.zeros(data.shape[0] + 1, dtype=np.uint16)
            padded[: data.shape[0]] = data.astype(np.uint16)
            byte_idx = (active_offsets // 8).astype(np.int64)
            shift = (active_offsets % 8).astype(np.int64)
            combined = padded[byte_idx] | (padded[byte_idx + 1] << np.uint16(8))
            mask = (np.left_shift(np.uint64(1), active_bits.astype(np.uint64)) - np.uint64(1))
            u = (combined.astype(np.uint64) >> shift.astype(np.uint64)) & mask
            sign_bit = np.left_shift(np.uint64(1), (active_bits - 1).astype(np.uint64))
            signed = u.astype(np.int64)
            neg_mask = (u & sign_bit) != 0
            signed[neg_mask] -= np.left_shift(1, active_bits[neg_mask]).astype(np.int64)
            out[active] = signed.astype(np.int32)
        return torch.from_numpy(out)
    b = bits.detach().cpu().to(torch.int16).flatten().tolist()
    out = unpack_ints_twos_complement(blob, b)
    return torch.tensor(out, dtype=torch.int32)


def pack_bits(mask: torch.Tensor) -> bytes:
    ext = _get_bitpack_cpp()
    if ext is not None:
        return ext.pack_bits_cpp(mask)
    if np is not None:
        m = mask.detach().cpu().to(torch.bool).flatten().numpy().astype(np.uint8)
        return np.packbits(m, bitorder="little").tobytes()
    m = mask.detach().cpu().to(torch.bool).flatten()
    out = bytearray()
    bitbuf = 0
    bitcount = 0
    for v in m.tolist():
        bitbuf |= (1 if v else 0) << bitcount
        bitcount += 1
        if bitcount == 8:
            out.append(bitbuf & 0xFF)
            bitbuf = 0
            bitcount = 0
    if bitcount:
        out.append(bitbuf & 0xFF)
    return bytes(out)


def unpack_bits(blob: bytes, nbits: int) -> torch.Tensor:
    ext = _get_bitpack_cpp()
    if ext is not None:
        return ext.unpack_bits_cpp(blob, nbits)
    if np is not None:
        data = np.frombuffer(blob, dtype=np.uint8)
        bits = np.unpackbits(data, bitorder="little")
        if bits.shape[0] < nbits:
            padded = np.zeros(nbits, dtype=np.uint8)
            padded[: bits.shape[0]] = bits
            bits = padded
        return torch.from_numpy(bits[:nbits].astype(np.bool_))
    out = torch.empty((nbits,), dtype=torch.bool)
    idx = 0
    for b in blob:
        for i in range(8):
            if idx >= nbits:
                return out
            out[idx] = bool((b >> i) & 1)
            idx += 1
    if idx < nbits:
        out[idx:] = False
    return out
