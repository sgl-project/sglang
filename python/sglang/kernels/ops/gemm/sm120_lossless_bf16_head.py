"""Lossless BF16 head storage; original GEMV operations and full vocabulary.

A 256-value block stores all sign/mantissa bits and a four-bit exponent offset
when its exponent range fits; otherwise it stores the original 16-bit words.
This is reversible bit packing, with no floating-point quantization step.
"""

import functools
import logging
import weakref

import numpy as np
import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

logger = logging.getLogger(__name__)


@functools.cache
def lossless_bf16_head_enabled():
    from sglang.srt.environ import envs

    return envs.SGLANG_SM120_LOSSLESS_BF16_HEAD.get()


@functools.cache
def _capability(device):
    return torch.cuda.get_device_capability(device)


@cache_once
def _module(n, k):
    args = make_cpp_args(n, k)
    return load_jit(
        "sm120_lossless_bf16_head",
        *args,
        cuda_files=["gemm/sm120_lossless_bf16_head.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::LosslessBf16<{args}>::run"),
            ("restore", f"sglang::LosslessBf16<{args}>::restore"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def _pack_bf16(weight):
    """CPU-only preprocessing; preserve all 65536 possible BF16 bit patterns."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.shape[1] % 256:
        raise ValueError("Expected a BF16 matrix with K divisible by 256")
    raw = weight.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
    n, k = raw.shape
    minimum = np.empty(n * k // 256, dtype=np.uint16)
    compressed = np.empty(minimum.size, dtype=np.bool_)
    for start in range(0, n, 1024):
        exponent = ((raw[start : start + 1024] >> 7) & 255).reshape(-1, 256)
        lo, hi = exponent.min(1), exponent.max(1)
        begin = start * k // 256
        minimum[begin : begin + lo.size] = lo
        compressed[begin : begin + lo.size] = hi - lo < 16
    offsets = np.zeros(minimum.size + 1, dtype=np.uint64)
    np.cumsum(np.where(compressed, 96, 128), out=offsets[1:])
    payload = np.empty(int(offsets[-1]), dtype=np.uint32)
    metadata = (
        offsets[:-1]
        | (minimum.astype(np.uint64) << 32)
        | (compressed.astype(np.uint64) << 40)
    )
    for start in range(0, n, 1024):
        block = raw[start : start + 1024].reshape(-1, 256)
        begin = start * k // 256
        select = compressed[begin : begin + block.shape[0]]
        offs = offsets[begin : begin + block.shape[0]]
        vals = block[select].astype(np.uint32)
        base = minimum[begin : begin + block.shape[0]][select, None].astype(np.uint32)
        code = (
            (vals & 127) | ((vals >> 8) & 128) | ((((vals >> 7) & 255) - base) << 8)
        ).reshape(-1, 32, 8)
        packed = np.stack(
            (
                code[:, :, 0] | (code[:, :, 1] << 12) | (code[:, :, 2] << 24),
                (code[:, :, 2] >> 8)
                | (code[:, :, 3] << 4)
                | (code[:, :, 4] << 16)
                | (code[:, :, 5] << 28),
                (code[:, :, 5] >> 4) | (code[:, :, 6] << 8) | (code[:, :, 7] << 20),
            ),
            -1,
        ).reshape(-1, 96)
        payload[offs[select, None] + np.arange(96, dtype=np.uint64)] = packed
        payload[offs[~select, None] + np.arange(128, dtype=np.uint64)] = (
            block[~select].copy().view(np.uint32)
        )
    return torch.from_numpy(payload), torch.from_numpy(metadata)


def _refresh_lossless_buffers(layer):
    """Repack after loading, preserving graph pointers when shape/device match."""
    weight = layer.weight
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Lossless head preparation must precede graph capture")
    payload, metadata = _pack_bf16(weight)
    # Reserve raw capacity so reprocessing (including deferred real-weight load)
    # never changes graph-visible pointers merely because compression changes.
    specs = (
        ("_lossless_bf16_payload", weight.numel() // 2, torch.uint32),
        ("_lossless_bf16_metadata", weight.numel() // 256, torch.uint64),
    )
    for name, size, dtype in specs:
        old = getattr(layer, name, None)
        if old is None or old.shape != (size,) or old.device != weight.device:
            layer.register_buffer(
                name,
                torch.empty(size, dtype=dtype, device=weight.device),
                persistent=False,
            )
    layer._lossless_bf16_payload[: payload.numel()].copy_(payload)
    layer._lossless_bf16_metadata.copy_(metadata)
    module = _module(*weight.shape)
    restored = torch.empty_like(weight)
    module.restore(
        layer._lossless_bf16_payload, layer._lossless_bf16_metadata, restored
    )
    if not torch.equal(restored.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Lossless head failed the full BF16 bit audit")
    layer._derived_weight_cache_error = "Online weight updates require rebuilding the lossless BF16 head before graph capture"
    layer._lossless_bf16_source = weakref.ref(weight)
    layer._lossless_bf16_used_bytes = payload.numel() * 4 + metadata.numel() * 8
    logger.info(
        "Prepared lossless BF16 head: %s, stream bytes=%d, every weight bit verified",
        tuple(weight.shape),
        layer._lossless_bf16_used_bytes,
    )


def prepare_lossless_bf16_head(layer):
    if not lossless_bf16_head_enabled():
        return
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    weight = getattr(layer, "weight", None)
    if (
        not isinstance(layer, ParallelLMHead)
        or weight is None
        or weight.shape != (131072, 4096)
        or weight.dtype != torch.bfloat16
        or weight.device.type != "cuda"
        or not weight.is_contiguous()
        or _capability(weight.device) != (12, 0)
    ):
        return
    _refresh_lossless_buffers(layer)


def try_lossless_bf16_head(x, layer):
    source = getattr(layer, "_lossless_bf16_source", None)
    if (
        not lossless_bf16_head_enabled()
        or source is None
        or source() is not layer.weight
        or x.ndim != 2
        or x.shape[1] != 4096
        or x.shape[0] != 1
        or x.dtype != torch.bfloat16
        or x.device != layer.weight.device
        or not x.is_contiguous()
        or layer.weight.shape != (131072, 4096)
    ):
        return None
    out = torch.empty((x.shape[0], 131072), dtype=x.dtype, device=x.device)
    module = _module(131072, 4096)
    module.run(x, layer._lossless_bf16_payload, layer._lossless_bf16_metadata, out)
    return out
