"""FlashInfer's batch-invariant quantizer kernels under the deterministic mode.

Under ``--enable-deterministic-inference`` (and the batch-invariant mode it
turns on) the activation quantizers and the dense MXFP8 GEMM run FlashInfer's
batch-invariant kernels (``batch_invariant=True``): a row's bits do not depend
on the batch it is in. Nothing here reads the batch size: a provider that
changed with M would break batch invariance even though each provider is
invariant on its own.
"""

from __future__ import annotations

import functools

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.mxfp8_input import (
    batch_invariant_or_deterministic,
)
from sglang.srt.utils.device import is_flashinfer_available

# Compute capabilities of FlashInfer's batch-invariant quantizer and GEMM kernels.
_FLASHINFER_BI_QUANT_ARCHS = ((10, 0), (10, 3), (10, 7))
# The longest padded input row FlashInfer's batch-invariant MXFP8 quantizer
# accepts (K <= 65536 for bf16 / fp16, 32768 for fp32); it raises above it, so
# `mxfp8_quantizer_takes_rows` mirrors its check. The GEMM rule bounds K at the
# bf16 figure for every activation dtype, so an fp32 activation with K above the
# quantizer's fp32 limit and a bf16 / fp16 output keeps the previous quantizer
# and takes the batch-invariant GEMM; each kernel is invariant on its own.
MXFP8_QUANT_MAX_ROW_BYTES = 128 * 1024
MXFP8_GEMM_MAX_K = MXFP8_QUANT_MAX_ROW_BYTES // 2


@functools.lru_cache(maxsize=None)
def flashinfer_bi_quant_supported(device_index: int) -> bool:
    if not is_flashinfer_available():
        return False
    try:
        # FlashInfer >= 0.7.4
        from flashinfer.batch_invariant import is_supported_device
    except ImportError:
        return False
    return is_supported_device(
        torch.device("cuda", device_index), _FLASHINFER_BI_QUANT_ARCHS
    )


def flashinfer_bi_quant_device_ok(device: torch.device) -> bool:
    """The device side of `use_flashinfer_bi_quantizers`: the opt-out
    ``SGLANG_ENABLE_DETERMINISTIC_FLASHINFER_BI_KERNELS=0`` is not set and ``device`` is a
    CUDA device the batch-invariant kernels cover."""
    if not envs.SGLANG_ENABLE_DETERMINISTIC_FLASHINFER_BI_KERNELS.get():
        return False
    if device.type != "cuda":
        return False
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return flashinfer_bi_quant_supported(index)


def use_flashinfer_bi_quantizers(device: torch.device) -> bool:
    """Whether activation quantization on ``device`` runs FlashInfer's
    batch-invariant kernels: the deterministic mode is on and
    `flashinfer_bi_quant_device_ok`. Depends on the device only, never on M."""
    return batch_invariant_or_deterministic() and flashinfer_bi_quant_device_ok(device)


def mxfp8_quantizer_takes_rows(x: torch.Tensor, alignment: int) -> bool:
    """Whether FlashInfer's batch-invariant MXFP8 quantizer takes the rows of
    ``x`` quantized with ``alignment``: it pads K to a multiple of ``alignment``
    and bounds the padded row bytes."""
    padded_k = -(-x.shape[-1] // alignment) * alignment
    return padded_k * x.element_size() <= MXFP8_QUANT_MAX_ROW_BYTES


def use_flashinfer_bi_mxfp8_quantizer(x: torch.Tensor, alignment: int) -> bool:
    """`use_flashinfer_bi_quantizers` for an MXFP8 activation ``x`` quantized
    with the call site's ``alignment``; longer rows keep the previous quantizer."""
    return mxfp8_quantizer_takes_rows(x, alignment) and use_flashinfer_bi_quantizers(
        x.device
    )


def mm_mxfp8_batch_invariant_shape_ok(
    weight: torch.Tensor, out_dtype: torch.dtype
) -> bool:
    """Whether FlashInfer's batch-invariant MXFP8 GEMM takes this weight: a
    row-major (N, K) tensor with N % 8 == 0, 128 <= K <= the quantizer's row
    limit and N >= 128, bf16 / fp16 output. Never reads M."""
    n, k = weight.shape
    return (
        out_dtype in (torch.bfloat16, torch.float16)
        and n % 8 == 0
        and n >= 128
        and 128 <= k <= MXFP8_GEMM_MAX_K
        and weight.is_contiguous()
    )
