from typing import Iterable, NamedTuple, Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)

# chunk to avoid too high GPU memory peak
CHUNK_NUMEL = 64 * 1024 * 1024


class CompareResult(NamedTuple):
    equal: bool
    max_abs_err: float
    mean_abs_err: float
    num_exceed: int  # elements past the combined per-side tolerance


class ComparableWeight:
    """Base class: a weight in comparable (dequantized) form, for all precisions."""

    @staticmethod
    def _quant_ulp(w_q: torch.Tensor) -> torch.Tensor:
        """Per-element ULP of w_q in its own dtype."""
        finfo = torch.finfo(w_q.dtype)
        x = w_q.to(torch.float32).abs()
        # frexp: x = m * 2^e, m in [0.5, 1), so 2^(e-1) is x's binade base.
        _, exponent = torch.frexp(x)
        binade = torch.exp2((exponent - 1).to(torch.float32))
        # Zeros and subnormals share the spacing of the smallest normal binade.
        binade = binade.masked_fill(x < finfo.smallest_normal, finfo.smallest_normal)
        return binade * finfo.eps

    def iter_chunks(self) -> Iterable[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        raise NotImplementedError

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        raise NotImplementedError


class Fp8BlockComparable(ComparableWeight):
    """Deepseek-style FP8 quantization."""

    def __init__(self, w_q: torch.Tensor, w_s: torch.Tensor):
        self.w_q = w_q
        self.w_s = w_s

    def __repr__(self) -> str:
        return f"fp8_block(shape={tuple(self.w_q.shape)} dtype={self.w_q.dtype})"

    @staticmethod
    def _normalize_scale(w_q: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
        if w_s.dtype == torch.int32:
            w_s = inverse_transform_scale_ue8m0(w_s, mn=w_q.shape[-2])
        return w_s.to(torch.float32)

    @staticmethod
    def _infer_block_size(w_q: torch.Tensor, w_s: torch.Tensor) -> list:
        k, s_k = w_q.shape[-1], w_s.shape[-1]
        assert k % s_k == 0, f"cannot infer block size from {w_q.shape=} {w_s.shape=}"
        block = k // s_k
        return [block, block]

    @staticmethod
    def _iter_quant_chunks(w_q: torch.Tensor, w_s: torch.Tensor, block_n: int):
        """Yields block-row-aligned (q_slice, s_slice) pairs of bounded size."""
        q3 = w_q.reshape(-1, *w_q.shape[-2:])
        s3 = w_s.reshape(-1, *w_s.shape[-2:])
        n, k = q3.shape[-2:]
        rows = max(block_n, CHUNK_NUMEL // k // block_n * block_n)
        for b in range(q3.shape[0]):
            for r0 in range(0, n, rows):
                r1 = min(r0 + rows, n)
                yield q3[b, r0:r1], s3[b, r0 // block_n : -(-r1 // block_n)]

    def _scale_and_block_size(self):
        s = self._normalize_scale(self.w_q, self.w_s)
        return s, self._infer_block_size(self.w_q, s)

    def iter_chunks(self):
        s, block_size = self._scale_and_block_size()
        for q, s_chunk in self._iter_quant_chunks(self.w_q, s, block_size[0]):
            q, s_chunk = q.cuda(), s_chunk.cuda()
            yield (
                block_quant_dequant(q, s_chunk, block_size, dtype=torch.bfloat16),
                block_quant_dequant(
                    self._quant_ulp(q), s_chunk, block_size, dtype=torch.float32
                ),
            )

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        s, block_size = self._scale_and_block_size()
        return block_quant_dequant(self.w_q, s, block_size, dtype=dtype)


class RawComparable(ComparableWeight):
    """Unquantized tensor: identity dequant, exact (no-tolerance) compare."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"raw(shape={tuple(self.tensor.shape)} dtype={self.tensor.dtype})"

    def iter_chunks(self):
        flat = self.tensor.reshape(-1)
        for start in range(0, flat.numel(), CHUNK_NUMEL):
            yield flat[start : start + CHUNK_NUMEL].cuda(), None

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return self.tensor


def compare_weights(
    expect: ComparableWeight, actual: ComparableWeight
) -> CompareResult:
    """Chunked element-wise compare in dequantized space."""
    equal = True
    max_abs_err = torch.zeros((), dtype=torch.float32)
    sum_abs_err = 0.0
    num_exceed = 0
    numel = 0
    for (expect_dq, expect_tol), (actual_dq, actual_tol) in zip(
        expect.iter_chunks(), actual.iter_chunks(), strict=True
    ):
        assert (
            expect_dq.shape == actual_dq.shape
        ), f"{expect_dq.shape=} {actual_dq.shape=}"
        numel += expect_dq.numel()
        abs_diff = (actual_dq.float() - expect_dq.float()).abs()
        if torch.all(abs_diff == 0):
            continue
        equal = False
        # |actual_dq - expect_dq| ≤ |actual_dq - w| + |expect_dq - w| ≤ actual_tol + expect_tol
        tol = (
            0.0 if expect_tol is None or actual_tol is None else expect_tol + actual_tol
        )
        max_abs_err = torch.maximum(max_abs_err, abs_diff.max().cpu())
        sum_abs_err += abs_diff.sum().item()
        # `~(diff <= tol)` instead of `diff > tol` so NaN counts as exceeding.
        num_exceed += int((~(abs_diff <= tol)).sum())
    return CompareResult(
        equal, max_abs_err.item(), sum_abs_err / max(numel, 1), num_exceed
    )


def select_comparable_weight(quant_method) -> Optional[type]:
    """Map a module's quant_method to its ComparableWeight. None means raw (bitwise equal) compare."""
    if (
        isinstance(quant_method, (Fp8LinearMethod, Fp8MoEMethod))
        and quant_method.block_quant
        and not quant_method.use_mxfp8
    ):
        return Fp8BlockComparable
    if isinstance(quant_method, (ModelOptFp4LinearMethod, ModelOptNvFp4FusedMoEMethod)):
        raise NotImplementedError(
            f"weight checker has no ComparableWeight for {type(quant_method).__name__}"
        )
    return None
