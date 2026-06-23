"""Reference weights for the weight checker: a quantized weight in dequantized,
encoding-invariant form. A new precision is a new ReferenceWeight subclass."""

from typing import Iterable, Optional, Tuple

import torch

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)

# Bound GPU memory: a full-size fp32 intermediate won't fit beside the KV-cache pool.
_CHUNK_NUMEL = 64 * 1024 * 1024


class ReferenceWeight:
    """A logical weight in encoding-invariant (dequantized) form."""

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
        """Yield (dequant, tolerance) cuda chunks bounded by _CHUNK_NUMEL. tolerance
        is this side's per-element abs tolerance (ULP in dequant space), or None
        when the format requires exact equality."""
        raise NotImplementedError

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Full dequantized tensor, for checksum hashing."""
        raise NotImplementedError


class Fp8BlockReference(ReferenceWeight):
    """fp8 block quant: a (weight, weight_scale_inv) pair, scale optionally
    UE8M0-packed. Square blocks with a possibly-partial last row block."""

    def __init__(self, w_q: torch.Tensor, w_s: torch.Tensor):
        self.w_q = w_q
        self.w_s = w_s

    def __repr__(self) -> str:
        return f"fp8_block(shape={tuple(self.w_q.shape)} dtype={self.w_q.dtype})"

    @staticmethod
    def _normalize_scale(w_q: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
        if w_s.dtype == torch.int32:
            # UE8M0 packed format (Blackwell DeepGEMM)
            w_s = inverse_transform_scale_ue8m0(w_s, mn=w_q.shape[-2])
        return w_s.to(torch.float32)

    @staticmethod
    def _block_size_of(w_q: torch.Tensor, w_s: torch.Tensor) -> list:
        # Square blocks. The row dim may have a partial last block (so n//s_n is wrong),
        # but the K dim is block-aligned, so k//s_k recovers the true block size exactly.
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
        rows = max(block_n, _CHUNK_NUMEL // k // block_n * block_n)
        for b in range(q3.shape[0]):
            for r0 in range(0, n, rows):
                r1 = min(r0 + rows, n)
                yield q3[b, r0:r1], s3[b, r0 // block_n : -(-r1 // block_n)]

    def _normalized(self):
        s = self._normalize_scale(self.w_q, self.w_s)
        return s, self._block_size_of(self.w_q, s)

    def iter_chunks(self):
        s, block_size = self._normalized()
        for q, s_chunk in self._iter_quant_chunks(self.w_q, s, block_size[0]):
            q, s_chunk = q.cuda(), s_chunk.cuda()
            yield (
                block_quant_dequant(q, s_chunk, block_size, dtype=torch.bfloat16),
                block_quant_dequant(
                    self._quant_ulp(q), s_chunk, block_size, dtype=torch.float32
                ),
            )

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        s, block_size = self._normalized()
        return block_quant_dequant(self.w_q, s, block_size, dtype=dtype)


def _compare_references(
    expect: ReferenceWeight, actual: ReferenceWeight
) -> Tuple[bool, float, float, int]:
    """Chunked compare of two ReferenceWeights in dequant space. Returns
    (equal, max_abs_err, mean_abs_err, num_exceed); num_exceed counts elements
    off by more than the combined per-side tolerance (NaN counts as exceeding)."""
    equal = True
    max_abs_err = torch.zeros((), dtype=torch.float32)
    sum_abs_err = 0.0
    num_exceed = 0
    numel = 0
    for (e_dq, e_tol), (a_dq, a_tol) in zip(
        expect.iter_chunks(), actual.iter_chunks(), strict=True
    ):
        assert e_dq.shape == a_dq.shape, f"{e_dq.shape=} {a_dq.shape=}"
        numel += e_dq.numel()
        abs_diff = (a_dq.float() - e_dq.float()).abs()
        if torch.all(abs_diff == 0):
            continue
        equal = False
        # Each side may deviate from the shared source weight by its own tolerance.
        tol = 0.0 if e_tol is None or a_tol is None else e_tol + a_tol
        # torch.maximum propagates NaN, unlike builtin max().
        max_abs_err = torch.maximum(max_abs_err, abs_diff.max().cpu())
        sum_abs_err += abs_diff.sum().item()
        # `~(diff <= tol)` instead of `diff > tol` so NaN counts as exceeding.
        num_exceed += int((~(abs_diff <= tol)).sum())
    return equal, max_abs_err.item(), sum_abs_err / max(numel, 1), num_exceed


def select_quantization_method(quant_method) -> Optional[type]:
    """Select the ReferenceWeight subclass for a module's quant_method: None if the
    module is not a quantized weight layer, the subclass for a supported format,
    else raise. Dispatching on quant_method (not param names) is robust to
    swizzled scales and per-format naming."""
    if not isinstance(quant_method, (LinearMethodBase, FusedMoEMethodBase)):
        return None
    if isinstance(quant_method, (UnquantizedLinearMethod, UnquantizedFusedMoEMethod)):
        return None
    # fp8 block quant has square blocks; mxfp8 ([1, 32]) is non-square and
    # per-tensor fp8 has no block scale, so neither is supported.
    if (
        isinstance(quant_method, (Fp8LinearMethod, Fp8MoEMethod))
        and quant_method.block_quant
        and not quant_method.use_mxfp8
    ):
        return Fp8BlockReference
    raise NotImplementedError(
        f"weight checker has no ReferenceWeight for {type(quant_method).__name__}"
    )
