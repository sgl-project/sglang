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
    """Base comparable-weight class; one subclass per precision or raw tensor."""

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
            # ue8m0 packing aligns k to a multiple of 4; drop the padding blocks.
            w_s = w_s[..., : -(-w_q.shape[-1] // 128)]
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


class Nvfp4TrtllmMoeComparable(ComparableWeight):
    """ModelOpt NVFP4 fused-MoE expert weights in the FlashInfer TRT-LLM layout.

    process_weights_after_loading row-shuffles the packed E2M1 weights and
    swizzles the E4M3 block scales in place; this comparable inverts those
    layout transforms (per expert) and compares in dequantized space, so two
    different (qweight, block_scale, weight_scale_2) factorizations of the
    same logical weights are equal within quantization ULP tolerance.
    """

    # E2M1 magnitudes by nibble & 0x7; the top nibble bit is the sign.
    _E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    # Grid spacing at each magnitude: 0.5 below 2, 1 in [2, 4), 2 in [4, 6].
    _E2M1_SPACING = (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0)

    _is_w13 = True
    # (shape, is_scale) -> inverse layout indices, shared across instances.
    _inverse_indices_cache: dict = {}

    def __init__(
        self, w_q: torch.Tensor, w_scale: torch.Tensor, w_scale_2: torch.Tensor
    ):
        self.w_q = w_q
        self.w_scale = w_scale
        self.w_scale_2 = w_scale_2

    def __repr__(self) -> str:
        return (
            f"nvfp4_trtllm(shape={tuple(self.w_q.shape)} "
            f"scale2_shape={tuple(self.w_scale_2.shape)} w13={self._is_w13})"
        )

    def _forward_weight_perm(self, expert_2d: torch.Tensor) -> torch.Tensor:
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        cache: dict = {}
        if self._is_w13:
            return _maybe_get_cached_w3_w1_permute_indices(
                cache, expert_2d, 128, is_gated_act_gemm=True
            )
        return get_w2_permute_indices_with_cache(cache, expert_2d, 128)

    def _forward_scale_transform(self, expert_2d: torch.Tensor) -> torch.Tensor:
        from flashinfer import nvfp4_block_scale_interleave
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        cache: dict = {}
        if self._is_w13:
            perm = _maybe_get_cached_w3_w1_permute_indices(
                cache, expert_2d, 128, num_elts_per_sf=16, is_gated_act_gemm=True
            )
        else:
            perm = get_w2_permute_indices_with_cache(
                cache, expert_2d, 128, num_elts_per_sf=16
            )
        return nvfp4_block_scale_interleave(
            expert_2d[perm.to(expert_2d.device)].contiguous()
        )

    def _inverse_indices(self, shape, is_scale: bool, device) -> torch.Tensor:
        """out_flat[p] == in_flat[idx[p]] for the forward layout transform,
        recovered by tracing byte planes of the flattened position index."""
        key = (self._is_w13, tuple(shape), is_scale)
        cached = self._inverse_indices_cache.get(key)
        if cached is not None:
            return cached
        numel = 1
        for d in shape:
            numel *= d
        assert numel < 1 << 32, f"position tracing needs numel < 2^32, got {numel}"
        positions = torch.arange(numel, device=device, dtype=torch.int64)
        traced = torch.zeros(numel, device=device, dtype=torch.int64)
        for shift in range(0, 32, 8):
            plane = ((positions >> shift) & 0xFF).to(torch.uint8).reshape(shape)
            if is_scale:
                out_plane = self._forward_scale_transform(plane)
            else:
                out_plane = plane[self._forward_weight_perm(plane).to(device)]
            traced |= out_plane.reshape(-1).to(torch.int64) << shift
        self._inverse_indices_cache[key] = traced
        return traced

    def _unshuffle(self, shuffled_2d: torch.Tensor, is_scale: bool) -> torch.Tensor:
        src = self._inverse_indices(shuffled_2d.shape, is_scale, shuffled_2d.device)
        flat = shuffled_2d.reshape(-1)
        linear = torch.empty_like(flat)
        linear[src] = flat
        return linear.reshape(shuffled_2d.shape)

    def _expert_scale_2(self, expert_idx: int, num_rows: int) -> torch.Tensor:
        """Per-row fp32 weight_scale_2 column; gated w13 keeps one scale per
        gate/up half."""
        s2 = self.w_scale_2[expert_idx].float()
        if s2.numel() == 1:
            return s2.reshape(1).expand(num_rows)
        assert s2.numel() == 2 and num_rows % 2 == 0, (s2.shape, num_rows)
        half = num_rows // 2
        return torch.cat([s2[0].expand(half), s2[1].expand(half)])

    def _dequant_expert(self, expert_idx: int):
        # Snapshot-side tensors live on CPU; everything is moved to cuda for
        # the unshuffle/dequant, so the LUTs must be cuda too.
        device = "cuda"
        lut = torch.tensor(
            [*self._E2M1_VALUES, *(-v for v in self._E2M1_VALUES)],
            dtype=torch.float32,
            device=device,
        )
        spacing_lut = torch.tensor(
            self._E2M1_SPACING * 2, dtype=torch.float32, device=device
        )
        q = self._unshuffle(self.w_q[expert_idx].view(torch.uint8).cuda(), False)
        s = self._unshuffle(
            self.w_scale[expert_idx].view(torch.uint8).cuda(), True
        ).view(torch.float8_e4m3fn)

        low, high = (q & 0x0F).to(torch.long), (q >> 4).to(torch.long)
        vals = torch.stack((lut[low], lut[high]), dim=-1).reshape(q.shape[0], -1)
        spacing = torch.stack((spacing_lut[low], spacing_lut[high]), dim=-1).reshape(
            q.shape[0], -1
        )

        num_rows = q.shape[0]
        scale_2_col = (
            self._expert_scale_2(expert_idx, num_rows).to(device="cuda").unsqueeze(-1)
        )
        scale = s.float().repeat_interleave(16, dim=-1) * scale_2_col
        # Tolerance floors the block scale at the smallest E4M3 subnormal:
        # quantizers legitimately differ on whether a tiny block flushes to a
        # zero scale or clamps to the subnormal grid.
        tol_scale = (
            s.float().clamp(min=2.0**-9).repeat_interleave(16, dim=-1) * scale_2_col
        )
        return vals * scale, spacing * tol_scale

    def iter_chunks(self):
        for expert_idx in range(self.w_q.shape[0]):
            yield self._dequant_expert(expert_idx)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.stack(
            [self._dequant_expert(i)[0].to(dtype) for i in range(self.w_q.shape[0])]
        )


class Nvfp4TrtllmW2Comparable(Nvfp4TrtllmMoeComparable):
    _is_w13 = False


class RawComparable(ComparableWeight):
    """Bitwise equal compare on raw tensor."""

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
    """Chunked element-wise compare in ComparableWeight space."""
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
    if isinstance(quant_method, ModelOptNvFp4FusedMoEMethod):
        if getattr(quant_method, "enable_flashinfer_trtllm_moe", False):
            return Nvfp4TrtllmMoeComparable
        raise NotImplementedError(
            f"weight checker has no ComparableWeight for {type(quant_method).__name__}"
        )
    if isinstance(quant_method, ModelOptFp4LinearMethod):
        raise NotImplementedError(
            f"weight checker has no ComparableWeight for {type(quant_method).__name__}"
        )
    return None
