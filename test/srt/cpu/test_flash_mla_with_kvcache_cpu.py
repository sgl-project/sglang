import itertools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

try:
    import quant_utils as flashmla_quant
    from sgl_kernel.flash_mla import flash_mla_with_kvcache_cpu

    _IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - exercised only when kernel missing
    flash_mla_with_kvcache_cpu = None  # type: ignore[assignment]
    flashmla_quant = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e


# Map ``fp8_layout`` integer (matches ``FP8KVCacheLayout`` C++ enum and
# ``flashmla_quant.FP8KVCacheLayout``) -> (d_qk, d_v).
_LAYOUT_DIMS = {
    1: (576, 512),  # V32_FP8Sparse
    2: (512, 512),  # MODEL1_FP8Sparse
}


def _ref_sparse_attn_decode(
    q: torch.Tensor,  # [B, S_q, H_q, D_qk] bf16
    k_dequant: torch.Tensor,  # [num_blocks, page_size, 1, D_qk] bf16
    indices: torch.Tensor,  # [B, S_q, topk] int32/int64
    topk_length,
    attn_sink,
    extra_k_dequant,
    extra_indices,
    extra_topk_length,
    sm_scale: float,
    d_v: int,
):
    """Pure-PyTorch reference for the sparse FP8 decode kernel.
    Mirrors :func:`sglang.srt.flashmla_tests.ref.ref_sparse_attn_decode`
    but accepts already-dequantized BF16 K caches and inlines only the
    parts needed for the decode comparison.
    """
    b, s_q, h_q, d_qk = q.shape

    def _gather(k_dq: torch.Tensor, idxs: torch.Tensor, tl):
        topk = idxs.size(-1)
        idxs_fixed = torch.clamp_min(idxs, 0).to(torch.int64)
        flat_k = k_dq.reshape(-1, d_qk)
        gathered = flat_k.index_select(0, idxs_fixed.reshape(-1)).reshape(
            b, s_q, topk, d_qk
        )
        invalid = idxs == -1
        if tl is not None:
            tl_mask = torch.arange(topk, device=invalid.device).view(1, 1, topk).expand(
                b, s_q, topk
            ) >= tl.view(b, 1, 1)
            invalid = invalid | tl_mask
        return gathered, invalid

    gathered_kv, invalid_mask = _gather(k_dequant, indices, topk_length)
    if extra_k_dequant is not None:
        gathered_kv1, invalid_mask1 = _gather(
            extra_k_dequant, extra_indices, extra_topk_length
        )
        gathered_kv = torch.cat([gathered_kv, gathered_kv1], dim=2)
        invalid_mask = torch.cat([invalid_mask, invalid_mask1], dim=2)

    gathered_kv_f = gathered_kv.reshape(b * s_q, -1, d_qk).float()
    gathered_kv_f[gathered_kv_f != gathered_kv_f] = 0.0  # NaN -> 0

    q_f = q.float().reshape(b * s_q, h_q, d_qk)
    attn_weight = q_f @ gathered_kv_f.transpose(-1, -2)  # [B*S_q, H_q, T]
    attn_weight = attn_weight * sm_scale
    full_invalid = invalid_mask.reshape(b * s_q, 1, -1).expand(
        b * s_q, h_q, invalid_mask.size(-1)
    )
    attn_weight = attn_weight.masked_fill(full_invalid, float("-inf"))

    lse = attn_weight.logsumexp(dim=-1)  # [B*S_q, H_q]
    probs = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = probs @ gathered_kv_f[..., :d_v]
    output = output.view(b, s_q, h_q, d_v)
    lse = lse.view(b, s_q, h_q)

    if attn_sink is not None:
        output = output * (
            1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))
        ).unsqueeze(-1)

    lonely_q_mask = lse == float("-inf")
    output = output.masked_fill(
        lonely_q_mask.unsqueeze(-1).expand(b, s_q, h_q, d_v), 0.0
    )
    lse = torch.where(lonely_q_mask, torch.full_like(lse, float("+inf")), lse)

    return output.to(torch.bfloat16), lse.transpose(1, 2).contiguous()


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"flash_mla_with_kvcache_cpu / flashmla_tests.quant unavailable: {_IMPORT_ERROR}",
)
class TestFlashMLAWithKVCacheCPU(CustomTestCase):
    """Tests for ``flash_mla_with_kvcache_cpu`` (sparse FP8 decode, CPU AMX).
    Only ``torch.bfloat16`` queries are supported by the kernel
    (see ``sgl-kernel/csrc/cpu/flash_mla.cpp``); ``page_size`` and the FP8
    KV-cache layout / index dtype are the dimensions we sweep here.
    """

    def setUp(self):
        torch.manual_seed(1234)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_quantized_kv_cache(
        self, num_blocks, page_size, d_qk, layout_enum, *, as_uint8=False
    ):
        # Generate random BF16 K cache then quantize using the project's
        # reference helper.  Shape: [num_blocks, page_size, 1, d_qk] bf16.
        k_bf16 = torch.randn(num_blocks, page_size, 1, d_qk, dtype=torch.bfloat16) * 0.5
        k_quant = flashmla_quant.quantize_k_cache(k_bf16, layout_enum)
        # Re-dequantize so the BF16 K used by the reference path matches the
        # one the kernel will see internally; this excludes quantization
        # error from the kernel-vs-ref comparison.
        k_dequant = flashmla_quant.dequantize_k_cache(k_quant, layout_enum)
        if as_uint8:
            k_quant = k_quant.view(torch.uint8)
        return k_quant.contiguous(), k_dequant

    def _make_bf16_kv_cache(self, num_blocks, page_size, d_qk):
        k_bf16 = torch.randn(num_blocks, page_size, 1, d_qk, dtype=torch.bfloat16) * 0.5
        k_bf16 = k_bf16.contiguous()
        return k_bf16, k_bf16

    def _make_indices(
        self, b, s_q, topk, total_tokens, *, dtype=torch.int32, invalid_ratio=0.0
    ):
        # Unique-per-row valid token indices, optionally with some -1 entries.
        idx = torch.empty(b, s_q, topk, dtype=dtype)
        for bi in range(b):
            for si in range(s_q):
                if total_tokens >= topk:
                    perm = torch.randperm(total_tokens)[:topk]
                else:
                    perm = torch.randint(0, total_tokens, (topk,))
                idx[bi, si] = perm.to(dtype)
        if invalid_ratio > 0.0:
            mask = torch.rand(b, s_q, topk) < invalid_ratio
            idx[mask] = -1
        return idx

    def _run_one(
        self,
        *,
        b,
        s_q,
        h_q,
        topk,
        page_size,
        num_blocks,
        fp8_layout,
        idx_dtype=torch.int32,
        have_attn_sink=False,
        have_topk_length=False,
        have_extra=False,
        have_extra_topk_length=False,
        extra_topk=0,
        extra_num_blocks=0,
        valid_tokens=None,
        extra_valid_tokens=None,
        invalid_ratio=0.0,
        is_fp8_kvcache=True,
        topk_length_value=None,
        extra_topk_length_value=None,
        fp8_cache_as_uint8=False,
    ):
        d_qk, d_v = _LAYOUT_DIMS[fp8_layout]
        layout_enum = flashmla_quant.FP8KVCacheLayout(fp8_layout)

        q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16) * 0.5

        if is_fp8_kvcache:
            k_cache, k_dequant = self._make_quantized_kv_cache(
                num_blocks,
                page_size,
                d_qk,
                layout_enum,
                as_uint8=fp8_cache_as_uint8,
            )
        else:
            k_cache, k_dequant = self._make_bf16_kv_cache(num_blocks, page_size, d_qk)
        total_tokens = (
            valid_tokens if valid_tokens is not None else (num_blocks * page_size)
        )
        indices = self._make_indices(
            b, s_q, topk, total_tokens, dtype=idx_dtype, invalid_ratio=invalid_ratio
        )

        attn_sink = torch.randn(h_q, dtype=torch.float32) if have_attn_sink else None
        topk_length = None
        if have_topk_length:
            if topk_length_value is None:
                lo = max(1, topk // 2)
                topk_length = torch.randint(
                    low=lo, high=topk + 1, size=(b,), dtype=torch.int32
                )
            else:
                topk_length = torch.full((b,), topk_length_value, dtype=torch.int32)

        extra_k_cache = None
        extra_k_dequant = None
        extra_indices = None
        extra_topk_length = None
        if have_extra:
            if is_fp8_kvcache:
                extra_k_cache, extra_k_dequant = self._make_quantized_kv_cache(
                    extra_num_blocks,
                    page_size,
                    d_qk,
                    layout_enum,
                    as_uint8=fp8_cache_as_uint8,
                )
            else:
                extra_k_cache, extra_k_dequant = self._make_bf16_kv_cache(
                    extra_num_blocks, page_size, d_qk
                )
            extra_total_tokens = (
                extra_valid_tokens
                if extra_valid_tokens is not None
                else (extra_num_blocks * page_size)
            )
            extra_indices = self._make_indices(
                b, s_q, extra_topk, extra_total_tokens, dtype=idx_dtype
            )
            if have_extra_topk_length:
                if extra_topk_length_value is None:
                    lo = max(1, extra_topk // 2)
                    extra_topk_length = torch.randint(
                        low=lo, high=extra_topk + 1, size=(b,), dtype=torch.int32
                    )
                else:
                    extra_topk_length = torch.full(
                        (b,), extra_topk_length_value, dtype=torch.int32
                    )

        sm_scale = d_qk**-0.5

        out_cpu, lse_cpu = flash_mla_with_kvcache_cpu(
            q=q,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=d_v,
            softmax_scale=sm_scale,
            causal=False,
            is_fp8_kvcache=is_fp8_kvcache,
            indices=indices,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
            fp8_layout=fp8_layout,
        )

        out_ref, lse_ref = _ref_sparse_attn_decode(
            q=q,
            k_dequant=k_dequant,
            indices=indices,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_k_dequant=extra_k_dequant,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
            sm_scale=sm_scale,
            d_v=d_v,
        )

        # Shape / dtype contract.
        self.assertEqual(tuple(out_cpu.shape), (b, s_q, h_q, d_v))
        self.assertEqual(tuple(lse_cpu.shape), (b, h_q, s_q))
        self.assertEqual(out_cpu.dtype, torch.bfloat16)
        self.assertEqual(lse_cpu.dtype, torch.float32)

        # Numerical check.  Tolerances mirror the loosened thresholds used by
        # ``debug_flash_mla_adapter._assert_close``.
        torch.testing.assert_close(out_cpu, out_ref, atol=1e-2, rtol=1e-2)

        # LSE: ignore positions that the reference left at +inf
        # (no attendable K) since the kernel may report a different sentinel
        # value for those slots.
        finite = torch.isfinite(lse_ref) & torch.isfinite(lse_cpu)
        if finite.any():
            torch.testing.assert_close(
                lse_cpu[finite], lse_ref[finite], atol=5e-3, rtol=1e-2
            )

    # ------------------------------------------------------------------
    # Tests: sweep page_size, layout (KV cache "dtype"), and index dtype.
    # ------------------------------------------------------------------
    def test_basic_layouts_page_sizes_and_idx_dtypes(self):
        configs = list(
            itertools.product(
                [64, 128, 256],  # page_size
                [1, 2],  # fp8_layout (V32 / MODEL1)
                [torch.int32, torch.int64],  # idx_dtype
            )
        )
        for page_size, fp8_layout, idx_dtype in configs:
            with self.subTest(
                page_size=page_size,
                fp8_layout=fp8_layout,
                idx_dtype=str(idx_dtype),
            ):
                self._run_one(
                    b=2,
                    s_q=1,
                    h_q=16,
                    topk=128,
                    page_size=page_size,
                    num_blocks=4,
                    fp8_layout=fp8_layout,
                    idx_dtype=idx_dtype,
                )

    def test_varying_batch_seq_and_topk(self):
        # Cover non-trivial batch / s_q / topk combinations that are not
        # exact multiples of the kernel's BLOCK_N (128).
        configs = [
            # (b, s_q, h_q, topk, page_size, num_blocks, fp8_layout)
            (1, 1, 16, 64, 64, 4, 1),
            (1, 2, 32, 100, 64, 4, 2),
            (3, 1, 16, 200, 128, 3, 1),
            (2, 2, 16, 256, 128, 4, 2),
            (1, 1, 64, 32, 64, 2, 2),
        ]
        for b, s_q, h_q, topk, page_size, num_blocks, layout in configs:
            with self.subTest(
                b=b,
                s_q=s_q,
                h_q=h_q,
                topk=topk,
                page_size=page_size,
                num_blocks=num_blocks,
                fp8_layout=layout,
            ):
                self._run_one(
                    b=b,
                    s_q=s_q,
                    h_q=h_q,
                    topk=topk,
                    page_size=page_size,
                    num_blocks=num_blocks,
                    fp8_layout=layout,
                )

    def test_with_attn_sink(self):
        for fp8_layout in (1, 2):
            with self.subTest(fp8_layout=fp8_layout):
                self._run_one(
                    b=2,
                    s_q=1,
                    h_q=16,
                    topk=128,
                    page_size=64,
                    num_blocks=4,
                    fp8_layout=fp8_layout,
                    have_attn_sink=True,
                )

    def test_with_topk_length(self):
        for fp8_layout in (1, 2):
            with self.subTest(fp8_layout=fp8_layout):
                self._run_one(
                    b=3,
                    s_q=1,
                    h_q=16,
                    topk=128,
                    page_size=128,
                    num_blocks=4,
                    fp8_layout=fp8_layout,
                    have_topk_length=True,
                )

    def test_with_short_topk_length(self):
        # Covers the optimized path that stops processing once topk_length is
        # reached instead of iterating over the full preallocated topk width.
        for is_fp8_kvcache in (True, False):
            with self.subTest(is_fp8_kvcache=is_fp8_kvcache):
                self._run_one(
                    b=2,
                    s_q=1,
                    h_q=16,
                    topk=256,
                    page_size=128,
                    num_blocks=4,
                    fp8_layout=2,
                    have_topk_length=True,
                    topk_length_value=17,
                    have_extra=True,
                    extra_topk=128,
                    extra_num_blocks=2,
                    have_extra_topk_length=True,
                    extra_topk_length_value=9,
                    is_fp8_kvcache=is_fp8_kvcache,
                )

    def test_with_invalid_indices(self):
        # Some indices set to -1; the kernel must mask them out and the
        # reference output must still match.
        for fp8_layout in (1, 2):
            with self.subTest(fp8_layout=fp8_layout):
                self._run_one(
                    b=2,
                    s_q=1,
                    h_q=16,
                    topk=128,
                    page_size=64,
                    num_blocks=4,
                    fp8_layout=fp8_layout,
                    invalid_ratio=0.25,
                )

    def test_with_all_invalid_tile(self):
        # A fully invalid sparse tile should be skipped without changing the
        # online-softmax state; the query becomes lonely and returns zero/+inf.
        for is_fp8_kvcache in (True, False):
            with self.subTest(is_fp8_kvcache=is_fp8_kvcache):
                self._run_one(
                    b=2,
                    s_q=1,
                    h_q=16,
                    topk=256,
                    page_size=128,
                    num_blocks=4,
                    fp8_layout=2,
                    invalid_ratio=1.0,
                    is_fp8_kvcache=is_fp8_kvcache,
                )

    def test_with_extra_kv_cache(self):
        # Exercise the "main + extra" KV-cache concatenation path
        # (used by DSv4's chunked KV cache layout).
        for fp8_layout in (1, 2):
            for have_topk_length, have_extra_topk_length in (
                (False, False),
                (True, True),
            ):
                with self.subTest(
                    fp8_layout=fp8_layout,
                    have_topk_length=have_topk_length,
                ):
                    self._run_one(
                        b=2,
                        s_q=1,
                        h_q=16,
                        topk=64,
                        page_size=64,
                        num_blocks=4,
                        fp8_layout=fp8_layout,
                        have_extra=True,
                        extra_topk=64,
                        extra_num_blocks=2,
                        have_topk_length=have_topk_length,
                        have_extra_topk_length=have_extra_topk_length,
                    )

    def test_with_uint8_fp8_kv_cache(self):
        # Production FP8 sparse caches may be passed as raw packed bytes.
        for fp8_layout in (1, 2):
            with self.subTest(fp8_layout=fp8_layout):
                self._run_one(
                    b=2,
                    s_q=1,
                    h_q=16,
                    topk=64,
                    page_size=64,
                    num_blocks=4,
                    fp8_layout=fp8_layout,
                    have_extra=True,
                    extra_topk=32,
                    extra_num_blocks=2,
                    fp8_cache_as_uint8=True,
                )

    def test_with_bf16_kv_cache(self):
        # Non-FP8 KV cache should skip dequantization and use BF16 cache rows
        # directly for both main and extra KV sources.
        for have_extra in (False, True):
            with self.subTest(have_extra=have_extra):
                self._run_one(
                    b=2,
                    s_q=1,
                    h_q=16,
                    topk=64,
                    page_size=64,
                    num_blocks=4,
                    fp8_layout=2,
                    have_extra=have_extra,
                    extra_topk=32 if have_extra else 0,
                    extra_num_blocks=2 if have_extra else 0,
                    have_topk_length=True,
                    have_extra_topk_length=have_extra,
                    is_fp8_kvcache=False,
                )

    def test_oversized_preallocated_kv_cache(self):
        # Only the token range implied by indices should be considered active;
        # the backing KV cache can be much larger because it is preallocated.
        for is_fp8_kvcache in (True, False):
            with self.subTest(is_fp8_kvcache=is_fp8_kvcache):
                self._run_one(
                    b=2,
                    s_q=1,
                    h_q=16,
                    topk=64,
                    page_size=64,
                    num_blocks=8,
                    fp8_layout=2,
                    have_extra=True,
                    extra_topk=32,
                    extra_num_blocks=4,
                    valid_tokens=96,
                    extra_valid_tokens=48,
                    invalid_ratio=0.1,
                    is_fp8_kvcache=is_fp8_kvcache,
                )

    def test_with_attn_sink_and_extra(self):
        # Combined: attn_sink + topk_length + extra K cache, both layouts.
        for fp8_layout in (1, 2):
            with self.subTest(fp8_layout=fp8_layout):
                self._run_one(
                    b=2,
                    s_q=2,
                    h_q=16,
                    topk=128,
                    page_size=128,
                    num_blocks=3,
                    fp8_layout=fp8_layout,
                    have_attn_sink=True,
                    have_topk_length=True,
                    have_extra=True,
                    extra_topk=64,
                    extra_num_blocks=2,
                )


if __name__ == "__main__":
    unittest.main()
