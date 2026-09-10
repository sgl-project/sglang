"""KV page offsets must not overflow int32 in the Triton sparse-MLA kernel.

The inner loop addresses the KV pool as `page * DIM`. Triton evaluates that in
int32, so once the pool passes (2**31 - 1) // DIM pages the offset wraps
negative and the load reads a wild address -- an out-of-bounds fault, or worse,
silently wrong attention output.

The pool is sized from whatever HBM is left after weights, so whether a run
crosses the limit depends on co-tenants rather than on anything in the request:
production logs show pools of 2.50M and 2.82M tokens (safe) alongside 4.00M
(over the 3.73M limit at DIM=576).
"""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=180, suite="nightly-amd-kernel-1-gpu")

D_V = 512
D_TAIL = 64
DIM = D_V + D_TAIL
INT32_PAGE_LIMIT = (2**31 - 1) // DIM


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(),
    "Triton sparse MLA prefill is gated to gfx950.",
)
class TestTritonSparseMLAKVOffset(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

        self.device = torch.device("cuda")
        self.fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
        self.sm_scale = 1.0 / (DIM**0.5)

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def _fp8_pool(self, n_pages: int, gen) -> torch.Tensor:
        """Fill the pool in slices; a fp32 staging buffer for the whole thing
        would need 15x the bytes of the fp8 result."""
        kv = torch.empty(n_pages, DIM, device=self.device, dtype=self.fp8_dtype)
        step = 131072
        for start in range(0, n_pages, step):
            stop = min(start + step, n_pages)
            raw = torch.randn(
                stop - start, DIM, device=self.device, generator=gen
            )
            kv[start:stop] = (raw * 0.3).to(self.fp8_dtype)
        return kv

    def _run(self, q_nope, q_rope, kv, indices, wide_kv_offset):
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            _launch_sparse_mla_prefill,
        )

        seq, heads, _ = q_nope.shape
        out = torch.full(
            (seq, heads, D_V), float("nan"), device=self.device, dtype=torch.bfloat16
        )
        _launch_sparse_mla_prefill(
            q_nope,
            q_rope,
            kv,
            indices,
            out,
            self.sm_scale,
            D_V,
            wide_kv_offset=wide_kv_offset,
        )
        torch.cuda.synchronize()
        return out

    def _reference(self, q_nope, q_rope, kv, indices):
        """Gather-then-attend in fp32. Gathers first so the fp32 copy is
        (seq, topk, DIM) rather than the whole pool."""
        idx = indices[:, 0, :].long()
        k = kv[idx].float()
        k_main, k_tail = k[..., :D_V], k[..., D_V:]
        qk = torch.einsum("shd,skd->shk", q_nope.float(), k_main)
        qk += torch.einsum("shd,skd->shk", q_rope.float(), k_tail)
        p = torch.softmax(qk * self.sm_scale, dim=-1)
        return torch.einsum("shk,skd->shd", p, k_main)

    def _assert_matches(self, got, want, ctx):
        self.assertFalse(torch.isnan(got).any(), f"{ctx}: output has NaN")
        got, want = got.float(), want.float()
        signal = want.pow(2).mean().sqrt()
        noise = (got - want).pow(2).mean().sqrt().clamp(min=1e-12)
        snr_db = float(20 * torch.log10(signal / noise))
        # The kernel requantises the probabilities to fp8 before the PV dot, so
        # this cannot be exact; a wrong base address lands far below 20 dB.
        self.assertGreater(snr_db, 20.0, f"{ctx}: SNR {snr_db:.1f} dB too low")

    def test_wide_offset_matches_narrow_on_small_pool(self):
        """Below the limit both paths are valid, so they must agree exactly."""
        seq, heads, topk, n_pages = 512, 16, 128, 4096
        gen = torch.Generator(device=self.device).manual_seed(7)

        def fp8(*shape):
            return (torch.randn(*shape, device=self.device, generator=gen) * 0.3).to(
                self.fp8_dtype
            )

        q_nope = fp8(seq, heads, D_V)
        q_rope = fp8(seq, heads, D_TAIL)
        kv = self._fp8_pool(n_pages, gen)
        indices = torch.randint(
            0, n_pages, (seq, 1, topk), device=self.device, dtype=torch.int32, generator=gen
        ).contiguous()

        narrow = self._run(q_nope, q_rope, kv, indices, wide_kv_offset=False)
        wide = self._run(q_nope, q_rope, kv, indices, wide_kv_offset=True)
        torch.testing.assert_close(wide, narrow, rtol=0, atol=0)

    def test_pool_past_int32_limit(self):
        """Pages above the limit must still read the rows they name.

        Only the widened path runs: the int32 path computes a negative offset
        here, and faulting the queue would take down the rest of the suite.
        """
        n_pages = INT32_PAGE_LIMIT + 71_730  # 3,800,000 pages at DIM=576
        pool_bytes = n_pages * DIM
        free, _ = torch.cuda.mem_get_info(self.device)
        if free < pool_bytes + (1 << 30):
            self.skipTest(
                f"needs ~{(pool_bytes >> 20) + 1024} MiB free, have {free >> 20} MiB"
            )

        seq, heads, topk = 64, 16, 128
        gen = torch.Generator(device=self.device).manual_seed(11)
        kv = self._fp8_pool(n_pages, gen)

        def fp8(*shape):
            return (torch.randn(*shape, device=self.device, generator=gen) * 0.3).to(
                self.fp8_dtype
            )

        q_nope = fp8(seq, heads, D_V)
        q_rope = fp8(seq, heads, D_TAIL)
        # Draw only from the region whose int32 offset has already wrapped.
        indices = torch.randint(
            INT32_PAGE_LIMIT + 1,
            n_pages,
            (seq, 1, topk),
            device=self.device,
            dtype=torch.int32,
            generator=gen,
        ).contiguous()
        self.assertGreater(int(indices.max()) * DIM, 2**31 - 1)

        got = self._run(q_nope, q_rope, kv, indices, wide_kv_offset=True)
        self._assert_matches(got, self._reference(q_nope, q_rope, kv, indices), "wide")

    def test_launcher_widens_exactly_past_limit(self):
        """Auto-selection must flip at the boundary, not near it.

        Uses meta tensors and a stub launch so the boundary can be probed at
        pool sizes that would not fit in HBM.
        """
        import sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill as mod

        captured = {}

        class StubKernel:
            def __getitem__(self, grid):
                def call(*args, **kwargs):
                    captured.update(kwargs)

                return call

        original = mod._sparse_mla_prefill_kernel
        mod._sparse_mla_prefill_kernel = StubKernel()
        try:
            for n_pages, expected in (
                (4096, False),
                (INT32_PAGE_LIMIT, False),
                (INT32_PAGE_LIMIT + 1, True),
                (4_000_576, True),
            ):
                meta = dict(device="meta", dtype=self.fp8_dtype)
                mod._launch_sparse_mla_prefill(
                    torch.empty(64, 16, D_V, **meta),
                    torch.empty(64, 16, D_TAIL, **meta),
                    torch.empty(n_pages, DIM, **meta),
                    torch.empty(64, 1, 128, device="meta", dtype=torch.int32),
                    torch.empty(64, 16, D_V, device="meta", dtype=torch.bfloat16),
                    self.sm_scale,
                    D_V,
                )
                self.assertEqual(
                    captured["WIDE_KV_OFFSET"],
                    expected,
                    f"n_pages={n_pages} (limit {INT32_PAGE_LIMIT})",
                )
        finally:
            mod._sparse_mla_prefill_kernel = original

    def test_sanitize_clamps_into_range_and_keeps_padding_masked(self):
        """Sanitizing must bound page ids without disturbing the -1 padding.

        It runs per prefill call on the serving path, so it is written as a
        single clamp over the whole tensor rather than a boolean-mask
        assignment, which would cost a nonzero() and a host sync each time.
        That relies on the kernel only testing `idx >= 0`, so pin both halves:
        oversized ids land inside the pool, and negatives stay negative.
        """
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            _sanitize_indices,
        )

        n_pages = 4096
        kv = torch.empty(n_pages, DIM, device=self.device, dtype=self.fp8_dtype)
        indices = torch.tensor(
            [[[0, 17, n_pages - 1, n_pages, n_pages + 12345, -1, -1, -7]]],
            device=self.device,
            dtype=torch.int32,
        )

        got = _sanitize_indices(indices, kv)

        self.assertTrue(bool((got[got >= 0] < n_pages).all()), f"out of range: {got}")
        was_pad = indices < 0
        self.assertTrue(bool((got[was_pad] < 0).all()), "padding stopped being masked")
        self.assertTrue(
            bool((got[~was_pad] == indices[~was_pad].clamp(max=n_pages - 1)).all()),
            "in-range ids were altered",
        )

    def test_sanitize_leaves_output_unchanged_for_any_negative_sentinel(self):
        """Which negative marks padding must not reach the result."""
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            triton_sparse_mla_prefill_fwd,
        )

        seq, heads, topk, n_pages = 128, 16, 64, 2048
        gen = torch.Generator(device=self.device).manual_seed(11)

        def fp8(*shape):
            return (torch.randn(*shape, device=self.device, generator=gen) * 0.3).to(
                self.fp8_dtype
            )

        q_nope = fp8(seq, heads, D_V)
        q_rope = fp8(seq, heads, D_TAIL)
        kv = self._fp8_pool(n_pages, gen)
        indices = torch.randint(
            0, n_pages, (seq, 1, topk), device=self.device, dtype=torch.int32, generator=gen
        ).contiguous()
        indices[:, :, topk // 2 :] = -1

        with_neg_one = triton_sparse_mla_prefill_fwd(
            q_nope, q_rope, kv, indices, self.sm_scale, D_V
        )
        other = indices.clone()
        other[other < 0] = -9999
        with_other = triton_sparse_mla_prefill_fwd(
            q_nope, q_rope, kv, other, self.sm_scale, D_V
        )
        torch.testing.assert_close(with_other, with_neg_one, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
