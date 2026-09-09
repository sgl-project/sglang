"""KV page offsets must not overflow int32 in the Triton sparse-MLA decode kernel."""

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
    "Triton sparse MLA decode is gated to gfx950.",
)
class TestTritonSparseMLADecodeKVOffset(CustomTestCase):
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
        kv = torch.empty(n_pages, 1, DIM, device=self.device, dtype=self.fp8_dtype)
        step = 131072
        for start in range(0, n_pages, step):
            stop = min(start + step, n_pages)
            raw = torch.randn(
                stop - start, 1, DIM, device=self.device, generator=gen
            )
            kv[start:stop] = (raw * 0.3).to(self.fp8_dtype)
        return kv

    def _run(self, q_all, kv, indices):
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
            triton_sparse_mla_decode_splitk_fwd,
        )

        out = triton_sparse_mla_decode_splitk_fwd(
            q_all[..., :D_V],
            q_all[..., D_V:],
            kv,
            indices,
            self.sm_scale,
            D_V,
        ).squeeze(0)
        torch.cuda.synchronize()
        return out

    def _reference(self, q_all, kv, indices):
        seq, heads, _ = q_all.shape
        n_pages = kv.shape[0]
        out = torch.zeros(seq, heads, D_V, device=self.device)
        q = q_all.float()
        kvf = kv.reshape(n_pages, DIM).float()
        for s in range(seq):
            pages = indices[s].reshape(-1)
            pages = pages[(pages >= 0) & (pages < n_pages)]
            if pages.numel() == 0:
                continue
            k = kvf[pages.long()]
            scores = (q[s] @ k.T) * self.sm_scale
            p = torch.softmax(scores, dim=-1)
            out[s] = p @ k[:, :D_V]
        return out

    def _assert_matches(self, got, want, ctx):
        self.assertFalse(torch.isnan(got).any(), f"{ctx}: output has NaN")
        got, want = got.float(), want.float()
        signal = want.pow(2).mean().sqrt()
        noise = (got - want).pow(2).mean().sqrt().clamp(min=1e-12)
        snr_db = float(20 * torch.log10(signal / noise))
        self.assertGreater(snr_db, 20.0, f"{ctx}: SNR {snr_db:.1f} dB too low")

    def test_auto_wide_path_on_small_pool(self):
        """Below the int32 limit the narrow path must run and match fp32 ref."""
        seq, heads, topk, n_pages = 12, 16, 128, 4096
        gen = torch.Generator(device=self.device).manual_seed(3)

        def fp8(*shape):
            return (torch.randn(*shape, device=self.device, generator=gen) * 0.3).to(
                self.fp8_dtype
            )

        q_all = fp8(seq, heads, DIM)
        kv = self._fp8_pool(n_pages, gen)
        indices = torch.randint(
            0, n_pages, (seq, 1, topk), device=self.device, dtype=torch.int32, generator=gen
        )
        got = self._run(q_all, kv, indices)
        self._assert_matches(got, self._reference(q_all, kv, indices), "narrow")

    def test_pool_past_int32_limit(self):
        n_pages = INT32_PAGE_LIMIT + 71_730
        pool_bytes = n_pages * DIM
        free, _ = torch.cuda.mem_get_info(self.device)
        if free < pool_bytes + (1 << 30):
            self.skipTest(
                f"needs ~{(pool_bytes >> 20) + 1024} MiB free, have {free >> 20} MiB"
            )

        seq, heads, topk = 12, 16, 128
        gen = torch.Generator(device=self.device).manual_seed(5)
        kv = self._fp8_pool(n_pages, gen)

        def fp8(*shape):
            return (torch.randn(*shape, device=self.device, generator=gen) * 0.3).to(
                self.fp8_dtype
            )

        q_all = fp8(seq, heads, DIM)
        indices = torch.randint(
            INT32_PAGE_LIMIT + 1,
            n_pages,
            (seq, 1, topk),
            device=self.device,
            dtype=torch.int32,
            generator=gen,
        )
        self.assertGreater(int(indices.max()) * DIM, 2**31 - 1)

        got = self._run(q_all, kv, indices)
        self._assert_matches(got, self._reference(q_all, kv, indices), "wide")

    def test_launcher_widens_exactly_past_limit(self):
        import sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode as mod

        captured = {}

        class StubKernel:
            def __getitem__(self, grid):
                def call(*args, **kwargs):
                    captured.update(kwargs)

                return call

        original = mod._sparse_mla_decode_splitk_partial_kernel
        mod._sparse_mla_decode_splitk_partial_kernel = StubKernel()
        try:
            meta = dict(device="meta", dtype=self.fp8_dtype)
            mod.triton_sparse_mla_decode_splitk_fwd(
                torch.empty(6, 16, D_V, **meta),
                torch.empty(6, 16, D_TAIL, **meta),
                torch.empty(4096, 1, DIM, **meta),
                torch.empty(6, 2048, device="meta", dtype=torch.int32),
                self.sm_scale,
                D_V,
            )
            self.assertFalse(captured["WIDE_KV_OFFSET"])

            captured.clear()
            mod.triton_sparse_mla_decode_splitk_fwd(
                torch.empty(6, 16, D_V, **meta),
                torch.empty(6, 16, D_TAIL, **meta),
                torch.empty(INT32_PAGE_LIMIT + 1, 1, DIM, **meta),
                torch.empty(6, 2048, device="meta", dtype=torch.int32),
                self.sm_scale,
                D_V,
            )
            self.assertTrue(captured["WIDE_KV_OFFSET"])
        finally:
            mod._sparse_mla_decode_splitk_partial_kernel = original


if __name__ == "__main__":
    unittest.main()
