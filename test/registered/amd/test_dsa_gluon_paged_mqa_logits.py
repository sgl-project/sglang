"""The in-tree gfx950 paged MQA logits kernel against aiter's, on DSA decode /
MTP-verify rows: same logits to fp32 rounding (the head sum is reordered) and
the same top-2048 selection."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

HEADS, DIM, PAGE, TOPK = 32, 128, 64, 2048
FP8 = torch.float8_e4m3fn


def _skip_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0]
    if arch != "gfx950":
        return f"kernel is gfx950-only, got {arch}"
    try:
        from aiter.ops.cache import indexer_k_quant_and_cache  # noqa: F401
        from aiter.ops.triton.pa_mqa_logits import (  # noqa: F401
            deepgemm_fp8_paged_mqa_logits,
        )

        from sglang.kernels.ops.attention.dsa import (  # noqa: F401
            gluon_paged_mqa_logits,
        )
    except ImportError as exc:  # pragma: no cover - build-environment dependent
        return f"unavailable: {exc}"
    return None


SKIP = _skip_reason()

# Same placement as test_dsa_gfx950_fused_indexer.py: a gfx950 suite.
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipIf(SKIP is not None, SKIP or "")
class TestGluonPagedMqaLogits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aiter.ops.cache import indexer_k_quant_and_cache
        from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

        from sglang.kernels.ops.attention.dsa import gluon_paged_mqa_logits

        cls.ref_fn = staticmethod(deepgemm_fp8_paged_mqa_logits)
        cls.mod = gluon_paged_mqa_logits
        cls.dev = torch.device("cuda", 0)
        # Index-K pool written by the production writer (preshuffled, ue8m0).
        cls.num_pages = 8192
        g = torch.Generator(device=cls.dev).manual_seed(0)
        k = torch.randn(cls.num_pages * PAGE, DIM, device=cls.dev, generator=g).to(
            torch.bfloat16
        )
        pool = torch.zeros(
            cls.num_pages, PAGE * (DIM + 4), dtype=torch.uint8, device=cls.dev
        )
        indexer_k_quant_and_cache(
            k,
            pool.view(-1, PAGE, DIM + 4).view(FP8),
            torch.arange(cls.num_pages * PAGE, device=cls.dev),
            DIM,
            "ue8m0",
            preshuffle=True,
        )
        cls.kv = pool.view(cls.num_pages, PAGE, 1, DIM + 4).view(FP8)

    def _case(self, lens, rows_per_request, share, width):
        """Rows grouped ``rows_per_request`` per request with lengths L+1..L+r; with
        ``share`` the rows of a request use one page table (MTP verify)."""
        g = torch.Generator(device="cpu").manual_seed(sum(lens) + width)
        rows = len(lens) * rows_per_request
        pt = torch.zeros(rows, width, dtype=torch.int32)
        seq = torch.empty(rows, dtype=torch.int32)
        for i, L in enumerate(lens):
            npg = -(-(L + rows_per_request) // PAGE)
            base = torch.randperm(self.num_pages, generator=g)[:npg].int()
            for j in range(rows_per_request):
                r = i * rows_per_request + j
                pt[r, :npg] = (
                    base
                    if share
                    else torch.randperm(self.num_pages, generator=g)[:npg].int()
                )
                seq[r] = L + j + 1
        q = torch.randn(rows, HEADS, DIM, generator=g).clamp_(-4, 4).to(FP8)
        w = torch.randn(rows, HEADS, generator=g) * HEADS**-0.5
        return q.to(self.dev), w.to(self.dev), seq.to(self.dev), pt.to(self.dev)

    def _logits(self, fn, q, w, seq, pt):
        out = torch.full(
            (q.shape[0], pt.shape[1] * PAGE), float("-inf"), device=self.dev
        )
        fn(
            q.unsqueeze(1),
            self.kv,
            w,
            out,
            seq,
            pt,
            out.shape[1],
            Preshuffle=True,
            KVBlockSize=PAGE,
        )
        torch.cuda.synchronize()
        return out

    def _assert_matches_aiter(self, lens, rows_per_request=6, share=True, width=4096):
        q, w, seq, pt = self._case(lens, rows_per_request, share, width)
        self.assertTrue(
            self.mod.supported(q, self.kv, preshuffle=True, kv_block_size=PAGE)
        )
        ref = self._logits(self.ref_fn, q, w, seq, pt)
        got = self._logits(self.mod.gluon_fp8_paged_mqa_logits, q, w, seq, pt)
        for r in range(q.shape[0]):
            n = int(seq[r])
            torch.testing.assert_close(
                got[r, :n], ref[r, :n], rtol=1e-5, atol=1e-5, msg=f"row {r}"
            )
            if n > TOPK:
                # Logits differ only by rounding, so the top-k selections may only
                # differ among elements within that rounding of the k-th score.
                eps = 1e-4
                kth = torch.topk(ref[r, :n], TOPK).values[-1]
                sel = torch.zeros(n, dtype=torch.bool, device=self.dev)
                sel[torch.topk(got[r, :n], TOPK).indices] = True
                self.assertTrue(
                    bool(sel[ref[r, :n] > kth + eps].all()),
                    f"row {r}: a clear top-k member was not selected",
                )
                self.assertTrue(
                    bool((ref[r, :n][sel] >= kth - eps).all()),
                    f"row {r}: selected an element clearly below the k-th",
                )

    def test_verify_rows_shared_tables(self):
        for lens in (
            [100_000] * 8,
            [3000, 50_000, 120_000, 200_000, 9000, 60_000, 130_000, 70_000],
            [250_000] * 14,
            [40_000] * 21,
        ):
            with self.subTest(n=len(lens), lens=lens[:3]):
                self._assert_matches_aiter(lens)

    def test_rows_without_shared_tables(self):
        """Paired rows whose page tables differ: the fused pair kernel must fall
        back to scoring them separately."""
        self._assert_matches_aiter([60_000] * 8, share=False)

    def test_plain_decode_rows(self):
        for n in (1, 7, 48, 100, 128):
            with self.subTest(rows=n):
                self._assert_matches_aiter(
                    [30_000 + 997 * i for i in range(n)], rows_per_request=1
                )

    def test_supported_bounds(self):
        q = torch.empty(129, HEADS, DIM, dtype=FP8, device=self.dev)
        self.assertFalse(
            self.mod.supported(q, self.kv, preshuffle=True, kv_block_size=PAGE)
        )
        self.assertFalse(
            self.mod.supported(q[:8], self.kv, preshuffle=False, kv_block_size=PAGE)
        )


if __name__ == "__main__":
    unittest.main()
