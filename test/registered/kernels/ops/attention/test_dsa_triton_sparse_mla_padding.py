"""Padded top-k rows must still give the attention of their valid keys only.

The inner loop no longer masks its KV loads. An invalid slot has its page id
clamped to 0, so the load is in bounds and returns real fp8, and correctness
rests entirely on the score for that column being forced to -inf afterwards:
the softmax then gives it weight zero and whatever was read multiplies out of
the PV dot. Masking the load instead cost a v_cndmask per element of a
[BLOCK_N, D_V] tile and was the largest single term in the loop.

Padding is what exercises that path, and the benchmark generators never
produce any -- they fill every row to topk. Prefill does produce it, on every
token whose context is shorter than topk, which at topk=2048 is the first
2048 positions of a sequence. A row of all -1 is the sharp case: every score
is -inf, the softmax denominator is zero, and the output has to be a finite
zero rather than a NaN.
"""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="nightly-amd-kernel-1-gpu")

D_V = 512
D_TAIL = 64
DIM = D_V + D_TAIL


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(),
    "Triton sparse MLA prefill is gated to gfx950.",
)
class TestTritonSparseMLAPadding(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

        self.device = torch.device("cuda")
        self.fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
        self.sm_scale = 1.0 / (DIM**0.5)
        self.gen = torch.Generator(device=self.device).manual_seed(0)

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def _inputs(self, seq, heads, n_pages):
        def rnd(*shape):
            raw = torch.randn(*shape, device=self.device, generator=self.gen)
            return (raw * 0.3).to(self.fp8_dtype)

        return rnd(seq, heads, D_V), rnd(seq, heads, D_TAIL), rnd(n_pages, DIM)

    def _reference(self, q_nope, q_rope, kv, indices):
        """Attention over each row's valid keys, in fp32.

        Deliberately not the same kernel with masking restored: that would pass
        even if both paths dropped the same keys.
        """
        seq, heads, _ = q_nope.shape
        out = torch.zeros(seq, heads, D_V, device=self.device, dtype=torch.float32)
        q = torch.cat([q_nope.float(), q_rope.float()], dim=-1)
        kvf = kv.float()
        for s in range(seq):
            pages = indices[s, 0]
            pages = pages[pages >= 0]
            if pages.numel() == 0:
                continue
            k = kvf[pages.long()]
            scores = (q[s] @ k.T) * self.sm_scale
            p = torch.softmax(scores, dim=-1)
            out[s] = p @ k[:, :D_V]
        return out

    def _check(self, indices, seq=96, heads=8, n_pages=2048):
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            triton_sparse_mla_prefill_fwd,
        )

        q_nope, q_rope, kv = self._inputs(seq, heads, n_pages)
        got = triton_sparse_mla_prefill_fwd(
            q_nope, q_rope, kv, indices, self.sm_scale, D_V
        ).squeeze(0)
        self.assertFalse(torch.isnan(got).any(), "kernel produced NaN")
        self.assertFalse(torch.isinf(got).any(), "kernel produced inf")

        want = self._reference(q_nope, q_rope, kv, indices)
        sig = want.pow(2).mean().sqrt()
        noise = (got.float() - want).pow(2).mean().sqrt().clamp(min=1e-12)
        snr = float(20 * torch.log10(sig.clamp(min=1e-12) / noise))
        # fp8 operands and an fp8 round trip on the probabilities put the floor
        # well below what a dropped or extra key would survive.
        self.assertGreater(snr, 30.0, f"SNR {snr:.1f} dB against fp32 reference")
        return snr

    def _empty(self, seq, topk):
        return torch.full(
            (seq, 1, topk), -1, device=self.device, dtype=torch.int32
        )

    def test_causal_prefix_padding(self):
        """Row s keeps min(s+1, topk) keys, the shape prefill actually makes."""
        seq, topk, n_pages = 96, 128, 2048
        idx = self._empty(seq, topk)
        for s in range(seq):
            n = min(s + 1, topk)
            idx[s, 0, :n] = torch.randperm(
                n_pages, device=self.device, generator=self.gen
            )[:n].to(torch.int32)
        self._check(idx, seq=seq)

    def test_all_rows_fully_padded(self):
        """Every score is -inf; the denominator is zero and must not divide."""
        seq, topk = 32, 128
        got_snr_free = self._empty(seq, topk)
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            triton_sparse_mla_prefill_fwd,
        )

        q_nope, q_rope, kv = self._inputs(seq, 8, 2048)
        got = triton_sparse_mla_prefill_fwd(
            q_nope, q_rope, kv, got_snr_free, self.sm_scale, D_V
        ).squeeze(0)
        self.assertFalse(torch.isnan(got).any(), "all-padded row gave NaN")
        self.assertFalse(torch.isinf(got).any(), "all-padded row gave inf")
        torch.testing.assert_close(
            got.float(), torch.zeros_like(got.float()), rtol=0, atol=0
        )

    def test_padding_interleaved_not_only_trailing(self):
        """-1 in the middle of a row, not just past its end.

        A tail-only layout would also pass if the kernel merely stopped early,
        so scatter the holes through the row.
        """
        seq, topk, n_pages = 64, 256, 2048
        idx = torch.randint(
            0, n_pages, (seq, 1, topk), device=self.device,
            generator=self.gen, dtype=torch.int32,
        )
        holes = (
            torch.rand(seq, 1, topk, device=self.device, generator=self.gen) < 0.4
        )
        idx[holes] = -1
        self._check(idx, seq=seq, n_pages=n_pages)

    def test_single_valid_key_per_row(self):
        """One key means the softmax is exactly 1 on it, so the output is that
        key -- an exact check that the surviving column is the right one."""
        seq, topk, n_pages = 48, 128, 2048
        idx = self._empty(seq, topk)
        slot = torch.randint(
            0, topk, (seq,), device=self.device, generator=self.gen
        )
        page = torch.randint(
            0, n_pages, (seq,), device=self.device, generator=self.gen
        )
        idx[torch.arange(seq, device=self.device), 0, slot] = page.to(torch.int32)

        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            triton_sparse_mla_prefill_fwd,
        )

        q_nope, q_rope, kv = self._inputs(seq, 8, n_pages)
        got = triton_sparse_mla_prefill_fwd(
            q_nope, q_rope, kv, idx, self.sm_scale, D_V
        ).squeeze(0)
        want = kv.float()[page.long()][:, :D_V]
        for h in range(got.shape[1]):
            torch.testing.assert_close(
                got[:, h].float(), want, rtol=2e-2, atol=2e-2
            )


if __name__ == "__main__":
    unittest.main()
