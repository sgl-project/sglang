"""Triton sparse-MLA decode, the kernel behind --dsa-decode-backend triton.

Three things here are not covered by eyeballing a benchmark, and each of them
was a real bug at some point in this kernel's history:

The query layout. dsa_backend hands the kernel two strided halves of one
[seq, heads, 576] tensor, because the gfx95 fused rope+cache path never
materialises the halves separately. The kernel reads them in place via a head
pitch rather than copying, so the strided and contiguous forms have to agree
bit for bit -- if the pitch were wrong the result would still look plausible,
just built from the wrong rows.

Ragged key groups. block_n is rounded down to a power of two, so a topk that
does not divide evenly leaves a short trailing tile. Nothing stops that tile
from reading into the next group's keys and counting them twice except its
group-boundary mask.

Index hygiene. The host-side clamp that used to sanitise the index tensor is
gone -- it cost a 1.57 us elementwise launch per call to do work the key loop
does for free -- so -1 padding and out-of-range page ids now have to be
rejected inside the loop.
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


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(),
    "Triton sparse MLA decode is gated to gfx950.",
)
class TestTritonSparseMLADecode(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

        self.device = torch.device("cuda")
        self.fp8_dtype = (
            torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
        )
        self.sm_scale = 1.0 / (DIM**0.5)
        self.gen = torch.Generator(device=self.device).manual_seed(0)

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def _fwd(self, q_nope, q_rope, kv, indices):
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
            triton_sparse_mla_decode_splitk_fwd,
        )

        return triton_sparse_mla_decode_splitk_fwd(
            q_nope, q_rope, kv, indices, self.sm_scale, D_V
        ).squeeze(0)

    def _inputs(self, seq, heads, topk, n_pages=8192):
        """q as the server holds it: one [seq, heads, 576] fp8 tensor."""
        raw = torch.randn(seq, heads, DIM, device=self.device, generator=self.gen)
        q_all = (raw * 0.3).to(self.fp8_dtype)
        kv = (
            torch.randn(n_pages, 1, DIM, device=self.device, generator=self.gen) * 0.3
        ).to(self.fp8_dtype)
        indices = torch.randint(
            0,
            n_pages,
            (seq, 1, topk),
            device=self.device,
            generator=self.gen,
            dtype=torch.int32,
        )
        return q_all, kv, indices

    def _reference(self, q_all, kv, indices):
        """Dense fp32 attention over each row's valid keys."""
        seq, _, _ = q_all.shape
        n_pages = kv.shape[0]
        out = torch.zeros(seq, q_all.shape[1], D_V, device=self.device)
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

    def _snr(self, want, got):
        sig = want.pow(2).mean().sqrt()
        noise = (got.float() - want).pow(2).mean().sqrt().clamp(min=1e-12)
        return float(20 * torch.log10(sig.clamp(min=1e-12) / noise))

    def _check(self, q_all, kv, indices, floor=30.0):
        got = self._fwd(q_all[..., :D_V], q_all[..., D_V:], kv, indices)
        self.assertFalse(torch.isnan(got).any(), "kernel produced NaN")
        self.assertFalse(torch.isinf(got).any(), "kernel produced inf")
        snr = self._snr(self._reference(q_all, kv, indices), got)
        self.assertGreater(snr, floor, f"SNR {snr:.1f} dB against fp32 reference")
        return snr

    def test_mtp_token_counts(self):
        """The token counts EAGLE actually produces, at both TP head counts.

        5 steps and 6 draft tokens means C drafted and 6*C verified, over the
        concurrencies the serving benchmark sweeps; heads is 16 at TP4 and 8 at
        TP8. The tuning helpers pick a different (block_n, warps, groups,
        stages) tuple across this range, so this is really a sweep over
        generated code, not over shapes -- three of those tuples used to
        miscompile and still return plausible-looking output.
        """
        for heads in (8, 16):
            for seq in (1, 2, 4, 6, 8, 12, 24, 48, 60, 84):
                with self.subTest(heads=heads, seq=seq):
                    q_all, kv, indices = self._inputs(seq, heads, 2048)
                    self._check(q_all, kv, indices)

    def test_strided_query_matches_contiguous(self):
        """The server's strided halves and copied halves must agree exactly.

        Not an SNR check: same inputs, same code path, so anything other than
        bit equality means the pitch walked to the wrong rows.
        """
        for heads in (8, 16):
            for seq in (1, 12, 84):
                with self.subTest(heads=heads, seq=seq):
                    q_all, kv, indices = self._inputs(seq, heads, 2048)
                    strided = self._fwd(
                        q_all[..., :D_V], q_all[..., D_V:], kv, indices
                    )
                    copied = self._fwd(
                        q_all[..., :D_V].contiguous(),
                        q_all[..., D_V:].contiguous(),
                        kv,
                        indices,
                    )
                    torch.testing.assert_close(copied, strided, rtol=0, atol=0)

    def test_ragged_key_groups(self):
        """topk values that leave a short trailing tile in each key group.

        A tile that overruns its group reads keys that are real and in range,
        so the only symptom is that they get counted twice -- which lands
        around 16 dB, not as a NaN.
        """
        for topk in (64, 128, 192, 256, 320, 512, 768, 1024, 1536):
            with self.subTest(topk=topk):
                q_all, kv, indices = self._inputs(6, 16, topk)
                self._check(q_all, kv, indices)

    def test_padded_and_out_of_range_indices(self):
        """-1 padding and page ids past the pool, both rejected in the loop."""
        q_all, kv, indices = self._inputs(12, 16, 2048)
        n_pages = kv.shape[0]
        holes = torch.rand(indices.shape, device=self.device, generator=self.gen)
        indices[holes < 0.2] = -1
        indices[holes > 0.95] = n_pages + 7
        self._check(q_all, kv, indices)

    def test_fully_padded_row_is_zero(self):
        """Every score -inf: the denominator is zero and must not divide."""
        q_all, kv, indices = self._inputs(8, 16, 2048)
        indices[3] = -1
        got = self._fwd(q_all[..., :D_V], q_all[..., D_V:], kv, indices)
        self.assertFalse(torch.isnan(got).any(), "all-padded row gave NaN")
        self.assertFalse(torch.isinf(got).any(), "all-padded row gave inf")
        torch.testing.assert_close(
            got[3].float(), torch.zeros_like(got[3].float()), rtol=0, atol=0
        )

    def test_matches_tilelang(self):
        """Against the backend it replaces, on the same inputs.

        The fp32 reference already covers correctness; this covers the two
        kernels being interchangeable behind --dsa-decode-backend, which is
        the claim the flag makes.
        """
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            tilelang_sparse_fwd,
        )

        for heads in (8, 16):
            for seq in (1, 12, 84):
                with self.subTest(heads=heads, seq=seq):
                    q_all, kv, indices = self._inputs(seq, heads, 2048)
                    mine = self._fwd(q_all[..., :D_V], q_all[..., D_V:], kv, indices)
                    theirs = tilelang_sparse_fwd(
                        q_all, kv, indices, self.sm_scale, D_V
                    ).squeeze(0)
                    snr = self._snr(theirs.float(), mine)
                    self.assertGreater(snr, 30.0, f"SNR {snr:.1f} dB against TileLang")


if __name__ == "__main__":
    unittest.main()
