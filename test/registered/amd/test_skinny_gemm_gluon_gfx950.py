"""Numerical checks for the gfx950 skinny bf16 GEMM against torch.mm.

The shapes are the GEMMs a GLM-5.2 decode step issues -- 3072x6144, 6144x1536,
3584x512 -- at the row counts speculative decode brings: one row while the draft
model proposes, six times the batch while the target verifies.

The split_k > 1 cases are the ones that exercise what this kernel adds over
`gemm.tiny_gemm`, which splits N only. The cache-policy and XCD-banding cases
cover the two axes that exist here because Gluon can express them and Triton
cannot; both must leave the result unchanged, since they move where bytes are
fetched from and not what is computed.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci


def _skip_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    if not torch.version.hip:
        return "ROCm-only kernel"
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0]
    if arch != "gfx950":
        return f"kernel is gfx950-only, got {arch}"
    try:
        from triton.experimental import gluon  # noqa: F401
    except ImportError as err:
        return f"Gluon unavailable: {err}"
    return None


def _snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.float()
    err = got.float() - ref
    return 10.0 * torch.log10(
        ref.square().mean() / err.square().mean().clamp_min(1e-30)
    ).item()


register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class TestSkinnyGemmGluonGfx950(unittest.TestCase):
    def _run(self, m, n, k, *, bias=False, **config):
        from sglang.kernels.ops.gemm import skinny_gemm_bf16

        torch.manual_seed(0)
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        c = torch.randn(n, dtype=torch.bfloat16, device="cuda") if bias else None
        got = skinny_gemm_bf16(a, b, c, **config)
        ref = a.float() @ b.float().t()
        if c is not None:
            ref = ref + c.float()
        self.assertEqual(tuple(got.shape), (m, n))
        self.assertEqual(got.dtype, torch.bfloat16)
        # bf16 accumulation over k is the floor here, not the kernel.
        self.assertGreater(_snr_db(ref, got), 35.0)

    def test_tuned_shapes(self):
        """Every shape in the tuned table, through the entry point, no overrides.

        This is the path a decode step takes, so it is the one that has to be
        right: a row whose configuration does not run leaves the caller on a
        fallback it was never measured against.
        """
        from sglang.kernels.ops.gemm.skinny_gemm_gluon import _TUNED

        for m, n, k in _TUNED:
            with self.subTest(m=m, n=n, k=k):
                self._run(m, n, k)

    def test_split_that_overshoots_k(self):
        """A split count whose rounded slice reaches past K.

        k = 6144 with split_k = 8 and block_k = 512 rounds each slice up to
        1024, so eight of them span 8192 and the last blocks start past the
        end. They must do nothing rather than read there: EVEN_K holds for this
        shape, so the bounds masks are gone and an unguarded block faults.
        """
        self._run(6, 3072, 6144, block_n=32, block_k=512, split_k=8)
        self._run(48, 3072, 6144, block_n=32, block_k=512, split_k=8,
                  reduce="atomic")

    def test_untabulated_shape_falls_back(self):
        """A shape with no tuned row still computes the right answer."""
        self._run(12, 2048, 4096)

    def test_split_k_reduction(self):
        """Splitting k is what this kernel has over gemm.tiny_gemm."""
        for split_k in (1, 2, 4, 8):
            with self.subTest(split_k=split_k):
                self._run(6, 3072, 6144, block_n=64, block_k=128, split_k=split_k)

    def test_fetch_axes_do_not_change_the_result(self):
        """Cache policy and XCD banding move bytes, not arithmetic."""
        for b_cpol in ("", ".cg"):
            for xcd_band in (1, 2, 4):
                with self.subTest(b_cpol=b_cpol, xcd_band=xcd_band):
                    self._run(6, 3072, 6144, block_n=64, block_k=128, split_k=4,
                              b_cpol=b_cpol, xcd_band=xcd_band)

    def test_k_rot_does_not_change_the_result(self):
        """Every block still visits every k tile, only in a rotated order."""
        for k_rot in (0, 1, 3):
            with self.subTest(k_rot=k_rot):
                self._run(6, 3072, 6144, block_n=64, block_k=128, split_k=4,
                          k_rot=k_rot)

    def test_bias(self):
        self._run(6, 3584, 512, block_n=32, block_k=128, split_k=1, bias=True)


if __name__ == "__main__":
    unittest.main()
