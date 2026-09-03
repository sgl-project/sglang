"""The fused fp8 q prep must be bit-exact against the concat + cast it replaces."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# The kernel is device-agnostic Triton; only the gate that reaches it is HIP-only,
# so both targets are worth running.
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="jit-kernel-unit-test-amd")

HEADS, NOPE, ROPE, FP8 = 16, 512, 64, torch.float8_e4m3fn


def _pair(tokens, heads=HEADS, seed=0):
    torch.manual_seed(seed)
    # e4m3 saturates near 448; neither path takes a scale, so keep the fixture
    # inside the range both of them assume.
    return (
        (torch.randn(tokens, heads, NOPE, device="cuda") * 8).bfloat16(),
        (torch.randn(tokens, heads, ROPE, device="cuda") * 8).bfloat16(),
    )


def _bytes_differ(a, b):
    return (a.view(torch.uint8) != b.view(torch.uint8)).sum().item()


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestHipFusedQPrep(CustomTestCase):
    def _backend(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        return DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)

    def _kv(self, dtype=FP8):
        return torch.empty((8, 1, NOPE + ROPE), dtype=dtype, device="cuda")

    def test_bit_exact_against_concat_then_cast(self):
        from sglang.kernels.ops.kvcache.cache_ops import concat_and_cast_q_fp8_pad
        from sglang.srt.layers.attention.dsa_backend import (
            concat_mla_absorb_q_general,
        )

        for tokens in (1, 6, 48, 512, 4096):
            with self.subTest(tokens=tokens):
                q_nope, q_rope = _pair(tokens, seed=tokens)
                want = concat_mla_absorb_q_general(q_nope, q_rope).to(FP8)
                got = torch.empty(
                    (tokens, HEADS, NOPE + ROPE), dtype=FP8, device="cuda"
                )
                concat_and_cast_q_fp8_pad(got, q_nope, q_rope, HEADS)
                self.assertEqual(_bytes_differ(got, want), 0)

    def test_halves_are_not_swapped(self):
        # A concat that swapped nope and rope would still be bit-exact against
        # itself, so pin the halves against their sources.
        from sglang.kernels.ops.kvcache.cache_ops import concat_and_cast_q_fp8_pad

        q_nope, q_rope = _pair(6, seed=1)
        got = torch.empty((6, HEADS, NOPE + ROPE), dtype=FP8, device="cuda")
        concat_and_cast_q_fp8_pad(got, q_nope, q_rope, HEADS)
        self.assertEqual(_bytes_differ(got[:, :, :NOPE], q_nope.to(FP8)), 0)
        self.assertEqual(_bytes_differ(got[:, :, NOPE:], q_rope.to(FP8)), 0)

    def test_cuda_is_untouched(self):
        # The fused path is gated on _is_hip. With it off, _sparse_q_all must
        # be the concat it replaced -- same call, same bytes, whatever q_all
        # already held.
        from unittest.mock import patch

        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.layers.attention.dsa_backend import (
            concat_mla_absorb_q_general,
        )

        b, kv = self._backend(), self._kv()
        q_nope, q_rope = _pair(6, seed=7)
        want = concat_mla_absorb_q_general(q_nope, q_rope)
        with patch.object(dsa_backend, "_is_hip", False):
            for prior in (None, torch.zeros_like(want)):
                got = b._sparse_q_all(prior, q_nope, q_rope, kv)
                self.assertEqual(got.dtype, want.dtype)
                self.assertEqual(_bytes_differ(got, want), 0)

    def test_falls_back_rather_than_guessing(self):
        b = self._backend()
        q_nope, q_rope = _pair(6, seed=5)
        for name, args in (
            ("bf16 kv", (q_nope, q_rope, self._kv(torch.bfloat16))),
            ("fp32 q", (q_nope.float(), q_rope.float(), self._kv())),
            ("12 heads", (*_pair(6, heads=12, seed=6), self._kv())),
        ):
            with self.subTest(case=name):
                out = b._sparse_q_all(None, *args)
                self.assertNotEqual(out.dtype, FP8, "should have kept the bf16 path")


if __name__ == "__main__":
    unittest.main()
