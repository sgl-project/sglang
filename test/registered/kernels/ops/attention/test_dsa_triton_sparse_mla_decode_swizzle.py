"""XCD swizzle in the Triton sparse-MLA decode kernel must cover every program."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="nightly-amd-kernel-1-gpu")

D_V = 512
D_TAIL = 64
DIM = D_V + D_TAIL
TOPK = 2048


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(),
    "Triton sparse MLA decode is gated to gfx950.",
)
class TestTritonSparseMLADecodeSwizzle(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

        self.device = torch.device("cuda")
        self.fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
        self.sm_scale = 1.0 / (DIM**0.5)

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def _inputs(self, seq: int, heads: int):
        gen = torch.Generator(device=self.device).manual_seed(seq * 97 + heads)

        def fp8(*shape):
            raw = torch.randn(*shape, device=self.device, generator=gen) * 0.3
            return raw.to(self.fp8_dtype)

        q_all = fp8(seq, heads, DIM)
        kv = fp8(8192, 1, DIM)
        indices = torch.randint(
            0,
            8192,
            (seq, 1, TOPK),
            device=self.device,
            dtype=torch.int32,
            generator=gen,
        )
        return q_all, kv, indices

    def _run(self, q_all, kv, indices, n_xcd: int) -> torch.Tensor:
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
            n_xcd=n_xcd,
        ).squeeze(0)
        torch.cuda.synchronize()
        return out

    def _check(self, seq: int, heads: int):
        q_all, kv, indices = self._inputs(seq, heads)
        swizzled = self._run(q_all, kv, indices, n_xcd=8)
        reference = self._run(q_all, kv, indices, n_xcd=1)

        self.assertFalse(torch.isnan(swizzled).any(), f"seq={seq} heads={heads}: NaN")
        torch.testing.assert_close(swizzled, reference, rtol=0, atol=0)

    def test_mtp_shapes_not_divisible_by_xcd(self):
        for seq in (1, 6, 12, 24, 84):
            with self.subTest(seq=seq):
                self._check(seq, heads=16)

    def test_eight_heads(self):
        for seq in (2, 48):
            with self.subTest(seq=seq):
                self._check(seq, heads=8)


if __name__ == "__main__":
    unittest.main()
