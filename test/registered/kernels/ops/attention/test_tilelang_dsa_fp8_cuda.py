"""CUDA raw-FP8 DSA: NoPE/RoPE, TP head counts, padded KPool tails and masks.

References use the same quantized inputs as the kernel, isolating kernel
correctness from model-level quantization accuracy. A wrong-index control
must fail the comparator rather than merely executing successfully.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reference(q, kv, indices, scale, d_v):
    rows = indices[:, 0].long()
    valid = rows >= 0
    selected = kv.float()[rows.clamp_min(0), 0]
    scores = torch.einsum("thd,tkd->thk", q.float(), selected) * scale
    scores.masked_fill_(~valid[:, None], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("thk,tkd->thd", probabilities, selected[..., :d_v])


@unittest.skipUnless(
    torch.cuda.is_available()
    and not is_hip()
    and torch.cuda.get_device_capability() >= (8, 9),
    "requires CUDA FP8 Tensor Cores",
)
class TestTileLangDsaFp8Cuda(CustomTestCase):
    def _inputs(self, heads, d_v, d_tail, topk=2112):
        torch.manual_seed(17)
        tokens, pool_size = 5, 4096
        dim = d_v + d_tail
        q = (torch.randn(tokens, heads, dim, device="cuda") * 0.5).bfloat16()
        kv = (torch.randn(pool_size, 1, dim, device="cuda") * 0.5).to(
            torch.float8_e4m3fn
        )
        indices = torch.randint(
            1, pool_size, (tokens, 1, topk), device="cuda", dtype=torch.int32
        )
        indices[..., -61:] = -1
        return q, kv, indices, dim**-0.5

    def test_one_hot_and_all_masked(self):
        for heads, d_v, d_tail in ((8, 512, 0), (16, 256, 0), (32, 512, 64)):
            with self.subTest(heads=heads, d_v=d_v, d_tail=d_tail):
                q, kv, indices, scale = self._inputs(heads, d_v, d_tail)
                indices.fill_(-1)
                selected = torch.tensor([1, 5, 17, 31, 63], device="cuda")
                indices[:, 0, 0] = selected.int()
                actual = tilelang_sparse_fwd(q, kv, indices, scale, d_v).squeeze(0)
                expected = kv.float()[selected, 0, :d_v][:, None].expand_as(actual)
                torch.testing.assert_close(
                    actual.float(), expected, atol=1e-3, rtol=1e-3
                )

                indices.fill_(-1)
                empty = tilelang_sparse_fwd(q, kv, indices, scale, d_v)
                self.assertTrue(torch.isfinite(empty).all().item())
                self.assertEqual(torch.count_nonzero(empty).item(), 0)

    def test_spread_and_wrong_index_negative_control(self):
        q, kv, indices, scale = self._inputs(8, 512, 0)
        actual = tilelang_sparse_fwd(q, kv, indices, scale, 512).squeeze(0).float()
        expected = _reference(q.to(kv.dtype), kv, indices, scale, 512)
        error = (actual - expected).abs().max() / expected.abs().max()
        self.assertLess(error.item(), 0.04)

        wrong = torch.where(indices >= 0, (indices + 2048) % 4096, indices)
        incorrect = tilelang_sparse_fwd(q, kv, wrong, scale, 512).squeeze(0).float()
        wrong_error = (incorrect - expected).abs().max() / expected.abs().max()
        self.assertGreater(wrong_error.item(), 0.04)

    def test_scaled_packed_rows_rejected(self):
        q, _, indices, scale = self._inputs(8, 512, 0)
        scaled_rows = torch.zeros(
            4096, 1, 528, device="cuda", dtype=torch.float8_e4m3fn
        )
        with self.assertRaisesRegex(ValueError, "raw KV rows"):
            tilelang_sparse_fwd(q, scaled_rows, indices, scale, 512)

    def test_rocm_fp8_format_rejected_on_cuda(self):
        q, _, indices, scale = self._inputs(8, 512, 0)
        kv = torch.empty(4096, 1, 512, device="cuda", dtype=torch.float8_e4m3fnuz)
        with self.assertRaisesRegex(ValueError, "requires float8_e4m3fn"):
            tilelang_sparse_fwd(q, kv, indices, scale, 512)


if __name__ == "__main__":
    unittest.main()
