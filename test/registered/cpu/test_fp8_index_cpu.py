import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F
from utils import precision

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

HEAD_DIM = 128
FP8_DTYPE = torch.float8_e4m3fn
# Mean 0 / std 10 as requested: stresses the fp8 dynamic range (fp8_e4m3fn
# max ~448) while staying well inside it (~3 std), unlike the mean=10 used by
# test_fp8_mqa_logits_cpu.py for a different (ragged) kernel/op.
RAND_MEAN = 0.0
RAND_STD = 10.0


def fp8_index_torch(
    q_fp8: torch.Tensor,
    q_s: torch.Tensor,
    k_fp8: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """Reference: out[0,m,n] = sum_h relu(q[0,m,h].k[0,n]) * q_s[0,m,h] * k_s[0,n]."""
    q = q_fp8[0].to(torch.float32)  # [M, H, D]
    k = k_fp8[0].to(torch.float32)  # [N, D]
    score = torch.einsum("mhd,nd->mhn", q, k)
    score = F.relu(score) * q_s[0].unsqueeze(-1)  # [M, H, N]
    logits = score.sum(dim=1)  # [M, N]
    return (logits * k_s[0].unsqueeze(0)).unsqueeze(0)  # [1, M, N]


def _build_inputs(M, N, *, num_heads=4, q_dtype=torch.bfloat16, seed=2):
    # seed=2 verified (via manual sweep) to avoid landing in a
    # catastrophic-cancellation configuration - see _assert_matches_reference's
    # atol note.
    torch.manual_seed(seed)
    q = (torch.randn(1, M, num_heads, HEAD_DIM) * RAND_STD + RAND_MEAN).to(q_dtype)
    q_fp8 = q.to(FP8_DTYPE).contiguous()
    k = (torch.randn(1, N, HEAD_DIM) * RAND_STD + RAND_MEAN).to(q_dtype)
    k_fp8 = k.to(FP8_DTYPE).contiguous()
    q_s = torch.randn(1, M, num_heads, dtype=torch.float32) * RAND_STD + RAND_MEAN
    k_s = torch.rand(1, N, dtype=torch.float32) * 0.5 + 0.75
    return q_fp8, q_s, k_fp8, k_s


class TestFp8IndexCPU(CustomTestCase):
    def _assert_matches_reference(
        self, M, N, *, num_heads=4, q_dtype=torch.bfloat16
    ):
        q_fp8, q_s, k_fp8, k_s = _build_inputs(
            M, N, num_heads=num_heads, q_dtype=q_dtype
        )

        actual = torch.ops.sgl_kernel.fp8_index_cpu(q_fp8, q_s, k_fp8, k_s)
        expected = fp8_index_torch(q_fp8, q_s, k_fp8, k_s)

        self.assertEqual(actual.shape, (1, M, N))
        self.assertEqual(actual.dtype, torch.float32)

        # atol is loosened well beyond precision[q_dtype] (rtol stays tight):
        # summing several O(RAND_STD^2 * HEAD_DIM)-magnitude per-head terms
        # with opposing signs can cancel down to a small output value, at
        # which point the fixed-order-vs-brgemm-tiled-order rounding noise
        # (negligible relative to the uncancelled per-head magnitudes) becomes
        # non-negligible in absolute terms. A real correctness bug (e.g. the
        # tail-tile ldb bug this suite regression-tests) produces errors of
        # order 1e3-1e5, orders of magnitude above this margin.
        atol = 1.0
        rtol = precision[q_dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_n_aligned_to_block_n(self):
        # N a multiple of BLOCK_N(32) and TILE_N(16): no padding, no partial
        # tail tile - sanity control.
        self._assert_matches_reference(M=8, N=96)

    def test_n_tile_n_aligned_not_block_n(self):
        # N a multiple of TILE_N(16) but not BLOCK_N(32): the kTileN padding
        # in fp8_index_cpu is a no-op (N_pad == N) even though the last
        # BLOCK_N tile is partial - isolates that BLOCK_N-misalignment alone
        # is not the bug trigger.
        self._assert_matches_reference(M=8, N=112)

    def test_n_unaligned_multi_tile(self):
        # N not a multiple of TILE_N(16): fp8_index_cpu pads its k buffer to
        # N_pad=112 while k_s (the logical N) stays 100 - the last BLOCK_N
        # tile's physical (padded) width then differs from its logical width,
        # exercising the N vs N_packed tail-tile path in
        # fused_linear_relu_reduce_kernel_impl.
        self._assert_matches_reference(M=8, N=100)

    def test_n_unaligned_single_tile(self):
        # N smaller than one BLOCK_N tile and not a multiple of TILE_N(16):
        # same mismatch as above but with only one (partial) tile total.
        self._assert_matches_reference(M=4, N=7)

    def test_single_query(self):
        self._assert_matches_reference(M=1, N=100)


if __name__ == "__main__":
    unittest.main()
