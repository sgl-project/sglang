"""Merging the DSA indexer's two projections into one GEMM, on ROCm.

``wk`` (N = index_head_dim) and ``weights_proj`` (N = index_n_heads) read the
same hidden state and are issued back to back. At decode row counts they are
almost entirely launch overhead, so ``SGLANG_ROCM_FUSE_INDEXER_GEMM`` runs them
as one N = head_dim + n_heads call and splits the result.

The merge is not bit-identical: the tuned GEMM table picks a different kernel
for the wider N, so the K reduction associates differently. What has to hold is
that the wider call is not the less accurate of the two -- it is measurably the
more accurate one on the key half -- and that the split lands on the right
columns.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

K = 6144  # GLM-5.2 hidden size
HEAD_DIM = 128  # index_head_dim
N_HEADS = 32  # index_n_heads
# Draft rows = concurrency, verify rows = 6 * concurrency; C=4 gives 4 and 24.
ROWS = [1, 4, 6, 8, 24, 48]


def _gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    return (
        str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"
    )


def _snr_db(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.float(), ref.float()
    return float(10 * torch.log10(ref.pow(2).sum() / (got - ref).pow(2).sum()))


@unittest.skipUnless(_gfx950(), "gfx950 only")
class TestIndexerGemmMerge(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from aiter.tuned_gemm import tgemm

        cls.mm = staticmethod(lambda a, b: tgemm.mm(a, b, None, otype=torch.bfloat16))

    def _case(self, m):
        torch.manual_seed(m)
        x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
        wk = torch.randn(HEAD_DIM, K, dtype=torch.bfloat16, device="cuda")
        wg = torch.randn(N_HEADS, K, dtype=torch.bfloat16, device="cuda")
        merged = torch.cat([wk, wg], dim=0).contiguous()
        return x, wk, wg, merged

    def test_split_lands_on_the_right_columns(self):
        # The head-gate half is the one the fused CUDA path also relies on being
        # the bottom n_heads rows; a transposed cat would still have the right
        # shapes and would only show up as garbage gates.
        for m in ROWS:
            with self.subTest(m=m):
                x, wk, wg, merged = self._case(m)
                key, gate = self.mm(x, merged).split([HEAD_DIM, N_HEADS], dim=-1)
                self.assertEqual(key.shape, (m, HEAD_DIM))
                self.assertEqual(gate.shape, (m, N_HEADS))
                ref_k = x.float() @ wk.float().t()
                ref_g = x.float() @ wg.float().t()
                # 30 dB is far above any rounding difference and far below what
                # a mis-split would produce.
                self.assertGreater(_snr_db(key, ref_k), 30.0)
                self.assertGreater(_snr_db(gate, ref_g), 30.0)

    def test_merged_key_is_not_less_accurate(self):
        for m in ROWS:
            with self.subTest(m=m):
                x, wk, wg, merged = self._case(m)
                ref_k = x.float() @ wk.float().t()
                split_db = _snr_db(self.mm(x, wk), ref_k)
                merged_db = _snr_db(
                    self.mm(x, merged).split([HEAD_DIM, N_HEADS], dim=-1)[0], ref_k
                )
                # Measured 9 dB better at every row count above 1; require only
                # that it does not regress, with 1 dB of slack.
                self.assertGreater(merged_db, split_db - 1.0)

    def test_head_gates_are_not_less_accurate(self):
        # Mostly bitwise equal to the separate call -- the bottom rows are
        # narrow enough that the wider call usually keeps the same kernel for
        # them -- but not always (m=8 and m=16 differ), so the contract is the
        # same one the key half gets: no accuracy regression.
        for m in ROWS:
            with self.subTest(m=m):
                x, wk, wg, merged = self._case(m)
                ref_g = x.float() @ wg.float().t()
                sep_db = _snr_db(self.mm(x, wg), ref_g)
                got_db = _snr_db(
                    self.mm(x, merged).split([HEAD_DIM, N_HEADS], dim=-1)[1], ref_g
                )
                self.assertGreater(got_db, sep_db - 1.0)


if __name__ == "__main__":
    unittest.main()
