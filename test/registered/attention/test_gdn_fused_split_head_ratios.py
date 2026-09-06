import unittest

import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=3, stage="base-b", runner_config="1-gpu-large")


def _reference_split(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
    """Plain-slicing reference for the contiguous [Q|K|V|Z] / [B|A] layouts."""
    batch = mixed_qkvz.shape[0]
    total_q = num_heads_qk * head_qk
    total_v = num_heads_v * head_v
    q = mixed_qkvz[:, :total_q]
    k = mixed_qkvz[:, total_q : 2 * total_q]
    v = mixed_qkvz[:, 2 * total_q : 2 * total_q + total_v]
    z = mixed_qkvz[:, 2 * total_q + total_v :]
    mixed_qkv = torch.cat((q, k, v), dim=-1).contiguous()
    b = mixed_ba[:, :num_heads_v].contiguous()
    a = mixed_ba[:, num_heads_v:].contiguous()
    return (
        mixed_qkv,
        z.reshape(batch, num_heads_v, head_v).contiguous(),
        b,
        a,
    )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGdnFusedSplitHeadRatios(unittest.TestCase):
    """The fused contiguous split must be exact for every supported v/k head
    ratio, including the non-power-of-two ratio 3 of the dense 27B hybrids
    (served by the per-head walk instead of one wide vector access)."""

    HEAD_QK = 128
    HEAD_V = 128
    NUM_HEADS_QK = 16

    def _run_ratio(self, ratio: int, batch: int = 33) -> None:
        torch.manual_seed(ratio)
        num_heads_v = self.NUM_HEADS_QK * ratio
        total_qkvz = (
            2 * self.NUM_HEADS_QK * self.HEAD_QK + 2 * num_heads_v * self.HEAD_V
        )
        mixed_qkvz = torch.randn(batch, total_qkvz, dtype=torch.bfloat16, device="cuda")
        mixed_ba = torch.randn(
            batch, 2 * num_heads_v, dtype=torch.bfloat16, device="cuda"
        )

        got_qkv, got_z, got_b, got_a = fused_qkvzba_split_reshape_cat_contiguous(
            mixed_qkvz,
            mixed_ba,
            self.NUM_HEADS_QK,
            num_heads_v,
            self.HEAD_QK,
            self.HEAD_V,
        )
        ref_qkv, ref_z, ref_b, ref_a = _reference_split(
            mixed_qkvz,
            mixed_ba,
            self.NUM_HEADS_QK,
            num_heads_v,
            self.HEAD_QK,
            self.HEAD_V,
        )

        # A pure data-movement kernel must be bitwise exact.
        torch.testing.assert_close(got_qkv.view(-1), ref_qkv.view(-1), rtol=0, atol=0)
        torch.testing.assert_close(got_z.reshape(-1), ref_z.reshape(-1), rtol=0, atol=0)
        torch.testing.assert_close(got_b.view(-1), ref_b.view(-1), rtol=0, atol=0)
        torch.testing.assert_close(got_a.view(-1), ref_a.view(-1), rtol=0, atol=0)

    def test_ratio_1(self):
        self._run_ratio(1)

    def test_ratio_2(self):
        self._run_ratio(2)

    def test_ratio_3(self):
        self._run_ratio(3)

    def test_ratio_4(self):
        self._run_ratio(4)


if __name__ == "__main__":
    unittest.main()
