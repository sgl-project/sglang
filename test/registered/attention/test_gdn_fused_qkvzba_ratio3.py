"""
Tests that fused_qkvzba_split_reshape_cat_contiguous matches the eager
split/reshape/cat reference for all supported num_v_heads/num_k_heads ratios,
including the non-power-of-2 ratio 3 used by Qwen3.5/3.6 dense 27B
(48 v-heads / 16 k-heads), which previously fell back to the eager path.
"""

import unittest

import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")


def _eager_reference(mixed_qkvz, mixed_ba, num_k_heads, num_v_heads, head_k, head_v):
    """Eager path from Qwen3_5GatedDeltaNet.fix_query_key_value_ordering +
    the reshape/cat that follows it in the model forward."""
    k_dim = num_k_heads * head_k
    v_dim = num_v_heads * head_v
    query, key, value, z = mixed_qkvz.split([k_dim, k_dim, v_dim, v_dim], dim=-1)
    b, a = mixed_ba.split([num_v_heads, num_v_heads], dim=-1)
    mixed_qkv = torch.cat(
        [
            query.reshape(query.size(0), -1),
            key.reshape(key.size(0), -1),
            value.reshape(value.size(0), -1),
        ],
        dim=-1,
    ).contiguous()
    z = z.reshape(z.size(0), num_v_heads, head_v).contiguous()
    return mixed_qkv, z, b.contiguous(), a.contiguous()


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFusedQkvzbaSplitContiguous(unittest.TestCase):
    def _run_case(self, num_k_heads, num_v_heads, head_k, head_v, seq_len):
        torch.manual_seed(0)
        qkvz_dim = 2 * num_k_heads * head_k + 2 * num_v_heads * head_v
        mixed_qkvz = torch.randn(seq_len, qkvz_dim, dtype=torch.bfloat16, device="cuda")
        mixed_ba = torch.randn(
            seq_len, 2 * num_v_heads, dtype=torch.bfloat16, device="cuda"
        )

        out = fused_qkvzba_split_reshape_cat_contiguous(
            mixed_qkvz, mixed_ba, num_k_heads, num_v_heads, head_k, head_v
        )
        ref = _eager_reference(
            mixed_qkvz, mixed_ba, num_k_heads, num_v_heads, head_k, head_v
        )
        for got, want, name in zip(out, ref, ["mixed_qkv", "z", "b", "a"]):
            self.assertTrue(
                torch.equal(got.view_as(want), want),
                f"{name} mismatch for nk={num_k_heads} nv={num_v_heads} "
                f"hk={head_k} hv={head_v} T={seq_len}",
            )

    def test_ratio3_qwen36_27b_shape(self):
        # Qwen3.5/3.6 dense 27B: 16 k-heads, 48 v-heads, head dim 128.
        for seq_len in [1, 7, 64, 8192]:
            self._run_case(16, 48, 128, 128, seq_len)

    def test_ratio2(self):
        for seq_len in [1, 7, 64, 8192]:
            self._run_case(8, 16, 128, 128, seq_len)
            self._run_case(4, 8, 64, 64, seq_len)

    def test_ratio1(self):
        for seq_len in [1, 7, 64, 8192]:
            self._run_case(4, 4, 128, 128, seq_len)


if __name__ == "__main__":
    unittest.main()
