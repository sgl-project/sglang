"""The split-prefix extend sweep must match the single-pass `extend_attention_fwd`."""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import (
    extend_attention_fwd,
    extend_attention_fwd_long_prefix,
    long_prefix_num_splits,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")

# observed on gfx950: ~5e-4 (bf16 KV), ~7e-3 (fp8 KV, the P.V rounding both kernels share)
ATOL = {torch.bfloat16: 1e-2, torch.float8_e4m3fn: 3e-2}


def _inputs(prefix_lens, extend_lens, h_q, h_kv, d, kv_dtype, device):
    B = len(prefix_lens)
    total_prefix = int(sum(prefix_lens))
    k_buffer = torch.randn(total_prefix, h_kv, d, device=device).to(kv_dtype)
    v_buffer = torch.randn(total_prefix, h_kv, d, device=device).to(kv_dtype)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.tensor(prefix_lens, dtype=torch.int32, device=device), 0
    )
    # a scrambled page table exercises the split sweep through kv_indices
    kv_indices = torch.randperm(total_prefix, device=device).to(torch.int64)
    n_ext = int(sum(extend_lens))
    q = torch.randn(n_ext, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(n_ext, h_kv, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(n_ext, h_kv, d, dtype=torch.bfloat16, device=device)
    qo_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(
        torch.tensor(extend_lens, dtype=torch.int32, device=device), 0
    )
    return q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestExtendLongPrefix(CustomTestCase):
    """A slice boundary, partial stride or combine weight error shows as a mismatch."""

    def _run(
        self,
        prefix_lens,
        extend_lens,
        h_q=16,
        h_kv=1,
        d=128,
        kv_dtype=torch.bfloat16,
        k_scale=1.0,
        v_scale=1.0,
        num_splits=None,
    ):
        device = "cuda"
        torch.manual_seed(0)
        q, k, v, kb, vb, qo, kvp, kvi = _inputs(
            prefix_lens, extend_lens, h_q, h_kv, d, kv_dtype, device
        )
        mle = max(extend_lens)
        sm_scale = 1.0 / (d**0.5)
        o_ref = torch.empty_like(q)
        extend_attention_fwd(
            q,
            k,
            v,
            o_ref,
            kb,
            vb,
            qo,
            kvp,
            kvi,
            None,
            True,
            None,
            mle,
            k_scale,
            v_scale,
            sm_scale=sm_scale,
            extend_seq_lens_cpu=extend_lens,
        )
        o = torch.empty_like(q)
        extend_attention_fwd_long_prefix(
            q,
            k,
            v,
            o,
            kb,
            vb,
            qo,
            kvp,
            kvi,
            True,
            mle,
            k_scale,
            v_scale,
            sm_scale=sm_scale,
            extend_seq_lens_cpu=extend_lens,
            num_splits=num_splits,
        )
        torch.cuda.synchronize()
        self.assertFalse(torch.isnan(o).any().item())
        diff = (o.float() - o_ref.float()).abs().max().item()
        self.assertLess(diff, ATOL[kv_dtype], f"max abs diff {diff}")

    def test_single_request_bf16(self):
        self._run([40000], [1000])

    def test_single_request_fp8(self):
        self._run([40000], [1000], kv_dtype=torch.float8_e4m3fn)

    def test_ragged_batch(self):
        self._run([12000, 300, 33000], [7, 129, 1500])

    def test_tiny_extend_many_splits(self):
        self._run([65536], [3], num_splits=16)

    def test_single_split_layout(self):
        # a single split still takes the 4-D partial layout and the combine
        self._run([20000], [2048], num_splits=1)

    def test_gqa_group_4(self):
        self._run([9000, 15000], [64, 500], h_q=8, h_kv=2)

    def test_kv_scales_fp8(self):
        self._run(
            [30000], [256], kv_dtype=torch.float8_e4m3fn, k_scale=0.7, v_scale=1.3
        )

    def test_head_dim_64(self):
        self._run([16384], [512], d=64)

    def test_prefix_shorter_than_one_tile_per_split(self):
        # 100 prefix tokens over 16 splits: most slices are empty
        self._run([100], [50], num_splits=16)

    def test_auto_split_count(self):
        self.assertEqual(long_prefix_num_splits(1, 16, 8), 16)
        self.assertEqual(long_prefix_num_splits(1, 16, 8192), 1)
        self.assertGreaterEqual(long_prefix_num_splits(24, 16, 4), 1)


if __name__ == "__main__":
    unittest.main()
