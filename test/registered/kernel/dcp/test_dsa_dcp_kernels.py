"""Exercise DCP KV writes and sparse partials beyond the per-rank watermark.

Emulates ranks on one GPU and checks their merged result against dense PyTorch
attention over the selected tokens, including a rank with an empty selection.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
from sglang.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_buffer_dcp_sharded_triton,
    set_mla_kv_buffer_triton,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestDSADCPKernels(CustomTestCase):
    def test_sharded_norope_sparse_attention(self):
        torch.manual_seed(23)
        device = "cuda"
        dim, heads, rows, per_rank_capacity = 512, 16, 3, 128
        q = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16)
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        for size in (1, 2, 4, 8):
            with self.subTest(dcp_size=size):
                capacity = per_rank_capacity * size
                # Write past the old per-rank bound into the last virtual page.
                locs = torch.arange(capacity - 32, capacity, device=device)
                k = torch.randn(32, 1, dim, device=device, dtype=torch.bfloat16)
                virtual = torch.full((rows, 64), -1, device=device, dtype=torch.int32)
                virtual[0, :32] = locs
                virtual[1, :8] = locs[:8]
                virtual[2, 0] = locs[-1]  # All but one rank own no selected KV.
                partials, lses = [], []
                for rank in range(size):
                    cache = torch.zeros(
                        per_rank_capacity, 1, dim, device=device, dtype=k.dtype
                    )
                    with get_parallel().override(
                        attn_dcp_size=size, attn_dcp_rank=rank
                    ):
                        set_mla_kv_buffer_dcp_sharded_triton(cache, locs, k, None)
                        local = backend._dcp_localize_page_table(virtual)
                    expected_cache = torch.zeros_like(cache)
                    owned = locs % size == rank
                    expected_cache[locs[owned] // size] = k[owned]
                    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
                    out, lse = tilelang_sparse_fwd(
                        q, cache, local.unsqueeze(1), dim**-0.5, return_lse=True
                    )
                    self.assertTrue(torch.isfinite(out).all())
                    partials.append(out.squeeze(0).float())
                    lses.append(lse.squeeze(0))
                lses = torch.stack(lses)
                weights = torch.softmax(
                    lses * torch.log(torch.tensor(2.0, device=device)), dim=0
                )
                merged = (torch.stack(partials) * weights.unsqueeze(-1)).sum(0)
                for row, selected in enumerate((k[:, 0], k[:8, 0], k[-1:, 0])):
                    scores = q[row].float() @ selected.float().T * dim**-0.5
                    expected = scores.softmax(-1) @ selected.float()
                    torch.testing.assert_close(
                        merged[row], expected, atol=0.025, rtol=0.025
                    )

                # The ordinary path retains its original signature and output.
                replicated = torch.zeros(capacity, 1, dim, device=device, dtype=k.dtype)
                set_mla_kv_buffer_triton(replicated, locs, k, None)
                out = tilelang_sparse_fwd(
                    q, replicated, virtual.unsqueeze(1), dim**-0.5
                )
                torch.testing.assert_close(
                    out.squeeze(0).float(), merged, atol=0.025, rtol=0.025
                )

    def test_norope_padding_does_not_write_reserved_slot(self):
        cache = torch.full((4, 1, 512), 7, dtype=torch.bfloat16, device="cuda")
        loc = torch.tensor([0, 3], device="cuda")
        kv = torch.ones(2, 1, 512, dtype=cache.dtype, device=cache.device)
        set_mla_kv_buffer_triton(cache, loc, kv, None)
        self.assertTrue((cache[0] == 7).all())
        self.assertTrue((cache[3] == 1).all())


if __name__ == "__main__":
    unittest.main()
