"""KPool fused metadata must retain live tails and refresh captured buffers."""

import unittest

import torch

from sglang.kernels.ops.attention.dsa_kpool_metadata.verify import (
    fused_dsa_target_verify_metadata,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestKPoolMetadataFusion(CustomTestCase):
    def test_verify_replay_boundaries_and_request_remapping(self):
        device = "cuda"
        bs, next_n, width, topk, pool_size = 4, 6, 16384, 2048, 4
        seq = torch.tensor([1, 61, 2047, 8191], device=device, dtype=torch.int64)
        req = torch.tensor([3, 1, 6, 0], device=device, dtype=torch.int64)
        table = torch.arange(8 * width, device=device, dtype=torch.int32).view(8, width)

        def empty(*shape):
            return torch.full(shape, -1, device=device, dtype=torch.int32)

        buffers = dict(
            cache_seqlens=empty(bs),
            cu_seqlens_k=empty(bs + 1),
            page_table_1=empty(bs * next_n, width),
            seqlens_expanded=empty(bs * next_n),
            dsa_cache_seqlens=empty(bs * next_n),
            dsa_cu_seqlens_k=empty(bs * next_n + 1),
            real_page_table=empty(bs * next_n, width // 64),
            paged_mqa_ctx_lens_2d=empty(bs, next_n),
        )
        addresses = {key: value.data_ptr() for key, value in buffers.items()}

        def refresh():
            fused_dsa_target_verify_metadata(
                seq_lens=seq,
                req_pool_indices=req,
                req_to_token=table,
                bs=bs,
                max_seqlen_k=width,
                dsa_index_topk=topk,
                real_page_size=64,
                next_n=next_n,
                index_kpool=pool_size,
                **buffers,
            )

        refresh()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            refresh()
        for lengths, requests in [
            ([2, 63, 2048, 8193], [0, 6, 1, 3]),
            ([64, 128, 2051, 8190], [5, 2, 7, 4]),
            ([0, 3, 2045, 9000], [7, 0, 3, 2]),
        ]:
            seq.copy_(torch.tensor(lengths, device=device))
            req.copy_(torch.tensor(requests, device=device))
            graph.replay()
            expanded = (
                seq[:, None] + torch.arange(1, next_n + 1, device=device)
            ).flatten()
            expected = torch.minimum(expanded, topk + expanded % pool_size).int()
            torch.testing.assert_close(buffers["seqlens_expanded"], expanded.int())
            torch.testing.assert_close(buffers["dsa_cache_seqlens"], expected)
            torch.testing.assert_close(
                buffers["dsa_cu_seqlens_k"][1:], expected.cumsum(0).int()
            )
            torch.testing.assert_close(buffers["cache_seqlens"], (seq + next_n).int())
            torch.testing.assert_close(
                buffers["paged_mqa_ctx_lens_2d"],
                (seq + next_n).int()[:, None].expand(bs, next_n),
            )
            expected_pages = table[req].repeat_interleave(next_n, dim=0)
            row_lens = (seq + next_n).repeat_interleave(next_n)
            live = torch.arange(width, device=device)[None, :] < row_lens[:, None]
            torch.testing.assert_close(
                buffers["page_table_1"][live], expected_pages[live]
            )
            real_live = (
                torch.arange(0, width, 64, device=device)[None, :] < row_lens[:, None]
            )
            torch.testing.assert_close(
                buffers["real_page_table"][real_live],
                (expected_pages[:, ::64] // 64)[real_live],
            )
            self.assertEqual(
                addresses, {key: value.data_ptr() for key, value in buffers.items()}
            )


if __name__ == "__main__":
    unittest.main()
