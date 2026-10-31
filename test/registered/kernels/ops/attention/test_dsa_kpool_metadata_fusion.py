"""KPool fused metadata must retain live tails and refresh captured buffers."""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.kpool_metadata.decode import (
    _fused_dsa_decode_metadata_kpool_kernel,
    fused_dsa_decode_metadata_kpool,
)
from sglang.kernels.ops.attention.dsa.kpool_metadata.draft_extend import (
    _fused_dsa_draft_extend_metadata_kpool_kernel,
    fused_dsa_draft_extend_metadata_kpool,
)
from sglang.kernels.ops.attention.dsa.kpool_metadata.verify import (
    _fused_dsa_target_verify_metadata_kpool_kernel,
    fused_dsa_target_verify_metadata_kpool,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestKPoolMetadataFusion(CustomTestCase):
    def test_batch_sizes_reuse_compiled_metadata_kernels(self):
        """New batch sizes in one tile bucket must not stall on compilation."""
        for mode, kernel in (
            ("decode", _fused_dsa_decode_metadata_kpool_kernel),
            ("verify", _fused_dsa_target_verify_metadata_kpool_kernel),
            ("draft_static", _fused_dsa_draft_extend_metadata_kpool_kernel),
            ("draft_variable", _fused_dsa_draft_extend_metadata_kpool_kernel),
        ):
            with self.subTest(mode=mode):
                kernel_cache = kernel.device_caches[torch.cuda.current_device()][0]
                for bs in (5, 6, 7):
                    self._check_batch_metadata(mode, bs)
                    if bs == 5:
                        compiled_after_first = len(kernel_cache)
                    self.assertEqual(len(kernel_cache), compiled_after_first)

    def _check_batch_metadata(self, mode, bs):
        device, width, next_n, topk, kpool = "cuda", 256, 4, 64, 4
        seq_values = [1, 64, 65, 68, 127, 128, 131][:bs]
        seq = torch.tensor(seq_values, device=device, dtype=torch.int64)
        req = torch.arange(bs - 1, -1, -1, device=device, dtype=torch.int64)
        table = torch.arange(bs * width, device=device, dtype=torch.int32).view(
            bs, width
        )
        if mode == "decode":
            repeats = [1] * bs
            expanded_values = seq_values
            cache = seq.int()
        elif mode == "verify":
            repeats = [next_n] * bs
            expanded_values = [s + i + 1 for s in seq_values for i in range(next_n)]
            cache = (seq + next_n).int()
        else:
            repeats = [
                next_n if mode == "draft_static" else 3 + i % 2 for i in range(bs)
            ]
            expanded_values = [
                max(s - n + i + 1, 0)
                for s, n in zip(seq_values, repeats)
                for i in range(n)
            ]
            cache = seq.int()
        expanded = torch.tensor(expanded_values, device=device, dtype=torch.int32)
        rows = len(expanded_values)

        def empty(*shape):
            return torch.full(shape, -1, device=device, dtype=torch.int32)

        buffers = dict(
            cache_seqlens=empty(bs),
            cu_seqlens_k=empty(bs + 1),
            page_table_1=empty(rows, width),
            dsa_cache_seqlens=empty(rows),
            dsa_cu_seqlens_k=empty(rows + 1),
            real_page_table=empty(rows, width // 64),
        )
        args = dict(
            seq_lens=seq,
            req_pool_indices=req,
            req_to_token=table,
            bs=bs,
            dsa_index_topk=topk,
            physical_page_size=64,
            index_kpool=kpool,
        )
        if mode == "decode":
            fused_dsa_decode_metadata_kpool(max_len=width, **args, **buffers)
        else:
            buffers["seqlens_expanded"] = empty(rows)
            if mode == "verify":
                buffers["paged_mqa_ctx_lens_2d"] = empty(bs, next_n)
                fused_dsa_target_verify_metadata_kpool(
                    max_seqlen_k=width, next_n=next_n, **args, **buffers
                )
                torch.testing.assert_close(
                    buffers["paged_mqa_ctx_lens_2d"], cache[:, None].expand(bs, next_n)
                )
            else:
                fused_dsa_draft_extend_metadata_kpool(
                    extend_seq_lens=torch.tensor(
                        repeats, device=device, dtype=torch.int32
                    ),
                    total_len=rows,
                    max_seqlen_k=width,
                    max_extend_len=next_n,
                    max_total_len=bs * next_n,
                    static_extend_len=mode == "draft_static",
                    **args,
                    **buffers,
                )
            torch.testing.assert_close(buffers["seqlens_expanded"], expanded)
        expected_dsa = torch.minimum(expanded, topk + expanded % kpool)
        torch.testing.assert_close(buffers["cache_seqlens"], cache)
        torch.testing.assert_close(buffers["cu_seqlens_k"][1:], cache.cumsum(0).int())
        torch.testing.assert_close(buffers["dsa_cache_seqlens"], expected_dsa)
        torch.testing.assert_close(
            buffers["dsa_cu_seqlens_k"][1:], expected_dsa.cumsum(0).int()
        )
        self.assertEqual(buffers["cu_seqlens_k"][0].item(), 0)
        self.assertEqual(buffers["dsa_cu_seqlens_k"][0].item(), 0)
        repeat_tensor = torch.tensor(repeats, device=device)
        expected_pages = table[req].repeat_interleave(repeat_tensor, dim=0)
        row_lens = cache.repeat_interleave(repeat_tensor)
        live = torch.arange(width, device=device)[None, :] < row_lens[:, None]
        torch.testing.assert_close(buffers["page_table_1"][live], expected_pages[live])
        real_live = (
            torch.arange(0, width, 64, device=device)[None, :] < row_lens[:, None]
        )
        torch.testing.assert_close(
            buffers["real_page_table"][real_live],
            (expected_pages[:, ::64] // 64)[real_live],
        )

    def test_verify_replay_boundaries_and_request_remapping(self):
        device = "cuda"
        bs, next_n, width, topk, kpool = 4, 6, 16384, 2048, 4
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
            fused_dsa_target_verify_metadata_kpool(
                seq_lens=seq,
                req_pool_indices=req,
                req_to_token=table,
                bs=bs,
                max_seqlen_k=width,
                dsa_index_topk=topk,
                physical_page_size=64,
                next_n=next_n,
                index_kpool=kpool,
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
            expected = torch.minimum(expanded, topk + expanded % kpool).int()
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
