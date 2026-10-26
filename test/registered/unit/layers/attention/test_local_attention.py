"""CPU tests for srt/layers/attention/local_attention: the virtual-batch
decomposition and the metadata builder."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.layers.attention.local_attention import (
    LocalAttentionMetadata,
    LocalAttentionMetadataBuilder,
    make_local_attention_virtual_batches,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CPU = torch.device("cpu")


def _page_table(bs: int, width: int = 32) -> torch.Tensor:
    return torch.arange(bs * width, dtype=torch.int32).view(bs, width)


class TestVirtualBatches(CustomTestCase):
    def test_worked_example_from_docstring(self):
        """chunk=4, q=[4,10,5], k=[6,17,9] must split into local_blocks=[2,4,2]."""
        q_start = np.array([0, 4, 14, 19], dtype=np.int32)
        seq_lens = np.array([6, 17, 9], dtype=np.int32)
        q_local, cu_q_local, k_local, block_table_local = (
            make_local_attention_virtual_batches(
                4, q_start, seq_lens, _page_table(3), 1
            )
        )
        self.assertEqual(q_local.tolist(), [2, 2, 1, 4, 4, 1, 4, 1])
        self.assertEqual(cu_q_local.tolist(), [0, 2, 4, 5, 9, 13, 14, 18, 19])
        # Every block sees a full chunk of keys except the last block of each request.
        self.assertEqual(k_local.tolist(), [4, 2, 4, 4, 4, 1, 4, 1])
        self.assertEqual(tuple(block_table_local.shape), (8, 4))

    def test_decode_is_one_block_per_request(self):
        q_start = np.array([0, 1, 2], dtype=np.int32)
        seq_lens = np.array([6, 9], dtype=np.int32)
        q_local, cu_q_local, k_local, _ = make_local_attention_virtual_batches(
            4, q_start, seq_lens, _page_table(2), 1
        )
        self.assertEqual(q_local.tolist(), [1, 1])
        self.assertEqual(cu_q_local.tolist(), [0, 1, 2])
        # Only the tail of the current chunk is attended: 6 % 4 == 2, 9 % 4 == 1.
        self.assertEqual(k_local.tolist(), [2, 1])

    def test_chunk_is_clamped_to_the_longest_sequence(self):
        q_start = np.array([0, 6], dtype=np.int32)
        seq_lens = np.array([6], dtype=np.int32)
        _, _, k_local, block_table_local = make_local_attention_virtual_batches(
            8, q_start, seq_lens, _page_table(1), 1
        )
        self.assertEqual(k_local.tolist(), [6])
        self.assertEqual(tuple(block_table_local.shape), (1, 6))

    def test_page_size_divides_the_block_table_width(self):
        q_start = np.array([0, 6], dtype=np.int32)
        seq_lens = np.array([6], dtype=np.int32)
        q_local, _, k_local, block_table_local = make_local_attention_virtual_batches(
            4, q_start, seq_lens, _page_table(1), 2
        )
        self.assertEqual(q_local.tolist(), [4, 2])
        self.assertEqual(k_local.tolist(), [4, 2])
        self.assertEqual(tuple(block_table_local.shape), (2, 2))  # chunk 4 / page 2


class TestLocalAttentionMetadataBuilder(CustomTestCase):
    def _builder(self, swa_translate=None, max_context_len=16):
        return LocalAttentionMetadataBuilder(
            attention_chunk_size=4,
            page_size=1,
            max_context_len=max_context_len,
            device=CPU,
            swa_translate=swa_translate,
        )

    def test_build_returns_none_without_a_complete_batch(self):
        b = self._builder()
        self.assertIsNone(
            b.build(
                cu_seqlens_q=None,
                cache_seqlens_int32=torch.tensor([6]),
                page_table=_page_table(1),
                device=CPU,
            )
        )

    def test_build_applies_the_swa_translation_to_the_block_table(self):
        args = dict(
            cu_seqlens_q=torch.tensor([0, 6], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([6], dtype=torch.int32),
            page_table=_page_table(1),
            device=CPU,
        )
        plain = self._builder().build(**args)
        swa = self._builder(swa_translate=lambda t: t + 100).build(**args)
        self.assertEqual(plain.local_max_query_len, 4)
        self.assertEqual(plain.local_max_seq_len, 4)
        torch.testing.assert_close(swa.local_block_table, plain.local_block_table + 100)

    def test_cuda_graph_buffers_fit_the_worst_case_decomposition(self):
        """Buffer sizing must match the most virtual batches max_bs x max_context_len can produce."""
        b = self._builder(max_context_len=16)
        bufs = b.alloc_cuda_graph_buffers(max_bs=2)
        md = b.build(
            cu_seqlens_q=torch.tensor([0, 16, 32], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([16, 16], dtype=torch.int32),
            page_table=_page_table(2),
            device=CPU,
        )
        self.assertEqual(
            md.local_query_start_loc.numel(), bufs["local_query_start_loc"].numel()
        )
        self.assertEqual(md.local_seqused_k.numel(), bufs["local_seqused_k"].numel())
        self.assertEqual(
            tuple(md.local_block_table.shape), tuple(bufs["local_block_table"].shape)
        )

    def test_capture_returns_views_of_the_preallocated_buffers(self):
        b = self._builder()
        bufs = b.alloc_cuda_graph_buffers(max_bs=2)
        metadata = SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([6, 9], dtype=torch.int32),
            page_table=_page_table(2),
        )
        md = b.build_for_capture(metadata, 2, buffers=bufs)
        self.assertEqual(md.local_query_start_loc.numel(), 3)
        self.assertEqual(md.local_seqused_k.numel(), 2)
        self.assertEqual(tuple(md.local_block_table.shape), (2, 4))
        self.assertEqual(md.local_max_query_len, 1)
        self.assertEqual(md.local_max_seq_len, 9)
        self.assertEqual(
            md.local_query_start_loc.data_ptr(),
            bufs["local_query_start_loc"].data_ptr(),
        )
        self.assertEqual(
            md.local_block_table.data_ptr(), bufs["local_block_table"].data_ptr()
        )

    def _paged_builder(self, swa_translate):
        # chunk 32 / page 16: two pages per local block.
        return LocalAttentionMetadataBuilder(
            attention_chunk_size=32,
            page_size=16,
            max_context_len=64,
            device=CPU,
            swa_translate=swa_translate,
        )

    def test_replay_reads_page_ids_from_swa_page_table(self):
        """Regression: with page_size > 1 the CUDA-graph replay must not push the
        page-granular table through the token-slot full->swa mapping."""
        # Any translation applied to page ids would land far outside {7, 8}.
        b = self._paged_builder(swa_translate=lambda t: t * 1000)
        bufs = b.alloc_cuda_graph_buffers(max_bs=2)
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([6, 9], dtype=torch.int32),
            page_table=torch.tensor([[3], [5]], dtype=torch.int32),
            swa_page_table=torch.tensor([[7], [8]], dtype=torch.int32),
            local_attn_metadata=LocalAttentionMetadata(),
        )
        b.update_for_replay(metadata, 2, buffers=bufs)
        table = bufs["local_block_table"][:2]
        self.assertEqual(table[:, 0].tolist(), [7, 8])
        self.assertTrue(set(table.flatten().tolist()) <= {0, 7, 8})

    def test_replay_without_swa_pool_reads_page_table(self):
        b = self._paged_builder(swa_translate=None)
        bufs = b.alloc_cuda_graph_buffers(max_bs=2)
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([6, 9], dtype=torch.int32),
            page_table=torch.tensor([[3], [5]], dtype=torch.int32),
            local_attn_metadata=LocalAttentionMetadata(),
        )
        b.update_for_replay(metadata, 2, buffers=bufs)
        self.assertEqual(bufs["local_block_table"][:2, 0].tolist(), [3, 5])

    def test_capture_and_replay_share_the_table_source(self):
        b = self._paged_builder(swa_translate=lambda t: t * 1000)
        bufs = b.alloc_cuda_graph_buffers(max_bs=2)
        metadata = SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([6, 9], dtype=torch.int32),
            page_table=torch.tensor([[3], [5]], dtype=torch.int32),
            swa_page_table=torch.tensor([[7], [8]], dtype=torch.int32),
        )
        md = b.build_for_capture(metadata, 2, buffers=bufs)
        self.assertEqual(
            tuple(md.local_block_table.shape), (2, 2)
        )  # 2 reqs x 2 pages per block

    def test_replay_refills_buffers_in_place_and_zeroes_the_tail(self):
        b = self._builder()
        bufs = b.alloc_cuda_graph_buffers(max_bs=2)
        for t in bufs.values():
            t.fill_(7)  # stale contents from an earlier, larger batch
        lam = LocalAttentionMetadata()
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([6, 9], dtype=torch.int32),
            page_table=_page_table(2),
            local_attn_metadata=lam,
        )
        b.update_for_replay(metadata, 2, buffers=bufs)
        self.assertEqual(bufs["local_query_start_loc"][:3].tolist(), [0, 1, 2])
        self.assertEqual(bufs["local_query_start_loc"][3:].abs().sum().item(), 0)
        self.assertEqual(bufs["local_seqused_k"][:2].tolist(), [2, 1])
        self.assertEqual(bufs["local_seqused_k"][2:].abs().sum().item(), 0)
        self.assertEqual(bufs["local_block_table"][2:].abs().sum().item(), 0)
        self.assertEqual((lam.local_max_query_len, lam.local_max_seq_len), (1, 2))


if __name__ == "__main__":
    unittest.main()
