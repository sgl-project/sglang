import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _verify_batch(prefix_lens, draft_token_num):
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=torch.tensor(prefix_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(prefix_lens, dtype=torch.int64),
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        extend_prefix_lens=None,
        spec_info=SimpleNamespace(draft_token_num=draft_token_num),
    )


class TestMiniMaxSparseVerifyMetadata(CustomTestCase):
    def test_resolve_extend_meta_uses_runtime_verify_width(self):
        backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        backend.is_npu = False
        backend.is_eagle3 = True
        backend.speculative_num_draft_tokens = 8
        batch = _verify_batch([7, 20], draft_token_num=2)

        cu_seqlens, seq_lens, prefix_lens = backend._resolve_extend_meta(
            batch, torch.empty(4, 1)
        )

        torch.testing.assert_close(
            cu_seqlens, torch.tensor([0, 2, 4], dtype=torch.int32)
        )
        torch.testing.assert_close(
            batch.extend_seq_lens, torch.tensor([2, 2], dtype=torch.int32)
        )
        self.assertEqual(batch.extend_seq_lens_cpu, [2, 2])
        torch.testing.assert_close(
            prefix_lens, torch.tensor([7, 20], dtype=torch.int32)
        )
        torch.testing.assert_close(seq_lens, torch.tensor([9, 22], dtype=torch.int32))

    def test_target_verify_with_precomputed_extend_metadata(self):
        backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        backend.is_npu = False
        backend.is_eagle3 = True
        backend.speculative_num_draft_tokens = 8
        batch = _verify_batch([7, 20], draft_token_num=3)
        batch.extend_seq_lens = torch.tensor([3, 3], dtype=torch.int32)
        batch.extend_seq_lens_cpu = [3, 3]
        batch.extend_prefix_lens = torch.tensor([7, 20], dtype=torch.int32)

        cu_seqlens, seq_lens, prefix_lens = backend._resolve_extend_meta(
            batch, torch.empty(6, 1)
        )

        torch.testing.assert_close(
            cu_seqlens, torch.tensor([0, 3, 6], dtype=torch.int32)
        )
        torch.testing.assert_close(
            prefix_lens, torch.tensor([7, 20], dtype=torch.int32)
        )
        torch.testing.assert_close(seq_lens, torch.tensor([10, 23], dtype=torch.int32))

    def test_regular_extend_metadata_is_unchanged(self):
        backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        backend.is_npu = False
        backend.is_eagle3 = True
        backend.speculative_num_draft_tokens = 8
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens=torch.tensor([7, 13], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([7, 13], dtype=torch.int64),
            extend_seq_lens=torch.tensor([2, 3], dtype=torch.int32),
            extend_seq_lens_cpu=[2, 3],
            extend_prefix_lens=torch.tensor([5, 10], dtype=torch.int32),
            spec_info=None,
        )

        cu_seqlens, seq_lens, prefix_lens = backend._resolve_extend_meta(
            batch, torch.empty(5, 1)
        )

        torch.testing.assert_close(
            cu_seqlens, torch.tensor([0, 2, 5], dtype=torch.int32)
        )
        torch.testing.assert_close(
            prefix_lens, torch.tensor([5, 10], dtype=torch.int32)
        )
        torch.testing.assert_close(seq_lens, torch.tensor([7, 13], dtype=torch.int32))

    def test_out_graph_metadata_includes_verify_width(self):
        backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        backend.is_npu = False
        backend.is_eagle3 = True
        backend.index_cache_enabled = False
        backend._msa_owns_decode = False
        backend.max_context_len = 1024
        backend.speculative_num_draft_tokens = 8
        batch = _verify_batch([17, 31], draft_token_num=4)

        backend.init_forward_metadata_out_graph(batch, in_capture=False)
        self.assertEqual(backend._max_seqlen_q, 4)
        self.assertEqual(backend._max_seqlen_k, 35)

        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        self.assertEqual(backend._max_seqlen_q, 4)
        self.assertEqual(backend._max_seqlen_k, 1024)

    def test_non_eagle3_target_verify_metadata_is_unchanged(self):
        backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        backend.is_npu = False
        backend.is_eagle3 = False
        backend.index_cache_enabled = False
        backend._msa_owns_decode = False
        backend.max_context_len = 1024
        backend.speculative_num_draft_tokens = 8
        batch = _verify_batch([17, 31], draft_token_num=4)

        backend.init_forward_metadata_out_graph(batch, in_capture=False)
        self.assertEqual(backend._max_seqlen_q, 1)
        self.assertEqual(backend._max_seqlen_k, 31)

        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        self.assertEqual(backend._max_seqlen_q, 1)
        self.assertEqual(backend._max_seqlen_k, 31)

    def test_graph_prep_invalidates_eagle3_verify_warmup_metadata(self):
        backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        backend.is_npu = False
        backend.is_eagle3 = True
        backend._prefill_seqblock_meta = object()
        batch = _verify_batch([17], draft_token_num=4)

        backend.init_forward_metadata_in_graph(batch)

        self.assertIsNone(backend._prefill_seqblock_meta)

    def test_graph_prep_keeps_non_verify_metadata(self):
        backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        backend.is_npu = False
        backend.is_eagle3 = True
        cached = object()
        backend._prefill_seqblock_meta = cached
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

        backend.init_forward_metadata_in_graph(batch)

        self.assertIs(backend._prefill_seqblock_meta, cached)


if __name__ == "__main__":
    unittest.main()
