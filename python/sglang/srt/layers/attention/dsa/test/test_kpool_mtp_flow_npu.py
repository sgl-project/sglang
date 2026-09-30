"""Flow tests for Ascend DSA KPool target-verify and draft-extend-v2."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

if torch_npu is not None:
    from sglang.srt.hardware_backend.npu.attention.ascend_dsa_backend import (
        AscendDSAAttnBackend,
        AscendDSAForwardMetadata,
        AscendDSAIndexerMetadata,
    )
    from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang.test.test_utils import CustomTestCase


POOL_SIZE = 4
NUM_DRAFT_TOKENS = 3
PAGE_SIZE = 128
HEAD_DIM = 128


def _npu_is_available() -> bool:
    if torch_npu is None or not hasattr(torch, "npu"):
        return False
    try:
        return torch.npu.is_available()
    except RuntimeError:
        return False


@unittest.skipUnless(_npu_is_available(), "Ascend NPU is required")
class TestKPoolMTPFlowNPU(CustomTestCase):
    device = "npu"

    def _make_backend(self):
        backend = object.__new__(AscendDSAAttnBackend)
        backend.page_size = PAGE_SIZE
        backend.dsa_index_kpool = POOL_SIZE
        backend.speculative_num_draft_tokens = NUM_DRAFT_TOKENS
        backend.speculative_step_id = 0
        backend.forward_metadata = AscendDSAForwardMetadata(page_size=PAGE_SIZE)

        req_to_token = torch.zeros(
            (2, PAGE_SIZE), dtype=torch.int64, device=self.device
        )
        req_to_token[0] = torch.arange(
            PAGE_SIZE, 2 * PAGE_SIZE, dtype=torch.int64, device=self.device
        )
        req_to_token[1] = torch.arange(
            2 * PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64, device=self.device
        )
        backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        return backend

    def _make_forward_batch(
        self,
        mode,
        seq_lens: list[int],
        num_accept_tokens: list[int] | None = None,
    ):
        spec_info = None
        if num_accept_tokens is not None:
            spec_info = SimpleNamespace(
                num_accept_tokens=torch.tensor(
                    num_accept_tokens, dtype=torch.int32, device=self.device
                )
            )
        return SimpleNamespace(
            forward_mode=mode,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            batch_size=len(seq_lens),
            req_pool_indices=torch.arange(
                len(seq_lens), dtype=torch.int64, device=self.device
            ),
            extend_seq_lens_cpu=None,
            extend_seq_lens=None,
            spec_info=spec_info,
        )

    def test_target_verify_and_draft_extend_v2_metadata(self):
        expected_seqlens = torch.tensor(
            [6, 7, 8, 10, 11, 12], dtype=torch.int32
        )
        expected_pool_seqlens = torch.tensor(
            [1, 1, 2, 2, 2, 3], dtype=torch.int32
        )

        cases = (
            (ForwardMode.TARGET_VERIFY, [5, 9], None),
            (ForwardMode.DRAFT_EXTEND_V2, [8, 12], [1, 3]),
        )
        for mode, seq_lens, accepted in cases:
            with self.subTest(mode=mode):
                backend = self._make_backend()
                forward_batch = self._make_forward_batch(mode, seq_lens, accepted)
                backend._populate_dsa_metadata(forward_batch)

                metadata = backend.forward_metadata
                plan = metadata.kpool_write_plan
                self.assertIsNotNone(plan)
                torch.testing.assert_close(
                    metadata.dsa_seqlens_expanded.cpu(), expected_seqlens
                )
                torch.testing.assert_close(
                    plan.seqlens_per_q.cpu(), expected_seqlens
                )
                torch.testing.assert_close(
                    plan.pool_seqlens_per_q.cpu(), expected_pool_seqlens
                )
                torch.testing.assert_close(
                    plan.write_start.cpu(),
                    torch.tensor([5, 9], dtype=torch.int32),
                )
                torch.testing.assert_close(
                    plan.tail_logical_start.cpu(),
                    torch.tensor([4, 8], dtype=torch.int32),
                )
                torch.testing.assert_close(
                    plan.write_loc[:, 0].cpu(),
                    torch.tensor([129, 258], dtype=torch.int64),
                )
                self.assertEqual(tuple(plan.write_loc.shape), (2, 1))
                self.assertEqual(tuple(metadata.real_page_table.shape), (2, 1))

                if mode.is_draft_extend_v2():
                    torch.testing.assert_close(
                        plan.effective_n_per_batch.cpu(),
                        torch.tensor([1, 3], dtype=torch.int32),
                    )
                else:
                    self.assertIsNone(plan.effective_n_per_batch)

    def test_draft_decode_metadata_does_not_require_extend_lengths(self):
        backend = self._make_backend()
        forward_batch = self._make_forward_batch(ForwardMode.DECODE, [5, 9])
        forward_batch.spec_info = SimpleNamespace()

        self.assertIsNone(forward_batch.extend_seq_lens)
        backend._populate_dsa_metadata(forward_batch, _graph_capture=True)

        metadata = backend.forward_metadata
        torch.testing.assert_close(
            metadata.token_to_batch_idx.cpu(),
            torch.tensor([0, 1], dtype=torch.int64),
        )
        self.assertIsNone(metadata.dsa_seqlens_expanded)

        replay_batch = self._make_forward_batch(ForwardMode.DECODE, [6, 10])
        replay_batch.spec_info = SimpleNamespace()
        backend._replay_dsa_metadata(replay_batch)

        self.assertIsNone(replay_batch.extend_seq_lens)
        torch.testing.assert_close(
            metadata.token_to_batch_idx.cpu(),
            torch.tensor([0, 1], dtype=torch.int64),
        )

    def test_mtp_paged_read_uses_per_query_lengths(self):
        backend = self._make_backend()
        forward_batch = self._make_forward_batch(
            ForwardMode.DRAFT_EXTEND_V2, [8, 12], [1, 3]
        )
        forward_batch.out_cache_loc = torch.arange(
            1, 7, dtype=torch.int64, device=self.device
        )
        backend._populate_dsa_metadata(forward_batch)
        metadata = AscendDSAIndexerMetadata(backend.forward_metadata)

        indexer = object.__new__(IndexerKPool)
        indexer.index_kpool = POOL_SIZE
        indexer.index_topk = 8
        indexer.n_heads = 2
        indexer.head_dim = HEAD_DIM

        cache = torch.zeros(
            (4, PAGE_SIZE, 1, HEAD_DIM),
            dtype=torch.bfloat16,
            device=self.device,
        )
        pool = SimpleNamespace(
            page_size=PAGE_SIZE,
            get_index_k_with_scale_buffer=lambda _layer_id: cache,
        )
        captured = {}

        def fake_lightning_indexer(**kwargs):
            captured.update(kwargs)
            rows = kwargs["query"].shape[0]
            indices = torch.zeros(
                (rows, 1, 2), dtype=torch.int32, device=self.device
            )
            return (indices,)

        query = torch.zeros(
            (6, indexer.n_heads, HEAD_DIM),
            dtype=torch.bfloat16,
            device=self.device,
        )
        weights = torch.ones(
            (6, indexer.n_heads, 1),
            dtype=torch.bfloat16,
            device=self.device,
        )

        with patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool."
            "get_token_to_kv_pool",
            return_value=pool,
        ), patch.object(
            torch_npu, "npu_lightning_indexer", side_effect=fake_lightning_indexer
        ):
            result = indexer._get_topk_paged_npu(
                forward_batch, 0, query, weights, metadata
            )

        torch.testing.assert_close(
            captured["actual_seq_lengths_key"].cpu(),
            torch.tensor([1, 1, 2, 2, 2, 3], dtype=torch.int32),
        )
        self.assertEqual(tuple(captured["block_table"].shape), (6, 1))
        self.assertEqual(tuple(result.shape), (6, 11))
        torch.testing.assert_close(
            result[:, 8:].cpu(),
            torch.tensor(
                [
                    [4, 5, -1],
                    [4, 5, 6],
                    [-1, -1, -1],
                    [8, 9, -1],
                    [8, 9, 10],
                    [-1, -1, -1],
                ],
                dtype=torch.int32,
            ),
        )

    def test_mtp_page_table_expansion_avoids_repeat_interleave(self):
        forward_batch = self._make_forward_batch(
            ForwardMode.DRAFT_EXTEND_V2, [8, 12], [1, 3]
        )
        block_tables = torch.tensor(
            [[11, 12], [21, 22]], dtype=torch.int32, device=self.device
        )

        with patch.object(
            torch,
            "repeat_interleave",
            side_effect=AssertionError("MTP graph path must not synchronize repeats"),
        ):
            expanded = IndexerKPool._expand_page_table_for_queries_npu(
                forward_batch, block_tables, 6
            )

        torch.testing.assert_close(
            expanded.cpu(),
            torch.tensor(
                [
                    [11, 12],
                    [11, 12],
                    [11, 12],
                    [21, 22],
                    [21, 22],
                    [21, 22],
                ],
                dtype=torch.int32,
            ),
        )

    def test_graph_replay_reuses_mtp_metadata_and_plan_buffers(self):
        backend = self._make_backend()
        backend.graph_mode = True
        backend.forward_metadata.block_tables = torch.tensor(
            [[1, 0], [2, 0]], dtype=torch.int32, device=self.device
        )
        first_batch = self._make_forward_batch(
            ForwardMode.DRAFT_EXTEND_V2, [8, 12], [1, 3]
        )
        backend._populate_dsa_metadata(first_batch, _graph_capture=True)

        metadata = backend.forward_metadata
        first_plan = metadata.kpool_write_plan
        pointers = {
            "cache_seqlens": metadata.cache_seqlens_int32.data_ptr(),
            "page_table": metadata.real_page_table.data_ptr(),
            "expanded": metadata.dsa_seqlens_expanded.data_ptr(),
            "plan_req": first_plan.req.data_ptr(),
            "plan_write_start": first_plan.write_start.data_ptr(),
            "plan_effective_n": first_plan.effective_n_per_batch.data_ptr(),
        }

        replay_batch = self._make_forward_batch(
            ForwardMode.DRAFT_EXTEND_V2, [9, 13], [2]
        )
        # Graph runners keep max-batch backing buffers and expose the active
        # capture bucket through batch_size. Replay metadata must ignore the
        # padded rows instead of reallocating capture-time tensors.
        replay_batch.seq_lens = torch.tensor(
            [9, 13, 100, 100, 100, 100, 100, 100],
            dtype=torch.int32,
            device=self.device,
        )
        replay_batch.seq_lens_cpu = torch.tensor(
            [9, 13, 100, 100, 100, 100, 100, 100], dtype=torch.int32
        )
        replay_batch.req_pool_indices = torch.tensor(
            [0, 1, 99, 99, 99, 99, 99, 99],
            dtype=torch.int64,
            device=self.device,
        )
        backend._replay_dsa_metadata(replay_batch)

        replay_plan = metadata.kpool_write_plan
        self.assertIs(replay_plan, first_plan)
        self.assertEqual(
            metadata.cache_seqlens_int32.data_ptr(), pointers["cache_seqlens"]
        )
        self.assertEqual(metadata.real_page_table.data_ptr(), pointers["page_table"])
        self.assertEqual(
            metadata.dsa_seqlens_expanded.data_ptr(), pointers["expanded"]
        )
        self.assertEqual(replay_plan.req.data_ptr(), pointers["plan_req"])
        self.assertEqual(
            replay_plan.write_start.data_ptr(), pointers["plan_write_start"]
        )
        self.assertEqual(
            replay_plan.effective_n_per_batch.data_ptr(), pointers["plan_effective_n"]
        )
        torch.testing.assert_close(
            metadata.dsa_seqlens_expanded.cpu(),
            torch.tensor([7, 8, 9, 11, 12, 13], dtype=torch.int32),
        )
        torch.testing.assert_close(
            replay_plan.write_start.cpu(), torch.tensor([6, 10], dtype=torch.int32)
        )
        torch.testing.assert_close(
            replay_plan.effective_n_per_batch.cpu(),
            torch.tensor([2, 0], dtype=torch.int32),
        )

    def test_forward_dispatches_both_mtp_stages_to_ring_plan(self):
        for mode, return_indices in (
            (ForwardMode.TARGET_VERIFY, True),
            (ForwardMode.DRAFT_EXTEND_V2, True),
            (ForwardMode.DRAFT_EXTEND_V2, False),
        ):
            with self.subTest(mode=mode, return_indices=return_indices):
                indexer = object.__new__(IndexerKPool)
                torch.nn.Module.__init__(indexer)
                indexer.head_dim = HEAD_DIM
                indexer.rope_head_dim = 64
                indexer.n_heads = 2
                indexer.index_topk = 8
                indexer.index_kpool = POOL_SIZE
                indexer.skip_rope = True
                indexer.index_kpool_compress_gate = torch.zeros(
                    (HEAD_DIM, 4), dtype=torch.bfloat16, device=self.device
                )
                indexer.wq_b = lambda _q_lora: (
                    torch.zeros(
                        (6, indexer.n_heads * HEAD_DIM),
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                    None,
                )
                indexer.wk = lambda _x: (
                    torch.zeros(
                        (6, HEAD_DIM), dtype=torch.bfloat16, device=self.device
                    ),
                    None,
                )
                indexer.k_norm = lambda value: value
                indexer._compress_write_mtp_npu = Mock()
                indexer._compress_write_decode_npu = Mock()
                indexer._compress_write_extend_npu = Mock()
                indexer._get_logits_head_gate = Mock(
                    return_value=torch.ones(
                        (6, indexer.n_heads, 1),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                )
                expected = torch.zeros(
                    (6, 11), dtype=torch.int32, device=self.device
                )
                indexer._get_topk_paged_npu = Mock(return_value=expected)

                forward_batch = SimpleNamespace(
                    forward_mode=mode,
                    seq_lens_cpu=torch.tensor([5, 9], dtype=torch.int32),
                )
                attention_backend = SimpleNamespace(
                    get_indexer_metadata=lambda _layer_id, _batch: SimpleNamespace()
                )
                x = torch.zeros((6, 4), dtype=torch.bfloat16, device=self.device)
                q_lora = torch.zeros(
                    (6, 4), dtype=torch.bfloat16, device=self.device
                )
                positions = torch.arange(6, dtype=torch.int64, device=self.device)

                with patch(
                    "sglang.srt.layers.attention.dsa.dsa_indexer_kpool."
                    "get_attn_backend",
                    return_value=attention_backend,
                ):
                    result = indexer.forward_npu(
                        x,
                        q_lora,
                        positions,
                        forward_batch,
                        layer_id=0,
                        return_indices=return_indices,
                    )

                indexer._compress_write_mtp_npu.assert_called_once()
                indexer._compress_write_decode_npu.assert_not_called()
                indexer._compress_write_extend_npu.assert_not_called()
                if return_indices:
                    self.assertIs(result, expected)
                    indexer._get_topk_paged_npu.assert_called_once()
                else:
                    self.assertIsNone(result)
                    indexer._get_topk_paged_npu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
