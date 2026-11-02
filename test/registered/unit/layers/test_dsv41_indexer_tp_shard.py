"""Eager-only TP row sharding and query views for the V4.1 prefill indexer."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.index_logits import (
    flat_index_logits_row_alignment,
    flat_index_logits_rows_per_tile,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import DeepGEMMPrefillData
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class IndexerInputSink:
    """Stop at the GPU backend boundary, retaining the actual dispatch inputs."""

    def topk_prefill(self, inputs):
        self.inputs = inputs

    def publish_prefill(self, inputs):
        self.inputs = inputs

    def publish_prefill_graph(self, inputs, padded_rows, table):
        self.inputs = inputs


class TestIndexerRowShardDispatch(CustomTestCase):
    def dispatch(self, *, graph, source, rows=512, tp=4, cp=1, enabled=True):
        backend = object.__new__(DeepseekV4AttnBackend)
        sink = IndexerInputSink()
        backend.full_topk_indexer = backend.prefill_candidates = sink
        backend.mega_attn = True
        out = torch.full((rows, 4), -1, dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(
            core_metadata=SimpleNamespace(
                page_table=torch.zeros((rows, 1), dtype=torch.int32),
                sparse_raw_indices=lambda ratio: out,
                sparse_page_indices=lambda ratio: out,
            ),
            late_layer_tail=None,
            low_ratio_k_slots={1: torch.arange(rows)},
            low_ratio_index_plans={},
            low_ratio_req_indices=torch.zeros(rows, dtype=torch.int32),
            candidate_metadata=None,
        )
        layer = SimpleNamespace(
            layer_id=0,
            compress_ratio=1,
            freqs_cis=torch.empty(0),
            indexer=SimpleNamespace(
                is_candidate_source=source, uses_candidates=False, n_heads=32
            ),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_num_token_non_padded_cpu=None,
            extend_num_tokens=rows,
            extend_seq_lens_cpu=[rows],
            extend_seq_lens=torch.tensor([rows]),
            seq_lens_cpu=[rows],
            req_pool_indices=torch.tensor([0]),
        )
        parallel = SimpleNamespace(
            attn_tp_size=tp,
            attn_tp_rank=0,
            attn_tp_group=object(),
            attn_cp_size=cp,
        )
        x = torch.empty(rows, 1)
        pos = torch.arange(rows)
        with (
            envs.SGLANG_OPT_DSV4_INDEXER_TP_SHARD.override(enabled),
            patch.object(deepseek_v4_backend, "get_parallel", return_value=parallel),
        ):
            if graph:
                backend._prefill_graph_full_topk_break(layer, x, x, pos, batch, None)
            else:
                backend._low_ratio_index_topk(
                    layer,
                    x,
                    x,
                    backend.forward_metadata.low_ratio_req_indices,
                    pos,
                    batch,
                )
        return sink.inputs

    def test_graph_break_keeps_full_topk_and_candidate_publish_replicated(self):
        # Both graph-break callers used to inherit a shard from
        # rows_per_request=None, despite neither opting in to TP collectives.
        for source in (False, True):
            for rows in (512, 513):
                with self.subTest(source=source, rows=rows):
                    eager = self.dispatch(graph=False, source=source, rows=rows)
                    captured = self.dispatch(graph=True, source=source, rows=rows)
                    self.assertIsNotNone(eager.row_shard)
                    self.assertIsNone(captured.row_shard)
                    self.assertEqual(captured.rows_per_request, [rows])
                    self.assertEqual(captured.out_raw_indices.shape[0], rows)

    def test_eager_sharding_requires_enabled_tp_and_no_cp(self):
        for enabled, tp, cp, rows, expected in (
            (True, 4, 1, 512, True),
            (True, 4, 1, 511, False),
            (False, 4, 1, 512, False),
            (True, 1, 1, 512, False),
            (True, 4, 2, 512, False),
        ):
            with self.subTest(enabled=enabled, tp=tp, cp=cp, rows=rows):
                inputs = self.dispatch(
                    graph=False, source=False, enabled=enabled, tp=tp, cp=cp, rows=rows
                )
                self.assertEqual(inputs.row_shard is not None, expected)
                if expected:
                    self.assertEqual(inputs.row_shard.local_rows(rows), slice(0, 128))


class TestPrefillQueryView(CustomTestCase):
    def test_narrow_preserves_global_k_and_slices_all_row_operands(self):
        data = DeepGEMMPrefillData(
            k_slots=torch.arange(30),
            request_starts=torch.tensor([0] * 5 + [10] * 3),
            lens_per_request=[10, 0, 20],
            rows_per_request=[5, 0, 3],
            compress_lens=torch.arange(8),
            q_fp4=torch.arange(8 * 2 * 64).view(8, 2, 64),
            q_sf=torch.arange(16).view(8, 2),
            weights=torch.arange(16).view(8, 2).float(),
        )
        for rows, counts in ((slice(3, 7), [2, 0, 2]), (slice(8, 8), [0, 0, 0])):
            with self.subTest(rows=rows):
                local = data.narrow(rows)
                self.assertEqual(local.rows_per_request, counts)
                self.assertEqual(local.num_rows, sum(counts))
                self.assertIs(local.k_slots, data.k_slots)
                self.assertIs(local.lens_per_request, data.lens_per_request)
                for field in (
                    "request_starts",
                    "compress_lens",
                    "q_fp4",
                    "q_sf",
                    "weights",
                ):
                    original = getattr(data, field)
                    view = getattr(local, field)
                    torch.testing.assert_close(view, original[rows])
                    self.assertEqual(
                        view.untyped_storage().data_ptr(),
                        original.untyped_storage().data_ptr(),
                    )
        self.assertEqual(data.rows_per_request, [5, 0, 3])

    @patch("torch.version.cuda", None)
    def test_shard_and_budget_tiles_share_mma_alignment(self):
        # Test the host-side alignment math, not GPU-specific wave rounding.
        # A CUDA-built torch can be present in CPU CI; probing its SM count
        # would initialize CUDA in the fork worker.
        for heads in (16, 32, 64):
            alignment = flat_index_logits_row_alignment(heads)
            tile = flat_index_logits_rows_per_tile(
                1024, 4096, heads=heads, budget_bytes=4096 * 4 * 13
            )
            self.assertEqual(tile % alignment, 0)
            self.assertLessEqual(tile, 13)


if __name__ == "__main__":
    unittest.main()
