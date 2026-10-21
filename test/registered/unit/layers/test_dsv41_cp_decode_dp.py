"""Query DP preserves global ordering, ghost ranks and page-table row ownership."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4Metadata,
)
from sglang.srt.layers.cp.cp_decode_attn_dp import (
    CPDecodeQueryLayout,
    cp_decode_query_layout,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCPDecodeDP(CustomTestCase):
    def test_uneven_verify_and_ghost_shards_round_trip_without_reordering(self):
        for size in (2, 4, 8):
            for rows in (0, 1, 5, 6, 17, 144, 384):
                source = torch.arange(rows * 3).view(rows, 3)
                shards, owned = [], []
                for rank in range(size):
                    layout = CPDecodeQueryLayout(rows, rank, size)
                    shard = layout.select(source, value=-1)
                    self.assertEqual(shard.shape[0], layout.local_rows)
                    torch.testing.assert_close(
                        shard[: layout.actual_rows], source[layout.query_rows]
                    )
                    self.assertTrue((shard[layout.actual_rows :] == -1).all())
                    owned.extend(range(layout.start, layout.stop))
                    shards.append(shard)
                self.assertEqual(owned, list(range(rows)))
                torch.testing.assert_close(torch.cat(shards)[:rows], source)
                with patch(
                    "sglang.srt.layers.cp.interleave.attn_cp_interleave_gather",
                    return_value=torch.cat(shards),
                ):
                    for rank, shard in enumerate(shards):
                        layout = CPDecodeQueryLayout(rows, rank, size)
                        torch.testing.assert_close(layout.gather(shard), source)

    def test_only_opted_in_target_modes_use_dp(self):
        parallel = SimpleNamespace(
            cp_decode_attn_mode="dp", attn_cp_rank=2, attn_cp_size=4
        )
        with patch(
            "sglang.srt.layers.cp.cp_decode_attn_dp.get_parallel", return_value=parallel
        ):
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                layout = cp_decode_query_layout(SimpleNamespace(forward_mode=mode), 17)
                self.assertEqual((layout.start, layout.stop), (10, 15))
            for mode in (
                ForwardMode.EXTEND,
                ForwardMode.DRAFT_EXTEND_V2,
                ForwardMode.IDLE,
            ):
                self.assertIsNone(
                    cp_decode_query_layout(SimpleNamespace(forward_mode=mode), 17)
                )
            parallel.cp_decode_attn_mode = None
            self.assertIsNone(
                cp_decode_query_layout(
                    SimpleNamespace(forward_mode=ForwardMode.DECODE), 17
                )
            )

    def test_indexer_page_transform_and_output_views_use_the_rank_offset(self):
        """DeepSelect must not translate every rank's picks through page rows zero."""
        backend = object.__new__(DeepseekV4AttnBackend)
        pages = torch.arange(11 * 7).view(11, 7).int()
        lens = torch.arange(11).int()
        sparse = torch.full((11, 5), -1, dtype=torch.int32)
        paged = SimpleNamespace(page_table=pages, compressed_seq_lens=lens)
        backend.forward_metadata = SimpleNamespace(
            c1_indexer_metadata=paged,
            core_metadata=SimpleNamespace(sparse_page_indices=lambda _: sparse),
        )
        layout = CPDecodeQueryLayout(11, 2, 4)
        layer = SimpleNamespace(
            compress_ratio=1, indexer=object(), layer_id=20, freqs_cis=torch.ones(1)
        )
        values = torch.arange(11)
        inputs = backend._make_low_ratio_decode_indexer_inputs(
            layer,
            values[layout.query_rows],
            values[layout.query_rows],
            values[layout.query_rows],
            values[layout.query_rows],
            ForwardMode.TARGET_VERIFY,
            query_rows=layout.query_rows,
        )
        torch.testing.assert_close(inputs.paged_metadata.page_table, pages[6:9])
        torch.testing.assert_close(inputs.paged_metadata.compressed_seq_lens, lens[6:9])
        inputs.out_page_indices.fill_(42)
        self.assertTrue((sparse[6:9] == 42).all())
        self.assertTrue((sparse[:6] == -1).all())
        self.assertTrue((sparse[9:] == -1).all())
        self.assertIs(paged.page_table, pages)
        self.assertEqual(paged.page_table.shape[0], 11)

    def test_prepare_replicates_kv_and_sources_but_projects_only_local_queries(self):
        """Slicing KV with Q would leave other requests' replicated caches stale."""
        for rows, rank in ((11, 2), (1, 3)):
            for fused in (False, True):
                with self.subTest(rows=rows, rank=rank, fused=fused):
                    x = torch.arange(rows * 3).view(rows, 3).float()
                    positions = torch.arange(rows)
                    quantized = x + 100
                    cache = torch.full_like(x, -1)
                    seen = {}

                    def write_kv(value, pos, batch, backend, *, qkv_a):
                        cache.copy_(value)
                        torch.testing.assert_close(pos, positions)

                    def sources(**kwargs):
                        seen["x"] = kwargs["x"].clone()
                        seen["q_lora"] = kwargs["q_lora"].clone()

                    def q_a(value, *, qkv_a):
                        normalized = (qkv_a if qkv_a is not None else value) + 10
                        return normalized, normalized

                    layer = SimpleNamespace(
                        fuse_wqa_wkv=fused,
                        wqkv_a=lambda value: (value + 20, None),
                        _compute_q_a=q_a,
                        _compute_kv_to_cache=write_kv,
                        _compute_q_b=lambda q, pos, out: q + pos[:, None],
                        compress_ratio=1,
                    )
                    layout = CPDecodeQueryLayout(rows, rank, 4)
                    q, kv = MQALayer._forward_prepare_decode_dp(
                        layer,
                        x,
                        positions,
                        object(),
                        SimpleNamespace(forward_low_ratio_sources=sources),
                        layout,
                        None,
                        quantized,
                    )
                    torch.testing.assert_close(cache, quantized)
                    torch.testing.assert_close(seen["x"], x)
                    expected = quantized + (20 if fused else 0) + 10
                    torch.testing.assert_close(seen["q_lora"], expected)
                    if rows == 11:
                        expected = expected[6:9] + positions[6:9, None]
                    else:
                        expected = torch.zeros(1, 3)
                    torch.testing.assert_close(q, expected)
                    self.assertIsNone(kv)

    def test_local_flashmla_schedule_resets_without_changing_captured_q_buffer(self):
        def metadata():
            core = SimpleNamespace(
                copy_=lambda other: None,
                refresh_for_breakable_cuda_graph_replay_=lambda other: None,
            )
            return DSV4Metadata(core_attn_metadata=core, indexer_metadata=None)

        held, live = metadata(), metadata()
        captured = torch.zeros(3, 64, 128)
        held.q_pad_buffer, held.q_pad_captured = captured, True
        live.decode_dp_flashmla_metadata[1] = object()
        held.decode_dp_flashmla_metadata[2] = object()
        held.copy_(live)
        self.assertEqual(held.decode_dp_flashmla_metadata, {})
        self.assertIn(1, live.decode_dp_flashmla_metadata)
        for two_level in (False, True):
            held.decode_dp_flashmla_metadata[1] = object()
            held.refresh_for_breakable_cuda_graph_replay_(live, two_level=two_level)
            self.assertEqual(held.decode_dp_flashmla_metadata, {})
            self.assertIs(held.q_pad_buffer, captured)
            self.assertTrue(held.q_pad_captured)


if __name__ == "__main__":
    unittest.main()
