"""Decode query shards must reconstruct the batch, including empty ranks."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DSV4DecodeDPMetadata,
    DSV4Metadata,
)
from sglang.srt.layers.cp.cp_decode_attn_dp import CPDecodeQueryLayout
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCPDecodeDP(CustomTestCase):
    def test_tp_attention_skips_query_dp_layout_and_configuration(self):
        layer = MQALayer.__new__(MQALayer)
        torch.nn.Module.__init__(layer)
        x = torch.arange(6).view(2, 3)
        metadata = DSV4Metadata(None, None, real_num_tokens=1)
        layer._forward_real_rows = lambda x, *args, **kwargs: x[:1]
        module = "sglang.srt.models.deepseek_v4"
        with (
            patch(
                f"{module}.get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            patch(
                f"{module}.get_attn_backend",
                return_value=SimpleNamespace(forward_metadata=metadata),
            ),
            patch(
                "sglang.srt.layers.cp.cp_decode_attn_dp.get_parallel",
                side_effect=AssertionError("TP queried the DP configuration"),
            ),
            patch(
                f"{module}.cp_decode_query_layout",
                side_effect=AssertionError("TP constructed a query-DP layout"),
            ),
        ):
            output = layer.forward(x, torch.arange(2), SimpleNamespace())
        torch.testing.assert_close(output, x[:1])

    def test_only_dp_metadata_tracks_query_schedules(self):
        state = []
        core = SimpleNamespace(
            copy_=lambda other: state.append(("copy", other)),
            refresh_for_breakable_cuda_graph_replay_=lambda other: state.append(
                ("refresh", other)
            ),
        )
        plain = DSV4Metadata(core, None)
        self.assertFalse(hasattr(plain, "decode_dp_flashmla_metadata"))
        dp = DSV4DecodeDPMetadata(core, None)
        dp.decode_dp_flashmla_metadata[1] = object()
        dp.copy_(plain)
        self.assertEqual(dp.decode_dp_flashmla_metadata, {})
        dp.decode_dp_flashmla_metadata[2] = object()
        dp.refresh_for_breakable_cuda_graph_replay_(plain)
        self.assertEqual(dp.decode_dp_flashmla_metadata, {})
        self.assertEqual(state, [("copy", core), ("refresh", core)])

    def test_uneven_verify_and_ghost_shards_round_trip_without_reordering(self):
        size = 4
        for rows in (1, 17, 144):
            source = torch.arange(rows * 3).view(rows, 3)
            shards = []
            for rank in range(size):
                layout = CPDecodeQueryLayout(rows, rank, size)
                shard = layout.select(source, value=-1)
                shards.append(shard)
            with patch(
                "sglang.srt.layers.cp.interleave.attn_cp_interleave_gather",
                return_value=torch.cat(shards),
            ):
                for rank, shard in enumerate(shards):
                    layout = CPDecodeQueryLayout(rows, rank, size)
                    torch.testing.assert_close(layout.gather(shard), source)


if __name__ == "__main__":
    unittest.main()
