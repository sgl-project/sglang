"""Decode query shards must reconstruct the batch, including empty ranks."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DSV4DecodeDPMetadata,
    DSV4Metadata,
)
from sglang.srt.layers.cp.cp_decode_attn_dp import CPDecodeQueryLayout
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _AttentionForwardHarness(MQALayer):
    """Exercise real forward dispatch with CPU stand-ins for GPU projections."""

    def __init__(self, *, is_dsv41, decode_dp):
        torch.nn.Module.__init__(self)
        self.is_dsv41 = is_dsv41
        self.use_cp_decode_attn_dp = decode_dp
        self.alt_streams = None
        self.compress_ratio = 4
        self.q_head_norm = True
        self.attn_tp_size = 1
        self.n_local_heads = 1
        self.head_dim = 3
        self.attn_sink = torch.nn.Parameter(torch.zeros(1))
        self.attn_mqa = SimpleNamespace(layer_id=0, v_head_dim=3)

    def _forward_prepare(self, x, *args, **kwargs):
        return x.unsqueeze(1), None

    def _forward_prepare_decode_dp(
        self, x, positions, forward_batch, attn_backend, layout, q_out, x_quant
    ):
        return layout.select(x).unsqueeze(1), None

    def _project_wo_a(self, o, **kwargs):
        return o.flatten(start_dim=1)

    def _project_wo_b(self, o, defer_all_reduce):
        return o


class TestCPDecodeDP(CustomTestCase):
    def _forward_with_backend(self, x, backend, mode, *, is_dsv41=True, layout=None):
        layer = _AttentionForwardHarness(
            is_dsv41=is_dsv41, decode_dp=layout is not None
        )
        module = "sglang.srt.models.deepseek_v4"
        with (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.override(False),
            envs.SGLANG_NPU_USE_MULTI_STREAM.override(False),
            patch(f"{module}.get_attn_backend", return_value=backend),
            patch(
                f"{module}.get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            patch(f"{module}.cp_decode_query_layout", return_value=layout),
            patch(f"{module}.is_in_breakable_cuda_graph", return_value=False),
            patch(f"{module}._is_npu", layout is None),
            patch(f"{module}._is_cuda", False),
            patch(f"{module}._is_hip", False),
            patch(f"{module}._is_gfx95_supported", False),
            patch(
                "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            return layer.forward(
                x, torch.arange(x.shape[0]), SimpleNamespace(forward_mode=mode)
            )

    def test_non_dp_forward_accepts_strict_backend_signature(self):
        """Ascend-style backends reject even query_layout=None."""

        def forward(
            q,
            k,
            v,
            layer,
            forward_batch,
            *,
            compress_ratio=0,
            attn_sink=None,
            save_kv_cache=True,
        ):
            return q * 2

        x = torch.arange(12, dtype=torch.float32).view(4, 3)
        backend = SimpleNamespace(forward_metadata=None, forward=forward)
        for is_dsv41 in (False, True):
            for mode in (
                ForwardMode.EXTEND,
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
            ):
                with self.subTest(is_dsv41=is_dsv41, mode=mode):
                    output = self._forward_with_backend(
                        x, backend, mode, is_dsv41=is_dsv41
                    )
                    torch.testing.assert_close(output, x * 2)

    def test_dp_forward_passes_layout_and_gathers_query_rows(self):
        def forward(
            q,
            k,
            v,
            layer,
            forward_batch,
            *,
            compress_ratio,
            attn_sink,
            save_kv_cache,
            query_layout,
        ):
            return q * query_layout.size

        x = torch.arange(12, dtype=torch.float32).view(4, 3)
        backend = SimpleNamespace(forward_metadata=None, forward=forward)
        layout = CPDecodeQueryLayout(rows=4, rank=1, size=2)
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with (
                self.subTest(mode=mode),
                patch(
                    "sglang.srt.layers.cp.interleave.attn_cp_interleave_gather",
                    side_effect=lambda local: torch.cat((x[:2] * 2, local)),
                ),
            ):
                output = self._forward_with_backend(x, backend, mode, layout=layout)
                torch.testing.assert_close(output, x * 2)

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
