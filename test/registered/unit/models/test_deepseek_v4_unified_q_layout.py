"""CPU-only tests for DeepSeek-V4 unified-KV Q layout selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_UNIFIED_UNPADDED_MODES = (
    ForwardMode.EXTEND,
    ForwardMode.MIXED,
    ForwardMode.TARGET_VERIFY,
    ForwardMode.SPLIT_PREFILL,
)


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        query = kwargs["q"]
        return query.new_zeros(query.shape[0], query.shape[1], 2)


class _AttentionHarness(deepseek_v4.MQALayer):
    """Runs the production layout branch without constructing model weights."""

    def __init__(self, *, enable_prefill_cp=False, enable_multi_stream=False):
        torch.nn.Module.__init__(self)
        self.attn_tp_rank = 0
        self.attn_tp_size = 4
        self.n_heads = 64
        self.n_local_heads = 16
        self.head_dim = 8
        self.n_local_groups = 1
        self.o_lora_rank = 3
        self.qk_rope_head_dim = 2
        self.freqs_cis = torch.empty(0)
        self.compress_ratio = 4
        self.attn_mqa = SimpleNamespace(layer_id=0, v_head_dim=2)
        self.attn_sink = torch.zeros(self.n_heads)
        self._attn_sink_local = None
        self.alt_streams = [object()] if enable_multi_stream else None
        self._multi_stream_bs_limit = 8
        self.dsa_enable_prefill_cp = enable_prefill_cp
        self.compressor = object()
        self.wo_a = SimpleNamespace(
            weight=torch.ones(
                self.n_local_groups,
                self.o_lora_rank,
                self.n_local_heads * 2,
            )
        )
        self.wo_b = lambda value: (value, None)
        self.prepare_calls = []

    def _prepare_query(self, path, x, q_out):
        self.prepare_calls.append((path, q_out))
        if q_out is None:
            query = x.new_zeros(x.shape[0], self.n_local_heads, self.head_dim)
        else:
            q_out.zero_()
            query = q_out
        return query

    def _forward_prepare(
        self,
        x,
        positions,
        forward_batch,
        attn_backend,
        q_out=None,
        x_quant=None,
    ):
        return self._prepare_query("single", x, q_out), None

    def _forward_prepare_multi_stream(
        self,
        x,
        positions,
        forward_batch,
        attn_backend,
        q_out=None,
        x_quant=None,
    ):
        return self._prepare_query("multi", x, q_out)

    def _forward_prepare_multi_stream_hip(
        self,
        x,
        positions,
        forward_batch,
        attn_backend,
        q_out=None,
        x_quant=None,
    ):
        return self._prepare_query("multi_hip", x, q_out)


class TestDeepseekV4UnifiedQLayout(unittest.TestCase):
    def _run_forward(
        self,
        mode,
        *,
        unified,
        multi_stream=False,
        prefill_cp=False,
        cp_active=False,
        hip=False,
    ):
        layer = _AttentionHarness(
            enable_prefill_cp=prefill_cp,
            enable_multi_stream=multi_stream,
        )
        backend = _RecordingBackend()
        forward_batch = SimpleNamespace(forward_mode=mode)
        x = torch.zeros(2, 4)
        positions = torch.arange(2)

        with (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.override(multi_stream),
            patch.object(env_gate, "is_unified_kv_triton", return_value=unified),
            patch.object(
                deepseek_v4,
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=True),
            ),
            patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            patch.object(deepseek_v4, "get_is_capture_mode", return_value=multi_stream),
            patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=cp_active),
            patch.object(deepseek_v4, "is_in_breakable_cuda_graph", return_value=False),
            patch.object(
                deepseek_v4,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=layer.attn_tp_size),
            ),
            patch.object(deepseek_v4, "fused_rope_inplace", return_value=None),
            patch.object(deepseek_v4, "_FP8_WO_A_GEMM", False),
            patch.object(deepseek_v4, "_is_gfx942_supported", False),
            patch.object(deepseek_v4, "_is_hip", hip),
            patch.object(deepseek_v4, "_is_npu", False),
        ):
            output = layer.forward(x, positions, forward_batch)

        self.assertEqual(output.shape, (2, layer.o_lora_rank))
        self.assertEqual(len(layer.prepare_calls), 1)
        self.assertEqual(len(backend.calls), 1)
        return layer, backend, layer.prepare_calls[0]

    def test_unified_prefill_modes_use_contiguous_local_head_query(self):
        for mode in _UNIFIED_UNPADDED_MODES:
            with self.subTest(mode=mode.name):
                layer, backend, (_, q_out) = self._run_forward(mode, unified=True)

                self.assertIsNone(q_out)
                query = backend.calls[0]["q"]
                self.assertEqual(query.shape, (2, layer.n_local_heads, layer.head_dim))
                self.assertTrue(query.is_contiguous())

    def test_unified_decode_keeps_64_head_padding(self):
        layer, backend, (_, q_out) = self._run_forward(ForwardMode.DECODE, unified=True)

        self.assertIsNotNone(q_out)
        self.assertIs(backend.calls[0]["q"], q_out)
        self.assertEqual(q_out.shape, (2, layer.n_local_heads, layer.head_dim))
        self.assertEqual(q_out.stride(), (64 * layer.head_dim, layer.head_dim, 1))
        self.assertFalse(q_out.is_contiguous())

    def test_non_unified_modes_keep_padded_query_behavior(self):
        for mode in ForwardMode:
            with self.subTest(mode=mode.name):
                layer, backend, (_, q_out) = self._run_forward(mode, unified=False)

                backend_query = backend.calls[0]["q"]
                self.assertIsNotNone(q_out)
                self.assertFalse(q_out.is_contiguous())
                self.assertEqual(backend_query.shape, (2, 64, layer.head_dim))
                self.assertTrue(backend_query.is_contiguous())

    def test_multi_stream_and_prefill_cp_routes_reach_unpadded_layout(self):
        for cp_active, expected_path in ((False, "multi_hip"), (True, "single")):
            with self.subTest(cp_active=cp_active):
                layer, backend, (path, q_out) = self._run_forward(
                    ForwardMode.EXTEND,
                    unified=True,
                    multi_stream=True,
                    prefill_cp=True,
                    cp_active=cp_active,
                    hip=True,
                )

                self.assertEqual(path, expected_path)
                self.assertIsNone(q_out)
                query = backend.calls[0]["q"]
                self.assertEqual(query.shape, (2, layer.n_local_heads, layer.head_dim))
                self.assertTrue(query.is_contiguous())


if __name__ == "__main__":
    unittest.main()
