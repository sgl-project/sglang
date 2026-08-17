"""Unit tests for DeepSeek-V4 attention-sink TP routing."""

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


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        query = kwargs["q"]
        return query.new_zeros(query.shape[0], query.shape[1], 2)


class _AttentionHarness(deepseek_v4.MQALayer):
    def __init__(self, rank):
        torch.nn.Module.__init__(self)
        self.attn_tp_rank = rank
        self.attn_tp_size = 8
        self.n_heads = 128
        self.n_local_heads = 16
        self.head_dim = 8
        self.n_local_groups = 1
        self.o_lora_rank = 3
        self.qk_rope_head_dim = 2
        self.freqs_cis = torch.empty(0)
        self.compress_ratio = 4
        self.attn_mqa = SimpleNamespace(layer_id=0, v_head_dim=2)
        self.attn_sink = torch.nn.Parameter(torch.arange(128, dtype=torch.float32))
        self._attn_sink_local = None
        self.alt_streams = None
        self.dsa_enable_prefill_cp = False
        self.compressor = object()
        self.wo_a = SimpleNamespace(
            weight=torch.ones(
                self.n_local_groups,
                self.o_lora_rank,
                self.n_local_heads * 2,
            )
        )
        self.wo_b = lambda value: (value, None)

    def _forward_prepare(
        self,
        x,
        positions,
        forward_batch,
        attn_backend,
        q_out=None,
        x_quant=None,
    ):
        q_out.zero_()
        return q_out, None


class TestDeepseekV4AttentionSink(unittest.TestCase):
    def test_unified_backend_receives_exact_tp_local_sink(self):
        for rank in (0, 3, 7):
            with self.subTest(rank=rank):
                layer = _AttentionHarness(rank)
                backend = _RecordingBackend()
                forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

                with (
                    envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.override(False),
                    patch.object(env_gate, "is_unified_kv_triton", return_value=True),
                    patch.object(
                        deepseek_v4,
                        "get_attn_tp_context",
                        return_value=SimpleNamespace(input_scattered=True),
                    ),
                    patch.object(
                        deepseek_v4,
                        "get_parallel",
                        return_value=SimpleNamespace(tp_size=8),
                    ),
                    patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
                    patch.object(deepseek_v4, "fused_rope_inplace", return_value=None),
                    patch.object(deepseek_v4, "_FP8_WO_A_GEMM", False),
                    patch.object(deepseek_v4, "_is_gfx942_supported", False),
                    patch.object(deepseek_v4, "_is_hip", True),
                    patch.object(deepseek_v4, "_is_npu", False),
                ):
                    layer.forward(
                        torch.zeros(2, 4),
                        torch.arange(2),
                        forward_batch,
                    )

                actual = backend.calls[0]["attn_sink"]
                start = rank * layer.n_local_heads
                expected = layer.attn_sink[start : start + layer.n_local_heads]
                self.assertEqual(tuple(actual.shape), (layer.n_local_heads,))
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
