"""DiT attention break points write their outputs into buffers allocated
before the break, so a replay refreshes the tensors the next captured segment
reads."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)


class _CountingImpl:
    def __init__(self):
        self.calls = 0

    def preprocess_qkv(self, qkv, _attn_metadata):
        return qkv

    def postprocess_output(self, output, _attn_metadata):
        return output

    def forward(self, q, k, v, attn_metadata=None):
        self.calls += 1
        return torch.full_like(q, float(self.calls))


def _attention(cls):
    attention = cls.__new__(cls)
    nn.Module.__init__(attention)
    attention.attn_impl = _CountingImpl()
    attention.backend = AttentionBackendEnum.FA
    attention.is_cross_attention = False
    attention.sp_attention_mode = "ulysses"
    attention.sp_attention_mode_is_auto = False
    return attention


class TestAttentionBCGOutputs(unittest.TestCase):
    def _capture_and_replay(self, attention, *args, **kwargs):
        graph = bcg.BreakableCUDAGraph()
        capture = SimpleNamespace(
            cuda_graph=graph,
            _end_current_segment=lambda: None,
            _begin_new_segment=lambda: None,
        )
        with (
            patch.object(layer, "is_in_breakable_cuda_graph", return_value=True),
            patch.object(
                layer,
                "get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch.object(layer, "get_sp_parallel_rank", return_value=0),
            patch.object(layer, "get_sp_world_size", return_value=1),
            patch.object(
                layer, "sequence_model_parallel_all_to_all_4D", lambda x, **_: x
            ),
            patch.object(layer, "sequence_model_parallel_all_gather", lambda x, **_: x),
        ):
            token = bcg._current_capture_var.set(capture)
            try:
                captured = attention(*args, **kwargs)
            finally:
                bcg._current_capture_var.reset(token)
            self.assertEqual(len(graph._break_fns), 1)
            self.assertIsNone(graph._break_fns[0]())
        return captured

    def test_local_attention_replays_into_captured_output(self):
        q = torch.zeros(1, 4, 2, 3)
        output = self._capture_and_replay(_attention(layer.LocalAttention), q, q, q)
        self.assertTrue(torch.all(output == 1))

    def test_ulysses_attention_replays_into_captured_tuple(self):
        q = torch.zeros(1, 4, 2, 3)
        replicated_q = torch.zeros(1, 2, 2, 3)
        attention = _attention(layer.UlyssesAttention)
        output, replicated_output = self._capture_and_replay(attention, q, q, q)
        self.assertTrue(torch.all(output == 1))
        self.assertIsNone(replicated_output)

        output, replicated_output = self._capture_and_replay(
            attention,
            q,
            q,
            q,
            replicated_q=replicated_q,
            replicated_k=replicated_q,
            replicated_v=replicated_q,
        )
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(replicated_output.shape, replicated_q.shape)
        self.assertTrue(torch.all(output == 2))
        self.assertTrue(torch.all(replicated_output == 2))


if __name__ == "__main__":
    unittest.main()
