import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    Mamba2AttnBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch
from sglang.srt.models import falcon_h1
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Capture:
    """Stands in for a breakable prefill-graph capture: keeps the eager breaks
    recorded between captured segments."""

    def __init__(self):
        self.cuda_graph = SimpleNamespace(_break_fns=[])

    def _end_current_segment(self):
        pass

    def _begin_new_segment(self):
        pass


def _batch(marker=0.0):
    mode = SimpleNamespace(is_idle=lambda: False, is_target_verify=lambda: False)
    return SimpleNamespace(forward_mode=mode, marker=marker)


def _layer_and_backend(calls):
    """A decoder layer with its mixers stubbed: attention returns zeros, the
    Mamba2 backend records the batch it ran with and writes its input plus
    that batch's marker into the output."""

    def mamba_forward(mixer, hidden_states, output, **kwargs):
        calls.append(kwargs["forward_batch"])
        output.copy_(hidden_states + kwargs["forward_batch"].marker)

    linear = MagicMock(spec=Mamba2AttnBackend)
    linear.forward.side_effect = mamba_forward
    backend = MagicMock(spec=HybridLinearAttnBackend)
    backend.linear_attn_backend = linear

    cls = falcon_h1.FalconH1HybridAttentionDecoderLayer
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    passthrough = SimpleNamespace(
        prepare=lambda h, forward_batch: h, finish=lambda h, forward_batch: h
    )
    layer.attn_boundary = layer.ffn_boundary = passthrough
    layer.self_attention = lambda positions, hidden_states, forward_batch: (
        torch.zeros_like(hidden_states)
    )
    layer.feed_forward = lambda h, forward_batch: h
    layer.attention_in_multiplier = layer.attn_out_multiplier = 1.0
    layer.ssm_in_multiplier, layer.ssm_out_multiplier = 2.0, 1.0
    layer.mamba, layer.layer_id, layer.mup_vector = MagicMock(), 3, None
    return layer, backend


class TestFalconH1MambaInBreakablePrefillGraph(CustomTestCase):
    """The Mamba2 extend reads each request's state slot and chunk metadata. In a
    breakable prefill-graph capture it must be an eager break that reads the
    batch live at replay; captured inside a segment, every replay reuses the
    capture-time batch (wrong outputs, state written to the capture-time slot)."""

    def test_capture_records_the_mamba_extend_as_a_break_run_with_the_live_batch(self):
        calls = []
        layer, backend = _layer_and_backend(calls)
        capture = _Capture()
        token = bcg._current_capture_var.set(capture)
        try:
            with (
                patch.object(falcon_h1, "get_attn_backend", return_value=backend),
                enable_breakable_cuda_graph(),
            ):
                layer.forward(None, torch.ones(4, 8), _batch())
        finally:
            bcg._current_capture_var.reset(token)
        self.assertEqual(calls, [], "the Mamba2 extend ran inside a captured segment")
        self.assertEqual(len(capture.cuda_graph._break_fns), 1)

        # Each replay runs the break with the batch live then, into the buffer
        # the rest of the captured layer reads.
        (brk,) = capture.cuda_graph._break_fns
        output = brk.args[-1]
        for marker in (1.0, 5.0):
            live = _batch(marker)
            with (
                patch.object(falcon_h1, "get_attn_backend", return_value=backend),
                set_forward_batch(live),
            ):
                brk()
            self.assertIs(calls[-1], live)
            torch.testing.assert_close(output, torch.full((4, 8), 2.0 + marker))
        self.assertEqual(len(calls), 2)

    def test_outside_a_breakable_graph_the_extend_runs_inline(self):
        calls = []
        layer, backend = _layer_and_backend(calls)
        batch = _batch()
        with patch.object(falcon_h1, "get_attn_backend", return_value=backend):
            out = layer.forward(None, torch.ones(4, 8), batch)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], batch)
        # Attention contributes zeros; Mamba echoes its scaled input.
        torch.testing.assert_close(out, torch.full((4, 8), 2.0))


if __name__ == "__main__":
    unittest.main()
