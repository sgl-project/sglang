"""The dual-stream MoE custom op must not let a cross-layer handoff escape
through its Tensor schema.

``dsv2_flashinfer_moe_dual_stream_graph`` is registered with a Tensor return
type, while ``forward_normal_dual_stream`` may hand back a
``MoeFinalizeHandoff``. ``ForwardFlags.scoped()`` overrides only the flags it
is given, so the decoder's ``defer_moe_finalize`` reaches inside the op unless
the op pins it off. The op is CUDA-only, so this is a GPU suite rather than a unit test.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-a", runner_config="1-gpu-small")


class _RecordingFusion:
    """Stands in for DeepseekV2MoE, recording the deferral the op published."""

    def __init__(self, handoff_if_deferring):
        self.handoff_if_deferring = handoff_if_deferring
        self.seen_defer = None

    def forward_normal_dual_stream(self, hidden_states):
        self.seen_defer = get_forward().defer_moe_finalize
        if self.seen_defer:
            # What the real MoE returns when the decoder's scope reaches in.
            return self.handoff_if_deferring
        return hidden_states + 1


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDsv2DualStreamOpContract(CustomTestCase):
    def test_registered_op_pins_the_deferral_off_and_returns_a_tensor(self):
        from sglang.srt.models.deepseek_v2 import (
            dsv2_flashinfer_moe_dual_stream_graph,
        )

        fusion = _RecordingFusion(handoff_if_deferring=object())
        hidden_states = torch.zeros(4, 8, device="cuda", dtype=torch.bfloat16)

        with patch(
            "sglang.srt.models.deepseek_v2.get_tc_piecewise_forward_context",
            return_value=SimpleNamespace(moe_fusions={0: fusion}),
        ):
            # The decoder's scope, which the op body runs inside.
            with get_forward().scoped(defer_moe_finalize=True):
                out = dsv2_flashinfer_moe_dual_stream_graph(
                    hidden_states, 0, True, False
                )
                # The op must not leak its pin back to the caller's scope.
                self.assertTrue(get_forward().defer_moe_finalize)

        self.assertIs(fusion.seen_defer, False)
        self.assertIsInstance(out, torch.Tensor)
        self.assertTrue(torch.equal(out, hidden_states + 1))

    def test_operands_still_carry_the_other_flags_into_the_op(self):
        """The pin must not disturb the flags the op deliberately republishes."""
        from sglang.srt.models.deepseek_v2 import (
            dsv2_flashinfer_moe_dual_stream_graph,
        )

        seen = {}

        class _FlagReader:
            def forward_normal_dual_stream(self, hidden_states):
                flags = get_forward()
                seen["fuse"] = flags.fuse_mlp_allreduce
                seen["scatter"] = flags.mlp_reduce_scatter
                return hidden_states

        hidden_states = torch.zeros(4, 8, device="cuda", dtype=torch.bfloat16)
        with patch(
            "sglang.srt.models.deepseek_v2.get_tc_piecewise_forward_context",
            return_value=SimpleNamespace(moe_fusions={0: _FlagReader()}),
        ):
            dsv2_flashinfer_moe_dual_stream_graph(hidden_states, 0, True, False)

        self.assertTrue(seen["fuse"])
        self.assertFalse(seen["scatter"])


if __name__ == "__main__":
    unittest.main()
