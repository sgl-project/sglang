"""FusedMoE.forward_impl must add ``output_add`` exactly once.

The shared-expert add is either folded into the runner's finalize epilogue or
done here as its own kernel. Getting that handoff wrong is silent: adding twice
or not at all still produces a tensor of the right shape, so pin the contract.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 16
TOKENS = 4


class _PassthroughDispatcher:
    def dispatch(self, *, hidden_states, topk_output):
        return StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

    def combine(self, combine_input):
        return combine_input


class _GatheringDispatcher(_PassthroughDispatcher):
    """Models a DP all-gather dispatch: extra rows in, local rows back out.

    The epilogue would then add output_add to the wrong rows, so the gate has to
    decline and leave the add to the caller side of the combine.
    """

    def dispatch(self, *, hidden_states, topk_output):
        return StandardDispatchOutput(
            hidden_states=hidden_states.repeat(2, 1),
            hidden_states_scale=None,
            topk_output=topk_output,
        )

    def combine(self, combine_input):
        return combine_input[:TOKENS]


class _StubLayer:
    """The slice of FusedMoE state that forward_impl reads."""

    def __init__(self, *, supports_fused_output_add, dispatcher=None):
        self.quant_method = object()
        self._dwdp_bound = False
        self.dispatcher = dispatcher or _PassthroughDispatcher()
        self.supports_fused_output_add = supports_fused_output_add
        self.hidden_size = HIDDEN
        self.params_dtype = torch.float32
        self.reduce_results = False
        self.moe_tp_size = 1
        self.moe_ep_size = 1
        self.seen_output_add = "unset"

    def _dispatch_with_pre_quant(self, hidden_states, topk_output, pre_quant_input):
        return FusedMoE._dispatch_with_pre_quant(
            self, hidden_states, topk_output, pre_quant_input
        )

    def run_moe_core(self, dispatch_output):
        # Stand in for a runner whose finalize epilogue folds output_add in.
        self.seen_output_add = dispatch_output.output_add
        out = dispatch_output.hidden_states * 2.0
        if dispatch_output.output_add is not None:
            out = out + dispatch_output.output_add
        return out


def _forward(layer, hidden_states, output_add):
    mod = "sglang.srt.layers.moe.fused_moe_triton.layer"
    with (
        patch(f"{mod}.get_parallel", return_value=SimpleNamespace(tp_group=None)),
        patch(f"{mod}.is_allocation_symmetric", return_value=False),
        patch(f"{mod}.use_symmetric_memory", lambda *a, **kw: nullcontext()),
    ):
        return FusedMoE.forward_impl(
            layer,
            hidden_states,
            topk_output=SimpleNamespace(),
            output_add=output_add,
        )


class TestFusedMoEOutputAddContract(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hidden_states = torch.randn(TOKENS, HIDDEN)
        self.output_add = torch.randn(TOKENS, HIDDEN)
        self.moe_out = self.hidden_states * 2.0

    def test_supporting_runner_gets_the_tensor_and_no_second_add(self):
        layer = _StubLayer(supports_fused_output_add=True)

        out = _forward(layer, self.hidden_states, self.output_add)

        self.assertIs(layer.seen_output_add, self.output_add)
        torch.testing.assert_close(out, self.moe_out + self.output_add)

    def test_non_supporting_runner_falls_back_to_an_explicit_add(self):
        layer = _StubLayer(supports_fused_output_add=False)

        out = _forward(layer, self.hidden_states, self.output_add)

        self.assertIsNone(layer.seen_output_add)
        torch.testing.assert_close(out, self.moe_out + self.output_add)

    def test_row_changing_dispatch_declines_the_fused_path(self):
        layer = _StubLayer(
            supports_fused_output_add=True, dispatcher=_GatheringDispatcher()
        )

        out = _forward(layer, self.hidden_states, self.output_add)

        self.assertIsNone(layer.seen_output_add)
        torch.testing.assert_close(out, self.moe_out + self.output_add)

    def test_dtype_mismatch_declines_the_fused_path(self):
        layer = _StubLayer(supports_fused_output_add=True)
        other_dtype_add = self.output_add.to(torch.bfloat16)

        out = _forward(layer, self.hidden_states, other_dtype_add)

        self.assertIsNone(layer.seen_output_add)
        torch.testing.assert_close(out, self.moe_out + other_dtype_add)

    def test_no_output_add_leaves_the_dispatch_untouched(self):
        layer = _StubLayer(supports_fused_output_add=True)

        out = _forward(layer, self.hidden_states, None)

        self.assertIsNone(layer.seen_output_add)
        torch.testing.assert_close(out, self.moe_out)


class TestFusedMoEWithLoRAOutputAdd(CustomTestCase):
    def test_active_lora_path_adds_output_add_once(self):
        wrapper = FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)
        torch.nn.Module.__init__(wrapper)
        wrapper.base_layer = SimpleNamespace(dispatcher=_PassthroughDispatcher())
        wrapper._quant_info = object()
        wrapper._lora_runner_backend = SimpleNamespace(
            is_experimental_sgl_trtllm=lambda: False,
            is_experimental_sgl_marlin=lambda: False,
        )
        wrapper._lora_runner = SimpleNamespace(
            run=lambda dispatch_output, quant_info, lora_info: (
                dispatch_output.hidden_states * 2.0
            )
        )
        hidden_states = torch.ones(TOKENS, HIDDEN)
        output_add = torch.full((TOKENS, HIDDEN), 10.0)

        out = wrapper._forward_with_lora(
            hidden_states, SimpleNamespace(), None, output_add=output_add
        )

        torch.testing.assert_close(out, torch.full((TOKENS, HIDDEN), 12.0))


if __name__ == "__main__":
    unittest.main()
