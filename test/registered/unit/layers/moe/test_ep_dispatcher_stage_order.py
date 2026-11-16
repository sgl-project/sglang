"""CPU tests for the dispatch/combine stage order of the EP token dispatchers."""

import unittest

from sglang.srt.layers.moe.token_dispatcher import deepep, mooncake, nixl, pplx
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_WRAPPERS = (
    (deepep.DeepEPDispatcher, deepep._Stage),
    (mooncake.MooncakeEPDispatcher, mooncake._Stage),
    (nixl.NixlEPDispatcher, nixl._Stage),
    (pplx.PplxDispatcher, pplx._Stage),
)


class _RecordingImpl:
    """Stands in for the all-to-all library behind a staged dispatcher."""

    def __init__(self):
        self.calls = []

    def dispatch_a(self, hidden_states, topk_output):
        self.calls.append("dispatch_a")
        return ("dispatch_state",)

    def dispatch_b(self, state):
        self.calls.append("dispatch_b")
        return "dispatch_output"

    def combine_a(self, hidden_states, topk_ids, topk_weights):
        self.calls.append("combine_a")
        return ("combine_state",)

    def combine_b(self, state):
        self.calls.append("combine_b")
        return "combined"


def _build(dispatcher_cls, stage_cls):
    # The constructors need the all-to-all library and a process group.
    dispatcher = dispatcher_cls.__new__(dispatcher_cls)
    BaseDispatcher.__init__(dispatcher)
    dispatcher.deepep_mode = DeepEPMode.LOW_LATENCY
    dispatcher._low_latency_dispatcher = _RecordingImpl()
    dispatcher._stage = stage_cls.INITIAL
    return dispatcher


_COMBINE_INPUT = ("hidden_states", "topk_ids", "topk_weights")


class TestEPDispatcherStageOrder(CustomTestCase):
    def test_out_of_order_stage_call_is_rejected(self):
        """A stage method called out of order must assert before it reaches the
        all-to-all library, on every staged dispatcher."""
        for dispatcher_cls, stage_cls in _WRAPPERS:
            with self.subTest(dispatcher=dispatcher_cls.__name__):
                dispatcher = _build(dispatcher_cls, stage_cls)
                dispatcher.dispatch_a(hidden_states=None, topk_output=None)
                with self.assertRaises(AssertionError):
                    dispatcher.dispatch_a(hidden_states=None, topk_output=None)
                with self.assertRaises(AssertionError):
                    dispatcher.combine_a(combine_input=_COMBINE_INPUT)
                self.assertEqual(
                    dispatcher._low_latency_dispatcher.calls, ["dispatch_a"]
                )

    def test_ordered_round_trip_returns_to_initial(self):
        """dispatch_a, dispatch_b, combine_a, combine_b must leave the dispatcher
        ready for the next forward pass of the layer."""
        for dispatcher_cls, stage_cls in _WRAPPERS:
            with self.subTest(dispatcher=dispatcher_cls.__name__):
                dispatcher = _build(dispatcher_cls, stage_cls)
                for _ in range(2):
                    dispatcher.dispatch_a(hidden_states=None, topk_output=None)
                    self.assertEqual(dispatcher.dispatch_b(), "dispatch_output")
                    dispatcher.combine_a(combine_input=_COMBINE_INPUT)
                    self.assertEqual(dispatcher.combine_b(), "combined")
                self.assertEqual(dispatcher._stage, stage_cls.INITIAL)
                self.assertEqual(
                    dispatcher._low_latency_dispatcher.calls,
                    ["dispatch_a", "dispatch_b", "combine_a", "combine_b"] * 2,
                )


if __name__ == "__main__":
    unittest.main()
