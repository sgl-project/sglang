"""CPU tests for the construction of the EP token dispatchers."""

import unittest
from unittest.mock import patch

from sglang.srt.layers.moe.token_dispatcher import nixl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


class TestNixlDispatcherBaseState(CustomTestCase):
    def test_hooks_register_on_a_constructed_dispatcher(self):
        """A constructed NIXL dispatcher must carry the BaseDispatcher state, so
        a post-dispatch hook registered on it sees the dispatch output."""
        with patch.object(
            nixl, "_NixlEPDispatcherImpl", lambda **kwargs: _RecordingImpl()
        ):
            dispatcher = nixl.NixlEPDispatcher(group=None, router_topk=8)
        self.assertEqual(dispatcher.quant_config, {})
        self.assertIsNone(dispatcher.overlap_args)

        seen = []
        dispatcher.register_post_dispatch_hook(
            lambda _dispatcher, dispatch_output: seen.append(dispatch_output)
        )
        output = dispatcher.dispatch(hidden_states=None, topk_output=None)

        self.assertEqual(output, "dispatch_output")
        self.assertEqual(seen, ["dispatch_output"])


if __name__ == "__main__":
    unittest.main()
