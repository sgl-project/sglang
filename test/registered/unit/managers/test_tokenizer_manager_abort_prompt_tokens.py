"""Unit tests for prompt-token accounting of aborted requests in TokenizerManager.

A request aborted before its first forward (e.g. a PD bootstrap or KV transfer
failure) processed no prompt tokens, so it must not add its prompt length to
sglang:prompt_tokens_total.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
)
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
    SchedulerReqTimeStats,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PROMPT_TOKENS = 1000


def _make_manager():
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.metrics_collector = MagicMock()
    tm.metrics_collector.labels = {}
    tm.enable_priority_scheduling = False
    tm.disaggregation_mode = DisaggregationMode.NULL
    tm._request_has_grammar = lambda obj: False
    return tm


def _make_state():
    obj = SimpleNamespace(custom_labels=None, stream=False)
    state = ReqState([], True, MagicMock(), obj, APIServerReqTimeStats())
    state.ttft_observed = True
    state.last_completion_tokens = 0
    return state


def _make_output(finish_type, forward_entry_time, with_time_stats=True):
    time_stats = None
    if with_time_stats:
        stats = SchedulerReqTimeStats()
        stats.forward_entry_time = forward_entry_time
        time_stats = [stats]
    return SimpleNamespace(
        completion_tokens=[0],
        prompt_tokens=[PROMPT_TOKENS],
        cached_tokens=[0],
        finished_reasons=[{"type": finish_type}],
        time_stats=time_stats,
    )


class TestAbortPromptTokens(CustomTestCase):
    def _observed_prompt_tokens(self, recv_obj):
        tm = _make_manager()
        tm.collect_metrics(_make_state(), recv_obj, 0)
        args = tm.metrics_collector.observe_one_finished_request.call_args.args
        return args[1]

    def test_abort_before_forward_counts_no_prompt_tokens(self):
        self.assertEqual(self._observed_prompt_tokens(_make_output("abort", 0.0)), 0)

    def test_abort_after_forward_counts_prompt_tokens(self):
        self.assertEqual(
            self._observed_prompt_tokens(_make_output("abort", 1.0)), PROMPT_TOKENS
        )

    def test_normal_finish_counts_prompt_tokens(self):
        self.assertEqual(
            self._observed_prompt_tokens(_make_output("stop", 1.0)), PROMPT_TOKENS
        )

    def test_abort_without_time_stats_keeps_counting(self):
        recv_obj = _make_output("abort", 0.0, with_time_stats=False)
        self.assertEqual(self._observed_prompt_tokens(recv_obj), PROMPT_TOKENS)


if __name__ == "__main__":
    unittest.main()
