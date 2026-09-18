import sys
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.admission_timing import parse_admission_wait
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("value", [None, "bad", "nan", "inf", "-1"])
def test_invalid_timing_is_zero(value):
    assert parse_admission_wait(value) == 0


@pytest.mark.parametrize(
    "role",
    [DisaggregationMode.NULL, DisaggregationMode.PREFILL, DisaggregationMode.DECODE],
)
@pytest.mark.parametrize("trusted_wait", [None, 0.0, 3.0])
@pytest.mark.parametrize("stream", [False, True])
def test_admission_metrics_are_separate_and_observed_once(role, trusted_wait, stream):
    collector = MagicMock()
    collector.labels = {}
    manager = NS(
        metrics_collector=collector,
        enable_priority_scheduling=False,
        disaggregation_mode=role,
        _request_has_grammar=lambda _: False,
    )
    time_stats = MagicMock()
    time_stats.get_first_token_latency.return_value = 2.0
    time_stats.get_e2e_latency.return_value = 8.0
    state = NS(
        obj=NS(stream=stream),
        ttft_observed=False,
        last_completion_tokens=0,
        admission_wait_seconds=trusted_wait,
        admission_ttft_observed=False,
        time_stats=time_stats,
        finished=False,
    )
    recv = NS(completion_tokens=[1], prompt_tokens=[20], cached_tokens=[10])
    TokenizerManager.collect_metrics(manager, state, recv, 0)
    state.finished = True
    recv.completion_tokens = [2]
    TokenizerManager.collect_metrics(manager, state, recv, 0)
    first = collector.histogram_admission_inclusive_ttft.labels.return_value.observe
    e2e = collector.histogram_admission_inclusive_e2e.labels.return_value.observe
    if role == DisaggregationMode.PREFILL or trusted_wait is None:
        first.assert_not_called()
        e2e.assert_not_called()
    else:
        first.assert_called_once_with(trusted_wait + 2.0)
        e2e.assert_called_once_with(trusted_wait + 8.0)
        collector.observe_time_to_first_token.assert_called_once_with(
            {}, 2.0, stream=stream
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
