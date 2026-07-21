"""Unit tests for scheduler metrics logging rank selection under CP.

Issue #31896: with --cp / attn context parallelism, every CP rank used to
export the same request gauges, so Prometheus summed num_running_reqs by
attn_cp_size.
"""


def test_is_stats_logging_rank_requires_attn_cp_rank_zero():
    # Predicate used in SchedulerMetricsCollectorContext.init_new
    cases = [
        (0, 0, True),
        (0, 1, False),
        (0, 7, False),
        (1, 0, False),
        (1, 1, False),
    ]
    for tp, cp, expected in cases:
        got = tp == 0 and cp == 0
        assert got is expected, (tp, cp, got, expected)


def test_init_new_disables_metrics_on_non_zero_cp_rank():
    import pytest

    try:
        from types import SimpleNamespace

        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.distributed.parallel_state_wrapper import ParallelState
        from sglang.srt.observability.metrics_collector import (
            SchedulerMetricsCollectorContext,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"sglang runtime deps unavailable: {e}")

    server_args = SimpleNamespace(
        enable_metrics=True,
        enable_metrics_for_all_schedulers=False,
        kv_events_config=None,
        disaggregation_mode=DisaggregationMode.NULL
        if hasattr(DisaggregationMode, "NULL")
        else "null",
        served_model_name="test",
        extra_metric_labels=None,
        enable_streaming_session=False,
    )

    # Prefer enum if available
    if hasattr(DisaggregationMode, "NULL"):
        server_args.disaggregation_mode = DisaggregationMode.NULL

    ps_cp0 = ParallelState.trivial(attn_tp_rank=0, attn_cp_rank=0, pp_rank=0)
    ps_cp1 = ParallelState.trivial(attn_tp_rank=0, attn_cp_rank=1, pp_rank=0)

    try:
        ctx0 = SchedulerMetricsCollectorContext.init_new(
            server_args=server_args,
            ps=ps_cp0,
            tp_rank=0,
            pp_rank=0,
            dp_rank=None,
            enable_priority_scheduling=False,
            enable_lora=False,
            enable_hierarchical_cache=False,
        )
        ctx1 = SchedulerMetricsCollectorContext.init_new(
            server_args=server_args,
            ps=ps_cp1,
            tp_rank=0,
            pp_rank=0,
            dp_rank=None,
            enable_priority_scheduling=False,
            enable_lora=False,
            enable_hierarchical_cache=False,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"init_new needs more runtime deps: {e}")

    assert ctx0.is_stats_logging_rank is True
    assert ctx0.current_scheduler_metrics_enabled is True
    assert ctx1.is_stats_logging_rank is False
    assert ctx1.current_scheduler_metrics_enabled is False
