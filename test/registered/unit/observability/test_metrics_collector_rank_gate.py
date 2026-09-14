"""Rank-gate tests for ``SchedulerMetricsCollector.init_new`` (issue #31896).

Under attention context parallelism every CP rank runs its own scheduler over
the same global requests, and with ``attn_tp_size == 1`` every CP rank has
``attn_tp_rank == 0``. Gating stats logging on ``attn_tp_rank`` alone therefore
made every CP rank export the same request gauges (``sglang:num_running_reqs``
and friends), inflating Prometheus sums by ``attn_cp_size``. The KV-cache event
export already gates on ``attn_cp_rank == 0``; these tests lock the same
predicate for request stats.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase


class _DummyCollector:
    """Stand-in so ``init_new`` runs without registering prometheus metrics."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _make_ps(**overrides) -> ParallelState:
    defaults = dict(dp_rank=None, moe_dp_rank=None)
    defaults.update(overrides)
    return ParallelState.trivial(**defaults)


class TestStatsLoggingRankGate(CustomTestCase):
    def _init_ctx(self, *, ps, enable_metrics=True, all_schedulers=False):
        override = get_context().override_server_args(
            enable_metrics=enable_metrics,
            enable_metrics_for_all_schedulers=all_schedulers,
            kv_events_config=None,
        )
        server_args = override.install()
        self.addCleanup(override.restore)
        with patch(
            "sglang.srt.observability.metrics_collector.resolve_collector_class",
            return_value=_DummyCollector,
        ):
            return SchedulerMetricsCollector.init_new(
                server_args=server_args,
                ps=ps,
                tp_rank=ps.tp_rank,
                pp_rank=ps.pp_rank,
                dp_rank=None,
                enable_priority_scheduling=False,
                enable_lora=False,
                enable_hierarchical_cache=False,
            )

    def test_cp_rank_zero_logs_and_exports(self):
        ctx = self._init_ctx(ps=_make_ps(attn_tp_rank=0, attn_cp_rank=0))
        self.assertTrue(ctx.is_stats_logging_rank)
        self.assertTrue(ctx.current_scheduler_metrics_enabled)

    def test_cp_nonzero_rank_does_not_log_or_export(self):
        # cp=8 with attn_tp_size == 1: tp_rank differs per CP rank while
        # attn_tp_rank == 0 on every one of them (issue #31896).
        ctx = self._init_ctx(
            ps=_make_ps(
                tp_rank=3, tp_size=8, attn_tp_rank=0, attn_cp_rank=3, attn_cp_size=8
            )
        )
        self.assertFalse(ctx.is_stats_logging_rank)
        self.assertFalse(ctx.current_scheduler_metrics_enabled)

    def test_enable_metrics_for_all_schedulers_still_exports_on_cp_ranks(self):
        ctx = self._init_ctx(
            ps=_make_ps(
                tp_rank=3, tp_size=8, attn_tp_rank=0, attn_cp_rank=3, attn_cp_size=8
            ),
            all_schedulers=True,
        )
        self.assertFalse(ctx.is_stats_logging_rank)
        self.assertTrue(ctx.current_scheduler_metrics_enabled)

    def test_metrics_disabled_never_exports(self):
        ctx = self._init_ctx(ps=_make_ps(), enable_metrics=False)
        self.assertTrue(ctx.is_stats_logging_rank)
        self.assertFalse(ctx.current_scheduler_metrics_enabled)
        self.assertIsNone(ctx.collector)


if __name__ == "__main__":
    unittest.main()
