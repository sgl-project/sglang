"""Unit tests for the Mamba/GDN retreat metrics (RFC sgl-project#40865).

Covers, without launching a server:
  - the budget policy helpers on Scheduler (absolute tail-max + Mamba pool
    watermark, both defaulting to a no-op);
  - the once-per-request first-admission accounting of the retreat gap;
  - that the Prometheus collector exposes the new counters/histogram;
  - the recorded mb16 fixture (real agentic-trace match records, first match
    per request) reproduces the measured aggregates from the RFC experiment.
"""

import os
import tempfile
import types
import unittest
from types import SimpleNamespace

# The collector imports prometheus_client conditioned on this env variable;
# it must be set before the first import below.
os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", tempfile.mkdtemp())

from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci = None
try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover - CI helpers unavailable upstream
    pass

if register_cpu_ci is not None:
    register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FIXTURE = os.path.join(
    os.path.dirname(__file__), "data", "mamba_retreat_mb16_fixture.jsonl"
)

_LABELS = {
    "model_name": "test-model",
    "engine_type": "standard",
    "tp_rank": 0,
    "pp_rank": 0,
    "moe_ep_rank": 0,
}


def _read_fixture():
    rows = []
    with open(FIXTURE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(__import__("json").loads(line))
    return rows


class _RecordingCollector:
    """Stands in for SchedulerMetricsCollector; records increment calls."""

    def __init__(self):
        self.calls = []

    def increment_mamba_retreat(self, gap_tokens, reprefill_tokens, collapsed, replay_admitted):
        self.calls.append(
            {
                "gap_tokens": gap_tokens,
                "reprefill_tokens": reprefill_tokens,
                "collapsed": collapsed,
                "replay_admitted": replay_admitted,
            }
        )


def _fake_scheduler(tail_max=0, watermark=1.0, mamba_usage=None):
    req_to_token_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(size=32) if mamba_usage is not None else None,
        mamba_allocator=(
            SimpleNamespace(available_size=lambda: int(32 * (1 - mamba_usage)))
            if mamba_usage is not None
            else None
        ),
    )
    sched = SimpleNamespace(
        server_args=SimpleNamespace(
            mamba_replay_tail_max=tail_max,
            mamba_replay_tail_watermark=watermark,
        ),
        req_to_token_pool=req_to_token_pool,
        tree_cache=SimpleNamespace(),  # no mamba_evictable_size attr
        metrics_reporter=SimpleNamespace(enable_metrics=True),
        metrics_collector=_RecordingCollector(),
        _mamba_pool_usage=lambda: mamba_usage,
    )
    # Bind the real unbound method so recording exercises the actual gate.
    sched._mamba_replay_budget_allows = types.MethodType(
        Scheduler._mamba_replay_budget_allows, sched
    )
    return sched


def _fake_req(full_kv_hit, accepted_len, host_hit=0, recorded=False):
    return SimpleNamespace(
        origin_input_ids=list(range(full_kv_hit + 8)),
        full_kv_hit_length=full_kv_hit,
        prefix_indices=list(range(accepted_len)),
        host_hit_length=host_hit,
        retreat_stats_recorded=recorded,
    )


class TestMambaRetreatBudgetPolicy(unittest.TestCase):
    def test_default_budget_is_disabled(self):
        sched = _fake_scheduler()
        self.assertFalse(Scheduler._mamba_replay_budget_allows(sched, 1024))

    def test_gap_over_budget_rejected(self):
        sched = _fake_scheduler(tail_max=4096)
        self.assertFalse(Scheduler._mamba_replay_budget_allows(sched, 4097))
        self.assertTrue(Scheduler._mamba_replay_budget_allows(sched, 4096))

    def test_watermark_blocks_when_pool_full(self):
        sched = _fake_scheduler(tail_max=65536, watermark=0.5, mamba_usage=0.75)
        self.assertFalse(Scheduler._mamba_replay_budget_allows(sched, 1024))
        sched_ok = _fake_scheduler(tail_max=65536, watermark=0.5, mamba_usage=0.25)
        self.assertTrue(Scheduler._mamba_replay_budget_allows(sched_ok, 1024))

    def test_no_mamba_pool_means_always_admitted(self):
        # Non-hybrid model: usage unknown -> policy preview counts as admitted.
        sched = _fake_scheduler(tail_max=4096)
        self.assertTrue(Scheduler._mamba_replay_budget_allows(sched, 1024))


class TestMambaRetreatRecording(unittest.TestCase):
    def test_recorded_once_per_request(self):
        sched = _fake_scheduler()
        req = _fake_req(full_kv_hit=8192, accepted_len=4096)
        Scheduler._record_mamba_retreat_stats(sched, req)
        Scheduler._record_mamba_retreat_stats(sched, req)
        self.assertEqual(len(sched.metrics_collector.calls), 1)

    def test_no_retreat_no_metric(self):
        sched = _fake_scheduler()
        req = _fake_req(full_kv_hit=8192, accepted_len=8192)
        Scheduler._record_mamba_retreat_stats(sched, req)
        self.assertEqual(sched.metrics_collector.calls, [])

    def test_gap_and_reprefill_accounting(self):
        sched = _fake_scheduler()
        # P=1000 tokens, full-KV hit H=900, accepted C=200: gap=700, the
        # re-prefill runs [200,1000) through all layers.
        req = _fake_req(full_kv_hit=900, accepted_len=200)
        req.origin_input_ids = list(range(1000))
        Scheduler._record_mamba_retreat_stats(sched, req)
        (call,) = sched.metrics_collector.calls
        self.assertEqual(call["gap_tokens"], 700)
        self.assertEqual(call["reprefill_tokens"], 800)
        self.assertFalse(call["collapsed"])
        self.assertFalse(call["replay_admitted"])  # budget disabled by default

    def test_collapse_when_accepted_zero(self):
        sched = _fake_scheduler()
        req = _fake_req(full_kv_hit=32768, accepted_len=0)
        Scheduler._record_mamba_retreat_stats(sched, req)
        (call,) = sched.metrics_collector.calls
        self.assertTrue(call["collapsed"])

    def test_metrics_disabled_skips_collector(self):
        sched = _fake_scheduler()
        sched.metrics_reporter.enable_metrics = False
        Scheduler._record_mamba_retreat_stats(sched, _fake_req(8192, 0))
        self.assertEqual(sched.metrics_collector.calls, [])


class TestMambaRetreatFixtureAggregates(unittest.TestCase):
    """The recorded mb16 cell must reproduce the measured RFC aggregates."""

    def test_first_admission_aggregates(self):
        rows = _read_fixture()
        self.assertEqual(len(rows), 151)  # one first-match record per request
        retreat = [r for r in rows if r["full_kv_hit"] > r["accepted"]]
        collapses = [r for r in retreat if r["accepted"] == 0]
        gap_sum = sum(r["full_kv_hit"] - r["accepted"] for r in retreat)
        reprefill_sum = sum(r["prompt_len"] - r["accepted"] for r in rows)
        self.assertEqual(len(retreat), 141)
        self.assertEqual(len(collapses), 141)
        self.assertEqual(gap_sum, 4204800)
        # 83% of the re-prefill volume is replay-eligible (report §3.1).
        self.assertAlmostEqual(gap_sum / reprefill_sum, 0.83, places=2)


class TestCollectorExposesRetreatMetrics(unittest.TestCase):
    def test_increment_mamba_retreat(self):
        from unittest import mock

        from sglang.srt.observability import metrics_collector as mc

        schedule_stub = SimpleNamespace(
            prefill_delayer_max_delay_passes=30,
            prefill_delayer_forward_passes_buckets=None,
            prefill_delayer_wait_seconds_buckets=None,
        )
        with mock.patch.object(
            mc, "exports_expert_balancedness_to_prometheus", return_value=False
        ), mock.patch.object(mc, "get_schedule", return_value=schedule_stub):
            collector = mc.SchedulerMetricsCollector(
                labels=_LABELS, server_args=SimpleNamespace()
            )
        collector.increment_mamba_retreat(
            gap_tokens=32768,
            reprefill_tokens=40960,
            collapsed=True,
            replay_admitted=False,
        )
        self.assertEqual(
            collector.mamba_retreat_requests_total.labels(**_LABELS)._value.get(), 1.0
        )
        self.assertEqual(
            collector.mamba_retreat_gap_tokens_total.labels(**_LABELS)._value.get(),
            32768.0,
        )
        self.assertEqual(
            collector.mamba_retreat_reprefill_tokens_total.labels(
                **_LABELS
            )._value.get(),
            40960.0,
        )
        self.assertEqual(
            collector.mamba_retreat_collapse_requests_total.labels(
                **_LABELS
            )._value.get(),
            1.0,
        )
        self.assertEqual(
            collector.mamba_replay_would_admit_total.labels(**_LABELS)._value.get(),
            0.0,
        )
        hist = collector.mamba_retreat_gap_histogram.labels(**_LABELS)
        self.assertEqual(hist._sum.get(), 32768.0)
        bucket_samples = [
            s for s in hist._child_samples() if s[0].endswith("_bucket")
        ]
        hit = [s for s in bucket_samples if s[1].get("le") == "32768.0"]
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0][2], 1.0)


if __name__ == "__main__":
    unittest.main()
