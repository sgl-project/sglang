"""Unit tests for runner_utilization_report.classify_job.

Pure-logic tests (no GitHub API, stdlib only) so they run in the
runner-utilization workflow without installing dependencies:

    python -m unittest discover -s scripts/ci/utils -p 'test_runner_utilization_report.py'

Regression guard for the queue-time underestimation bug: jobs still
waiting in the runner queue (or still running) used to be dropped because
the old code required a runner_name and a completed_at, so multi-hour
8-gpu waits never showed up in max/avg queue time.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner_utilization_report as rur  # noqa: E402

NOW = datetime(2026, 5, 27, 21, 50, 56, tzinfo=timezone.utc)
CREATED = NOW - timedelta(hours=4)  # entered the queue 4h ago


def _job(**kw):
    base = {
        "name": "base-c-test-8-gpu-h200 / base-c-test-8-gpu-h200 (3)",
        "status": "completed",
        "conclusion": "success",
        "runner_name": "h200-wk03",
        "labels": ["self-hosted", "X64", "8-gpu-h200"],
        "created_at": CREATED.isoformat().replace("+00:00", "Z"),
        "started_at": None,
        "completed_at": None,
        "html_url": "https://github.com/o/r/actions/runs/1/job/2",
    }
    base.update(kw)
    return base


def _iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


class TestClassifyJob(unittest.TestCase):
    def test_queued_job_counts_ongoing_wait(self):
        """The core bug: a still-queued job reports started_at == created_at
        (placeholder) and no completed_at. Its wait must be now - created_at,
        not 0, and it must not be dropped."""
        job = _job(
            status="queued",
            runner_name="",
            started_at=_iso(CREATED),  # GitHub placeholder == created_at
            completed_at=None,
        )
        info = rur.classify_job(job, NOW)
        self.assertIsNotNone(info)
        self.assertAlmostEqual(info["queue_time"], 4 * 3600, delta=1)
        self.assertIsNone(info["start"])  # no runner occupied yet
        self.assertEqual(info["labels"], ["8-gpu-h200"])  # generic labels dropped

    def test_in_progress_job_counts_final_wait(self):
        """A running job's wait is final (started - created); old code dropped
        it for lacking completed_at."""
        started = CREATED + timedelta(hours=3)
        job = _job(status="in_progress", started_at=_iso(started), completed_at=None)
        info = rur.classify_job(job, NOW)
        self.assertIsNotNone(info)
        self.assertAlmostEqual(info["queue_time"], 3 * 3600, delta=1)
        self.assertEqual(info["start"], started)
        self.assertEqual(info["end"], NOW)  # still occupying the runner

    def test_completed_job_unchanged(self):
        started = CREATED + timedelta(minutes=30)
        completed = CREATED + timedelta(minutes=90)
        job = _job(started_at=_iso(started), completed_at=_iso(completed))
        info = rur.classify_job(job, NOW)
        self.assertAlmostEqual(info["queue_time"], 30 * 60, delta=1)
        self.assertAlmostEqual(info["duration"], 60 * 60, delta=1)
        self.assertEqual(info["end"], completed)

    def test_skipped_job_dropped(self):
        """Skipped / cancelled-before-start jobs never waited for a runner."""
        job = _job(status="completed", runner_name="", started_at=None)
        self.assertIsNone(rur.classify_job(job, NOW))

    def test_queued_without_created_dropped(self):
        job = _job(status="queued", runner_name="", created_at=None, started_at=None)
        self.assertIsNone(rur.classify_job(job, NOW))


class TestWallclockBusySeconds(unittest.TestCase):
    """Guards the merge that keeps per-host busy time <= window_seconds
    (and therefore per-label utilization <= 100%). Before this merge,
    job-interval overlap from GitHub timestamp slop / `filter=all`
    retries / in_progress `end=now` was double-counted -- busy pools
    rendered above 100% in the report."""

    WS = NOW - timedelta(hours=24)
    WE = NOW

    def _intv(self, h0: float, h1: float):
        """Build a job_info-shaped dict spanning [WS+h0h, WS+h1h]."""
        return {
            "start": self.WS + timedelta(hours=h0),
            "end": self.WS + timedelta(hours=h1),
        }

    def test_disjoint_intervals_sum_straight(self):
        # Two non-overlapping 1h intervals -> 2h busy.
        jobs = [self._intv(0, 1), self._intv(2, 3)]
        busy = rur._wallclock_busy_seconds(jobs, self.WS, self.WE)
        self.assertEqual(busy, 2 * 3600)

    def test_overlap_merged_not_double_counted(self):
        # Two 1h jobs overlapping by 30min -> wall-clock 1.5h, not 2h.
        # This is the canonical timestamp-slop / retry-row case.
        jobs = [self._intv(0, 1), self._intv(0.5, 1.5)]
        busy = rur._wallclock_busy_seconds(jobs, self.WS, self.WE)
        self.assertEqual(busy, 1.5 * 3600)

    def test_fully_contained_interval_absorbed(self):
        # A 30-min job entirely inside a 2h job -> still 2h busy.
        jobs = [self._intv(0, 2), self._intv(0.5, 1)]
        busy = rur._wallclock_busy_seconds(jobs, self.WS, self.WE)
        self.assertEqual(busy, 2 * 3600)

    def test_intervals_clipped_to_window(self):
        # Job spans before window_start and after window_end -> clipped to 24h.
        jobs = [
            {
                "start": self.WS - timedelta(hours=5),
                "end": self.WE + timedelta(hours=5),
            }
        ]
        busy = rur._wallclock_busy_seconds(jobs, self.WS, self.WE)
        self.assertEqual(busy, 24 * 3600)

    def test_busy_bounded_by_window(self):
        # Many overlapping jobs covering the whole window -> busy capped
        # at 24h regardless of how many overlap. This is the property
        # that guarantees utilization <= 100% per label.
        jobs = [self._intv(0, 24) for _ in range(50)]
        busy = rur._wallclock_busy_seconds(jobs, self.WS, self.WE)
        self.assertEqual(busy, 24 * 3600)

    def test_empty_jobs(self):
        self.assertEqual(rur._wallclock_busy_seconds([], self.WS, self.WE), 0.0)


class TestConcurrencyMergesPerRunnerSlop(unittest.TestCase):
    """The Avg Concurrent column had the same timestamp-slop overcount
    that Utilization did -- jobs on the SAME runner_name reporting
    overlapping intervals briefly pushed `current_running` above the
    real per-instant count, dragging Avg Concurrent above num_runners.
    These tests guard the per-runner merge inside
    calculate_concurrency_metrics."""

    WS = NOW - timedelta(hours=24)
    WE = NOW

    def _running_job(self, runner, h0, h1):
        return {
            "start": self.WS + timedelta(hours=h0),
            "end": self.WS + timedelta(hours=h1),
            "runner_name": runner,
            "created_at": self.WS + timedelta(hours=h0),
            "queue_end": self.WS + timedelta(hours=h0),
            "status": "pass",
            "queue_time": 0,
        }

    def test_same_runner_overlap_does_not_inflate_avg(self):
        # Two reported intervals on the same runner overlap by 0.5h
        # (timestamp slop / retry-row). Real wall-clock busy = 1.5h,
        # avg_concurrent over 24h = 1.5/24 = 0.0625. Without merge,
        # the sum-of-intervals view would give 2.0h busy and a higher
        # avg.
        jobs = [
            self._running_job("R1", 0, 1),
            self._running_job("R1", 0.5, 1.5),
        ]
        conc = rur.calculate_concurrency_metrics(jobs, self.WS, self.WE, num_runners=1)
        # peak should stay at 1: at any real instant only one job on R1
        self.assertEqual(conc["peak_concurrent"], 1)
        # avg = 1.5h / 24h
        self.assertAlmostEqual(conc["avg_concurrent"], 1.5 / 24, places=4)

    def test_different_runners_overlap_does_inflate_avg(self):
        # Same intervals but on TWO different runners -> legitimately
        # concurrent. Merge must NOT collapse across runners.
        jobs = [
            self._running_job("R1", 0, 1),
            self._running_job("R2", 0.5, 1.5),
        ]
        conc = rur.calculate_concurrency_metrics(jobs, self.WS, self.WE, num_runners=2)
        # 0.5h window of overlap when both runners were running
        self.assertEqual(conc["peak_concurrent"], 2)
        # avg = (1h + 1h) / 24h
        self.assertAlmostEqual(conc["avg_concurrent"], 2.0 / 24, places=4)

    def test_avg_concurrent_capped_by_num_runners_with_full_overlap_slop(self):
        # 10 reported intervals all on the same runner, all covering the
        # full window -- after merge, the runner is busy for 24h, so avg
        # concurrent = 1 (not 10). This is the property that prevents
        # the Concurrency column from rendering >100%.
        jobs = [self._running_job("R1", 0, 24) for _ in range(10)]
        conc = rur.calculate_concurrency_metrics(jobs, self.WS, self.WE, num_runners=1)
        self.assertEqual(conc["peak_concurrent"], 1)
        self.assertAlmostEqual(conc["avg_concurrent"], 1.0, places=4)


class TestConcurrencyHandlesQueuedJobs(unittest.TestCase):
    def test_queued_job_does_not_crash_and_counts_in_peak_queue(self):
        window_start = NOW - timedelta(hours=24)
        queued = rur.classify_job(
            _job(status="queued", runner_name="", started_at=_iso(CREATED)), NOW
        )
        ran = rur.classify_job(
            _job(
                started_at=_iso(CREATED + timedelta(hours=1)),
                completed_at=_iso(CREATED + timedelta(hours=2)),
            ),
            NOW,
        )
        conc = rur.calculate_concurrency_metrics(
            [queued, ran], window_start, NOW, num_runners=2
        )
        # Both jobs were waiting at CREATED before either started -> peak 2.
        self.assertEqual(conc["peak_queue"], 2)


class TestStatusAndFormatting(unittest.TestCase):
    def test_status_mapping_and_url(self):
        for conclusion, expected in (
            ("success", "pass"),
            ("failure", "fail"),
            ("timed_out", "fail"),
            ("cancelled", "cancel"),
        ):
            info = rur.classify_job(
                _job(
                    conclusion=conclusion,
                    started_at=_iso(CREATED + timedelta(minutes=5)),
                    completed_at=_iso(CREATED + timedelta(minutes=10)),
                ),
                NOW,
            )
            self.assertEqual(info["status"], expected)
            self.assertEqual(
                info["html_url"], "https://github.com/o/r/actions/runs/1/job/2"
            )

        self.assertEqual(
            rur.classify_job(
                _job(status="queued", runner_name="", started_at=_iso(CREATED)), NOW
            )["status"],
            "queued",
        )
        self.assertEqual(
            rur.classify_job(
                _job(
                    status="in_progress", started_at=_iso(CREATED + timedelta(hours=1))
                ),
                NOW,
            )["status"],
            "running",
        )

    def test_format_status_counts(self):
        self.assertEqual(rur.format_status_counts({}), "—")
        cell = rur.format_status_counts({"pass": 5, "queued": 2, "fail": 0})
        self.assertIn("✅5", cell)
        self.assertIn("⏳2", cell)
        self.assertNotIn("❌", cell)  # zero counts omitted

    def test_format_report_has_links_and_status(self):
        results = [
            {
                "label": "8-gpu-h200",
                "num_runners": 4,
                "effective_runners": 4,
                "num_jobs": 2,
                "total_active_hours": 1.0,
                "utilization_pct": 50.0,
                "avg_queue_min": 100.0,
                "max_queue_min": 264.0,
                "peak_concurrent": 1,
                "avg_concurrent": 0.5,
                "saturation_hours": 0.0,
                "saturation_pct": 0.0,
                "peak_queue": 2,
                "status_counts": {"pass": 1, "queued": 1},
            }
        ]
        url = "https://github.com/sgl-project/sglang/actions/runs/1/job/2"
        waits = [
            {
                "queue_time": 264 * 60,
                "status": "queued",
                "labels": ["8-gpu-h200"],
                "job_name": "base-c-test-8-gpu-h200 (3)",
                "html_url": url,
            }
        ]
        report = rur.format_report(results, 24, 0.0, longest_waits=waits)
        self.assertIn(rur.TZ_LABEL, report)  # generated-time stamped with tz
        self.assertIn("| Status |", report)  # new main-table column
        self.assertIn("Longest Queue Waits", report)
        self.assertIn(f"]({url})", report)  # clickable job link
        self.assertIn("264m", report)
        self.assertIn("⏳", report)


class TestChartBuilders(unittest.TestCase):
    @staticmethod
    def _info(created, queue_end, start=None, end=None):
        return {
            "created_at": created,
            "queue_end": queue_end,
            "start": start,
            "end": end,
        }

    def test_queue_timeline_tracks_ongoing_wait(self):
        # One job queued the whole 4h window (still waiting at the end).
        lj = {"8-gpu-h200": [self._info(CREATED, NOW)]}
        labels, series = rur.build_queue_timeline(lj, CREATED, NOW)
        self.assertEqual(len(series), 1)
        pool, vals = series[0]
        self.assertEqual(pool, "8-gpu-h200")
        self.assertEqual(len(vals), len(labels))  # x/y aligned for mermaid
        self.assertAlmostEqual(vals[0], 0, delta=2)  # ~0 at window start
        self.assertAlmostEqual(vals[-1], 240, delta=2)  # full 4h wait at end

    def test_queue_timeline_caps_and_sorts_by_peak(self):
        lj = {
            f"pool{i}": [self._info(CREATED, CREATED + timedelta(minutes=10 * i + 1))]
            for i in range(12)
        }
        _, series = rur.build_queue_timeline(lj, CREATED, NOW, max_series=8)
        self.assertEqual(len(series), 8)  # capped at palette size
        peaks = [max(v) for _, v in series]
        self.assertEqual(peaks, sorted(peaks, reverse=True))  # busiest first

    def test_load_buckets_counts_running_and_queued(self):
        ws = NOW - timedelta(hours=8)  # 8 hourly buckets
        start = ws + timedelta(hours=1)
        end = ws + timedelta(hours=3)
        lj = {"p": [self._info(ws, start, start=start, end=end)]}
        labels, pools = rur.build_load_buckets(lj, ws, NOW)
        self.assertEqual(len(labels), 8)
        lbl, running, queued = pools[0]
        self.assertEqual(lbl, "p")
        self.assertEqual(len(running), 8)
        self.assertEqual((queued[0], running[0]), (1, 0))  # hour 0: waiting
        self.assertEqual((queued[1], running[1]), (0, 1))  # hour 1: running
        self.assertEqual(running[3], 0)  # ended by hour 3

    def test_empty_inputs_safe(self):
        self.assertEqual(rur.build_queue_timeline({}, CREATED, NOW), ([], []))
        self.assertEqual(rur.build_load_buckets({}, CREATED, NOW), ([], []))

    def test_load_buckets_hybrid_includes_high_util_low_peak_pool(self):
        """The motivating case: a pool with only 1 runner serving 1 job
        the entire window (peak running+queued = 1) is invisible under
        pure peak-ranking but has 100% utilization. With the hybrid it
        must appear alongside the high-peak pools."""
        ws = NOW - timedelta(hours=8)
        lj = {}
        # 8 noisy pools each peaking at 5 concurrent jobs in one bucket.
        # Without the hybrid these claim the top 8 slots.
        for i in range(8):
            jobs = []
            for _ in range(5):
                start = ws + timedelta(hours=2)
                end = ws + timedelta(hours=2, minutes=30)
                jobs.append(self._info(start, start, start=start, end=end))
            lj[f"noisy-{i}"] = jobs
        # One "8-gpu-b200-like" pool: a single job pinning a single
        # runner for the whole window -> peak = 1, util = 100%.
        start = ws
        end = NOW
        lj["small-saturated"] = [self._info(start, start, start=start, end=end)]

        util = {f"noisy-{i}": 10.0 for i in range(8)}
        util["small-saturated"] = 100.0

        _, pools_no_util = rur.build_load_buckets(lj, ws, NOW, max_pools=8)
        names_no_util = [lbl for lbl, _, _ in pools_no_util]
        self.assertNotIn("small-saturated", names_no_util)  # legacy: cut

        _, pools_hybrid = rur.build_load_buckets(
            lj, ws, NOW, max_pools=8, utilization_by_label=util, max_total=12
        )
        names_hybrid = [lbl for lbl, _, _ in pools_hybrid]
        self.assertIn("small-saturated", names_hybrid)  # hybrid: surfaced
        self.assertLessEqual(len(names_hybrid), 12)

    def test_load_buckets_hybrid_peak_first_then_util_only(self):
        """Display order: pools that won by peak come first, then any
        pools only added via utilization. Within each group the order
        mirrors the rank that included them."""
        ws = NOW - timedelta(hours=4)
        lj = {
            "peaky-A": [
                self._info(
                    ws,
                    ws,
                    start=ws + timedelta(hours=1),
                    end=ws + timedelta(hours=2),
                )
            ]
            * 10,  # peak ~10
            "peaky-B": [
                self._info(
                    ws,
                    ws,
                    start=ws + timedelta(hours=1),
                    end=ws + timedelta(hours=2),
                )
            ]
            * 5,  # peak ~5
            "util-only": [
                self._info(ws, ws, start=ws, end=NOW)
            ],  # peak 1 but 100% util
        }
        util = {"peaky-A": 30.0, "peaky-B": 20.0, "util-only": 100.0}
        _, pools = rur.build_load_buckets(
            lj, ws, NOW, max_pools=2, utilization_by_label=util, max_total=12
        )
        names = [lbl for lbl, _, _ in pools]
        # peaky-A, peaky-B by peak (top 2). util-only added via util.
        self.assertEqual(names[:2], ["peaky-A", "peaky-B"])
        self.assertIn("util-only", names[2:])


if __name__ == "__main__":
    unittest.main()
