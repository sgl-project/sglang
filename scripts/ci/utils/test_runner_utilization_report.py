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
        self.assertIn("| Status |", report)  # new main-table column
        self.assertIn("Longest Queue Waits", report)
        self.assertIn(f"]({url})", report)  # clickable job link
        self.assertIn("264m", report)
        self.assertIn("⏳", report)


class TestGetWorkflowRuns(unittest.TestCase):
    """Regression guard for the run-listing truncation bugs.

    Two distinct failure modes are covered:
    - the original silent 50-page cap (5000 runs) on an UNfiltered listing
      while ~18k runs fire per busy 24h — fixed by server-side `created`
      filtering, with honest truncation reporting when a budget is hit;
    - the GitHub API's 1000-result cap on any `created`-FILTERED listing
      (page 11 comes back empty despite a larger total_count) — fixed by
      walking the range's upper bound down past each cap.
    """

    def _fake_gh_api(self, all_runs):
        """Emulate the runs-listing API over `all_runs` (newest-first),
        including the 1000-result-per-filtered-query cap."""
        calls = []

        def fake(args):
            path = args[0]
            calls.append(path)
            page = int(path.split("&page=")[1].split("&")[0])
            created = path.split("&created=")[1]
            if created.startswith(">="):
                lo, hi = created[2:], None
            else:
                lo, hi = created.split("..")
            matching = [
                r
                for r in all_runs
                if r["created_at"] >= lo and (hi is None or r["created_at"] <= hi)
            ]
            offset = (page - 1) * 100
            # The API serves at most 1000 results per filtered query even
            # though total_count reports the full match count.
            page_runs = matching[offset : offset + 100] if offset < 1000 else []
            return {"total_count": len(matching), "workflow_runs": page_runs}

        return fake, calls

    def _make_runs(self, n):
        """n runs, newest-first, one per minute."""
        return [
            {
                "id": i,
                "created_at": _iso(NOW - timedelta(minutes=i + 1)),
            }
            for i in range(n)
        ]

    def _run(self, all_runs, **kw):
        fake, calls = self._fake_gh_api(all_runs)
        orig = rur.run_gh_command
        rur.run_gh_command = fake
        try:
            result = rur.get_workflow_runs("o/r", since=NOW - timedelta(hours=24), **kw)
        finally:
            rur.run_gh_command = orig
        return result, calls

    def test_uses_created_filter_and_stops_when_exhausted(self):
        (runs, truncated), calls = self._run(self._make_runs(130))
        self.assertEqual(len(runs), 130)
        self.assertFalse(truncated)
        self.assertEqual(len(calls), 2)
        self.assertIn("created=>=", calls[0])

    def test_walks_past_the_1000_result_cap(self):
        # 1327 runs in-window (the exact live-validation failure): a single
        # filtered query serves only the newest 1000; the cursor walkdown
        # must fetch the remaining 327 via a narrowed created range.
        (runs, truncated), calls = self._run(self._make_runs(1327))
        self.assertEqual(len(runs), 1327)
        self.assertEqual(len({r["id"] for r in runs}), 1327)  # deduped
        self.assertFalse(truncated)
        # Second sweep queries a bounded range, not the open-ended filter.
        self.assertTrue(any(".." in c.split("&created=")[1] for c in calls))

    def test_reports_truncation_at_request_budget(self):
        (runs, truncated), _ = self._run(self._make_runs(1000), max_pages=3)
        self.assertEqual(len(runs), 300)
        self.assertTrue(truncated)


class TestPreWindowJobFiltering(unittest.TestCase):
    """Lookback runs bring in jobs that finished before the window started.

    Those jobs must be droppable by comparing their latest activity
    (busy end or queue end) against window_start — kept only when they
    touch the window. This mirrors the inline filter in
    calculate_utilization.
    """

    def _latest_activity(self, info):
        return max(t for t in (info["end"], info["queue_end"]) if t is not None)

    def test_finished_pre_window_job_is_droppable(self):
        window_start = NOW - timedelta(hours=24)
        info = rur.classify_job(
            _job(
                created_at=_iso(NOW - timedelta(hours=30)),
                started_at=_iso(NOW - timedelta(hours=29)),
                completed_at=_iso(NOW - timedelta(hours=26)),
            ),
            NOW,
        )
        self.assertLess(self._latest_activity(info), window_start)

    def test_job_spanning_window_start_is_kept(self):
        window_start = NOW - timedelta(hours=24)
        info = rur.classify_job(
            _job(
                created_at=_iso(NOW - timedelta(hours=30)),
                started_at=_iso(NOW - timedelta(hours=26)),
                completed_at=_iso(NOW - timedelta(hours=20)),
            ),
            NOW,
        )
        self.assertGreaterEqual(self._latest_activity(info), window_start)

    def test_still_queued_pre_window_job_is_kept(self):
        # queue_end anchors to `now` for still-queued jobs, so an old run's
        # still-waiting job always touches the window.
        window_start = NOW - timedelta(hours=24)
        info = rur.classify_job(
            _job(
                status="queued",
                runner_name="",
                created_at=_iso(NOW - timedelta(hours=30)),
                started_at=_iso(NOW - timedelta(hours=30)),
                completed_at=None,
            ),
            NOW,
        )
        self.assertGreaterEqual(self._latest_activity(info), window_start)


class TestCoverageWarning(unittest.TestCase):
    def test_banner_when_coverage_below_requested(self):
        report = rur.format_report([], 24, 0.0, coverage_hours=9.3)
        self.assertIn("Coverage warning", report)
        self.assertIn("9.3h", report)

    def test_no_banner_at_full_coverage(self):
        report = rur.format_report([], 24, 0.0, coverage_hours=24)
        self.assertNotIn("Coverage warning", report)
        report = rur.format_report([], 24, 0.0)  # default: full coverage
        self.assertNotIn("Coverage warning", report)


if __name__ == "__main__":
    unittest.main()
