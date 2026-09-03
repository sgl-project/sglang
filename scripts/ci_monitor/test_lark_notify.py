import unittest
from datetime import datetime, timedelta, timezone

import lark_notify as ln


def _job(
    name,
    conclusion="success",
    status="completed",
    labels=None,
    created=None,
    started=None,
):
    return {
        "name": name,
        "conclusion": conclusion,
        "status": status,
        "html_url": f"https://example.com/{name}",
        "labels": labels or [],
        "created_at": created,
        "started_at": started,
    }


def _runner(name, labels, status="online", busy=False):
    return {
        "name": name,
        "status": status,
        "busy": busy,
        "labels": [{"name": l} for l in labels],
    }


NOW = datetime(2026, 9, 2, 20, 0, tzinfo=timezone.utc)


class LabelTest(unittest.TestCase):
    def test_primary_cuda_label_ignores_aliases(self):
        self.assertEqual(
            ln.primary_cuda_label(
                ["self-hosted", "1-gpu-runner", "1-gpu-large", "1-gpu-h100"]
            ),
            "1-gpu-h100",
        )
        self.assertEqual(
            ln.primary_cuda_label(["8-gpu-h200", "8-gpu-h200-deepep"]), "8-gpu-h200"
        )
        self.assertIsNone(ln.primary_cuda_label(["linux-mi300-8gpu-sglang"]))
        self.assertIsNone(ln.primary_cuda_label(["1-gpu-runner"]))
        self.assertIsNone(ln.primary_cuda_label(["intel-bmg", "xeon-spr"]))


class CiStatusTest(unittest.TestCase):
    def _run(self, attempt=1, conclusion="failure"):
        return {
            "name": "Nightly Test (Nvidia)",
            "run_attempt": attempt,
            "conclusion": conclusion,
            "run_started_at": "2026-09-02T14:00:00Z",
            "updated_at": "2026-09-02T16:41:00Z",
            "head_sha": "2d799c28f4abcdef",
            "head_commit": {"message": "first line\nsecond line"},
            "html_url": "https://example.com/run",
        }

    def test_aggregator_jobs_are_ignored(self):
        jobs = [
            _job("a", "failure"),
            _job("check-all-jobs", "failure"),
            _job("pr-test-finish", "failure"),
            _job("check-changes", "success"),
        ]
        self.assertEqual(list(ln.failed_job_names(jobs)), ["a"])
        self.assertEqual(
            [j["name"] for j in jobs if ln.is_reportable_job(j)], ["a", "check-changes"]
        )

    def test_failed_conclusions(self):
        jobs = [_job("a", "failure"), _job("b", "timed_out"), _job("c", "cancelled")]
        self.assertEqual(sorted(ln.failed_job_names(jobs)), ["a", "b"])

    def test_diff_attempts(self):
        cur = ln.failed_job_names([_job("a", "failure"), _job("b", "failure")])
        prev = ln.failed_job_names([_job("b", "failure"), _job("d", "failure")])
        diff = ln.diff_attempts(cur, prev)
        self.assertEqual([j["name"] for j in diff["fixed"]], ["d"])
        self.assertEqual([j["name"] for j in diff["still"]], ["b"])
        self.assertEqual([j["name"] for j in diff["new"]], ["a"])

    def test_first_attempt_lists_failures_without_comparison(self):
        jobs = [
            _job("a", "failure"),
            _job("b", "success"),
            _job("c", "skipped"),
            _job("check-all-jobs", "failure"),
        ]
        card = ln.render_ci_status(self._run(), jobs, None)
        header = card["card"]["header"]
        self.assertEqual(header["template"], "red")
        self.assertEqual(
            header["title"]["content"],
            "Nightly Test (Nvidia): FAILED (1 failed / 2 jobs, 2h41m)",
        )
        body = card["card"]["elements"][0]["text"]["content"]
        self.assertIn("`2d799c28f` first line", body)
        self.assertIn("**Failed jobs (1)**", body)
        self.assertNotIn("second line", body)
        self.assertNotIn("Still failing", body)
        self.assertNotIn("check-all-jobs", body)
        buttons = card["card"]["elements"][1]["actions"]
        self.assertEqual([b["text"]["content"] for b in buttons], ["Run"])

    def test_first_attempt_passed(self):
        card = ln.render_ci_status(
            self._run(conclusion="success"), [_job("a"), _job("b")], None
        )
        self.assertEqual(card["card"]["header"]["template"], "green")
        self.assertEqual(
            card["card"]["header"]["title"]["content"],
            "Nightly Test (Nvidia): PASSED (2 jobs, 2h41m)",
        )

    def test_rerun_compares_with_previous_attempt(self):
        jobs = [_job("a", "success"), _job("b", "failure"), _job("c", "success")]
        prev_failed = ln.failed_job_names([_job("a", "failure"), _job("b", "failure")])
        card = ln.render_ci_status(self._run(attempt=2), jobs, prev_failed)
        header = card["card"]["header"]
        self.assertEqual(header["template"], "red")
        self.assertEqual(
            header["title"]["content"],
            "Rerun #2 - Nightly Test (Nvidia): FAILED (1 failed / 3 jobs, 2h41m)",
        )
        body = card["card"]["elements"][0]["text"]["content"]
        self.assertIn("**Fixed by rerun (1)**\n  - [a]", body)
        self.assertIn("**Still failing (1)**\n  - [b]", body)
        self.assertNotIn("New failures", body)
        buttons = card["card"]["elements"][1]["actions"]
        self.assertEqual(
            [(b["text"]["content"], b["url"]) for b in buttons],
            [
                ("Run", "https://example.com/run"),
                ("Attempt 1", "https://example.com/run/attempts/1"),
            ],
        )

    def test_rerun_all_green(self):
        prev_failed = ln.failed_job_names([_job("a", "failure")])
        card = ln.render_ci_status(
            self._run(attempt=3, conclusion="success"),
            [_job("a"), _job("b")],
            prev_failed,
        )
        self.assertEqual(card["card"]["header"]["template"], "green")
        self.assertTrue(
            card["card"]["header"]["title"]["content"].startswith("Rerun #3 - ")
        )
        self.assertIn(
            "Fixed by rerun (1)", card["card"]["elements"][0]["text"]["content"]
        )

    def test_job_list_truncation(self):
        jobs = [_job(f"j{i}", "failure") for i in range(20)]
        md = ln.list_jobs_md(jobs)
        self.assertEqual(md.count("\n") + 1, ln.MAX_LISTED_JOBS + 1)
        self.assertIn("and 5 more", md)


class RunnerHealthTest(unittest.TestCase):
    def _pools(self, offline):
        runners = [
            _runner(
                f"5090-d-runner-{i}",
                ["1-gpu-5090"],
                "offline" if i < offline else "online",
                busy=(i == 7),
            )
            for i in range(8)
        ] + [_runner("mi300-1", ["linux-mi300-8gpu-sglang"], "offline")]
        return ln.summarize_pools(runners)

    def test_summarize_pools_only_cuda(self):
        pools = self._pools(offline=3)
        self.assertEqual(list(pools), ["1-gpu-5090"])
        p = pools["1-gpu-5090"]
        self.assertEqual(
            (p["total"], p["online"], p["offline"], p["busy"]), (8, 5, 3, 1)
        )
        self.assertEqual(p["offline_names"], [f"5090-d-runner-{i}" for i in range(3)])

    def test_transitions(self):
        healthy = self._pools(offline=1)
        degraded = self._pools(offline=6)

        events, state = ln.plan_health_events(healthy, {}, NOW, 0.5, 1.0)
        self.assertEqual(events, [])
        self.assertEqual(state, {})

        events, state = ln.plan_health_events(degraded, state, NOW, 0.5, 1.0)
        self.assertEqual([e[0] for e in events], ["degraded"])
        self.assertEqual(state["1-gpu-5090"]["degraded_since"], "2026-09-02T20:00:00Z")

        # 15 minutes later: still degraded, no reminder yet
        events, state = ln.plan_health_events(
            degraded, state, NOW + timedelta(minutes=15), 0.5, 1.0
        )
        self.assertEqual(events, [])
        self.assertEqual(state["1-gpu-5090"]["degraded_since"], "2026-09-02T20:00:00Z")

        # 61 minutes later: hourly reminder
        events, state = ln.plan_health_events(
            degraded, state, NOW + timedelta(minutes=61), 0.5, 1.0
        )
        self.assertEqual([e[0] for e in events], ["still_degraded"])
        self.assertEqual(state["1-gpu-5090"]["last_notified"], "2026-09-02T21:01:00Z")

        # recovered: one event, state cleared
        events, state = ln.plan_health_events(
            healthy, state, NOW + timedelta(hours=2), 0.5, 1.0
        )
        self.assertEqual([e[0] for e in events], ["recovered"])
        self.assertEqual(events[0][3], NOW)
        self.assertEqual(state, {})

    def test_render_health_cards(self):
        pool = self._pools(offline=8)["1-gpu-5090"]
        card = ln.render_health_event("degraded", "1-gpu-5090", pool, NOW, NOW, "o/r")
        self.assertEqual(card["card"]["header"]["template"], "red")
        self.assertIn("DOWN", card["card"]["header"]["title"]["content"])
        self.assertIn(
            "5090-d-runner-{0,1,2,3,4,5,6,7}",
            card["card"]["elements"][0]["text"]["content"],
        )

        pool = self._pools(offline=5)["1-gpu-5090"]
        card = ln.render_health_event(
            "recovered",
            "1-gpu-5090",
            pool,
            NOW - timedelta(hours=6, minutes=29),
            NOW,
            "o/r",
        )
        self.assertEqual(card["card"]["header"]["template"], "green")
        self.assertIn(
            "Degraded for 6h29m", card["card"]["elements"][0]["text"]["content"]
        )

    def test_compress_runner_names(self):
        self.assertEqual(
            ln.compress_runner_names(
                ["h100-b", "5090-d-runner-3", "5090-d-runner-1", "5090-e-runner-0"]
            ),
            "5090-d-runner-{1,3}, 5090-e-runner-0, h100-b",
        )


class QueueDigestTest(unittest.TestCase):
    def test_summarize_queue(self):
        t0 = "2026-09-02T19:00:00Z"
        jobs = [
            _job(
                "a",
                labels=["1-gpu-h100", "1-gpu-runner"],
                created=t0,
                started="2026-09-02T19:01:00Z",
            ),
            _job(
                "b", labels=["1-gpu-h100"], created=t0, started="2026-09-02T19:11:00Z"
            ),
            _job(
                "c",
                labels=["1-gpu-h100"],
                status="queued",
                created="2026-09-02T19:40:00Z",
                started=None,
            ),
            _job(
                "d",
                labels=["linux-mi300-8gpu-sglang"],
                created=t0,
                started="2026-09-02T19:30:00Z",
            ),
            _job("e", labels=["1-gpu-5090"], created=t0, started=None),
        ]
        stats = ln.summarize_queue(jobs, NOW)
        self.assertEqual(sorted(stats), ["1-gpu-h100"])
        s = stats["1-gpu-h100"]
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["p50"], 60)
        self.assertEqual(s["max"], 660)
        self.assertEqual(s["queued_now"], 1)
        self.assertEqual(s["oldest_queued"], 20 * 60)

    def test_render_queue_digest_flags_slow(self):
        stats = {
            "1-gpu-h100": {
                "n": 10,
                "p50": 60,
                "p90": 45 * 60,
                "max": 3600,
                "queued_now": 2,
                "oldest_queued": 300,
            },
            "1-gpu-5090": {
                "n": 5,
                "p50": 10,
                "p90": 20,
                "max": 30,
                "queued_now": 0,
                "oldest_queued": None,
            },
        }
        card = ln.render_queue_digest(stats, 6, 30, "o/r")
        self.assertEqual(card["card"]["header"]["template"], "orange")
        body = card["card"]["elements"][0]["text"]["content"]
        self.assertTrue(body.startswith("**1-gpu-h100** (!)"))
        self.assertIn("queued now 2 (oldest 5m00s)", body)
        self.assertNotIn("1-gpu-5090** (!)", body)

    def test_fmt_duration_and_percentile(self):
        self.assertEqual(ln.fmt_duration(None), "-")
        self.assertEqual(ln.fmt_duration(59), "59s")
        self.assertEqual(ln.fmt_duration(61), "1m01s")
        self.assertEqual(ln.fmt_duration(3661), "1h01m")
        self.assertIsNone(ln.percentile([], 0.9))
        self.assertEqual(ln.percentile([5, 1, 3], 0.5), 3)
        self.assertEqual(ln.percentile(list(range(1, 11)), 0.9), 9)


if __name__ == "__main__":
    unittest.main()
