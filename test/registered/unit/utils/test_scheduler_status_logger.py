"""Unit tests for srt/utils/scheduler_status_logger.py — no server, no model.

Covers the SchedulerStatusLogger contracts:
- maybe_create's env gating (unset target -> None, target without
  --enable-metrics -> hard error),
- target-string parsing (comma split, whitespace strip, empty segments),
- maybe_dump's interval throttling and its timestamp bookkeeping,
- the scheduler.status event schema consumed by log scrapers.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _req(rid: str) -> SimpleNamespace:
    return SimpleNamespace(rid=rid)


def _make_logger(dump_interval: float = 60.0) -> SchedulerStatusLogger:
    logger = SchedulerStatusLogger(targets=["stdout"], dump_interval=dump_interval)
    return logger


class TestMaybeCreate(CustomTestCase):
    def test_unset_target_returns_none(self):
        """Negative gate: without SGLANG_LOG_SCHEDULER_STATUS_TARGET the
        scheduler must not pay for a status logger at all."""
        with envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.override(""):
            self.assertIsNone(SchedulerStatusLogger.maybe_create(enable_metrics=True))

    def test_target_without_metrics_fails_fast(self):
        """Config-misuse guard: setting the target env var while
        --enable-metrics is off would silently produce no dumps, so it must
        raise at startup instead."""
        with envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.override("stdout"):
            with self.assertRaises(ValueError):
                SchedulerStatusLogger.maybe_create(enable_metrics=False)

    def test_target_string_is_split_stripped_and_de_emptied(self):
        """Derived property of target parsing: comma-separated entries are
        stripped and empty segments dropped, so a trailing comma or spaces in
        the env var never creates a bogus log target."""
        with (
            envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.override(" stdout , /tmp/x ,, "),
            envs.SGLANG_LOG_SCHEDULER_STATUS_INTERVAL.override(5.0),
            patch(
                "sglang.srt.utils.scheduler_status_logger.create_log_targets"
            ) as mock_create,
        ):
            logger = SchedulerStatusLogger.maybe_create(enable_metrics=True)
        self.assertEqual(mock_create.call_args.kwargs["targets"], ["stdout", "/tmp/x"])
        self.assertEqual(logger.dump_interval, 5.0)


class TestMaybeDump(CustomTestCase):
    def test_dump_is_throttled_by_interval(self):
        """Derived property: dumps are rate-limited — a second call inside
        dump_interval is dropped, a call after the interval fires again, and
        the throttle clock only advances on an actual dump."""
        logger = _make_logger(dump_interval=10.0)
        batch = SimpleNamespace(reqs=[_req("r1")])
        with (
            patch("sglang.srt.utils.scheduler_status_logger.log_json") as mock_log_json,
            patch("sglang.srt.utils.scheduler_status_logger.time") as mock_time,
        ):
            mock_time.time.side_effect = [100.0, 105.0, 111.0]
            logger.maybe_dump(batch, waiting_queue=[])  # t=100: first dump
            logger.maybe_dump(batch, waiting_queue=[])  # t=105: throttled
            logger.maybe_dump(batch, waiting_queue=[])  # t=111: fires again

        self.assertEqual(mock_log_json.call_count, 2)
        self.assertEqual(logger.last_dump_time, 111.0)

    def test_dump_payload_schema(self):
        """Bookkeeping: log scrapers parse the scheduler.status event with
        running_rids/queued_rids lists; renaming any of them silently breaks
        the consumers."""
        logger = _make_logger(dump_interval=0.0)
        batch = SimpleNamespace(reqs=[_req("run-1"), _req("run-2")])
        waiting = [_req("wait-1")]
        with patch(
            "sglang.srt.utils.scheduler_status_logger.log_json"
        ) as mock_log_json:
            logger.maybe_dump(batch, waiting_queue=waiting)

        loggers_arg, event, payload = mock_log_json.call_args.args
        self.assertEqual(loggers_arg, logger.loggers)
        self.assertEqual(event, "scheduler.status")
        self.assertEqual(payload["running_rids"], ["run-1", "run-2"])
        self.assertEqual(payload["queued_rids"], ["wait-1"])
        self.assertEqual(payload["rank"], 0)


if __name__ == "__main__":
    unittest.main()
