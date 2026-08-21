"""Unit tests for the sjf (shortest-job-first with aging) schedule policy."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(rid, input_len, matched=0, entry_time=0.0):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(range(input_len)),
        num_matched_prefix_tokens=matched,
        time_stats=SimpleNamespace(wait_queue_entry_time=entry_time),
    )


NOW = 10_000.0
PERF_COUNTER = "sglang.srt.managers.schedule_policy.time.perf_counter"


class TestSortByShortestJob(CustomTestCase):
    def test_orders_by_input_length(self):
        queue = [_req("long", 8192), _req("short", 50), _req("mid", 512)]
        with patch(PERF_COUNTER, return_value=NOW):
            SchedulePolicy._sort_by_shortest_job(queue)

        self.assertEqual([r.rid for r in queue], ["short", "mid", "long"])

    def test_uncached_length_is_the_job_size(self):
        # 8192 input with 8100 cached beats a fully-uncached 512 request
        queue = [_req("cold-512", 512), _req("warm-8192", 8192, matched=8100)]
        with patch(PERF_COUNTER, return_value=NOW):
            SchedulePolicy._sort_by_shortest_job(queue)

        self.assertEqual([r.rid for r in queue], ["warm-8192", "cold-512"])

    def test_aging_prevents_starvation(self):
        # an 8192-token request that has waited 10s outranks a fresh
        # 50-token request (8192 - 1024*10 < 50)
        queue = [
            _req("fresh-short", 50, entry_time=NOW),
            _req("old-long", 8192, entry_time=NOW - 10.0),
        ]
        with patch(PERF_COUNTER, return_value=NOW):
            SchedulePolicy._sort_by_shortest_job(queue)

        self.assertEqual([r.rid for r in queue], ["old-long", "fresh-short"])

    def test_aging_rate_parameter_scales_credit(self):
        # at 8192 tok/s aging, a 1s-old long request already overtakes
        queue = [
            _req("fresh-short", 50, entry_time=NOW),
            _req("old-long", 8192, entry_time=NOW - 1.001),
        ]
        with patch(PERF_COUNTER, return_value=NOW):
            SchedulePolicy._sort_by_shortest_job(queue, aging_tokens_per_s=8192.0)

        self.assertEqual([r.rid for r in queue], ["old-long", "fresh-short"])

    def test_unset_entry_time_gets_no_aging_credit(self):
        queue = [
            _req("unset-long", 8192, entry_time=0.0),
            _req("short", 50, entry_time=NOW),
        ]
        with patch(PERF_COUNTER, return_value=NOW):
            SchedulePolicy._sort_by_shortest_job(queue)

        self.assertEqual([r.rid for r in queue], ["short", "unset-long"])


if __name__ == "__main__":
    unittest.main()
