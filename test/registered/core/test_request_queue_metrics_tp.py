"""TP > 1 accounting for sglang:num_queue_rejected_requests_total.

RequestReceiver.recv_requests() broadcasts every request to every TP rank, so
each scheduler runs the queue-rejection paths for the same logical request. Only
ranks with current_scheduler_metrics_enabled may report, otherwise the exported
aggregate is inflated by tp_size. Single-rank behavior lives in
test_request_queue_validation.py.
"""

import asyncio
import os
import unittest
from typing import Dict, Tuple

import requests
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests,
)

register_cuda_ci(est_time=100, stage="base-b", runner_config="2-gpu-large")
register_amd_ci(est_time=120, suite="stage-b-test-2-gpu-large-amd")

TP_SIZE = 2
METRIC_NAME = "sglang:num_queue_rejected_requests_total"


class TestQueueRejectionMetricUnderTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--tp-size",
                str(TP_SIZE),
                "--max-running-requests",  # Enforce max request concurrency is 1
                "1",
                "--max-queued-requests",  # Enforce max queued request number is 1
                "1",
                "--attention-backend",
                "triton",
                "--enable-metrics",
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def _queue_full_by_rank(self) -> Dict[Tuple[str, ...], float]:
        """queue_full counter keyed by label set, so per-rank contributions stay visible."""
        response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(response.status_code, 200)

        return {
            tuple(sorted(sample.labels.items())): sample.value
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
            if sample.name == METRIC_NAME
            and sample.labels.get("reason") == "queue_full"
        }

    def test_rejections_are_counted_once_across_tp_ranks(self):
        """The aggregate must equal the number of logically rejected requests.

        Gating on enable_metrics instead of current_scheduler_metrics_enabled makes
        every TP rank report, so this delta would come out as tp_size * num_rejected.
        """
        before = self._queue_full_by_rank()

        status_codes = asyncio.run(
            send_concurrent_generate_requests(self.base_url, num_requests=10)
        )
        num_rejected = status_codes.count(503)
        self.assertGreater(num_rejected, 0, "expected the queue cap to reject requests")

        after = self._queue_full_by_rank()
        delta = sum(after.values()) - sum(before.values())
        self.assertEqual(delta, num_rejected)

        # Sharper form of the same check: exactly one rank may account for them.
        # Every rank still pre-seeds its own zeroed series, so the number of
        # exported series is not a substitute for this.
        reporting_ranks = [
            labels
            for labels, value in after.items()
            if value - before.get(labels, 0.0) > 0
        ]
        self.assertEqual(
            len(reporting_ranks),
            1,
            f"expected a single reporting rank, got {len(reporting_ranks)}: "
            f"{reporting_ranks}",
        )


if __name__ == "__main__":
    unittest.main()
