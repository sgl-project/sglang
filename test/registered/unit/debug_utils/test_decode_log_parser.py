import unittest

from sglang.srt.debug_utils.log_parser import parse
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecodeLogParser(CustomTestCase):
    def test_queue_count_with_and_without_extra_metrics(self):
        prefix = (
            "[2026-04-07 13:24:13 TP0] Decode batch, #running-req: 10, "
            "#token: 485248, token usage: 0.94, cuda graph: True, "
            "gen throughput (token/s): 259.82, #queue-req: "
        )
        lines = [
            prefix + "0",
            prefix + "12, est. decode TFLOPS/s (per GPU): 1.23",
        ]

        self.assertEqual(parse("\n".join(lines))["queue_req"].to_list(), [0, 12])


if __name__ == "__main__":
    unittest.main()
