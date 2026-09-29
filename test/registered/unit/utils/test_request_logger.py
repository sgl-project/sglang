import unittest
from unittest.mock import patch

import numpy as np

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils.request_logger import RequestLogger
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")


class TestRequestLoggerNumpyRows(unittest.TestCase):
    def test_finished_request_logs_numpy_rows_as_lists(self):
        """Finished-request logs write numpy rows exactly as the lists they stand for,
        in both formats, at every level, and past the level's truncation length."""
        obj = GenerateReqInput(rid="r", text="hi")
        for masks in ([[5, 3, 9], [7]] * 1500, [list(range(5000))]):
            logprobs = [[-1 / (token + 3) for token in row] for row in masks]
            for log_format in ("text", "json"):
                for level in (0, 1, 2, 3):
                    with self.subTest(
                        num_rows=len(masks), log_format=log_format, level=level
                    ):
                        request_logger = RequestLogger(
                            log_requests=True,
                            log_requests_level=level,
                            log_requests_format=log_format,
                            log_requests_target=None,
                        )
                        messages = []
                        for to_row in (list, np.array):
                            meta_info = {
                                "id": "r",
                                "output_token_sampling_mask": [
                                    to_row(row) for row in masks
                                ],
                                "output_token_sampling_logprobs": [
                                    to_row(row) for row in logprobs
                                ],
                            }
                            with (
                                patch("sglang.srt.utils.log_utils.datetime") as clock,
                                self.assertLogs(request_logger.targets[0]) as logs,
                            ):
                                clock.now.return_value.isoformat.return_value = "t"
                                request_logger.log_finished_request(
                                    obj, {"text": "hello", "meta_info": meta_info}
                                )
                            (record,) = logs.records
                            messages.append(record.getMessage())
                        from_lists, from_arrays = messages
                        self.assertEqual(from_arrays, from_lists)


if __name__ == "__main__":
    unittest.main()
