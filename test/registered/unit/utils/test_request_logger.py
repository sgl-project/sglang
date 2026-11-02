"""The request-level ``no_logs`` opt-out must suppress request logging."""

import os
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import List

from sglang.srt.utils.request_logger import RequestLogger
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


@dataclass
class _FakeReq:
    rid: str = "req-1"
    text: str = "secret prompt"
    input_ids: List[int] = field(default_factory=lambda: [1, 2, 3])
    no_logs: bool = False


@dataclass
class _ReqWithoutNoLogsField:
    """Mirrors ``EmbeddingReqInput``, which does not carry ``no_logs``."""

    rid: str = "req-2"
    text: str = "secret prompt"
    input_ids: List[int] = field(default_factory=lambda: [1, 2])


class TestNoLogsOptOut(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._logger = RequestLogger(
            log_requests=True,
            log_requests_level=3,
            log_requests_format="text",
            log_requests_target=[self._tmp.name],
        )
        self.addCleanup(self._logger.targets[0].handlers[0].close)

    def _log_contents(self) -> str:
        names = [f for f in os.listdir(self._tmp.name) if f.endswith(".log")]
        if not names:
            return ""
        with open(os.path.join(self._tmp.name, names[0]), encoding="utf-8") as fh:
            return fh.read()

    def _finished_out(self):
        return {"meta_info": {"e2e_latency": 0.01}}

    def test_received_request_with_no_logs_is_not_logged(self):
        self._logger.log_received_request(_FakeReq(no_logs=True))
        self.assertEqual(self._log_contents(), "")

    def test_finished_request_with_no_logs_is_not_logged(self):
        self._logger.log_finished_request(_FakeReq(no_logs=True), self._finished_out())
        self.assertEqual(self._log_contents(), "")

    def test_request_without_the_flag_still_logs_its_content(self):
        self._logger.log_received_request(_FakeReq(no_logs=False))
        self.assertIn("secret prompt", self._log_contents())

    def test_request_type_without_the_field_still_logs(self):
        self._logger.log_received_request(_ReqWithoutNoLogsField())
        self.assertIn("secret prompt", self._log_contents())


if __name__ == "__main__":
    unittest.main()
