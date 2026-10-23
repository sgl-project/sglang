"""Request logging must not traverse hidden token lists when watermarking is off."""

import dataclasses
import unittest

from sglang.srt.runtime_context import get_context
from sglang.srt.utils.request_logger import RequestLogger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _UnreadableList(list):
    def __iter__(self):
        raise AssertionError("logging traversed a skipped token list")


@dataclasses.dataclass
class _Request:
    rid: str
    input_ids: list
    sampling_params: dict


class TestRequestLogger(CustomTestCase):
    def test_disabled_watermark_does_not_read_hidden_tokens(self):
        """Level-1 Receive/Finish logs must leave skipped token payloads untouched."""
        req = _Request("r1", _UnreadableList([1, 2, 3]), {"temperature": 0.7})
        out = {"output_ids": _UnreadableList([4]), "meta_info": {"id": "r1"}}
        with get_context().override_server_args():
            for log_format in ("text", "json"):
                with self.subTest(log_format=log_format):
                    logger = RequestLogger(True, 1, log_format, None)
                    with self.assertLogs(logger.targets[0], level="INFO") as logs:
                        logger.log_received_request(req)
                        logger.log_finished_request(req, out)
                    self.assertEqual(len(logs.records), 2)
                    for record in logs.records:
                        self.assertIn("r1", record.getMessage())
                        self.assertNotIn("input_ids", record.getMessage())
                        self.assertNotIn("output_ids", record.getMessage())

    def test_watermark_redaction_follows_server_flag(self):
        """The flag controls redaction for native and raw OpenAI request logs."""
        params = {"watermark": {"key": "0123456789abcdef", "enabled": True}}
        req = _Request("r1", [1, 2, 3], params)
        out = {"meta_info": {"id": "r1"}}
        for enabled in (False, True):
            with get_context().override_server_args(enable_watermark=enabled):
                for log_format in ("text", "json"):
                    with self.subTest(enabled=enabled, log_format=log_format):
                        logger = RequestLogger(True, 1, log_format, None)
                        with self.assertLogs(logger.targets[0], level="INFO") as logs:
                            logger.log_received_request(req)
                            logger.log_finished_request(req, out)
                            logger.log_openai_received_request(params)
                        self.assertEqual(len(logs.records), 3)
                        for record in logs.records:
                            message = record.getMessage()
                            if enabled:
                                self.assertIn("<redacted>", message)
                                self.assertNotIn("0123456789abcdef", message)
                            else:
                                self.assertIn("0123456789abcdef", message)
                                self.assertNotIn("<redacted>", message)
        self.assertEqual(params["watermark"]["key"], "0123456789abcdef")


if __name__ == "__main__":
    unittest.main()
