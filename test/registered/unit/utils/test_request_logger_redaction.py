"""Unit tests for request-log formatting: watermark secrets stay redacted, and the
redaction walk is bounded by max_length instead of the prompt length."""

import unittest
from unittest import mock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import request_logger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

SECRET = "super-secret-watermark-key"


def _request(num_ids: int) -> GenerateReqInput:
    return GenerateReqInput(
        input_ids=list(range(num_ids)),
        sampling_params={
            "max_new_tokens": 8,
            "watermark": {"enabled": True, "key": SECRET},
            "watermark_key": SECRET,
        },
    )


class _LongestListSpy:
    """Wraps redact_watermark_secrets and records the longest list it is handed."""

    def __init__(self):
        self.real = request_logger.redact_watermark_secrets
        self.longest = 0

    def __call__(self, value, **kwargs):
        self._scan(value)
        return self.real(value, **kwargs)

    def _scan(self, value):
        if isinstance(value, (list, tuple)):
            self.longest = max(self.longest, len(value))
        elif isinstance(value, dict):
            for v in value.values():
                self._scan(v)


class TestRequestLogRedaction(CustomTestCase):
    def test_string_format_redacts_secrets(self):
        for num_ids in (16, 100_000):
            out = request_logger._dataclass_to_string_truncated(_request(num_ids))
            self.assertNotIn(SECRET, out)
            self.assertIn("<redacted>", out)

    def test_transform_redacts_secrets(self):
        for num_ids in (16, 100_000):
            out = request_logger._transform_data_for_logging(_request(num_ids))
            self.assertNotIn(SECRET, repr(out))

    def test_long_list_inside_sampling_params_is_still_redacted(self):
        req = GenerateReqInput(
            input_ids=[1, 2, 3],
            sampling_params=[{"watermark_key": SECRET}] * 5000,
        )
        self.assertNotIn(SECRET, request_logger._dataclass_to_string_truncated(req))
        self.assertNotIn(SECRET, repr(request_logger._transform_data_for_logging(req)))

    def test_redaction_walk_is_bounded_at_every_log_level(self):
        # Each level's real (max_length, skip_names): levels 0/1 do not truncate
        # (max_length=1<<30) but omit input_ids, so the walk must skip it.
        for level in (0, 1, 2, 3):
            max_length, skip_names, _ = request_logger.RequestLogger(
                log_requests=True,
                log_requests_level=level,
                log_requests_format="text",
                log_requests_target=None,
            ).metadata
            bound = 2048 if max_length >= 1 << 30 else max_length
            for fn in (
                request_logger._dataclass_to_string_truncated,
                request_logger._transform_data_for_logging,
            ):
                spy = _LongestListSpy()
                with mock.patch.object(request_logger, "redact_watermark_secrets", spy):
                    fn(_request(100_000), max_length, skip_names)
                if level == 3:  # level 3 prints every input_id, so it must redact them
                    continue
                self.assertLessEqual(spy.longest, bound, f"level={level} {fn.__name__}")

    def test_printed_sampling_params_redacted_at_level_1(self):
        max_length, skip_names, _ = request_logger.RequestLogger(
            log_requests=True,
            log_requests_level=1,
            log_requests_format="text",
            log_requests_target=None,
        ).metadata
        req = _request(100_000)
        out = request_logger._dataclass_to_string_truncated(req, max_length, skip_names)
        self.assertIn("sampling_params", out)
        self.assertNotIn(SECRET, out)
        self.assertNotIn(
            SECRET,
            repr(
                request_logger._transform_data_for_logging(req, max_length, skip_names)
            ),
        )

    def test_truncated_output_keeps_head_and_tail(self):
        out = request_logger._dataclass_to_string_truncated(
            _request(100_000), max_length=8
        )
        self.assertIn("[0, 1, 2, 3] ... [99996, 99997, 99998, 99999]", out)


if __name__ == "__main__":
    unittest.main()
