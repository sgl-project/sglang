import json
import unittest

from sglang.srt.observability.request_logger import (
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MAX_LENGTH = 8
BIG = "x" * 10_000


class TestRequestLoggerListTruncation(CustomTestCase):
    def test_json_elided_list_truncates_elements(self):
        out = _transform_data_for_logging([BIG] * (MAX_LENGTH + 1), MAX_LENGTH)

        self.assertEqual(len(out), MAX_LENGTH + 1)
        self.assertEqual(out[MAX_LENGTH // 2], "...")
        self.assertLess(len(json.dumps(out)), 1000)

    def test_json_short_list_truncates_elements(self):
        out = _transform_data_for_logging([BIG] * MAX_LENGTH, MAX_LENGTH)

        self.assertLess(len(json.dumps(out)), 1000)

    def test_text_list_truncates_elements(self):
        for data in ([BIG] * MAX_LENGTH, [BIG] * (MAX_LENGTH + 1)):
            with self.subTest(length=len(data)):
                out = _dataclass_to_string_truncated(data, MAX_LENGTH)

                self.assertLess(len(out), 1000)

    def test_text_list_keeps_scalar_and_tuple_format(self):
        self.assertEqual(
            _dataclass_to_string_truncated([1, 2, 3], MAX_LENGTH), "[1, 2, 3]"
        )
        self.assertEqual(
            _dataclass_to_string_truncated(list(range(10)), MAX_LENGTH),
            "[0, 1, 2, 3] ... [6, 7, 8, 9]",
        )
        self.assertEqual(
            _dataclass_to_string_truncated(("a", "b"), MAX_LENGTH), "('a', 'b')"
        )

    def test_text_deeply_nested_list(self):
        # `str()` printed this before list elements were rendered one by one.
        depth = 500
        nested = 1
        for _ in range(depth):
            nested = [nested]

        self.assertEqual(
            _dataclass_to_string_truncated({"x": nested}, MAX_LENGTH),
            "{'x': " + "[" * depth + "1" + "]" * depth + "}",
        )

    def test_text_list_redacts_watermark_secrets(self):
        self.assertEqual(
            _dataclass_to_string_truncated(
                [{"watermark_key": "s", "watermark": {"key": "s"}}], 2048
            ),
            "[{'watermark_key': '<redacted>', 'watermark': {'key': '<redacted>'}}]",
        )


if __name__ == "__main__":
    unittest.main()
