import base64
import unittest
import zlib

from sglang.srt.entrypoints.openai.responses_adapters import (
    decode_reasoning_state,
    encode_reasoning_state,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestReasoningState(CustomTestCase):
    def test_round_trip(self):
        self.assertEqual(
            decode_reasoning_state(encode_reasoning_state("hello")), "hello"
        )

    def test_output_at_limit_is_accepted(self):
        blob = encode_reasoning_state("abcd")
        self.assertEqual(decode_reasoning_state(blob, max_output_size=4), "abcd")

    def test_output_over_limit_is_rejected(self):
        blob = encode_reasoning_state("abcde")
        self.assertIsNone(decode_reasoning_state(blob, max_output_size=4))

    def test_truncated_stream_is_rejected(self):
        raw = zlib.compress(b"hello")[:-1]
        blob = "sglang-reasoning-v1." + base64.urlsafe_b64encode(raw).decode("ascii")
        self.assertIsNone(decode_reasoning_state(blob))

    def test_trailing_stream_is_rejected(self):
        raw = zlib.compress(b"hello") + b"trailing"
        blob = "sglang-reasoning-v1." + base64.urlsafe_b64encode(raw).decode("ascii")
        self.assertIsNone(decode_reasoning_state(blob))


if __name__ == "__main__":
    unittest.main()
