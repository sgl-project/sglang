import signal
import unittest
from contextlib import contextmanager

from sglang.srt.function_call.gemma4_detector import _parse_gemma4_array
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1.0, suite="base-a-test-cpu")


@contextmanager
def _time_limit(seconds: int):
    # signal.alarm is POSIX-only; on other platforms run without a hard timeout
    # (the assertions still catch a wrong/corrupted parse).
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise(_signum, _frame):
        raise TimeoutError("gemma4 parser did not terminate")

    signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


class TestGemma4ArrayParser(CustomTestCase):
    def test_nested_array_string_elements_with_brackets(self):
        cases = [
            (
                '[<|"|>a]b<|"|>,<|"|>c<|"|>],<|"|>tail<|"|>',
                [["a]b", "c"], "tail"],
            ),
            (
                '[<|"|>a[b<|"|>,<|"|>c<|"|>],<|"|>tail<|"|>',
                [["a[b", "c"], "tail"],
            ),
            ('[<|"|>a[b]c<|"|>,<|"|>d<|"|>]', [["a[b]c", "d"]]),
        ]
        for payload, expected in cases:
            with self.subTest(payload=payload):
                with _time_limit(5):
                    result = _parse_gemma4_array(payload)
                self.assertEqual(result, expected)

    def test_stray_closing_bracket_terminates(self):
        with _time_limit(5):
            result = _parse_gemma4_array("42,]trailing")
        self.assertEqual(result, [42])


if __name__ == "__main__":
    unittest.main()
