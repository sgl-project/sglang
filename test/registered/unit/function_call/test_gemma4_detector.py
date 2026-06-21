import signal
from contextlib import contextmanager

import pytest

from sglang.srt.function_call.gemma4_detector import _parse_gemma4_array
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


@contextmanager
def _time_limit(seconds: int):
    # signal.alarm is POSIX-only; on other platforms run without a hard timeout
    # (the assertions still catch a wrong/corrupted parse).
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise(*_):
        raise TimeoutError("gemma4 parser did not terminate")

    signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def test_nested_array_string_element_with_bracket():
    # A ']' inside a quoted string element must not corrupt nested-array depth.
    with _time_limit(5):
        result = _parse_gemma4_array('[<|"|>a]b<|"|>,<|"|>c<|"|>],<|"|>tail<|"|>')
    assert result == [["a]b", "c"], "tail"]


def test_stray_closing_bracket_terminates():
    # A stray ']' must abort the array parse instead of looping forever.
    with _time_limit(5):
        result = _parse_gemma4_array("42,]trailing")
    assert result == [42]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
