import pytest

from sglang.srt.debug_utils.comparator.output_types import report_sink


@pytest.fixture(autouse=True)
def _reset_report_sink() -> None:
    yield
    report_sink._reset()
