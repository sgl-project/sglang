import warnings

warnings.filterwarnings(
    "ignore", message="builtin type Swig.*", category=DeprecationWarning
)

import pytest

from sglang.srt.debug_utils.comparator.output_types import report_sink

collect_ignore_glob: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Unknown config option. asyncio_mode:pytest.PytestConfigWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:builtin type Swig.*:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Named tensors and all their associated APIs:UserWarning",
    )


@pytest.fixture(autouse=True)
def _reset_report_sink() -> None:
    yield
    report_sink._reset()
