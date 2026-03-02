import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore", message="builtin type Swig.*", category=DeprecationWarning
)

# Add the test root to sys.path so `registered.debug_utils.comparator.testing_helpers`
# can be imported by test modules.
_TEST_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

import pytest

from sglang.srt.debug_utils.comparator.report_sink import report_sink

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
