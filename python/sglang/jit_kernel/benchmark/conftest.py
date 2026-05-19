from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run benchmark cases marked pytest.mark.slow (full cartesian sweep).",
    )
    parser.addoption(
        "--bench-full",
        action="store_true",
        default=False,
        help="Alias for --runslow.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: full-cartesian benchmark case; deselected unless --runslow / --bench-full is passed.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runslow") or config.getoption("--bench-full"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow or --bench-full to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
