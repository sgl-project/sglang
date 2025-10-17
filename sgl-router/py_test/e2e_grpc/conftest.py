"""
Pytest configuration for gRPC router e2e tests.

This module provides shared fixtures that can be used across all gRPC router tests.
"""

import sys
from pathlib import Path

import pytest

# Ensure sglang Python package is importable
_SGLANG_ROOT = Path(__file__).resolve().parents[3]
_SGLANG_PYTHON = _SGLANG_ROOT / "python"
if str(_SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SGLANG_PYTHON))

# Ensure sglang test utilities are importable
_SGLANG_TEST = _SGLANG_ROOT / "test"
if str(_SGLANG_TEST) not in sys.path:
    sys.path.insert(0, str(_SGLANG_TEST))

# Ensure router py_src is importable
_ROUTER_ROOT = Path(__file__).resolve().parents[2]
_ROUTER_SRC = _ROUTER_ROOT / "py_src"
if str(_ROUTER_SRC) not in sys.path:
    sys.path.insert(0, str(_ROUTER_SRC))


@pytest.fixture(scope="session")
def e2e_model():
    """Default model for e2e testing."""
    # Override to use llama-3.1-8b-instruct instead of the small model
    return "/home/ubuntu/models/llama-3.1-8b-instruct"


@pytest.fixture(scope="session")
def e2e_timeout():
    """Default timeout for server launches."""
    from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    return DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH


# Pytest markers for test organization
def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end tests with real workers")
    config.addinivalue_line("markers", "grpc: gRPC-specific tests")
    config.addinivalue_line("markers", "slow: slow-running tests")
    config.addinivalue_line("markers", "pd: prefill-decode disaggregation tests")
