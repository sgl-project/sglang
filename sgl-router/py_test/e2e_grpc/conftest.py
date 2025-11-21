"""
Pytest configuration for gRPC router e2e tests.

This module provides shared fixtures that can be used across all gRPC router tests.
"""

import sys
from pathlib import Path

import pytest  # noqa: F401

# Ensure router bindings/python is importable
_ROUTER_ROOT = Path(__file__).resolve().parents[2]
_ROUTER_SRC = _ROUTER_ROOT / "bindings" / "python"
if str(_ROUTER_SRC) not in sys.path:
    sys.path.insert(0, str(_ROUTER_SRC))

# Ensure e2e_grpc test utilities are importable
_E2E_GRPC_DIR = Path(__file__).parent
if str(_E2E_GRPC_DIR) not in sys.path:
    sys.path.insert(0, str(_E2E_GRPC_DIR))


# Pytest markers for test organization
def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end tests with real workers")
    config.addinivalue_line("markers", "grpc: gRPC-specific tests")
    config.addinivalue_line("markers", "slow: slow-running tests")
    config.addinivalue_line("markers", "pd: prefill-decode disaggregation tests")
