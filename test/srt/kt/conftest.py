"""
Pytest configuration for KT-Kernel tests
"""

import pytest


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "basic: Basic functionality tests")
    config.addinivalue_line("markers", "performance: Performance and throughput tests")
    config.addinivalue_line(
        "markers", "correctness: Model correctness and accuracy tests"
    )
    config.addinivalue_line("markers", "gpu1: Tests that require 1 GPU")
    config.addinivalue_line("markers", "gpu4: Tests that require 4 GPUs")
    config.addinivalue_line("markers", "gpu8: Tests that require 8 GPUs")


@pytest.fixture(scope="session", autouse=True)
def verify_kt_installation():
    """Verify KT-kernel is installed and AMX is supported"""
    try:
        from kt_kernel import KTMoEWrapper

        print("✓ KT-kernel installed")
    except ImportError as e:
        pytest.skip(f"KT-kernel not installed: {e}")

    try:
        import torch

        if not torch._C._cpu._is_amx_tile_supported():
            pytest.skip("AMX instructions not supported on this CPU")
        print("✓ AMX supported")
    except Exception as e:
        pytest.skip(f"Failed to verify AMX support: {e}")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on test file name
        if "test_kt_basic" in item.nodeid:
            item.add_marker(pytest.mark.basic)
        if "test_kt_long_context" in item.nodeid:
            item.add_marker(pytest.mark.basic)
        if "test_kt_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        if "test_kt_correctness" in item.nodeid:
            item.add_marker(pytest.mark.correctness)

        # Add GPU count markers based on test class name
        if "1GPU" in item.nodeid or "1gpu" in item.nodeid:
            item.add_marker(pytest.mark.gpu1)
        elif "4GPU" in item.nodeid or "4gpu" in item.nodeid:
            item.add_marker(pytest.mark.gpu4)
        elif "8GPU" in item.nodeid or "8gpu" in item.nodeid:
            item.add_marker(pytest.mark.gpu8)
