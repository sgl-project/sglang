"""
Shared pytest fixtures and configuration for E2E tests.
"""

import os
import pytest
import asyncio
from pathlib import Path

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Set asyncio mode to auto for all async tests
def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "requires_sidecar: marks tests that require sidecar"
    )

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Default configuration
DEFAULT_SERVER_URL = os.environ.get("SGLANG_SERVER_URL", "http://localhost:30000")
DEFAULT_SIDECAR_URL = os.environ.get("SIDECAR_URL", "http://localhost:9009")
DEFAULT_TIMEOUT = 120.0
DEFAULT_TOP_K = 32


def pytest_addoption(parser):
    """Add custom pytest options for E2E tests."""
    try:
        parser.addoption(
            "--server",
            action="store",
            default=DEFAULT_SERVER_URL,
            help="SGLang server URL (default: http://localhost:30000)",
        )
    except ValueError:
        pass  # Option already added

    try:
        parser.addoption(
            "--sidecar",
            action="store",
            default=DEFAULT_SIDECAR_URL,
            help="Rapids sidecar URL (default: http://localhost:9009)",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--model",
            action="store",
            default=None,
            help="Model name to test against",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--full-validation",
            action="store_true",
            default=False,
            help="Run full validation suite (slower)",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--attention-top-k",
            action="store",
            type=int,
            default=DEFAULT_TOP_K,
            help="Number of top-k attention tokens",
        )
    except ValueError:
        pass


@pytest.fixture
def server_url(request):
    """Get server URL from pytest options or environment."""
    return request.config.getoption("--server", default=DEFAULT_SERVER_URL)


@pytest.fixture
def sidecar_url(request):
    """Get sidecar URL from pytest options or environment."""
    return request.config.getoption("--sidecar", default=DEFAULT_SIDECAR_URL)


@pytest.fixture
def model_name(request):
    """Get model name from pytest options."""
    return request.config.getoption("--model", default=None)


@pytest.fixture
def full_validation(request):
    """Check if full validation is enabled."""
    return request.config.getoption("--full-validation", default=False)


@pytest.fixture
def attention_top_k(request):
    """Get attention top-k from pytest options."""
    return request.config.getoption("--attention-top-k", default=DEFAULT_TOP_K)


@pytest.fixture
def e2e_output_dir(tmp_path):
    """Create temporary output directory for E2E test results."""
    output_dir = tmp_path / "e2e_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# Model configurations for different test scenarios
MODEL_CONFIGS = {
    "small": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens": 256,
        "expected_tps": 100,  # Expected tokens per second
    },
    "medium": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "max_tokens": 512,
        "expected_tps": 50,
    },
    "large": {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "max_tokens": 1024,
        "expected_tps": 20,
    },
    "qwen3_fp8": {
        "name": "Qwen/Qwen3-Next-FP8",
        "max_tokens": 1024,
        "expected_tps": 30,
    },
}


@pytest.fixture(params=["small"])  # Default to small for quick tests
def model_config(request):
    """Get model configuration for parameterized tests."""
    config_name = request.param
    return MODEL_CONFIGS.get(config_name, MODEL_CONFIGS["small"])


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and options."""
    # Skip slow tests unless full-validation is enabled
    if not config.getoption("--full-validation", default=False):
        skip_slow = pytest.mark.skip(reason="Slow test - use --full-validation")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
