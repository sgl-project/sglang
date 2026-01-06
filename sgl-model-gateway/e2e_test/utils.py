"""Consolidated utilities for E2E tests.

This module provides common utilities used across E2E tests:
- Tokenizer loading (get_tokenizer)
- Test base classes (CustomTestCase for unittest compatibility)
- Model path resolution
- Process management utilities

Import examples:
    from utils import get_tokenizer, CustomTestCase
    from utils import DEFAULT_MODEL_PATH, DEFAULT_TIMEOUT
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Re-export commonly used items from submodules
from backends import kill_process_tree  # noqa: F401
from infra.model_specs import (  # noqa: F401; Default model paths
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_ENABLE_THINKING_MODEL_PATH,
    DEFAULT_GPT_OSS_MODEL_PATH,
    DEFAULT_MISTRAL_FUNCTION_CALLING_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_QWEN_FUNCTION_CALLING_MODEL_PATH,
    DEFAULT_REASONING_MODEL_PATH,
    DEFAULT_SMALL_MODEL_PATH,
    MODEL_SPECS,
    ROUTER_LOCAL_MODEL_PATH,
    _resolve_model_path,
)

# =============================================================================
# Constants
# =============================================================================

# Server startup timeout (seconds)
DEFAULT_TIMEOUT = 600
DEFAULT_STARTUP_TIMEOUT = 300

# Default test port range
DEFAULT_PORT_BASE = 20000

# File paths for test output
STDOUT_FILENAME = "/tmp/sglang_test_stdout.txt"
STDERR_FILENAME = "/tmp/sglang_test_stderr.txt"

# =============================================================================
# Tokenizer Utilities
# =============================================================================

# Lazy import transformers to avoid import errors in environments without it
_transformers_available = None
_AutoTokenizer = None
_PreTrainedTokenizer = None
_PreTrainedTokenizerBase = None
_PreTrainedTokenizerFast = None


def _ensure_transformers():
    """Lazy load transformers module."""
    global _transformers_available, _AutoTokenizer
    global _PreTrainedTokenizer, _PreTrainedTokenizerBase, _PreTrainedTokenizerFast

    if _transformers_available is not None:
        return _transformers_available

    try:
        from transformers import (
            AutoTokenizer,
            PreTrainedTokenizer,
            PreTrainedTokenizerBase,
            PreTrainedTokenizerFast,
        )

        _AutoTokenizer = AutoTokenizer
        _PreTrainedTokenizer = PreTrainedTokenizer
        _PreTrainedTokenizerBase = PreTrainedTokenizerBase
        _PreTrainedTokenizerFast = PreTrainedTokenizerFast
        _transformers_available = True
    except ImportError:
        _transformers_available = False

    return _transformers_available


def check_gguf_file(model_path: str) -> bool:
    """Check if the model path points to a GGUF file."""
    if not isinstance(model_path, str):
        return False
    return model_path.endswith(".gguf")


def is_remote_url(path: str) -> bool:
    """Check if the path is a remote URL."""
    if not isinstance(path, str):
        return False
    return path.startswith("http://") or path.startswith("https://")


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: str | None = None,
    **kwargs,
):
    """Gets a tokenizer for the given model name via Huggingface.

    Args:
        tokenizer_name: Name or path of the tokenizer
        tokenizer_mode: Mode for tokenizer loading ("auto", "slow")
        trust_remote_code: Whether to trust remote code
        tokenizer_revision: Specific revision to use
        **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained

    Returns:
        Loaded tokenizer instance

    Raises:
        ImportError: If transformers is not installed
        RuntimeError: If tokenizer loading fails
    """
    if not _ensure_transformers():
        raise ImportError(
            "transformers is required for tokenizer utilities. "
            "Install with: pip install transformers"
        )

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    # Handle special model name mapping
    if tokenizer_name == "mistralai/Devstral-Small-2505":
        tokenizer_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = tokenizer_name
        tokenizer_name = str(Path(tokenizer_name).parent)

    try:
        tokenizer = _AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs,
        )
    except TypeError as e:
        err_msg = (
            "Failed to load the tokenizer. If you are running a model with "
            "a custom tokenizer, please set the --trust-remote-code flag."
        )
        raise RuntimeError(err_msg) from e

    if not isinstance(tokenizer, _PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a performance "
            "degradation. Consider using a fast tokenizer instead."
        )

    return tokenizer


def get_tokenizer_from_processor(processor):
    """Extract tokenizer from a processor object."""
    if not _ensure_transformers():
        raise ImportError("transformers is required for tokenizer utilities.")

    if isinstance(processor, _PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


# =============================================================================
# Environment Utilities
# =============================================================================


def is_ci_environment() -> bool:
    """Check if running in CI environment."""
    ci_vars = ["CI", "GITHUB_ACTIONS", "JENKINS_URL", "GITLAB_CI", "CIRCLECI"]
    return any(os.environ.get(var) for var in ci_vars)


def get_test_timeout() -> int:
    """Get test timeout from environment or default."""
    return int(os.environ.get("E2E_TEST_TIMEOUT", str(DEFAULT_TIMEOUT)))


def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    import pytest

    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
    except ImportError:
        # Try nvidia-ml-py
        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            if count == 0:
                pytest.skip("No GPU available")
        except Exception:
            pytest.skip("Cannot detect GPU (torch and pynvml not available)")
