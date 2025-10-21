"""
Standalone utilities for e2e_grpc tests.

This module provides all necessary utilities without depending on sglang Python package.
Extracted and adapted from:
- sglang.srt.utils.kill_process_tree
- sglang.srt.utils.hf_transformers_utils.get_tokenizer
- sglang.test.test_utils (constants and CustomTestCase)
"""

import os
import signal
import threading
import unittest
from pathlib import Path
from typing import Optional, Union

import psutil

try:
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizer,
        PreTrainedTokenizerBase,
        PreTrainedTokenizerFast,
    )
except ImportError:
    raise ImportError(
        "transformers is required for tokenizer utilities. "
        "Install with: pip install transformers"
    )


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 600
DEFAULT_PORT_FOR_SRT_TEST_RUNNER = 20000
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"


# ============================================================================
# Process Management
# ============================================================================


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """
    Kill the process and all its child processes.

    Args:
        parent_pid: PID of the parent process
        include_parent: Whether to kill the parent process itself
        skip_pid: Optional PID to skip during cleanup
    """
    # Remove sigchld handler to avoid spammy logs
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            itself.kill()
        except psutil.NoSuchProcess:
            pass


# ============================================================================
# Tokenizer Utilities
# ============================================================================


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
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    Gets a tokenizer for the given model name via Huggingface.

    Args:
        tokenizer_name: Name or path of the tokenizer
        tokenizer_mode: Mode for tokenizer loading ("auto", "slow")
        trust_remote_code: Whether to trust remote code
        tokenizer_revision: Specific revision to use
        **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained

    Returns:
        Loaded tokenizer instance
    """
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
        tokenizer_name = Path(tokenizer_name).parent

    # Note: Removed remote URL handling and local directory download
    # as they depend on sglang-specific utilities

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs,
        )
    except TypeError as e:
        # Handle specific errors
        err_msg = (
            "Failed to load the tokenizer. If you are running a model with "
            "a custom tokenizer, please set the --trust-remote-code flag."
        )
        raise RuntimeError(err_msg) from e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        print(
            f"Warning: Using a slow tokenizer. This might cause a performance "
            f"degradation. Consider using a fast tokenizer instead."
        )

    return tokenizer


def get_tokenizer_from_processor(processor):
    """Extract tokenizer from a processor object."""
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


# ============================================================================
# Test Utilities
# ============================================================================


class CustomTestCase(unittest.TestCase):
    """
    Custom test case base class with retry support.

    This provides automatic test retry functionality based on environment variables.
    """

    def _callTestMethod(self, method):
        """Override to add retry logic."""
        max_retry = int(os.environ.get("SGLANG_TEST_MAX_RETRY", "0"))

        if max_retry == 0:
            # No retry, just run once
            return super(CustomTestCase, self)._callTestMethod(method)

        # Retry logic
        last_exception = None
        for attempt in range(max_retry + 1):
            try:
                return super(CustomTestCase, self)._callTestMethod(method)
            except Exception as e:
                last_exception = e
                if attempt < max_retry:
                    print(
                        f"Test failed on attempt {attempt + 1}/{max_retry + 1}, retrying..."
                    )
                    continue
                else:
                    raise

        # If we get here, all retries failed
        if last_exception:
            raise last_exception

    def setUp(self):
        """Print test method name at the start of each test."""
        print(f"[Test Method] {self._testMethodName}", flush=True)
