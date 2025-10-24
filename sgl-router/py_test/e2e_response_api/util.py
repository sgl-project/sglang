"""
Utility functions for Response API e2e tests.
"""

import logging
import os
import signal
import threading
import unittest

import psutil

logger = logging.getLogger(__name__)


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
        for attempt in range(max_retry + 1):
            try:
                return super(CustomTestCase, self)._callTestMethod(method)
            except Exception as e:
                if attempt < max_retry:
                    logger.info(
                        f"Test failed on attempt {attempt + 1}/{max_retry + 1}, retrying..."
                    )
                    continue
                else:
                    # Last attempt, re-raise the exception
                    raise
