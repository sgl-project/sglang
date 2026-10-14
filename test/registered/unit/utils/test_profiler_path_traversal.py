"""
Security tests for profiler filename sanitization (issue #44189).

These tests verify that the profiler's ``profile_id`` and ``profile_prefix``
inputs cannot be used to escape the configured output directory via path
traversal, absolute paths, null bytes, or other path-oriented tricks.

Usage:
    python -m pytest test_profiler_path_traversal.py -v
    python -m unittest test_profiler_path_traversal -v
"""

import os
import unittest

from sglang.srt.managers.scheduler_components.profiler_manager import (
    _sanitize_filename_component,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFilenameSanitizer(CustomTestCase):
    """Unit tests for the ``_sanitize_filename_component`` helper."""

    # Malicious inputs that must all raise ValueError.
    _MALICIOUS_CASES = [
        # Classic traversal through OS path separator.
        ("../../../etc", "profile_prefix"),
        ("../../../tmp/pwned", "profile_id"),
        ("a/../../etc", "profile_id"),
        # Absolute paths (Unix style).
        ("/etc/passwd", "profile_prefix"),
        ("/var/tmp/evil", "profile_id"),
        # Windows-style separators and absolute paths.
        ("..\\..\\windows", "profile_prefix"),
        ("C:\\Windows\\Temp", "profile_id"),
        ("\\Windows\\System32", "profile_prefix"),
        # Mixed separators.
        ("a/b\\c", "profile_id"),
        # Null byte injection.
        ("foo\x00bar", "profile_id"),
        ("valid\x00../../etc", "profile_prefix"),
        # Embedded .. segments (even if they don't contain slashes they
        # still trigger the explicit traversal check when split).
        ("a/../b", "profile_id"),
        ("../a", "profile_prefix"),
        # Special dot components.
        (".", "profile_id"),
        ("..", "profile_prefix"),
        # Disallowed characters (whitespace, shell metachars).
        ("my profile", "profile_id"),
        ("profile!@#", "profile_prefix"),
        ("with$dollar", "profile_id"),
        ("has,semicolon", "profile_prefix"),
    ]

    # Benign inputs that must pass through unchanged.
    _VALID_CASES = [
        ("", "profile_prefix"),  # optional, empty by default
        ("valid_id", "profile_id"),
        ("my-profile_1.test", "profile_prefix"),
        ("A-Za-z0-9._-", "profile_id"),
        ("trace.2024_01-15", "profile_prefix"),
        ("model_v2", "profile_id"),
    ]

    def test_malicious_inputs_rejected(self):
        for value, field in self._MALICIOUS_CASES:
            with self.subTest(value=value, field=field):
                with self.assertRaises(ValueError) as ctx:
                    _sanitize_filename_component(value, field)
                # Error message names the offending field so callers can
                # surface a helpful message in the API response.
                self.assertIn(field, str(ctx.exception))

    def test_valid_inputs_pass_through(self):
        for value, field in self._VALID_CASES:
            with self.subTest(value=value, field=field):
                self.assertEqual(_sanitize_filename_component(value, field), value)

    def test_non_string_input_rejected(self):
        with self.assertRaises(ValueError):
            _sanitize_filename_component(None, "profile_id")
        with self.assertRaises(ValueError):
            _sanitize_filename_component(123, "profile_prefix")


if __name__ == "__main__":
    unittest.main()
