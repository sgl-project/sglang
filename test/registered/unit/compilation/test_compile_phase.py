"""Unit tests for phase-flag context managers in srt/compilation/compile_phase.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.compilation.compile_phase import (
    enable_torch_compile_warmup,
    get_pcg_capture_stream,
    is_in_torch_compile_warmup,
    set_pcg_capture_stream,
)
from sglang.test.test_utils import CustomTestCase


class TestEnableTorchCompileWarmup(CustomTestCase):
    def test_flag_is_false_before_context(self):
        self.assertFalse(is_in_torch_compile_warmup())

    def test_flag_is_true_inside_context(self):
        with enable_torch_compile_warmup():
            self.assertTrue(is_in_torch_compile_warmup())

    def test_flag_is_false_after_context_exits(self):
        with enable_torch_compile_warmup():
            pass
        self.assertFalse(is_in_torch_compile_warmup())

    def test_flag_restored_to_false_after_exception_inside_context(self):
        try:
            with enable_torch_compile_warmup():
                raise RuntimeError("intentional")
        except RuntimeError:
            pass
        self.assertFalse(is_in_torch_compile_warmup())

    def test_nested_inner_context_sees_true(self):
        with enable_torch_compile_warmup():
            self.assertTrue(is_in_torch_compile_warmup())
            with enable_torch_compile_warmup():
                self.assertTrue(is_in_torch_compile_warmup())

    def test_after_all_nested_contexts_exit_flag_is_false(self):
        with enable_torch_compile_warmup():
            with enable_torch_compile_warmup():
                pass
        self.assertFalse(is_in_torch_compile_warmup())


class TestSetPcgCaptureStream(CustomTestCase):
    def test_get_returns_none_before_context(self):
        self.assertIsNone(get_pcg_capture_stream())

    def test_get_returns_stream_inside_context(self):
        mock_stream = MagicMock()
        with set_pcg_capture_stream(mock_stream):
            self.assertIs(get_pcg_capture_stream(), mock_stream)

    def test_get_returns_none_after_context_exits(self):
        mock_stream = MagicMock()
        with set_pcg_capture_stream(mock_stream):
            pass
        self.assertIsNone(get_pcg_capture_stream())

    def test_get_returns_none_after_exception_inside_context(self):
        mock_stream = MagicMock()
        try:
            with set_pcg_capture_stream(mock_stream):
                raise ValueError("intentional")
        except ValueError:
            pass
        self.assertIsNone(get_pcg_capture_stream())

    def test_nested_inner_stream_overrides_outer_inside_inner(self):
        outer = MagicMock()
        inner = MagicMock()
        with set_pcg_capture_stream(outer):
            self.assertIs(get_pcg_capture_stream(), outer)
            with set_pcg_capture_stream(inner):
                self.assertIs(get_pcg_capture_stream(), inner)

    def test_after_all_nested_contexts_exit_stream_is_none(self):
        outer = MagicMock()
        inner = MagicMock()
        with set_pcg_capture_stream(outer):
            with set_pcg_capture_stream(inner):
                pass
        self.assertIsNone(get_pcg_capture_stream())


if __name__ == "__main__":
    unittest.main()
