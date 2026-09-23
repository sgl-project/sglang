"""CPU coverage for runner warmup helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.runner import base_runner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDummyRequestWindowHistory(CustomTestCase):
    def _initialize(self, request_window, *, enabled=True):
        model_runner = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(request_window=request_window)
        )
        exec_config = SimpleNamespace(
            features=SimpleNamespace(enable_encoder_swa_bounded_replay=enabled)
        )
        with patch.object(base_runner, "get_exec", return_value=exec_config):
            base_runner._initialize_dummy_request_window_history(model_runner)

    def test_disabled_replay_does_not_initialize_request_window(self):
        request_window = SimpleNamespace(initialize_dummy_history=Mock())

        self._initialize(request_window=request_window, enabled=False)

        request_window.initialize_dummy_history.assert_not_called()

    def test_paged_swa_tail_without_request_window_is_supported(self):
        self._initialize(request_window=None)

    def test_legacy_request_window_history_is_initialized(self):
        request_window = SimpleNamespace(initialize_dummy_history=Mock())

        self._initialize(request_window=request_window)

        request_window.initialize_dummy_history.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
