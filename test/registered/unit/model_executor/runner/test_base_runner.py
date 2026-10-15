"""CPU coverage for runner warmup helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.runner import base_runner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDummyRequestWindowHistory(CustomTestCase):
    def test_paged_swa_tail_without_request_window_is_supported(self):
        model_runner = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(request_window=None)
        )
        exec_config = SimpleNamespace(
            features=SimpleNamespace(enable_encoder_swa_bounded_replay=True)
        )
        with patch.object(base_runner, "get_exec", return_value=exec_config):
            base_runner._initialize_dummy_request_window_history(model_runner)


if __name__ == "__main__":
    unittest.main()
