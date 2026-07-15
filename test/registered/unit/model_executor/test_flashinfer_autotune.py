import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.model_executor.runner.flashinfer_autotune import (
    should_run_flashinfer_autotune,
)


class TestFlashInferAutotune(CustomTestCase):
    def _should_run(
        self,
        *,
        target="flashinfer_trtllm",
        target_a2a="none",
        draft="flashinfer_cutedsl",
        draft_a2a="deepep",
    ):
        runner = SimpleNamespace(
            device="cuda",
            server_args=SimpleNamespace(
                disable_flashinfer_autotune=False,
                moe_runner_backend=target,
                moe_a2a_backend=target_a2a,
                speculative_moe_runner_backend=draft,
                speculative_moe_a2a_backend=draft_a2a,
            ),
            model_config=SimpleNamespace(quantization="mxfp8"),
            spec_algorithm=SimpleNamespace(is_speculative=lambda: True),
            is_draft_worker=True,
        )
        with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
            return should_run_flashinfer_autotune(runner, for_speculative_draft=True)

    def test_speculative_backend_selection(self):
        self.assertFalse(self._should_run())
        self.assertTrue(
            self._should_run(
                target="flashinfer_cutedsl",
                target_a2a="deepep",
                draft_a2a="flashinfer",
            )
        )
        self.assertFalse(
            self._should_run(
                target="flashinfer_cutedsl",
                target_a2a="deepep",
                draft=None,
                draft_a2a=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
