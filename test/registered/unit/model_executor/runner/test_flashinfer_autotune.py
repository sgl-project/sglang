import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.runner.flashinfer_autotune import (
    should_run_flashinfer_autotune,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFlashinferAutotune(CustomTestCase):
    def test_skip_cutedsl_flashinfer_a2a_speculative_draft(self):
        model_runner = SimpleNamespace(
            device="cuda",
            server_args=SimpleNamespace(
                disable_flashinfer_autotune=False,
                moe_runner_backend="flashinfer_cutedsl",
                moe_a2a_backend="flashinfer",
                speculative_moe_runner_backend="flashinfer_cutedsl",
                speculative_moe_a2a_backend="flashinfer",
            ),
        )

        self.assertFalse(
            should_run_flashinfer_autotune(
                model_runner,
                for_speculative_draft=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
