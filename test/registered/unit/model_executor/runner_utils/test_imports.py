"""Exercise import order in fresh interpreters, without a warmed module cache."""

import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class TestRunnerUtilsImports(unittest.TestCase):
    def test_execution_state_does_not_load_runner_dependencies(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "from sglang.srt.model_executor.runner_utils.forward_batch "
                "import get_forward_batch, set_forward_batch\n"
                "from sglang.srt.model_executor.runner_utils.prefill_graph "
                "import prefill_graph_scope\n"
                "assert 'sglang.srt.model_executor.runner_utils.deepep_adapter' "
                "not in sys.modules\n"
                "assert 'sglang.srt.model_executor.runner_utils.buffers' "
                "not in sys.modules\n"
                "batch = object()\n"
                "with set_forward_batch(batch), prefill_graph_scope(full_graph=True):\n"
                "    assert get_forward_batch() is batch\n",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_test_utils_imports_before_attention(self):
        # CPU suites enter through test_utils -> disaggregation -> quantization
        # -> radix_attention. Importing runner_utils must not loop back through
        # DeepEP and metrics_collector into partially initialized disaggregation.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from sglang.test.test_utils import CustomTestCase\n"
                "from sglang.srt.layers.radix_attention import RadixAttention\n",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
