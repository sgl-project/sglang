import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    _should_capture_decode_cuda_graph,
)
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    should_run_flashinfer_autotune,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestVPPStartupGates(unittest.TestCase):
    def test_vpp_prefill_skips_decode_graph_and_flashinfer_autotune(self):
        with get_context().override_server_args(
            pp_virtual_stages=2,
            disaggregation_mode="prefill",
        ):
            self.assertFalse(_should_capture_decode_cuda_graph(True))
            self.assertFalse(
                should_run_flashinfer_autotune(SimpleNamespace(device="cuda"))
            )

    def test_regular_pipeline_preserves_decode_graph_request(self):
        with get_context().override_server_args(
            pp_virtual_stages=1,
            disaggregation_mode="prefill",
        ):
            self.assertTrue(_should_capture_decode_cuda_graph(True))
            self.assertFalse(_should_capture_decode_cuda_graph(False))


if __name__ == "__main__":
    unittest.main()
