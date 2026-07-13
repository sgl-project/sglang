"""Integration test for the full prefill CUDA graph backend.

Spins up Qwen3-8B with --cuda-graph-backend-prefill=full and checks
mgsm_en accuracy, mirroring the breakable-CG integration test.

The attention backend is pinned to flashinfer: plain EXTEND under full
CUDA graph requires the backend's init_forward_metadata_out_graph to
support extend (capture-stable plan state). flashinfer and the
FlashAttention backend (fa4; fa3 untested — needs SM90 hardware)
implement it; flashinfer is pinned here for CI-hardware portability
(fa4 requires Blackwell).
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# CI Registration — large suite to fit the integration test's server startup.
register_cuda_ci(est_time=91, stage="base-b", runner_config="1-gpu-large")


class TestFullCudaGraphPrefill(CustomTestCase):
    """Integration: Qwen3-8B with --cuda-graph-backend-prefill=full on mgsm_en."""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--cuda-graph-backend-prefill=full",
                "--attention-backend=flashinfer",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=1319,
            num_threads=1024,
        )

        metrics = run_eval(args)
        score = metrics["score"]
        print(f"mgsm_en accuracy with full prefill CUDA graph: {score:.3f}")

        self.assertGreaterEqual(score, 0.80)


if __name__ == "__main__":
    unittest.main()
