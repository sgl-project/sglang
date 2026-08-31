"""
End-to-end accuracy test for the unified memory pool on a hybrid-SWA MoE model.

Launches gpt-oss-20b with ``--enable-unified-memory`` on the Triton attention
backend and checks that GSM8K accuracy holds. This exercises the SWA +
full-attention KV sub-pools stored as per-layer views in the unified
page-major envelope.

Registered to the label-gated ``run-ci-extra`` suite (opt-in, not per-commit).

Usage:
    python3 -m unittest test_page_major_gpt_oss
"""

import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE

register_cuda_ci(est_time=109, stage="extra-a", runner_config="1-gpu-large")

_UNIFIED_COMMON_ARGS = [
    "--enable-unified-memory",
    "--mem-fraction-static",
    "0.70",
    "--cuda-graph-backend-prefill=disabled",
]


class TestUnifiedGptOssTriton(DefaultServerBase):
    """Unified pool on gpt-oss-20b (hybrid-SWA MoE), Triton pinned: dense
    MHA/SWA views through the reference backend."""

    model = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE

    gsm8k_threshold = 0.45
    num_gsm8k_questions = 200
    num_shots = 5
    parallel = 32

    other_args = _UNIFIED_COMMON_ARGS + ["--attention-backend", "triton"]

    def test_gsm8k(self):
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=self.num_shots,
            data_path=None,
            num_questions=self.num_gsm8k_questions,
            max_new_tokens=512,
            parallel=self.parallel,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_few_shot_gsm8k(args)
        print(
            f"[{self.__class__.__name__}] GSM8K accuracy: {metrics['accuracy']:.3f} "
            f"(threshold: {self.gsm8k_threshold})"
        )
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_threshold)


if __name__ == "__main__":
    unittest.main()
