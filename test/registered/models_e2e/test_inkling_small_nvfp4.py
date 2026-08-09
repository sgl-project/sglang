"""Per-commit accuracy + logprob-consistency test for Inkling-Small-NVFP4.

``test_inkling.py`` boots a shrunken checkpoint, so it can only guard that the
code paths run -- an undertrained model has no answer quality to gate on. This
one serves the real NVFP4 checkpoint, which is what catches a weight-load or
FP4-kernel regression that keeps the server healthy while the outputs go wrong.

gsm8k here is few-shot completion, so it never renders a chat turn and never
reaches the reasoning path -- it gates the FP4 numerics, not answer quality
under thinking. tp=4 to match the runner; the checkpoint does not fit on fewer
cards.
"""

import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-b200")

_MODEL_PATH = os.environ.get(
    "INKLING_SMALL_TEST_MODEL_PATH", "thinkingmachines/Inkling-Small-NVFP4"
)

# Measured 0.900 (10-shot, 200 questions, tp=4, invalid=0.000) -- completion,
# so no thinking. The floor sits ~2.5 sigma of the 200-question sampling noise
# below that: a real accuracy collapse trips it, the sample spread does not.
GSM8K_THRESHOLD = 0.85


class TestInklingSmallNvfp4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "4",
                "--trust-remote-code",
                "--quantization",
                "modelopt_fp4",
                "--attention-backend",
                "fa4",
                "--page-size",
                "128",
                "--fp4-gemm-backend",
                "flashinfer_trtllm",
                "--moe-runner-backend",
                "flashinfer_trtllm_routed",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--swa-full-tokens-ratio",
                "0.1",
                "--mamba-full-memory-ratio",
                "0.1",
                "--mem-fraction-static",
                "0.85",
            ],
            env={**os.environ, "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        """Answer quality on the real checkpoint: guards the modelopt_fp4 weight
        load and the FP4 GEMM/MoE kernels against changes that keep the server
        healthy but corrupt the numerics."""
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        metrics = run_few_shot_gsm8k(
            SimpleNamespace(
                num_shots=10,
                data_path=None,
                num_questions=200,
                max_new_tokens=16000,
                parallel=128,
                host=f"http://{url.hostname}",
                port=int(url.port),
            )
        )
        print(f"[{self.__class__.__name__}] gsm8k: {metrics['accuracy']:.3f}")
        self.assertGreaterEqual(metrics["accuracy"], GSM8K_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
