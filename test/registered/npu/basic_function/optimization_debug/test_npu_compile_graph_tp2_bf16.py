import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.npu_eval_accuracy_kit import NPUGSM8KMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1800, suite="full-2-npu-a3", nightly=True)

COMPILE_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.8",
    "--attention-backend",
    "ascend",
    "--dtype",
    "bfloat16",
    "--enable-torch-compile",
    # Capture buckets have to cover the eval's concurrency, otherwise decode
    # falls back to eager and the compiled graphs are never replayed.
    "--torch-compile-max-bs",
    "32",
    "--cuda-graph-max-bs-decode",
    "32",
    "--tp-size",
    "2",
]

# Same as the TorchAir compile test: without it the ascend backend runs paged
# attention through ATB, which has been observed to fail graph capture on A3.
os.environ["ASCEND_USE_FIA"] = "true"


class TestNpuCompileGraphTp2Bf16(CustomTestCase, NPUGSM8KMixin):
    # The only lane that compiles and captures HCOM collectives, which
    # capture and replay handle differently from single-rank graphs.
    # Measured 0.970 on this model and harness; 0.94 leaves three points
    # of margin. Compiling must not move accuracy.
    gsm8k_score_threshold = 0.94
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    # At 2048 this model truncates 28% of its answers and scores 0.820;
    # 8192 brings the truncated rate to 0.010.
    gsm8k_max_tokens = 8192

    @classmethod
    def setUpClass(cls):
        # Qwen3-8B, not the Qwen2.5-7B-Instruct of the sibling ACLGraph tests:
        # sgl-eval's gsm8k prompt asks for the answer alone, which that model
        # obeys literally and scores 0.32 on, leaving no usable bar.
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            # Compiling the decode graphs takes ~565s on this model, which the
            # 600s default does not cover. 1500s matches the sibling NPU tests.
            timeout=1500,
            other_args=COMPILE_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
