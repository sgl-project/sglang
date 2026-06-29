import subprocess
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=570, stage="extra-b", runner_config="8-gpu-h200")

GLM5_FP8_MODEL_PATH = "zai-org/GLM-5-FP8"


class TestGLM5HiSparse(DefaultServerBase, GSM8KMixin):
    """GLM-5 FP8 with HiSparse (host-to-device sparse KV offload) on DSA decode.

    HiSparse targets the high-concurrency regime and is not used together with
    EAGLE MTP, so this variant runs without speculative decoding (unlike the
    DSA-MTP variants in test_dsa_glm5_{dp,tp}_mtp.py).
    """

    model = GLM5_FP8_MODEL_PATH
    other_args = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--dp",
        "8",
        "--enable-dp-attention",
        "--page-size",
        "64",
        "--max-running-requests",
        "200",
        "--mem-fraction-static",
        "0.85",
        "--disable-radix-cache",
        "--kv-cache-dtype",
        "bfloat16",
        "--dsa-decode-backend",
        "flashmla_sparse",
        "--enable-hisparse",
        "--hisparse-config",
        '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 5}',
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    ]

    # Match the original standalone hisparse eval config.
    gsm8k_accuracy_thres = 0.94
    gsm8k_num_questions = 500
    gsm8k_num_threads = 100
    gsm8k_num_shots = 24

    @classmethod
    def tearDownClass(cls):
        # HiSparse's large pinned host buffer stalls an external SIGKILL teardown
        # (kernel unpin). Drive the server's own graceful shutdown so each rank
        # unregisters in userspace; hard-kill as a fallback.
        cls.process.terminate()
        try:
            cls.process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            kill_process_tree(cls.process.pid, wait_timeout=60)
        time.sleep(2)


if __name__ == "__main__":
    unittest.main()
