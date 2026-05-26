"""MI355X GLM-5.1-MXFP4 TP=2 GSM8K accuracy gate.

This is a PR Test (AMD) regression test for the GLM-5.1-MXFP4 TP=2
accuracy drop seen on MI355X/gfx950 when aiter selected a bad BF16 GEMM path.

Registry: stage-c-test-large-8-gpu-amd-mi35x suite
"""

import os
import resource
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=3600,
    suite="stage-c-test-large-8-gpu-amd-mi35x",
)

GLM51_MXFP4_MODEL_ID = "amd/GLM-5.1-MXFP4"
GLM51_MXFP4_LOCAL_PATHS = (
    "/data2/models/amd-GLM-5.1-MXFP4",
    "/data/huggingface/hub/amd/GLM-5.1-MXFP4",
)

GSM8K_ACCURACY_THRESHOLD = 0.92
GSM8K_INVALID_THRESHOLD = 0.02
DEFAULT_NUM_QUESTIONS = 1200
DEFAULT_PARALLEL = 1200


def _raise_nofile_limit() -> None:
    """GSM8K with high parallelism can exceed the default soft nofile=1024."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65535 if hard == resource.RLIM_INFINITY else min(hard, 65535)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


def _get_model_path() -> str:
    env_path = os.environ.get("GLM51_MXFP4_MODEL_PATH")
    if env_path:
        return env_path
    for path in GLM51_MXFP4_LOCAL_PATHS:
        if os.path.exists(path):
            return path
    return GLM51_MXFP4_MODEL_ID


class TestGLM51MXFP4TP2GSM8KMI35x(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _raise_nofile_limit()
        cls.model = _get_model_path()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            model=cls.model,
            base_url=cls.base_url,
            timeout=5400,
            other_args=[
                "--tp",
                "2",
                "--trust-remote-code",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--watchdog-timeout",
                "1200",
                "--mem-fraction-static",
                "0.85",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--disable-radix-cache",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 8}',
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--tokenizer-worker-num",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        num_questions = int(
            os.environ.get("GLM51_MXFP4_GSM8K_NUM_QUESTIONS", DEFAULT_NUM_QUESTIONS)
        )
        parallel = int(os.environ.get("GLM51_MXFP4_GSM8K_PARALLEL", DEFAULT_PARALLEL))
        url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=num_questions,
            max_new_tokens=512,
            parallel=parallel,
            host=url.hostname or "127.0.0.1",
            port=url.port or 30000,
            temperature=0.0,
        )

        metrics = run_gsm8k_eval(args)
        accuracy = metrics["accuracy"]
        invalid = metrics["invalid"]
        summary = (
            "### GLM-5.1-MXFP4 TP=2 GSM8K (MI355X)\n\n"
            "| Model | TP | Questions | Accuracy | Invalid | Threshold | Status |\n"
            "| ----- | -- | --------- | -------- | ------- | --------- | ------ |\n"
        )
        passed = (
            accuracy >= GSM8K_ACCURACY_THRESHOLD
            and invalid <= GSM8K_INVALID_THRESHOLD
        )
        status = "PASS" if passed else "FAIL"
        summary += (
            f"| {self.model} | 2 | {num_questions} | {accuracy:.3f} | "
            f"{invalid:.3f} | accuracy >= {GSM8K_ACCURACY_THRESHOLD:.2f} | {status} |\n"
        )
        if is_in_ci():
            write_github_step_summary(summary)

        self.assertGreaterEqual(accuracy, GSM8K_ACCURACY_THRESHOLD)
        self.assertLessEqual(invalid, GSM8K_INVALID_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
