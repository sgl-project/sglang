"""MI355X GLM-5.1-MXFP4 TP=2 GSM8K accuracy gate.

This is a nightly AMD regression test for the GLM-5.1-MXFP4 TP=2
accuracy drop seen on MI355X/gfx950 when aiter selected a bad BF16 GEMM path.

Registry: nightly-amd-2-gpu-mi35x-glm51-mxfp4 suite
"""

import resource
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=3600,
    suite="nightly-amd-2-gpu-mi35x-glm51-mxfp4",
    nightly=True,
)

GLM51_MXFP4_MODEL_ID = "amd/GLM-5.1-MXFP4"
SERVER_LAUNCH_TIMEOUT = 5400

GSM8K_ACCURACY_THRESHOLD = 0.92
GSM8K_NUM_EXAMPLES = None
GSM8K_NUM_THREADS = 512


def _raise_nofile_limit() -> None:
    """GSM8K with high parallelism can exceed the default soft nofile=1024."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65535 if hard == resource.RLIM_INFINITY else min(hard, 65535)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


class TestGLM51MXFP4TP2GSM8KMI35x(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _raise_nofile_limit()
        cls.model = GLM51_MXFP4_MODEL_ID
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            model=cls.model,
            base_url=cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
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

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            num_examples=GSM8K_NUM_EXAMPLES,
            num_threads=GSM8K_NUM_THREADS,
            max_tokens=512,
            temperature=0.0,
        )

        metrics = run_eval(args)
        print(f"{metrics=}", flush=True)
        score = metrics["score"]

        if is_in_ci():
            write_github_step_summary(
                "### GLM-5.1-MXFP4 TP=2 GSM8K (MI355X)\n\n"
                "| Model | Examples | Max Parallel | Score | Threshold | Latency |\n"
                "| ----- | --------- | ------------ | ----- | --------- | ------- |\n"
                f"| {self.model} | full | default ({GSM8K_NUM_THREADS}) | "
                f"{score:.3f} | {GSM8K_ACCURACY_THRESHOLD:.2f} | "
                f"{metrics.get('latency', 0):.1f}s |\n"
            )

        self.assertGreaterEqual(score, GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
