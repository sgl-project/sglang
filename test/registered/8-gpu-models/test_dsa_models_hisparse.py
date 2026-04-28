import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=720, suite="stage-c-test-8-gpu-h200", nightly=True)
register_amd_ci(est_time=3600, suite="stage-c-test-large-8-gpu-amd-mi35x")

GLM5_MODEL_PATH = os.environ.get("GLM5_MODEL_PATH", "zai-org/GLM-5-FP8")
GLM5_HISPARSE_GSM8K_NUM_EXAMPLES = int(
    os.environ.get("GLM5_HISPARSE_GSM8K_NUM_EXAMPLES", "500")
)
GLM5_HISPARSE_ROCM_MEM_FRACTION_STATIC = os.environ.get(
    "GLM5_HISPARSE_ROCM_MEM_FRACTION_STATIC", "0.75"
)
GLM5_HISPARSE_ROCM_HOST_TO_DEVICE_RATIO = os.environ.get(
    "GLM5_HISPARSE_ROCM_HOST_TO_DEVICE_RATIO", "1"
)
GLM5_HISPARSE_ROCM_DEVICE_BUFFER_SIZE = os.environ.get(
    "GLM5_HISPARSE_ROCM_DEVICE_BUFFER_SIZE", "8192"
)
GLM5_HISPARSE_GSM8K_MAX_TOKENS = int(
    os.environ.get("GLM5_HISPARSE_GSM8K_MAX_TOKENS", "4000")
)
GLM5_HISPARSE_GSM8K_NUM_THREADS = int(
    os.environ.get("GLM5_HISPARSE_GSM8K_NUM_THREADS", "100")
)
GLM5_HISPARSE_ROCM_DISABLE_OVERLAP = (
    os.environ.get("GLM5_HISPARSE_ROCM_DISABLE_OVERLAP", "1").lower()
    not in ("0", "false", "no")
)

ROCM_INFERENCEX_ENV_DEFAULTS = {
    "SGLANG_ROCM_FUSED_DECODE_MLA": "0",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
    "SAFETENSORS_FAST_GPU": "1",
}


class TestGLM5DPHiSparse(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GLM5_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls._old_env = {}
        if is_hip():
            for key, value in ROCM_INFERENCEX_ENV_DEFAULTS.items():
                cls._old_env[key] = os.environ.get(key)
                os.environ.setdefault(key, value)

        nsa_backend = "tilelang" if is_hip() else "flashmla_sparse"
        loader_threads = 8 if is_hip() else 64
        mem_fraction_static = (
            GLM5_HISPARSE_ROCM_MEM_FRACTION_STATIC if is_hip() else "0.85"
        )
        max_running_requests = "64" if is_hip() else "200"
        hisparse_config = (
            f'{{"top_k": 2048, "device_buffer_size": {GLM5_HISPARSE_ROCM_DEVICE_BUFFER_SIZE}, '
            f'"host_to_device_ratio": {GLM5_HISPARSE_ROCM_HOST_TO_DEVICE_RATIO}}}'
            if is_hip()
            else '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 5}'
        )
        other_args = [
            "--trust-remote-code",
            "--tool-call-parser",
            "glm47",
            "--reasoning-parser",
            "glm45",
            "--tp",
            "8",
            "--dp",
            "8",
            "--enable-dp-attention",
            "--page-size",
            "64",
            "--max-running-requests",
            max_running_requests,
            "--mem-fraction-static",
            mem_fraction_static,
            "--disable-radix-cache",
            "--kv-cache-dtype",
            "bfloat16",
            "--nsa-prefill-backend",
            nsa_backend,
            "--nsa-decode-backend",
            nsa_backend,
            "--enable-hisparse",
            "--hisparse-config",
            hisparse_config,
            "--model-loader-extra-config",
            f'{{"enable_multithread_load": true, "num_threads": {loader_threads}}}',
        ]
        if is_hip() and GLM5_HISPARSE_ROCM_DISABLE_OVERLAP:
            other_args.append("--disable-overlap-schedule")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=7200,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)
        for key, value in getattr(cls, "_old_env", {}).items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=GLM5_HISPARSE_GSM8K_MAX_TOKENS,
            num_examples=GLM5_HISPARSE_GSM8K_NUM_EXAMPLES,
            num_threads=GLM5_HISPARSE_GSM8K_NUM_THREADS,
            num_shots=24,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (glm-5 hisparse)\n" f'{metrics["score"]=:.3f}\n'
            )
            self.assertGreater(metrics["score"], 0.94)


if __name__ == "__main__":
    unittest.main()
