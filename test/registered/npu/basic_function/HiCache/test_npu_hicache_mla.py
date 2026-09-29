import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

# Nightly runs the image's preinstalled sglang. On a diagnostic PR, use this
# checkout in both the test process and its server subprocess so new probes run.
if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
    source_python = Path(__file__).resolve().parents[5] / "python"
    sys.path.insert(0, str(source_python))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(source_python) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    print(f"NPU diagnostic checkout source: {source_python}", flush=True)

from sglang.srt.utils.npu_pinned_host_diagnostics import log_npu_host_baseline
from sglang.test.ascend.npu_eval_accuracy_kit import _is_pr_pipeline, run_npu_pr_smoke
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=400, suite="base-b-test-4-npu-a3")
register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    os.environ.get(
        "SGLANG_NPU_HICACHE_MLA_MODEL_PATH",
        "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V2-Lite-W8A8",
    ): {
        "accuracy": 0.34,
        "latency": 1000,
        "output_throughput": 6,
    },
}


class TestAscendMlaHicache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--tp-size",
            4,
        ]
        if os.environ.get("SGLANG_NPU_HICACHE_MLA_DISABLE_HICACHE") != "1":
            cls.common_args.extend(
                [
                    "--enable-hierarchical-cache",
                    "--hicache-size",
                    int(os.environ.get("SGLANG_NPU_HICACHE_MLA_SIZE_GB", "30")),
                ]
            )

    def test_a_gsm8k(self):
        self.addCleanup(log_npu_host_baseline, "after_mla_test")
        log_npu_host_baseline("before_mla_test")
        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    env={"SGLANG_NPU_PINNED_HOST_DEBUG": "1"},
                    other_args=[
                        *self.common_args,
                    ],
                )

                try:
                    if (
                        _is_pr_pipeline
                        or os.environ.get("SGLANG_NPU_HICACHE_MLA_SMOKE") == "1"
                    ):
                        run_npu_pr_smoke(self.base_url)
                    else:
                        print(f"##=== Testing accuracy: {model} ===##")

                        args = SimpleNamespace(
                            num_shots=5,
                            data_path=os.environ.get(
                                "SGLANG_NPU_HICACHE_MLA_GSM8K_DATA_PATH"
                            ),
                            num_questions=1319,
                            max_new_tokens=512,
                            parallel=128,
                            host=self.url.hostname,
                            port=int(self.url.port),
                        )

                        metrics = run_eval_few_shot_gsm8k(args)
                        self.assertGreaterEqual(
                            metrics["accuracy"],
                            TEST_MODEL_MATRIX[model]["accuracy"],
                        )
                finally:
                    terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
