import os
import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=1800, suite="nightly-4-gpu-b200", nightly=True)

QWEN3_NVFP4_MODEL = "nvidia/Qwen3-30B-A3B-NVFP4"
GSM8K_QUESTION_COUNT = 200
GSM8K_NUM_SHOTS = 8
GSM8K_MIN_ACCURACY = 0.88
MAX_ALLOWED_ACCURACY_DROP = 0.05

CASE_MATRIX = [
    {
        "name": "case1_trtllm_moe_trtllm_gemm_tp1",
        "other_args": [
            "--quantization",
            "modelopt_fp4",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--fp4-gemm-backend",
            "flashinfer_trtllm",
            "--tp",
            "1",
            "--moe-a2a-backend",
            "none",
            "--trust-remote-code",
        ],
    },
    {
        "name": "case2_flashinfer_cutlass_moe_cutlass_gemm_tp1",
        "other_args": [
            "--quantization",
            "modelopt_fp4",
            "--moe-runner-backend",
            "flashinfer_cutlass",
            "--fp4-gemm-backend",
            "flashinfer_cutlass",
            "--tp",
            "1",
            "--moe-a2a-backend",
            "none",
            "--trust-remote-code",
        ],
    },
    {
        "name": "case3_cutlass_moe_cutlass_gemm_tp1",
        "other_args": [
            "--quantization",
            "modelopt_fp4",
            "--moe-runner-backend",
            "cutlass",
            "--fp4-gemm-backend",
            "flashinfer_cutlass",
            "--tp",
            "1",
            "--moe-a2a-backend",
            "none",
            "--trust-remote-code",
        ],
    },
    {
        "name": "case4_flashinfer_a2a_cutlass_moe_tp2_ep2",
        "other_args": [
            "--quantization",
            "modelopt_fp4",
            "--moe-runner-backend",
            "flashinfer_cutlass",
            "--fp4-gemm-backend",
            "flashinfer_cutlass",
            "--moe-a2a-backend",
            "flashinfer",
            "--tp",
            "2",
            "--ep",
            "2",
            "--trust-remote-code",
        ],
    },
    {
        "name": "case5_dp_allgather_flashinfer_cutlass_tp4_dp4_ep4",
        "other_args": [
            "--quantization",
            "modelopt_fp4",
            "--moe-runner-backend",
            "flashinfer_cutlass",
            "--fp4-gemm-backend",
            "flashinfer_cutlass",
            "--moe-a2a-backend",
            "none",
            "--tp",
            "4",
            "--dp",
            "4",
            "--ep",
            "4",
            "--enable-dp-attention",
            "--trust-remote-code",
        ],
    },
]


@unittest.skipIf(
    get_device_sm() < 100, "Test requires CUDA SM 100 or higher (Blackwell)"
)
@unittest.skipIf(torch.cuda.device_count() < 4, "Test requires at least 4 CUDA GPUs")
class TestQwen3Nvfp4OnlineInputScaleUpdateWeights(CustomTestCase):
    model = QWEN3_NVFP4_MODEL
    base_url = DEFAULT_URL_FOR_TEST
    port = int(base_url.split(":")[-1])

    def _run_gsm8k_eval(self, case_name):
        args = SimpleNamespace(
            num_shots=GSM8K_NUM_SHOTS,
            data_path=None,
            num_questions=GSM8K_QUESTION_COUNT,
            max_new_tokens=512,
            parallel=GSM8K_QUESTION_COUNT,
            host="http://127.0.0.1",
            port=self.port,
        )
        metrics = run_eval(args)
        print(f"{case_name=}, {metrics=}")
        return metrics

    def _update_weights_from_disk(self, case_name):
        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={
                "model_path": self.model,
                "flush_cache": True,
                "abort_all_requests": False,
            },
            timeout=300,
        )
        self.assertEqual(response.status_code, 200, msg=f"{case_name}: {response.text}")

        result = response.json()
        self.assertTrue(result.get("success"), msg=f"{case_name}: {result}")

    def _assert_model_path_is_expected(self, case_name):
        response = requests.get(self.base_url + "/get_model_info", timeout=30)
        self.assertEqual(response.status_code, 200, msg=f"{case_name}: {response.text}")
        self.assertEqual(
            response.json()["model_path"],
            self.model,
            msg=f"{case_name}: model mismatch",
        )

    def _run_single_case(self, case):
        process = None
        case_name = case["name"]
        try:
            process = popen_launch_server(
                self.model,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=case["other_args"],
                env={
                    **os.environ,
                    "SGLANG_NVFP4_ONLINE_INPUT_SCALE": "1",
                },
            )

            metrics_before = self._run_gsm8k_eval(case_name)
            self.assertGreaterEqual(
                metrics_before["accuracy"],
                GSM8K_MIN_ACCURACY,
                msg=f"{case_name}: before accuracy too low",
            )

            self._update_weights_from_disk(case_name)
            self._assert_model_path_is_expected(case_name)

            metrics_after = self._run_gsm8k_eval(case_name)
            self.assertGreaterEqual(
                metrics_after["accuracy"],
                GSM8K_MIN_ACCURACY,
                msg=f"{case_name}: after accuracy too low",
            )
            self.assertGreaterEqual(
                metrics_after["accuracy"],
                metrics_before["accuracy"] - MAX_ALLOWED_ACCURACY_DROP,
                msg=f"{case_name}: post-update accuracy regressed too much",
            )
        finally:
            if process is not None:
                kill_process_tree(process.pid)

    def test_update_weights_with_online_input_scale_matrix(self):
        for case in CASE_MATRIX:
            with self.subTest(case=case["name"]):
                self._run_single_case(case)


if __name__ == "__main__":
    unittest.main()
