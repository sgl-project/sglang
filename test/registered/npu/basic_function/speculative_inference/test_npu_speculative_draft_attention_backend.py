import os
import unittest
from types import SimpleNamespace
from typing import Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.npu_eval_accuracy_kit import _is_pr_pipeline, run_npu_pr_smoke
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_WEIGHTS_PATH,
    
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="base-b-test-16-npu-a3")
register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)
register_npu_ci(est_time=400, suite="nightly-8-npu-a5", nightly=True)

MODEL_PATH = QWEN3_30B_A3B_WEIGHTS_PATH

def get_server_info(base_url: str) -> dict:
    response = requests.get(base_url + "/server_info", timeout=10)
    response.raise_for_status()
    return response.json()


def get_avg_spec_accept_length(base_url: str) -> Optional[float]:
    try:
        info = get_server_info(base_url)
    except Exception:
        return None
    internal_states = info.get("internal_states") or []
    if not internal_states:
        return None
    value = internal_states[0].get("avg_spec_accept_length")
    if value is None:
        return None
    return float(value)


class TestAscendSpeculativeDraftAttentionAndMoeRunner(CustomTestCase):
    """Testcase: Test configuration '--speculative-draft-attention-backend' and '--speculative-moe-runner-backend' on the GSM8K dataset is no less than 0.9.
    Also validate ACL graph health, eagle-topk constraint, and MTP accept length on Ascend NPU.

    [Test Category] Parameter
    [Test Target] --speculative-draft-attention-backend; --speculative-moe-runner-backend
    """

    #os.environ["DEEP_NORMAL_MODE_USE_INT8_QUANT"] = "1"
    os.environ["HCCL_BUFFSIZE"] = "2048"
    os.environ["SGLANG_ENABLE_OVERLAP_PLAN_SITEAM"] = "1"
    os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
    os.environ["ASCEND_USE_FIA"] = "1"
    #os.environ["SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT"] = "1"
    env = os.environ.copy()

    @classmethod
    def setUpClass(cls):
        cls.models = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            #"--quantization",
            #"modelslim",
            "--mem-fraction-static",
            0.7,
            "--disable-radix-cache",
            "--chunked-prefill-size",
            32768,
            "--tp-size",
            16,
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            "/root/.cache/modelscope/hub/models/Qwen/Qwen3-a3B_eagle3",
            "--speculative-num-steps",
            3,
            "--speculative-eagle-topk",
            1,
            "--speculative-num-draft-tokens",
            4,
            "--speculative-draft-model-quantization",
            "unquant",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "auto",
            "--max-running-requests",
            64,
            "--speculative-draft-attention-backend",
            "ascend",
            "--speculative-moe-runner-backend",
            "auto",
        ]

        cls.process = popen_launch_server(
            MODEL_PATH,
            cls.base_url,
            timeout=1500,
            other_args=cls.common_args,
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k_and_mtp_mechanism(self):
        # GSM8K evaluation (configuration + accuracy)
        if _is_pr_pipeline:
            run_npu_pr_smoke(self.base_url)
            return
        args = SimpleNamespace(
            base_url=self.base_url,
            eval_name="gsm8k",
            api="completion",
            num_examples=1319,
            num_threads=128,
            max_tokens=512,
            num_shots=5,
        )

        metrics = run_eval(args)
        score = metrics["score"]
        print(f"GSM8K score for {MODEL_PATH}: {score:.4f}")
        self.assertIsNotNone(score, "GSM8K evaluation returned no score")
        self.assertIsInstance(score, float, "Score should be a float")
        self.assertGreaterEqual(
            score,
            0.9,
            f"GSM8K score {score} below threshold 0.9",
        )

        # ------------------------------------------------------------
        # MTP mechanism assertions (added on top of config/accuracy)
        # ------------------------------------------------------------
        server_info = get_server_info(self.base_url)

        # 1. ACL graph must not be disabled (critical for NPU performance)
        self.assertFalse(
            server_info.get("disable_cuda_graph", False),
            "CUDA/ACL graph is disabled while MTP expects it enabled",
        )

        # 2. Defensive check: eagle_topk == 1 (NPU hardware constraint)
        if "speculative_eagle_topk" in server_info:
            self.assertEqual(
                server_info.get("speculative_eagle_topk"),
                1,
                "speculative-eagle-topk is not 1 (NPU constraint)",
            )

        # 3. MTP acceptance length (key indicator of speculative decoding effectiveness)
        avg_accept = get_avg_spec_accept_length(self.base_url)
        print(f"avg_spec_accept_length: {avg_accept}")

        self.assertIsNotNone(
            avg_accept,
            "avg_spec_accept_length is None",
        )
        self.assertGreaterEqual(
            avg_accept,
            1.6,
            f"MTP accept length too low: {avg_accept}",
        )


if __name__ == "__main__":
    unittest.main()
