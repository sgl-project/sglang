import logging
import os
import socket
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_pd_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-8-npu-a3",
    nightly=True,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAscendSpeculativeAttentionMode(TestDisaggregationBase):
    """Testcase: Verify that in the PD disaggregation + MTP scenario, the model inference accuracy remains
    uncompromised when the Prefill service is launched with the parameter --speculative-attention-mode decode
    and the Decode service is configured with --speculative-attention-mode prefill.

    [Test Category] Parameter
    [Test Target] --speculative-attention-mode
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH
        cls.accuracy = 0.86
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            cls.store_port = s.getsockname()[1]
        os.environ["ASCEND_MF_STORE_URL"] = f"tcp://127.0.0.1:{cls.store_port}"

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disable-cuda-graph",
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--quantization",
            "modelslim",
            "--disable-radix-cache",
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            QWEN3_32B_EAGLE3_WEIGHTS_PATH,
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--speculative-attention-mode",
            "decode",
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--dtype",
            "bfloat16",
        ]
        cls.extra_envs = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
        os.environ.update(cls.extra_envs)
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--base-gpu-id",
            4,
            "--disaggregation-transfer-backend",
            "ascend",
            "--num-reserved-decode-tokens",
            128,
            "--disaggregation-decode-polling-interval",
            2,
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--quantization",
            "modelslim",
            "--disable-radix-cache",
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            QWEN3_32B_EAGLE3_WEIGHTS_PATH,
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--speculative-attention-mode",
            "prefill",
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
        ]
        cls.extra_envs = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
        os.environ.update(cls.extra_envs)
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        logger.info(f"##=== Testing accuracy: {self.model} ===##")
        args = SimpleNamespace(
            base_url=self.base_url,
            eval_name="gsm8k",
            api="completion",
            num_examples=1319,
            num_threads=128,
            max_new_tokens=512,
            num_shots=5,
            temperature=0.0,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(
            metrics["score"],
            self.accuracy,
            f"GSM8K score {metrics['score']} below threshold {self.accuracy}",
        )

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
