import os
import unittest
from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class TestDisaggregationHybridAttentionBase(PDDisaggregationServerBase):
    """
    Base class for PD-disaggregation tests on Ascend NPU.
    Subclasses specify variant parameters via class variables.
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
    prefill_extra_args = []
    decode_extra_args = []

    gsm8k_score_threshold = 1.0

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24667"

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
        ] + cls.prefill_extra_args
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
        ] + cls.decode_extra_args
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL", None)
        super().tearDownClass()

    def run_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)

        self.assertGreater(metrics["score"], self.gsm8k_score_threshold)


class TestDisaggregationHybridAttentionGDN(TestDisaggregationHybridAttentionBase):
    """In a PD-split scenario, verify the accuracy on the GSM8K dataset using Ascend NPU transmission with TP set to 4."""

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
    prefill_extra_args = [
        "--tp-size",
        4,
    ]
    decode_extra_args = [
        "--tp-size",
        4,
        "--base-gpu-id",
        4,
    ]
    gsm8k_score_threshold = 0.93

    def test_gsm8k(self):
        self.run_gsm8k()


class TestDisaggregationHybridAttentionGDNDPDecode(
    TestDisaggregationHybridAttentionBase
):
    """Test with prefill tp=4 and decode tp=4/dp=2 with dp-attention enabled."""

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
    prefill_extra_args = [
        "--tp-size",
        4,
    ]
    decode_extra_args = [
        "--tp-size",
        4,
        "--dp-size",
        2,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--base-gpu-id",
        4,
    ]
    gsm8k_score_threshold = 0.9

    def test_gsm8k(self):
        self.run_gsm8k()


if __name__ == "__main__":
    unittest.main()
