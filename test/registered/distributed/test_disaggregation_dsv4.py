import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=400, suite="stage-c-test-8-gpu-h20")

DSV4_FLASH_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

DSV4_FLASH_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}

_EAGLE_SPEC_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]


class TestDisaggregationDSV4(PDDisaggregationServerBase):
    PREFILL_DP_SIZE = 4
    DECODE_DP_SIZE = 4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        cls.model = try_cached_model(DSV4_FLASH_MODEL)

        cls.start_prefill()
        cls.start_decode()

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
            "--tp",
            str(cls.PREFILL_DP_SIZE),
            "--dp",
            str(cls.PREFILL_DP_SIZE),
            "--enable-dp-attention",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "256",
            "--mem-fraction-static",
            "0.7",
            *_EAGLE_SPEC_ARGS,
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=DSV4_FLASH_ENV,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.DECODE_DP_SIZE),
            "--dp",
            str(cls.DECODE_DP_SIZE),
            "--enable-dp-attention",
            "--base-gpu-id",
            str(cls.PREFILL_DP_SIZE),
            "--moe-a2a-backend",
            "deepep",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "256",
            "--mem-fraction-static",
            "0.7",
            *_EAGLE_SPEC_ARGS,
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=DSV4_FLASH_ENV,
        )

    def test_gsm8k(self):
        """End-to-end PD-disagg accuracy through the LB."""
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=64,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_gsm8k_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
