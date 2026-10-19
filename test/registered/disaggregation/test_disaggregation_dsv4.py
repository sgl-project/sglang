import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=190, stage="base-c", runner_config="8-gpu-h200")

DSV4_FLASH_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
}

# Force retraction on decode; the interval of 20 passes is arbitrary, sparse
# enough to keep the run short.
DSV4_FLASH_DECODE_ENV = {
    **DSV4_FLASH_ENV,
    "SGLANG_TEST_RETRACT": "1",
    "SGLANG_TEST_RETRACT_INTERVAL": "20",
}

_EAGLE_SPEC_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "1",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "2",
]


class TestDisaggregationDSV4(SpecDecodingMixin, PDDisaggregationServerBase, GSM8KMixin):
    gsm8k_accuracy_thres = 0.93
    accept_length_thres = 1.8
    bs_1_speed_thres = 140

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

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
            "--nccl-port",
            cls.prefill_nccl_port,
            "--tp",
            4,
            "--attn-dp-size",
            4,
            "--moe-a2a-backend",
            "deepep",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--cuda-graph-max-bs-decode",
            "128",
            "--max-running-requests",
            "128",
            *_EAGLE_SPEC_ARGS,
            "--watchdog-timeout",
            "900",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices_for(range(4))
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
            "--nccl-port",
            cls.decode_nccl_port,
            "--tp",
            4,
            "--attn-dp-size",
            4,
            "--base-gpu-id",
            4,
            "--moe-a2a-backend",
            "deepep",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--cuda-graph-max-bs-decode",
            "128",
            "--max-running-requests",
            "128",
            *_EAGLE_SPEC_ARGS,
            "--watchdog-timeout",
            "900",
            "--enable-metrics",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices_for(range(4, 8))
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=DSV4_FLASH_DECODE_ENV,
        )

    def test_gsm8k(self):
        super().test_gsm8k()
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)
        metrics = requests.get(self.decode_url + "/metrics", timeout=30).text
        retracted = sum(
            float(line.rsplit(" ", 1)[1])
            for line in metrics.splitlines()
            if line.startswith("sglang:num_retracted_requests_total")
        )
        self.assertGreater(retracted, 0, "decode never retracted a request")


if __name__ == "__main__":
    unittest.main()
