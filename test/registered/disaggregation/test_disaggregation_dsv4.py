import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=311, stage="base-c", runner_config="deepep-8-gpu-h200")

DSV4_FLASH_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
DSV4_FLASH_LOADER_CONFIG = '{"enable_multithread_load": true, "num_threads": 64}'
DSV4_HISPARSE_CONFIG = (
    '{"top_k":512,"device_buffer_size":4096,"host_to_device_ratio":2}'
)

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
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
            "--tp",
            4,
            "--dp",
            4,
            "--enable-dp-attention",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "128",
            *_EAGLE_SPEC_ARGS,
            "--watchdog-timeout",
            "900",
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
            4,
            "--dp",
            4,
            "--enable-dp-attention",
            "--base-gpu-id",
            4,
            "--moe-a2a-backend",
            "deepep",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "128",
            *_EAGLE_SPEC_ARGS,
            "--watchdog-timeout",
            "900",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=DSV4_FLASH_ENV,
        )


class TestDisaggregationDSV4HiSparseMooncake(PDDisaggregationServerBase, GSM8KMixin):
    gsm8k_accuracy_thres = 0.93
    gsm8k_num_questions = 200
    gsm8k_num_shots = 20

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
            "--tp",
            4,
            "--page-size",
            256,
            "--chunked-prefill-size",
            8192,
            "--max-running-requests",
            16,
            "--mem-fraction-static",
            0.9,
            "--skip-server-warmup",
            "--reasoning-parser",
            "deepseek-v4",
            "--tool-call-parser",
            "deepseekv4",
            "--model-loader-extra-config",
            DSV4_FLASH_LOADER_CONFIG,
            "--watchdog-timeout",
            "900",
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
            4,
            "--base-gpu-id",
            4,
            "--page-size",
            256,
            "--chunked-prefill-size",
            8192,
            "--max-running-requests",
            16,
            "--mem-fraction-static",
            0.9,
            "--skip-server-warmup",
            "--reasoning-parser",
            "deepseek-v4",
            "--tool-call-parser",
            "deepseekv4",
            "--model-loader-extra-config",
            DSV4_FLASH_LOADER_CONFIG,
            "--enable-hisparse",
            "--hisparse-config",
            DSV4_HISPARSE_CONFIG,
            "--watchdog-timeout",
            "900",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=DSV4_FLASH_ENV,
        )


if __name__ == "__main__":
    unittest.main()
