import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=403, stage="extra-b", runner_config="deepep-8-gpu-h200")

DSV4_FLASH_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
DSV4_FLASH_LOADER_CONFIG = '{"enable_multithread_load": true, "num_threads": 64}'
DSV4_HISPARSE_CONFIG = (
    '{"top_k":512,"device_buffer_size":4096,"host_to_device_ratio":2}'
)

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
}
DSV4_NIXL_SERVER_LAUNCH_TIMEOUT = 1800


def _has_nixl():
    try:
        import nixl._api  # noqa: F401
    except Exception:
        return False
    return True


class TestDisaggregationDSV4HiSparseBase(PDDisaggregationServerBase, GSM8KMixin):
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


@unittest.skipIf(is_in_ci(), "Flaky in CI — skip until stabilized.")
class TestDisaggregationDSV4HiSparseNixl(TestDisaggregationDSV4HiSparseBase):
    @classmethod
    def setUpClass(cls):
        PDDisaggregationServerBase.setUpClass.__func__(cls)

        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.model = try_cached_model(DSV4_FLASH_MODEL)

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(
            cls.prefill_url + "/health",
            timeout=DSV4_NIXL_SERVER_LAUNCH_TIMEOUT,
            process=cls.process_prefill,
        )
        cls.wait_server_ready(
            cls.decode_url + "/health",
            timeout=DSV4_NIXL_SERVER_LAUNCH_TIMEOUT,
            process=cls.process_decode,
        )

        cls.launch_lb()


if __name__ == "__main__":
    unittest.main()
