"""Stage-a basic sanity: small-but-broad server smoke that downstream
stages depend on. Multiple sanity-kit mixins driving one shared server,
covering protocol, decode correctness, scheduler stress, occupancy, and
hellaswag accuracy."""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.kits.fwd_occupancy_kit import FwdOccupancyMixin
from sglang.test.kits.hellaswag_kit import HellaswagMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=160, stage="base-a", runner_config="1-gpu-small")
register_amd_ci(est_time=160, suite="stage-a-test-1-gpu-small-amd")


class TestBasicSanity(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    FwdOccupancyMixin,
    HellaswagMixin,
    CustomTestCase,
):
    served_model_name = DEFAULT_MODEL_NAME_FOR_TEST
    # 5090 + Llama-3.1-8B single-batch decode with overlap scheduler +
    # cuda graph measured ~99 median in CI; keep ~2pp headroom.
    fwd_occupancy_threshold = 97.0

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--cuda-graph-max-bs",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--enable-metrics",
            ],
            env={"SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
