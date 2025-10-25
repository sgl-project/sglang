from sglang.test.test_disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)


class TestDisaggregationNixl(TestDisaggregationBase):
    """Test NIXL disaggregation functionality"""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        super().setUpClass()

    @classmethod
    def start_prefill(cls, prefill_args):
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls, decode_args):
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    @classmethod
    def _get_prefill_args(
        self, prefill_tp=1, prefill_pp=1, disaggregation_ib_device="mlx5_roce0"
    ):
        return [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            str(prefill_tp),
            "--pp-size",
            str(prefill_pp),
            "--disaggregation-transfer-backend",
            "nixl",
            "--disaggregation-ib-device",
            disaggregation_ib_device,
        ]

    @classmethod
    def _get_decode_args(self, decode_tp=1, disaggregation_ib_device="mlx5_roce0"):
        return [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp",
            str(decode_tp),
            "--disaggregation-transfer-backend",
            "nixl",
            "--base-gpu-id",
            "4",
            "--disaggregation-ib-device",
            disaggregation_ib_device,
        ]
