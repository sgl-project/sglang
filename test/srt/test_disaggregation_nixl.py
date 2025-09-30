import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_pd_server,
)


class TestDisaggregationNixl(TestDisaggregationBase):
    """Test NIXL disaggregation functionality"""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        print(f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} {cls.decode_port=}")

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

    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy with NIXL disaggregation"""
        test_cases = [
            (1, 1, 2),  # 1 prefill TP, 1 prefill PP, 2 decode TP
            (2, 2, 2),  # 2 prefill TP, 2 prefill PP, 2 decode TP
            (1, 2, 2),  # 1 prefill TP, 2 prefill PP, 2 decode TP
            (2, 1, 4),  # 2 prefill TP, 1 prefill PP, 4 decode TP
            (2, 2, 4),  # 2 prefill TP, 2 prefill PP, 4 decode TP
        ]

        expected_accuracy = 0.70

        for prefill_tp, prefill_pp, decode_tp in test_cases:
            with self.subTest(
                prefill_tp=prefill_tp, prefill_pp=prefill_pp, decode_tp=decode_tp
            ):

                prefill_args = self._get_prefill_args(prefill_tp, prefill_pp)
                decode_args = self._get_decode_args(decode_tp)

                # Non blocking start servers
                self.start_prefill(prefill_args)
                self.start_decode(decode_args)

                # Block until both
                self.wait_server_ready(self.prefill_url + "/health")
                self.wait_server_ready(self.decode_url + "/health")

                self.launch_lb()

                # Run GSM8K evaluation
                args = SimpleNamespace(
                    num_shots=5,
                    data_path=None,
                    num_questions=100,
                    max_new_tokens=512,
                    parallel=64,
                    host=f"http://{self.base_host}",
                    port=int(self.lb_port),
                )

                metrics = run_eval_few_shot_gsm8k(args)
                print(
                    f"Evaluation metrics for config {prefill_tp}/{prefill_pp}/{decode_tp}: {metrics}"
                )
                self.assertGreater(metrics["accuracy"], expected_accuracy)

                self.tearDownClass()


if __name__ == "__main__":
    unittest.main()
