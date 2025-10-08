import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.test_disaggregation_nixl_utils import (
    TestDisaggregationNixl,
)
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k


class TestDisaggregationVariableParallelismNixl(TestDisaggregationNixl):
    """Test NIXL disaggregation functionality with variable TP"""

    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy with NIXL disaggregation"""
        test_cases = [
            (1, 1, 2),  # 1 prefill TP, 1 prefill PP, 2 decode TP
            (2, 1, 4),  # 2 prefill TP, 1 prefill PP, 4 decode TP
            (2, 2, 2),  # 2 prefill TP, 2 prefill PP, 2 decode TP
        ]

        expected_accuracy = 0.65

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
                    num_questions=50,
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
