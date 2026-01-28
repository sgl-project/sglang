"""
Test explicit decode_dp_rank and prefill_dp_rank routing in disaggregated mode.
"""

import unittest

import requests

from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)


class TestExternalDPRanks(PDDisaggregationServerBase):
    """
    Test explicit DP rank routing with heterogeneous prefill/decode DP sizes.

    Setup:
    - Prefill: 2 DP ranks
    - Decode: 4 DP ranks

    This tests the core functionality: independent control of decode and prefill
    DP ranks for heterogeneous configurations.
    """

    PREFILL_DP_SIZE = 2
    DECODE_DP_SIZE = 4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "1",
            "--dp",
            str(cls.PREFILL_DP_SIZE),
            "--enable-dp-attention",
        ]
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
            "--tp",
            "1",
            "--dp",
            str(cls.DECODE_DP_SIZE),
            "--enable-dp-attention",
            "--base-gpu-id",
            str(cls.PREFILL_DP_SIZE),
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_heterogeneous_dp_ranks(self):
        """
        Test heterogeneous DP configuration: 2 prefill workers, 4 decode workers.

        Tests that decode_dp_rank and prefill_dp_rank can be specified independently
        to route requests across workers with different DP sizes.
        """
        test_cases = [
            # (decode_rank, prefill_rank, description)
            (0, 0, "Decode#0 <- Prefill#0"),
            (1, 1, "Decode#1 <- Prefill#1"),
            (2, 0, "Decode#2 <- Prefill#0"),  # Wrap to prefill worker 0
            (3, 1, "Decode#3 <- Prefill#1"),  # Wrap to prefill worker 1
        ]

        for decode_rank, prefill_rank, desc in test_cases:
            with self.subTest(desc=desc):
                response = requests.post(
                    self.lb_url + "/generate",
                    json={
                        "text": f"Test: {desc}",
                        "sampling_params": {"max_new_tokens": 10, "temperature": 0},
                        "decode_dp_rank": decode_rank,
                        "prefill_dp_rank": prefill_rank,
                    },
                )
                self.assertEqual(response.status_code, 200, f"Failed for {desc}")
                result = response.json()
                self.assertIn("text", result)

    def test_backward_compatibility(self):
        """Test that old data_parallel_rank field still works."""
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Backward compatibility test",
                "sampling_params": {"max_new_tokens": 10, "temperature": 0},
                "data_parallel_rank": 1,  # Old field should still work
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("text", result)

    def test_new_fields_override_old(self):
        """Test that new fields take precedence over data_parallel_rank."""
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Priority test",
                "sampling_params": {"max_new_tokens": 10, "temperature": 0},
                "data_parallel_rank": 0,  # Should be ignored
                "decode_dp_rank": 2,  # Should be used
                "prefill_dp_rank": 1,  # Should be used
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("text", result)


if __name__ == "__main__":
    unittest.main()
