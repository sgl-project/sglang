"""Sending weights through a group that was never initialized fails cleanly."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_exporter import (
    WeightExporter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSendWeightsToRemoteInstance(unittest.TestCase):
    def test_unknown_group_returns_failure_instead_of_raising(self):
        exporter = WeightExporter(
            tp_rank=0,
            tp_size=1,
            gpu_id=0,
            get_model_path=lambda: "model",
            get_model=lambda: torch.nn.Linear(2, 2),
        )
        with patch.object(torch.distributed, "is_initialized", return_value=True):
            success, message = exporter.send_weights_to_remote_instance(
                master_address="127.0.0.1",
                ports="29500",
                group_name="weight_send_group",
            )

        self.assertFalse(success)
        self.assertIn("weight_send_group_29500_0", message)
        self.assertIn("init_weights_send_group_for_remote_instance", message)


if __name__ == "__main__":
    unittest.main()
