"""Unit tests for the remote-instance weight export component."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_exporter import (
    WeightExporter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSendWeightsToRemoteInstance(CustomTestCase):
    def test_send_before_group_init_reports_failure(self):
        """Sending weights for a group that was never initialized must return a
        failure message to the control request, not raise out of the handler."""
        model = torch.nn.Linear(2, 2)
        exporter = WeightExporter(
            tp_rank=0,
            tp_size=1,
            gpu_id=0,
            get_model_path=lambda: "",
            get_model=lambda: model,
        )

        with patch.object(torch.distributed, "is_initialized", return_value=True):
            success, message = exporter.send_weights_to_remote_instance(
                master_address="127.0.0.1", ports="29500", group_name="sync"
            )

        self.assertFalse(success)
        self.assertIn("init_weights_send_group_for_remote_instance", message)


if __name__ == "__main__":
    unittest.main()
