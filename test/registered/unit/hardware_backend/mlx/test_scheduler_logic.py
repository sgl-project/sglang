import unittest
from unittest.mock import MagicMock
import torch
from sglang.srt.managers.overlap_utils import resolve_forward_inputs

class TestMlxSchedulerLogic(unittest.TestCase):
    def test_resolve_forward_inputs_mlx_prefill(self):
        # Mock ScheduleBatch
        batch = MagicMock()
        batch.prefill_input_ids_cpu = torch.tensor([1, 2, 3])
        batch.mix_running_indices = None
        batch.device = "cpu"
        batch.input_ids = None
        batch.is_spec_v2 = False

        # MLX case: future_map is None
        future_map = None

        # Call the function
        resolve_forward_inputs(batch, future_map)

        # Verify input_ids materialized
        self.assertIsNotNone(batch.input_ids)
        self.assertTrue(torch.equal(batch.input_ids, torch.tensor([1, 2, 3])))
        self.assertIsNone(batch.prefill_input_ids_cpu)

    def test_resolve_forward_inputs_mlx_mixed_no_future_map(self):
        # Even if mix_running_indices is set, if future_map is None (MLX),
        # it should fall back to just prefill instead of crashing.
        batch = MagicMock()
        batch.prefill_input_ids_cpu = torch.tensor([1, 2, 3])
        batch.mix_running_indices = torch.tensor([0])
        batch.device = "cpu"
        batch.input_ids = None
        batch.is_spec_v2 = False

        future_map = None

        resolve_forward_inputs(batch, future_map)

        self.assertIsNotNone(batch.input_ids)
        self.assertTrue(torch.equal(batch.input_ids, torch.tensor([1, 2, 3])))

if __name__ == "__main__":
    unittest.main()
