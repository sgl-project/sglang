"""An IDLE attention-DP rank must execute the same PDMux layer slices."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPDMuxSplitDispatch(unittest.TestCase):
    def _runner(self):
        runner = object.__new__(ModelRunner)
        runner.device = "cuda"
        runner.is_draft_worker = False
        runner.hisparse_coordinator = None
        runner.decode_cuda_graph_runner = SimpleNamespace(
            can_run_graph=Mock(return_value=True),
            execute=Mock(return_value="decode graph output"),
        )
        runner._prepare_eager_forward_batch = Mock()
        runner._maybe_execute_deferred_mamba_cow_and_clear = Mock()
        runner.forward_split_prefill = Mock(return_value=None)
        return runner

    def test_explicit_idle_split_bypasses_decode_graph(self):
        runner = self._runner()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE, global_num_tokens_cpu=None
        )
        with (
            patch(
                "sglang.srt.model_executor.model_runner.has_forward_context",
                return_value=True,
            ),
            patch(
                "sglang.srt.model_executor.model_runner.get_global_dwdp_manager",
                return_value=None,
            ),
        ):
            output = runner._forward_raw(batch, None, split_forward_count=3)
        self.assertIsNone(output.logits_output)
        self.assertFalse(output.can_run_graph)
        runner.decode_cuda_graph_runner.execute.assert_not_called()
        runner.forward_split_prefill.assert_called_once_with(
            batch, reinit_attn_backend=False, forward_count=3
        )

    def test_ordinary_idle_keeps_decode_graph_path(self):
        runner = self._runner()
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)
        with patch(
            "sglang.srt.model_executor.model_runner.has_forward_context",
            return_value=True,
        ):
            output = runner._forward_raw(batch, None)
        self.assertEqual(output.logits_output, "decode graph output")
        self.assertTrue(output.can_run_graph)
        runner.forward_split_prefill.assert_not_called()

    def test_idle_first_slice_initializes_backend_to_clear_stale_state(self):
        runner = self._runner()
        del runner.forward_split_prefill
        runner.model_config = SimpleNamespace(num_hidden_layers=6)
        runner.attn_backend = SimpleNamespace(init_forward_metadata=Mock())
        runner.device_timer = None
        runner.model = SimpleNamespace(forward_split_prefill=Mock(return_value=None))
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            split_index=0,
            input_ids=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
        )
        runner.forward_split_prefill(batch, forward_count=2)
        self.assertEqual(batch.split_index, 2)
        runner.attn_backend.init_forward_metadata.assert_called_once_with(batch)
        runner.model.forward_split_prefill.assert_called_once_with(
            batch.input_ids, batch.positions, batch, (0, 2)
        )


if __name__ == "__main__":
    unittest.main()
