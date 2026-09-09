import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch

import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as runner_module
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_context_size = None
        runner.max_num_tokens = 16
        return runner

    def _make_forward_batch(self, num_tokens):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
            extend_prefix_lens_cpu=[0],
            seq_lens_cpu=torch.tensor([num_tokens], dtype=torch.int64),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int64),
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))

    def test_replay_snapshot_uses_padded_token_count(self):
        runner = self._make_runner()
        runner.use_captured_attn_metadata = False
        attn_backend = mock.Mock()
        runner.model_runner = SimpleNamespace(attn_backend=attn_backend)
        forward_batch = self._make_forward_batch(8)
        static_forward_batch = self._make_forward_batch(16)

        runner._prepare_forward_metadata_for_replay(
            forward_batch,
            static_forward_batch,
            shape_key=ShapeKey(size=16),
        )

        attn_backend.init_forward_metadata.assert_called_once_with(forward_batch)
        attn_backend.prepare_prefill_shared_read_snapshot.assert_called_once_with(
            forward_batch, num_qo_tokens=16
        )

    def test_max_context_size_is_not_a_graph_key_axis(self):
        runner = self._make_runner()
        runner.max_context_size = 1024

        short = self._make_forward_batch(4)
        short.seq_lens_cpu.fill_(200)
        self.assertTrue(runner.can_run_graph(short))

        medium = self._make_forward_batch(4)
        medium.seq_lens_cpu.fill_(600)
        self.assertTrue(runner.can_run_graph(medium))
        self.assertEqual(runner._shape_key(4, short), runner._shape_key(4, medium))

    def test_rejects_context_above_fixed_maximum(self):
        runner = self._make_runner()
        runner.max_context_size = 700

        much_shorter = self._make_forward_batch(4)
        much_shorter.seq_lens_cpu.fill_(200)
        self.assertTrue(runner.can_run_graph(much_shorter))

        uncovered = self._make_forward_batch(4)
        uncovered.seq_lens_cpu.fill_(701)
        self.assertFalse(runner.can_run_graph(uncovered))

    def test_max_context_size_is_bounded_by_runtime_pool_capacity(self):
        PrefillCudaGraphRunner._validate_max_context_capacity(
            max_context_size=704,
            table_width=704,
        )
        with self.assertRaisesRegex(ValueError, "request-to-token pool capacity"):
            PrefillCudaGraphRunner._validate_max_context_capacity(
                max_context_size=704,
                table_width=700,
            )

    def test_capture_has_only_token_axis(self):
        runner = self._make_runner()
        runner.max_context_size = 1024
        runner.model_runner = SimpleNamespace(device="cpu", gpu_id=0)
        calls = []
        runner.capture_one_shape = lambda size, **kwargs: calls.append(size)

        with (
            patch.object(runner_module, "get_available_gpu_memory", return_value=1.0),
            patch.object(
                runner_module,
                "get_parallel",
                return_value=SimpleNamespace(tp_rank=1),
            ),
        ):
            runner._capture_one_stream()

        self.assertEqual(calls, [16, 4])

    def test_unsupported_path_ignores_max_context_size(self):
        runner = self._make_runner()
        runner.max_context_size = 1024

        with self.assertLogs(runner_module.logger, level="WARNING") as logs:
            runner._ignore_max_context_size("test path")

        self.assertIsNone(runner.max_context_size)
        self.assertIn("fixed metadata extent", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
