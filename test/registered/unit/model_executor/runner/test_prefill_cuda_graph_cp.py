import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeEmbedding:
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        values = input_ids.to(torch.float32)
        return torch.stack((values, values + 100), dim=-1)


class TestPrefillCudaGraphCPStaticInputs(CustomTestCase):
    def _runner(self) -> PrefillCudaGraphRunner:
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.model_runner = SimpleNamespace(
            model=SimpleNamespace(get_input_embeddings=lambda: _FakeEmbedding())
        )
        runner.cp_input_embeds = torch.full((16, 2), -1.0)
        runner.cp_positions = torch.full((16,), -1, dtype=torch.int64)
        runner.cp_bucket_local_tokens = {}
        return runner

    @staticmethod
    def _batch(num_tokens: int, batch_size: int = 1):
        input_ids = torch.arange(num_tokens, dtype=torch.int64)
        return SimpleNamespace(
            batch_size=batch_size,
            input_ids=input_ids,
            input_embeds=None,
            positions=torch.arange(num_tokens, dtype=torch.int64),
            extend_num_tokens=num_tokens,
            attn_cp_metadata=object(),
        )

    @patch(
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
        "cp_split_before_forward",
        create=True,
    )
    @patch(
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
        "prepare_cp_forward",
        create=True,
    )
    def test_capture_and_replay_use_fixed_local_buffers(self, mock_prepare, mock_split):
        runner = self._runner()
        capture_batch = self._batch(16)
        capture_input_ids = capture_batch.input_ids
        capture_metadata = object()

        def prepare(batch):
            self.assertIsNone(batch.attn_cp_metadata)
            batch.attn_cp_metadata = capture_metadata

        mock_prepare.side_effect = prepare
        capture_embeds = torch.tensor(
            [[1.0, 101.0], [4.0, 104.0], [11.0, 111.0], [14.0, 114.0]]
        )
        capture_positions = torch.tensor([1, 4, 11, 14], dtype=torch.int64)
        mock_split.return_value = (capture_embeds, capture_positions)

        live_rows = runner._prepare_cp_static_inputs(
            capture_batch,
            static_num_tokens=16,
            capture=True,
        )

        self.assertEqual(live_rows, 4)
        self.assertIs(capture_batch.input_ids, capture_input_ids)
        self.assertIs(capture_batch.attn_cp_metadata, capture_metadata)
        self.assertEqual(runner.cp_bucket_local_tokens, {16: 4})
        self.assertEqual(
            capture_batch.input_embeds.data_ptr(), runner.cp_input_embeds.data_ptr()
        )
        self.assertEqual(
            capture_batch.positions.data_ptr(), runner.cp_positions.data_ptr()
        )
        torch.testing.assert_close(capture_batch.input_embeds, capture_embeds)
        torch.testing.assert_close(capture_batch.positions, capture_positions)

        replay_batch = self._batch(12, batch_size=2)
        replay_metadata = object()
        seen_metadata = []

        def prepare_replay(batch):
            seen_metadata.append(batch.attn_cp_metadata)
            batch.attn_cp_metadata = replay_metadata

        mock_prepare.side_effect = prepare_replay
        replay_embeds = torch.tensor([[2.0, 102.0], [7.0, 107.0], [10.0, 110.0]])
        replay_positions = torch.tensor([2, 7, 10], dtype=torch.int64)
        mock_split.return_value = (replay_embeds, replay_positions)

        live_rows = runner._prepare_cp_static_inputs(
            replay_batch,
            static_num_tokens=16,
            capture=False,
        )

        self.assertEqual(live_rows, 3)
        self.assertEqual(seen_metadata, [None])
        self.assertIs(replay_batch.attn_cp_metadata, replay_metadata)
        self.assertEqual(
            replay_batch.input_embeds.data_ptr(), runner.cp_input_embeds.data_ptr()
        )
        self.assertEqual(
            replay_batch.positions.data_ptr(), runner.cp_positions.data_ptr()
        )
        torch.testing.assert_close(
            replay_batch.input_embeds[:3],
            replay_embeds,
        )
        torch.testing.assert_close(
            replay_batch.positions[:3],
            replay_positions,
        )
        torch.testing.assert_close(
            replay_batch.input_embeds[3],
            torch.zeros(2),
        )
        self.assertEqual(replay_batch.positions[3].item(), 0)

    @patch(
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
        "cp_split_before_forward",
        create=True,
    )
    @patch(
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
        "prepare_cp_forward",
        create=True,
    )
    def test_replay_rebuilds_metadata_when_request_count_changes(
        self, mock_prepare, mock_split
    ):
        runner = self._runner()
        metadata = []

        def prepare(batch):
            self.assertIsNone(batch.attn_cp_metadata)
            current = object()
            metadata.append(current)
            batch.attn_cp_metadata = current

        mock_prepare.side_effect = prepare
        mock_split.side_effect = (
            (torch.ones((4, 2)), torch.arange(4)),
            (torch.ones((3, 2)), torch.arange(3)),
        )

        capture_batch = self._batch(16)
        runner._prepare_cp_static_inputs(
            capture_batch, static_num_tokens=16, capture=True
        )
        replay_batch = self._batch(12, batch_size=2)
        runner._prepare_cp_static_inputs(
            replay_batch, static_num_tokens=16, capture=False
        )

        self.assertIsNot(metadata[0], metadata[1])
        self.assertIs(capture_batch.attn_cp_metadata, metadata[0])
        self.assertIs(replay_batch.attn_cp_metadata, metadata[1])
        self.assertEqual(
            capture_batch.input_embeds.data_ptr(),
            replay_batch.input_embeds.data_ptr(),
        )
        self.assertEqual(
            capture_batch.positions.data_ptr(),
            replay_batch.positions.data_ptr(),
        )

    @patch(
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
        "cp_split_before_forward",
        create=True,
    )
    @patch(
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
        "prepare_cp_forward",
        create=True,
    )
    def test_replay_rejects_more_local_rows_than_captured(
        self, mock_prepare, mock_split
    ):
        runner = self._runner()
        mock_prepare.side_effect = lambda batch: setattr(
            batch, "attn_cp_metadata", object()
        )
        mock_split.return_value = (torch.ones((4, 2)), torch.arange(4))
        runner._prepare_cp_static_inputs(
            self._batch(16), static_num_tokens=16, capture=True
        )

        mock_split.return_value = (torch.ones((5, 2)), torch.arange(5))
        with self.assertRaisesRegex(
            RuntimeError, "5 local CP rows.*captured capacity 4"
        ):
            runner._prepare_cp_static_inputs(
                self._batch(16), static_num_tokens=16, capture=False
            )


if __name__ == "__main__":
    unittest.main()
