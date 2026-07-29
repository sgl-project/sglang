import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
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


class TestPrefillCudaGraphCPBodyCapture(CustomTestCase):
    def test_capture_prepares_cp_before_attention_metadata(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        forward_batch = SimpleNamespace(lora_ids=None)
        attn_backend = MagicMock()
        events = []
        runner.enable_cp_v2_body_capture = True
        runner._is_full_backend = False
        runner._capture_chunked_prefix = False
        runner.use_captured_attn_metadata = False
        runner.capture_prepare = MagicMock(return_value=(forward_batch, attn_backend))
        runner._prepare_cp_static_inputs = MagicMock(
            side_effect=lambda *args, **kwargs: events.append("cp") or 4
        )
        runner._validate_cp_flashinfer_dispatch_capacity = MagicMock(
            side_effect=lambda *args, **kwargs: events.append("capacity")
        )
        runner._init_forward_metadata_for_capture = MagicMock(
            side_effect=lambda *args, **kwargs: events.append("metadata")
        )
        runner._run_forward = MagicMock(
            side_effect=lambda *args, **kwargs: events.append("forward")
            or torch.zeros((4, 2))
        )
        runner.backend = MagicMock()
        runner.backend.capture_one.side_effect = (
            lambda shape_key, run_once, **kwargs: run_once()
        )

        runner.capture_one_shape(16)

        self.assertEqual(events, ["cp", "capacity", "metadata", "forward"])
        runner._prepare_cp_static_inputs.assert_called_once_with(
            forward_batch,
            static_num_tokens=16,
            capture=True,
        )
        runner._validate_cp_flashinfer_dispatch_capacity.assert_called_once_with(
            global_bucket_tokens=16,
            required_local_tokens=4,
        )
        self.assertEqual(
            runner.backend.capture_one.call_args.args[0],
            ShapeKey(size=16),
        )

    @patch(
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
        "cp_gather_after_forward",
        create=True,
    )
    @patch("torch.cuda.current_stream")
    def test_replay_trims_local_rows_gathers_and_runs_live_logits(
        self, mock_current_stream, mock_gather
    ):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        local_padded = torch.arange(8, dtype=torch.float32).view(4, 2)
        gathered = torch.arange(12, dtype=torch.float32).view(6, 2)
        output = object()
        stream = object()
        logits_processor = MagicMock(return_value=output)
        model = SimpleNamespace(
            capture_aux_hidden_states=False,
            forward=MagicMock(),
            lm_head=object(),
            logits_processor=logits_processor,
            pp_group=SimpleNamespace(is_last_rank=True),
        )
        runner.model_runner = SimpleNamespace(model=model)
        runner.backend = MagicMock()
        runner.backend.replay.return_value = local_padded
        runner._cp_live_local_tokens = 3
        forward_batch = SimpleNamespace(input_ids=torch.arange(6))
        static_forward_batch = SimpleNamespace()
        mock_current_stream.return_value = stream
        mock_gather.return_value = gathered

        actual = runner._execute_cp_body_capture(
            forward_batch,
            static_forward_batch,
            static_num_tokens=16,
        )

        self.assertIs(actual, output)
        runner.backend.replay.assert_called_once_with(
            ShapeKey(size=16),
            static_forward_batch,
        )
        mock_gather.assert_called_once()
        torch.testing.assert_close(mock_gather.call_args.args[0], local_padded[:3])
        self.assertIs(mock_gather.call_args.args[1], static_forward_batch)
        self.assertIs(mock_gather.call_args.args[2], stream)
        logits_processor.assert_called_once_with(
            forward_batch.input_ids,
            gathered,
            model.lm_head,
            forward_batch,
            None,
        )
        model.forward.assert_not_called()

    @patch(
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
        "cp_gather_after_forward",
        create=True,
    )
    @patch("torch.cuda.current_stream")
    def test_replay_preserves_auxiliary_hidden_states(
        self, mock_current_stream, mock_gather
    ):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        local_hidden = torch.ones((4, 2))
        local_aux = [torch.full((4, 2), 2.0)]
        gathered = torch.full((6, 2), 3.0)
        output = object()
        logits_processor = MagicMock(return_value=output)
        model = SimpleNamespace(
            capture_aux_hidden_states=True,
            lm_head=object(),
            logits_processor=logits_processor,
            pp_group=SimpleNamespace(is_last_rank=True),
        )
        runner.model_runner = SimpleNamespace(model=model)
        runner.backend = MagicMock()
        runner.backend.replay.return_value = (local_hidden, local_aux)
        runner._cp_live_local_tokens = 3
        forward_batch = SimpleNamespace(input_ids=torch.arange(6))
        static_forward_batch = SimpleNamespace()
        mock_gather.return_value = gathered

        actual = runner._execute_cp_body_capture(
            forward_batch,
            static_forward_batch,
            static_num_tokens=16,
        )

        self.assertIs(actual, output)
        logits_processor.assert_called_once()
        logits_args = logits_processor.call_args.args
        self.assertIs(logits_args[0], forward_batch.input_ids)
        self.assertIs(logits_args[1], gathered)
        self.assertIs(logits_args[2], model.lm_head)
        self.assertIs(logits_args[3], forward_batch)
        self.assertEqual(len(logits_args[4]), 1)
        torch.testing.assert_close(logits_args[4][0], local_aux[0][:3])


class TestPrefillCudaGraphCPFlashInferCapacity(CustomTestCase):
    @staticmethod
    def _dispatcher(capacity: int) -> FlashinferDispatcher:
        dispatcher = FlashinferDispatcher.__new__(FlashinferDispatcher)
        dispatcher.max_num_tokens = capacity
        return dispatcher

    def test_accepts_equal_or_larger_flashinfer_capacity(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        dispatcher = self._dispatcher(4)
        runner.moe_layers = [
            SimpleNamespace(dispatcher=dispatcher),
            SimpleNamespace(dispatcher=dispatcher),
        ]

        runner._validate_cp_flashinfer_dispatch_capacity(
            global_bucket_tokens=16,
            required_local_tokens=4,
        )
        dispatcher.max_num_tokens = 8
        runner._validate_cp_flashinfer_dispatch_capacity(
            global_bucket_tokens=16,
            required_local_tokens=4,
        )

    def test_rejects_flashinfer_capacity_smaller_than_local_capture(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.moe_layers = [
            SimpleNamespace(dispatcher=self._dispatcher(3)),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "global bucket 16.*required local rows 4.*configured capacity 3.*"
            "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
        ):
            runner._validate_cp_flashinfer_dispatch_capacity(
                global_bucket_tokens=16,
                required_local_tokens=4,
            )

    def test_ignores_non_flashinfer_dispatchers(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.moe_layers = [
            SimpleNamespace(
                dispatcher=SimpleNamespace(max_num_tokens=1),
            )
        ]

        runner._validate_cp_flashinfer_dispatch_capacity(
            global_bucket_tokens=16,
            required_local_tokens=4,
        )


if __name__ == "__main__":
    unittest.main()
