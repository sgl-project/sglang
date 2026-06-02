"""Regression test for EAGLE draft CUDA graph seq_lens_sum padding."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RAW_SEQ_LENS = [10, 11]
RAW_BS = len(RAW_SEQ_LENS)
CAPTURE_BS = 4
SEQ_LEN_FILL_VALUE = 1
RAW_SEQ_LENS_SUM = sum(RAW_SEQ_LENS)
PADDED_SEQ_LENS = RAW_SEQ_LENS + [SEQ_LEN_FILL_VALUE] * (CAPTURE_BS - RAW_BS)
PADDED_SEQ_LENS_SUM = sum(PADDED_SEQ_LENS)


class _RecordingDraftBackend:
    def __init__(self):
        self.metadata_seq_lens_sum = None
        self.metadata_seq_lens = None

    def init_forward_metadata_replay_cuda_graph(self, forward_batch, bs):
        self.metadata_bs = bs
        self.metadata_batch_size = forward_batch.batch_size
        self.metadata_seq_lens_sum = forward_batch.seq_lens_sum
        self.metadata_seq_lens = forward_batch.seq_lens.clone()


class TestEagleDraftCudaGraphRunner(CustomTestCase):
    def _make_runner(self, backend, *, replay_error=None):
        runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner.buffers = SimpleNamespace(
            seq_lens=torch.empty(CAPTURE_BS, dtype=torch.int32),
            out_cache_loc=torch.empty(12, dtype=torch.int64),
            positions=torch.empty(CAPTURE_BS, dtype=torch.int64),
            rids_int=None,
            bootstrap_room_ids_int=None,
            topk_p=torch.empty(CAPTURE_BS, 1, dtype=torch.float32),
            topk_index=torch.empty(CAPTURE_BS, 1, dtype=torch.int64),
            hidden_states=torch.empty(CAPTURE_BS, 2, dtype=torch.float32),
            req_pool_indices=torch.empty(CAPTURE_BS, dtype=torch.int32),
            seq_lens_cpu=torch.empty(CAPTURE_BS, dtype=torch.int32),
        )
        runner.capture_bs = [1, CAPTURE_BS]
        runner.num_tokens_per_bs = 1
        runner.speculative_num_steps = 3
        runner.seq_len_fill_value = SEQ_LEN_FILL_VALUE
        runner.require_mlp_tp_gather = False
        runner.require_gathered_buffer = False
        runner.model_runner = SimpleNamespace(
            model_config=SimpleNamespace(vocab_size=8),
            draft_attn_backend=backend,
        )
        runner.draft_attn_backend = backend
        runner.output_buffers = {4: object()}
        runner._postprocess_output_to_raw_bs = lambda out, raw_bs: out

        def replay(forward_batch):
            runner.replay_seq_lens_sum = forward_batch.seq_lens_sum
            if replay_error is not None:
                raise replay_error

        runner._replay = replay
        return runner

    def _make_forward_batch(self):
        return SimpleNamespace(
            batch_size=RAW_BS,
            seq_lens=torch.tensor(RAW_SEQ_LENS, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(RAW_SEQ_LENS, dtype=torch.int32),
            seq_lens_sum=RAW_SEQ_LENS_SUM,
            out_cache_loc=torch.arange(6, dtype=torch.int64),
            positions=torch.tensor(RAW_SEQ_LENS, dtype=torch.int64),
            req_pool_indices=torch.arange(RAW_BS, dtype=torch.int32),
            rids_int=None,
            bootstrap_room_ids_int=None,
            spec_info=SimpleNamespace(
                topk_p=torch.ones(RAW_BS, 1, dtype=torch.float32),
                topk_index=torch.zeros(RAW_BS, 1, dtype=torch.int64),
                hidden_states=torch.zeros(RAW_BS, 2, dtype=torch.float32),
            ),
        )

    def test_pads_seq_lens_sum_during_metadata_and_replay(self):
        # raw_bs=2 with seq_lens [10, 11] is padded to captured bs=4 as
        # [10, 11, 1, 1], so replay metadata must see seq_lens_sum=23.
        backend = _RecordingDraftBackend()
        runner = self._make_runner(backend)
        forward_batch = self._make_forward_batch()

        runner.replay(forward_batch)

        self.assertEqual(backend.metadata_bs, CAPTURE_BS)
        self.assertEqual(backend.metadata_batch_size, CAPTURE_BS)
        self.assertEqual(
            backend.metadata_seq_lens_sum,
            PADDED_SEQ_LENS_SUM,
        )
        self.assertEqual(
            runner.replay_seq_lens_sum,
            PADDED_SEQ_LENS_SUM,
        )
        self.assertEqual(backend.metadata_seq_lens.tolist(), PADDED_SEQ_LENS)
        self.assertEqual(forward_batch.seq_lens_sum, RAW_SEQ_LENS_SUM)
        self.assertEqual(forward_batch.batch_size, RAW_BS)

    def test_restores_seq_lens_sum_when_replay_raises(self):
        backend = _RecordingDraftBackend()
        error = RuntimeError("boom")
        runner = self._make_runner(backend, replay_error=error)
        forward_batch = self._make_forward_batch()

        with self.assertRaisesRegex(RuntimeError, "boom"):
            runner.replay(forward_batch)

        self.assertEqual(
            backend.metadata_seq_lens_sum,
            PADDED_SEQ_LENS_SUM,
        )
        self.assertEqual(forward_batch.seq_lens_sum, RAW_SEQ_LENS_SUM)


if __name__ == "__main__":
    unittest.main()
