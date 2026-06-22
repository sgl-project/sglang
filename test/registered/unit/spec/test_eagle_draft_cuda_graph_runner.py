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
    """Records what the draft attention backend observes while building replay
    metadata, so the test can assert seq_lens_sum reflects the padded rows."""

    def __init__(self):
        self.metadata_batch_size = None
        self.metadata_seq_lens_sum = None
        self.metadata_seq_lens = None

    def init_forward_metadata_out_graph(self, forward_batch):
        self.metadata_batch_size = forward_batch.batch_size
        self.metadata_seq_lens_sum = forward_batch.seq_lens_sum
        self.metadata_seq_lens = forward_batch.seq_lens.clone()


class TestEagleDraftCudaGraphRunner(CustomTestCase):
    def _make_runner(self, backend):
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
            draft_probs=None,
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
            server_args=SimpleNamespace(speculative_use_rejection_sampling=False),
            device_timer=None,
            draft_attn_backend=backend,
        )
        runner.draft_attn_backend = backend
        runner._postprocess_output_to_raw_bs = lambda out, raw_bs: out

        def replay_graph(shape_key, forward_batch):
            runner.replay_seq_lens_sum = forward_batch.seq_lens_sum
            return None

        runner._replay_graph = replay_graph
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
            sampling_info=None,
            spec_info=SimpleNamespace(
                topk_p=torch.ones(RAW_BS, 1, dtype=torch.float32),
                topk_index=torch.zeros(RAW_BS, 1, dtype=torch.int64),
                draft_probs=None,
                hidden_states=torch.zeros(RAW_BS, 2, dtype=torch.float32),
            ),
        )

    def test_pads_seq_lens_sum_during_metadata_and_replay(self):
        # raw_bs=2 with seq_lens [10, 11] is padded to captured bs=4 as
        # [10, 11, 1, 1], so replay metadata must observe seq_lens_sum=23, and
        # the raw value must be restored on the forward_batch afterwards.
        backend = _RecordingDraftBackend()
        runner = self._make_runner(backend)
        forward_batch = self._make_forward_batch()

        runner.execute(forward_batch)

        # Backend builds replay metadata against the padded batch.
        self.assertEqual(backend.metadata_batch_size, CAPTURE_BS)
        self.assertEqual(backend.metadata_seq_lens_sum, PADDED_SEQ_LENS_SUM)
        self.assertEqual(backend.metadata_seq_lens.tolist(), PADDED_SEQ_LENS)
        # Graph replay also sees the padded seq_lens_sum.
        self.assertEqual(runner.replay_seq_lens_sum, PADDED_SEQ_LENS_SUM)
        # The raw batch shape is restored once replay finishes.
        self.assertEqual(forward_batch.seq_lens_sum, RAW_SEQ_LENS_SUM)
        self.assertEqual(forward_batch.batch_size, RAW_BS)


if __name__ == "__main__":
    unittest.main()
