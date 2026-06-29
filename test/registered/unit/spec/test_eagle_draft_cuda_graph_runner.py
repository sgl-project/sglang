"""Regression test for EAGLE draft CUDA graph seq_lens_sum padding.

This is a white-box unit test: it constructs a bare ``EAGLEDraftCudaGraphRunner``
via ``__new__`` and stubs the fields/methods that ``execute()`` touches, so the
padding bookkeeping can be exercised on CPU without a captured CUDA graph. It is
therefore coupled to ``execute()``'s internals and may need updating when that
method changes -- that coupling is intentional and the price of testing the
padding path in isolation.

The contract under test is the invariant ``seq_lens_sum == seq_lens.sum()``:
while the runner pads ``raw_bs`` up to a captured ``bs`` with fake rows, the
draft attention backends read ``seq_lens_sum`` to size/slice draft kv_indices,
so it must stay consistent with the padded ``seq_lens`` they are handed -- and
the raw value must be restored once replay finishes.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CAPTURE_BS = 4
SEQ_LEN_FILL_VALUE = 1
NUM_STEPS = 3


class _RecordingDraftBackend:
    """Records ``(batch_size, seq_lens_sum, seq_lens)`` at every point the runner
    asks for replay metadata, so a test can check they stay consistent.

    The same recorder is also invoked from the stubbed graph replay, so both the
    metadata-build and the graph-replay phases are observed.
    """

    def __init__(self):
        self.observations = []

    def init_forward_metadata_out_graph(self, forward_batch):
        self.observe("metadata_build", forward_batch)

    def observe(self, phase, forward_batch):
        seq_lens = forward_batch.seq_lens
        self.observations.append(
            SimpleNamespace(
                phase=phase,
                batch_size=forward_batch.batch_size,
                seq_lens_sum=forward_batch.seq_lens_sum,
                seq_lens=None if seq_lens is None else seq_lens.clone(),
            )
        )


class TestEagleDraftCudaGraphRunner(CustomTestCase):
    def _build_runner(self, backend):
        runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner.buffers = SimpleNamespace(
            seq_lens=torch.empty(CAPTURE_BS, dtype=torch.int32),
            out_cache_loc=torch.empty(CAPTURE_BS * NUM_STEPS, dtype=torch.int64),
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
        runner.speculative_num_steps = NUM_STEPS
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

        def replay_graph_stub(shape_key, forward_batch):
            # Observe again during graph replay to catch a premature restore.
            backend.observe("graph_replay", forward_batch)
            return None

        runner._replay_graph = replay_graph_stub
        return runner

    def _build_forward_batch(self, seq_lens, seq_lens_sum):
        raw_bs = len(seq_lens)
        return SimpleNamespace(
            batch_size=raw_bs,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            seq_lens_sum=seq_lens_sum,
            out_cache_loc=torch.arange(raw_bs * NUM_STEPS, dtype=torch.int64),
            positions=torch.arange(raw_bs, dtype=torch.int64),
            req_pool_indices=torch.arange(raw_bs, dtype=torch.int32),
            rids_int=None,
            bootstrap_room_ids_int=None,
            sampling_info=None,
            spec_info=SimpleNamespace(
                topk_p=torch.ones(raw_bs, 1, dtype=torch.float32),
                topk_index=torch.zeros(raw_bs, 1, dtype=torch.int64),
                draft_probs=None,
                hidden_states=torch.zeros(raw_bs, 2, dtype=torch.float32),
            ),
        )

    def _execute_and_observe(self, seq_lens, seq_lens_sum):
        backend = _RecordingDraftBackend()
        runner = self._build_runner(backend)
        forward_batch = self._build_forward_batch(seq_lens, seq_lens_sum)
        runner.execute(forward_batch)
        # Both the metadata build and the graph replay must have been observed.
        self.assertEqual(
            [observation.phase for observation in backend.observations],
            ["metadata_build", "graph_replay"],
        )
        return backend, forward_batch

    def test_padded_replay_keeps_seq_lens_sum_consistent(self):
        # raw_bs=2 < CAPTURE_BS=4: the batch is padded with fake rows of length
        # SEQ_LEN_FILL_VALUE, so every observation must see seq_lens_sum equal to
        # the sum of the padded seq_lens it was handed.
        raw_seq_lens = [10, 11]
        num_fake_rows = CAPTURE_BS - len(raw_seq_lens)
        expected_padded_seq_lens = raw_seq_lens + [SEQ_LEN_FILL_VALUE] * num_fake_rows
        expected_padded_sum = sum(expected_padded_seq_lens)
        backend, forward_batch = self._execute_and_observe(
            raw_seq_lens, sum(raw_seq_lens)
        )

        for observation in backend.observations:
            # Padding actually happened (guards against "sum matches an
            # un-padded seq_lens" false negatives).
            self.assertEqual(observation.batch_size, CAPTURE_BS, msg=observation.phase)
            self.assertEqual(
                observation.seq_lens.tolist(),
                expected_padded_seq_lens,
                msg=observation.phase,
            )
            # The invariant the fix maintains, plus the independently-derived
            # expected value.
            self.assertEqual(
                observation.seq_lens_sum,
                int(observation.seq_lens.sum()),
                msg=observation.phase,
            )
            self.assertEqual(
                observation.seq_lens_sum, expected_padded_sum, msg=observation.phase
            )

        # The raw batch shape is restored once replay finishes.
        self.assertEqual(forward_batch.batch_size, len(raw_seq_lens))
        self.assertEqual(forward_batch.seq_lens_sum, sum(raw_seq_lens))

    def test_unpadded_replay_leaves_seq_lens_sum_untouched(self):
        # raw_bs == CAPTURE_BS: no fake rows, so the padding branch is skipped
        # and seq_lens_sum must pass through unchanged.
        seq_lens = [3, 4, 5, 6]
        backend, forward_batch = self._execute_and_observe(seq_lens, sum(seq_lens))

        for observation in backend.observations:
            self.assertEqual(
                observation.seq_lens_sum, sum(seq_lens), msg=observation.phase
            )
            self.assertEqual(
                observation.seq_lens_sum,
                int(observation.seq_lens.sum()),
                msg=observation.phase,
            )
        self.assertEqual(forward_batch.seq_lens_sum, sum(seq_lens))

    def test_none_seq_lens_sum_is_preserved(self):
        # seq_lens_sum may be intentionally absent; padding must keep it None
        # rather than coerce it into an int.
        backend, forward_batch = self._execute_and_observe([10, 11], None)

        for observation in backend.observations:
            self.assertIsNone(observation.seq_lens_sum, msg=observation.phase)
        self.assertIsNone(forward_batch.seq_lens_sum)


if __name__ == "__main__":
    unittest.main()
