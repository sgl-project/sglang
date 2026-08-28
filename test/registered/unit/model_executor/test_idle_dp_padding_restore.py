"""An idle DP rank must leave the forward with the empty batch it arrived with.

Under DP attention every rank joins every forward, including the ones with no
requests. Whether an idle rank's 0-row inputs get PADDED is decided by
arithmetic it does not control: ``DpPaddingMode.get_dp_padding_mode`` switches
from SUM_LEN to MAX_LEN once ``sum_len * 2 >= max_len * dp_size``, which for one
draft row per busy rank means "half the ranks are busy" -- 4 of 8 flips it, 1 of
8 does not.

``post_forward_mlp_sync_batch`` truncates the OUTPUT back to zero rows. It used
to leave the padded INPUT tensors on the batch, which is fatal because the batch
is reused: ``EagleDraftWorker.draft_forward`` runs ``speculative_num_steps``
forwards on one ForwardBatch and pairs ``forward_batch.positions`` with the
logits that forward returned. With a 1-row ``positions`` and 0-row logits, the
topk=1 draft kernel's precondition fails, and four of eight DP ranks died on the
first decode step after four requests arrived.

CPU-only: this is shape bookkeeping, so a stub attention backend supplying the
seq-len fill value is the only collaborator needed. The batch is built with
``__new__`` + explicit fields rather than ``init_new`` so the test states exactly
which fields it is talking about.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SEQ_LEN_FILL_VALUE = 1
_MODEL_RUNNER = SimpleNamespace(
    attn_backend=SimpleNamespace(
        get_cuda_graph_seq_len_fill_value=lambda: SEQ_LEN_FILL_VALUE
    )
)


class _StubDraftInput:
    """The EagleDraftInput fields `_pad_inputs_to_size` pads."""

    def __init__(self, rows, *, hidden=4):
        self.hidden_states = torch.zeros((rows, hidden), dtype=torch.float32)
        self.topk_p = torch.zeros((rows, 1), dtype=torch.float32)
        self.topk_index = torch.zeros((rows, 1), dtype=torch.int64)
        self.draft_probs = None
        self.num_tokens_per_req = 1

    def is_draft_input(self):
        return True


def _make_batch(rows, *, forward_mode):
    batch = ForwardBatch.__new__(ForwardBatch)
    batch.forward_mode = forward_mode
    batch.batch_size = rows
    batch.input_ids = torch.arange(rows, dtype=torch.int64)
    batch.req_pool_indices = torch.arange(rows, dtype=torch.int64)
    batch.positions = torch.arange(rows, dtype=torch.int64)
    batch.out_cache_loc = torch.arange(rows, dtype=torch.int64)
    batch.seq_lens = torch.full((rows,), 7, dtype=torch.int64)
    batch.seq_lens_cpu = torch.full((rows,), 7, dtype=torch.int64)
    batch.seq_lens_sum = 7 * rows
    batch.lora_ids = None
    batch.encoder_lens = None
    batch.mamba_track_indices = None
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None
    batch.mrope_positions = None
    batch.extend_seq_lens = None
    batch.rids_int = None
    batch.bootstrap_room_ids_int = None
    batch.sampling_info = None
    batch.spec_info = _StubDraftInput(rows)
    batch._original_forward_mode = None
    batch._original_batch_size = None
    return batch


def _pad_like_mlp_sync(batch, *, num_tokens, bs):
    """The two bookkeeping writes prepare_mlp_sync_batch makes around padding.

    `_original_forward_mode` stays None: that is the plain-IDLE case, the one the
    draft loop reuses. The fabricated-row conversions (hybrid-SSM, prefill
    breakable graph) set it, and keep their fabricated shape by design.
    """
    batch._original_batch_size = batch.batch_size
    batch.batch_size = bs
    batch._pad_inputs_to_size(_MODEL_RUNNER, num_tokens, bs)


def _logits(rows, *, vocab=8, hidden=4):
    return SimpleNamespace(
        next_token_logits=torch.zeros((rows, vocab), dtype=torch.float32),
        hidden_states=torch.zeros((rows, hidden), dtype=torch.float32),
    )


class TestIdleDpPaddingRestore(CustomTestCase):
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_dp_size", return_value=8)
    def test_max_len_is_reached_at_half_the_ranks(self, _mock_dp_size):
        """The trigger is a threshold, so it stays invisible until it is hit."""
        self.assertEqual(
            DpPaddingMode.get_dp_padding_mode(
                is_extend_in_batch=False, global_num_tokens=[1, 0, 0, 0, 0, 0, 0, 0]
            ),
            DpPaddingMode.SUM_LEN,
        )
        self.assertEqual(
            DpPaddingMode.get_dp_padding_mode(
                is_extend_in_batch=False, global_num_tokens=[0, 0, 1, 1, 1, 1, 0, 0]
            ),
            DpPaddingMode.MAX_LEN,
        )

    def test_idle_rank_comes_back_empty(self):
        batch = _make_batch(0, forward_mode=ForwardMode.IDLE)
        _pad_like_mlp_sync(batch, num_tokens=1, bs=1)
        # The padding really happened, else the restore below proves nothing.
        self.assertEqual(batch.positions.shape[0], 1)
        self.assertEqual(batch.seq_lens.shape[0], 1)
        self.assertEqual(batch.seq_lens_sum, SEQ_LEN_FILL_VALUE)
        self.assertEqual(batch.spec_info.hidden_states.shape[0], 1)

        logits_output = _logits(1)
        batch.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(logits_output.next_token_logits.shape[0], 0)
        for field in (
            "input_ids",
            "req_pool_indices",
            "positions",
            "out_cache_loc",
            "seq_lens",
            "seq_lens_cpu",
        ):
            self.assertEqual(getattr(batch, field).shape[0], 0, f"{field} left padded")
        self.assertEqual(batch.seq_lens_sum, 0)
        self.assertEqual(batch.batch_size, 0)
        self.assertEqual(batch.spec_info.topk_p.shape[0], 0)
        self.assertEqual(batch.spec_info.topk_index.shape[0], 0)
        self.assertEqual(batch.spec_info.hidden_states.shape[0], 0)

    def test_positions_and_logits_agree_after_restore(self):
        """The invariant the draft loop asserts on, stated directly."""
        batch = _make_batch(0, forward_mode=ForwardMode.IDLE)
        _pad_like_mlp_sync(batch, num_tokens=1, bs=1)
        logits_output = _logits(1)
        batch.post_forward_mlp_sync_batch(logits_output)
        self.assertEqual(
            batch.positions.shape[0], logits_output.next_token_logits.shape[0]
        )

    def test_busy_decode_rank_is_untouched(self):
        """No collateral damage: a rank WITH requests keeps the narrower restore
        it always had (per-token tensors to the pre-padding token count, per-request
        tensors to the real batch size) and must not be emptied."""
        batch = _make_batch(1, forward_mode=ForwardMode.DECODE)
        _pad_like_mlp_sync(batch, num_tokens=2, bs=2)
        self.assertEqual(batch.positions.shape[0], 2)

        logits_output = _logits(2)
        batch.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(batch.positions.shape[0], 1)
        self.assertEqual(batch.seq_lens.shape[0], 1)
        self.assertEqual(batch.req_pool_indices.shape[0], 1)
        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(logits_output.next_token_logits.shape[0], 1)

    def test_converted_idle_rank_keeps_its_fabricated_request(self):
        """A hybrid-SSM / prefill-graph idle rank is given a whole dummy request
        by prepare_mlp_sync_batch instead of having an empty one padded. Undoing
        only the padding half would leave a batch that is neither, so the restore
        deliberately skips it -- pinned here so the skip stays a decision."""
        batch = _make_batch(0, forward_mode=ForwardMode.IDLE)
        _pad_like_mlp_sync(batch, num_tokens=1, bs=1)
        batch._original_forward_mode = ForwardMode.IDLE
        batch.forward_mode = ForwardMode.TARGET_VERIFY

        batch.post_forward_mlp_sync_batch(_logits(1))

        self.assertEqual(batch.forward_mode, ForwardMode.IDLE)
        self.assertEqual(batch.positions.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
