"""BUG REGRESSION: a FlashInfer draft extend replayed from a CUDA graph must
read each row's whole sliding window, as the same draft extend planned eagerly
does."""

import unittest

import torch

from sglang.srt.layers.attention import flashinfer_backend as fb
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WINDOW = 8
NUM_TOKENS_PER_REQ = 4


def _planned_window(*, seq_lens, prefix_lens, spec_info):
    """(lengths, start offsets) the window wrapper is planned with."""
    updater = object.__new__(fb.FlashInferIndicesUpdaterPrefill)
    updater.sliding_window_size = WINDOW
    updater.prefill_wrapper_ragged = None
    updater._swa_kv_pool = None
    updater.kv_indptr = [torch.zeros(4, dtype=torch.int32) for _ in range(2)]
    updater.qo_indptr = [torch.zeros(4, dtype=torch.int32) for _ in range(2)]
    planned = []
    updater.call_begin_forward = lambda *args, **kwargs: planned.append(args)
    updater.update_sliding_window(
        torch.arange(seq_lens.numel()),
        seq_lens,
        seq_lens.clone(),
        int(seq_lens.sum()),
        prefix_lens,
        [None, None],
        False,
        None,
        spec_info,
        plan=None,
    )
    window_args = planned[0]
    # call_begin_forward(..., paged_kernel_lens, ..., kv_start_idx, ...)
    return window_args[3], window_args[7]


class TestFlashInferDraftExtendWindow(CustomTestCase):
    def test_a_graph_replay_reads_the_window_of_every_row(self):
        # Lengths after the draft extend wrote its NUM_TOKENS_PER_REQ rows, the
        # accepted ones first; a graph replay plans without prefix lengths.
        seq_lens = torch.tensor([30, 5, 12])
        num_accept_tokens = torch.tensor([1, 4, 2])
        spec_info = EagleDraftExtendInput(
            num_correct_drafts=num_accept_tokens - 1,
            num_accept_tokens=num_accept_tokens,
            num_tokens_per_req=NUM_TOKENS_PER_REQ,
        )
        replay = _planned_window(
            seq_lens=seq_lens, prefix_lens=None, spec_info=spec_info
        )
        eager = _planned_window(
            seq_lens=seq_lens,
            prefix_lens=seq_lens - NUM_TOKENS_PER_REQ,
            spec_info=None,
        )
        lens = torch.clamp(seq_lens, max=WINDOW + NUM_TOKENS_PER_REQ)
        for planned in (replay, eager):
            torch.testing.assert_close(planned[0], lens, rtol=0, atol=0)
            torch.testing.assert_close(planned[1], seq_lens - lens, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
