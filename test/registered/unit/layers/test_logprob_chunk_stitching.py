"""Chunked input-logprob processing must match the non-chunked reference.

Regression for the cross-chunk stitching accounting: zero-logprob-row
sequences (logprob opt-outs in mixed batches, mid-chunked-prefill segments)
were skipped or double-emitted, drifting the per-request entry counts that
the scheduler asserts on.
"""

import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logprob_processor import InputLogprobProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

VOCAB = 11
# Heterogeneous per-sequence parameters; uniform ones hide misalignment.
TOPK_CYCLE = [2, 0, 3]
# [] is a valid probe set distinct from None (opt-out).
TOKEN_IDS_CYCLE = [[0, 3], None, [1], []]


def _build_batch(seq_specs, with_token_ids):
    """seq_specs: list of (extend_len, logprob_start_len). Mirrors
    LogitsProcessor._get_pruned_states for the extend-with-logprobs path."""
    pruned_rows = []
    token_to_seq_idx = []
    sample_indices = []
    input_logprob_indices = []
    pruned_lens = []
    sample_pt = -1
    lp_pt = 0
    for idx, (extend_len, start) in enumerate(seq_specs):
        eff_start = start - 1 if extend_len == start else start
        rows = extend_len - eff_start
        pruned_rows.append(torch.randn(rows, VOCAB, dtype=torch.float32))
        token_to_seq_idx.extend([idx] * rows)
        sample_pt += rows
        sample_indices.append(sample_pt)
        n_lp = extend_len - start
        input_logprob_indices.extend([lp_pt + i for i in range(n_lp)])
        lp_pt += rows
        pruned_lens.append(n_lp)
    metadata = SimpleNamespace(
        extend_return_top_logprob=True,
        extend_token_ids_logprob=with_token_ids,
        top_logprobs_nums=[TOPK_CYCLE[i % 3] for i in range(len(seq_specs))],
        extend_logprob_pruned_lens_cpu=pruned_lens,
        extend_input_logprob_token_ids_gpu=torch.zeros(
            len(input_logprob_indices), dtype=torch.int64
        ),
        token_ids_logprobs=(
            [TOKEN_IDS_CYCLE[i % len(TOKEN_IDS_CYCLE)] for i in range(len(seq_specs))]
            if with_token_ids
            else [None] * len(seq_specs)
        ),
    )
    return (
        torch.cat(pruned_rows),
        torch.tensor(sample_indices, dtype=torch.int64),
        torch.tensor(input_logprob_indices, dtype=torch.int64),
        token_to_seq_idx,
        metadata,
    )


def _run(proc, batch, chunked, chunk_size):
    pruned_states, sample_indices, input_logprob_indices, t2s, metadata = batch
    proc.enable_logprobs_chunk = chunked
    proc.logprobs_chunk_size = chunk_size

    def get_logits_fn(states, lm_head, logits_metadata, **kwargs):
        return states.float()

    return proc.forward(
        pruned_states=pruned_states,
        sample_indices=sample_indices,
        input_logprob_indices=input_logprob_indices,
        token_to_seq_idx=t2s,
        lm_head=None,
        get_logits_fn=get_logits_fn,
        logits_metadata=metadata,
    )


class TestLogprobChunkStitching(CustomTestCase):
    def _sweep(self, with_token_ids):
        torch.manual_seed(0)
        proc = InputLogprobProcessor()
        # (extend_len, start); start == extend_len is the degenerate
        # zero-logprob-row shape.
        menu = [(1, 1), (2, 2), (3, 0), (4, 1), (5, 5), (2, 0), (6, 2)]
        tried = 0
        for n_seqs in (1, 2, 3, 4):
            for combo in itertools.product(menu, repeat=n_seqs):
                batch = _build_batch(list(combo), with_token_ids)
                # Same unit as the production gate: grid rows, not logprob rows.
                total_rows = batch[0].shape[0]
                for chunk_size in (1, 2, 3, 5):
                    if total_rows <= chunk_size:
                        continue
                    tried += 1
                    ref, ref_sampled = _run(proc, batch, False, 10**9)
                    got, got_sampled = _run(proc, batch, True, chunk_size)
                    label = f"specs={list(combo)} chunk={chunk_size}"
                    self.assertEqual(ref.top_logprobs_val, got.top_logprobs_val, label)
                    self.assertEqual(ref.top_logprobs_idx, got.top_logprobs_idx, label)
                    if with_token_ids:
                        self.assertEqual(
                            ref.token_ids_logprobs_val,
                            got.token_ids_logprobs_val,
                            label,
                        )
                        self.assertEqual(
                            ref.token_ids_logprobs_idx,
                            got.token_ids_logprobs_idx,
                            label,
                        )
                    torch.testing.assert_close(
                        ref.token_logprobs, got.token_logprobs, msg=label
                    )
                    torch.testing.assert_close(ref_sampled, got_sampled, msg=label)
        self.assertGreater(tried, 1000)

    def test_top_logprobs_stitching(self):
        self._sweep(with_token_ids=False)

    def test_token_ids_logprobs_stitching(self):
        self._sweep(with_token_ids=True)


if __name__ == "__main__":
    unittest.main()
