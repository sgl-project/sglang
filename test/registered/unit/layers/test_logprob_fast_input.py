"""Fast input-logprob path must match the log-softmax reference.

The fast path (SGLANG_ENABLE_FAST_INPUT_LOGPROBS) computes token / top-k /
token-ids logprobs directly from logits with a per-row logsumexp normalizer,
never materializing the full-vocab log-softmax. Same math, so results must
agree with the reference path to floating-point tolerance, with identical
top-k indices, across chunk splits and heterogeneous per-sequence params.
"""

import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logprob_processor import (
    InputLogprobProcessor,
    _logits_topk,
    compute_row_log_normalizer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

VOCAB = 11
# Heterogeneous per-sequence parameters; uniform ones hide misalignment.
TOPK_CYCLE = [2, 0, 3]
# [] is a valid probe set distinct from None (opt-out).
TOKEN_IDS_CYCLE = [[0, 3], None, [1], []]


def _build_batch(seq_specs, dtype):
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
        pruned_rows.append(torch.randn(rows, VOCAB).to(dtype))
        token_to_seq_idx.extend([idx] * rows)
        sample_pt += rows
        sample_indices.append(sample_pt)
        n_lp = extend_len - start
        input_logprob_indices.extend([lp_pt + i for i in range(n_lp)])
        lp_pt += rows
        pruned_lens.append(n_lp)
    metadata = SimpleNamespace(
        extend_return_top_logprob=True,
        extend_token_ids_logprob=True,
        top_logprobs_nums=[TOPK_CYCLE[i % 3] for i in range(len(seq_specs))],
        extend_logprob_pruned_lens_cpu=pruned_lens,
        extend_input_logprob_token_ids_gpu=torch.zeros(
            len(input_logprob_indices), dtype=torch.int64
        ),
        token_ids_logprobs=[
            TOKEN_IDS_CYCLE[i % len(TOKEN_IDS_CYCLE)] for i in range(len(seq_specs))
        ],
    )
    return (
        torch.cat(pruned_rows),
        torch.tensor(sample_indices, dtype=torch.int64),
        torch.tensor(input_logprob_indices, dtype=torch.int64),
        token_to_seq_idx,
        metadata,
    )


def _run(proc, batch, fast, chunk_size):
    pruned_states, sample_indices, input_logprob_indices, t2s, metadata = batch
    proc.enable_logprobs_chunk = chunk_size is not None
    proc.logprobs_chunk_size = chunk_size if chunk_size is not None else 10**9
    proc.enable_fast_input_logprobs = fast

    def get_logits_fn(states, lm_head, logits_metadata, **kwargs):
        return states

    return proc.forward(
        pruned_states=pruned_states,
        sample_indices=sample_indices,
        input_logprob_indices=input_logprob_indices,
        token_to_seq_idx=t2s,
        lm_head=None,
        get_logits_fn=get_logits_fn,
        logits_metadata=metadata,
    )


def _assert_nested_close(test, ref, got, label, rtol, atol):
    test.assertEqual(_shape_of(ref), _shape_of(got), label)
    ref_flat = _flatten(ref)
    got_flat = _flatten(got)
    if ref_flat:
        torch.testing.assert_close(
            torch.tensor(ref_flat, dtype=torch.float64),
            torch.tensor(got_flat, dtype=torch.float64),
            rtol=rtol,
            atol=atol,
            msg=label,
        )


def _flatten(nested):
    if isinstance(nested, list):
        return [x for item in nested for x in _flatten(item)]
    return [nested]


def _shape_of(nested):
    if isinstance(nested, list):
        return [_shape_of(item) for item in nested]
    return None


class TestFastInputLogprobs(CustomTestCase):
    def _sweep(self, dtype, rtol, atol):
        torch.manual_seed(0)
        proc = InputLogprobProcessor()
        # (extend_len, start); start == extend_len is the degenerate
        # zero-logprob-row shape.
        menu = [(1, 1), (3, 0), (4, 1), (5, 5), (6, 2)]
        tried = 0
        for n_seqs in (1, 2, 3):
            for combo in itertools.product(menu, repeat=n_seqs):
                batch = _build_batch(list(combo), dtype)
                for chunk_size in (None, 1, 2, 3, 5):
                    tried += 1
                    ref, ref_sampled = _run(proc, batch, False, chunk_size)
                    got, got_sampled = _run(proc, batch, True, chunk_size)
                    label = f"specs={list(combo)} chunk={chunk_size} dtype={dtype}"
                    # Top-k order comes from the same values shifted by a
                    # per-row constant, so indices must match exactly.
                    self.assertEqual(ref.top_logprobs_idx, got.top_logprobs_idx, label)
                    self.assertEqual(
                        ref.token_ids_logprobs_idx, got.token_ids_logprobs_idx, label
                    )
                    _assert_nested_close(
                        self,
                        ref.top_logprobs_val,
                        got.top_logprobs_val,
                        label,
                        rtol,
                        atol,
                    )
                    _assert_nested_close(
                        self,
                        ref.token_ids_logprobs_val,
                        got.token_ids_logprobs_val,
                        label,
                        rtol,
                        atol,
                    )
                    torch.testing.assert_close(
                        ref.token_logprobs.float(),
                        got.token_logprobs.float(),
                        rtol=rtol,
                        atol=atol,
                        msg=label,
                    )
                    torch.testing.assert_close(ref_sampled, got_sampled, msg=label)
        self.assertGreater(tried, 100)

    def test_fast_matches_reference_fp32(self):
        self._sweep(torch.float32, rtol=1e-5, atol=1e-5)

    def test_fast_matches_float64_truth_bf16(self):
        # bf16 log_softmax rounds near-ties together, so the reference path's
        # top-k ORDER is not reproducible from raw logits; validate the fast
        # path against float64 ground truth instead. The fast path only
        # rounds at the bf16 logits themselves (normalizer is fp32), so it
        # sits much closer to the truth than bf16 resolution.
        torch.manual_seed(0)
        proc = InputLogprobProcessor()
        menu = [(1, 1), (3, 0), (4, 1), (5, 5), (6, 2)]
        for n_seqs in (1, 2, 3):
            for combo in itertools.product(menu, repeat=n_seqs):
                batch = _build_batch(list(combo), torch.bfloat16)
                pruned_states, _, input_logprob_indices, _, metadata = batch
                truth = torch.log_softmax(pruned_states.double(), dim=-1)[
                    input_logprob_indices
                ]
                for chunk_size in (None, 2, 5):
                    got, _ = _run(proc, batch, True, chunk_size)
                    label = f"specs={list(combo)} chunk={chunk_size}"
                    self._assert_rows_match_truth(
                        got, truth, metadata, label, atol=1e-4
                    )

    def _assert_rows_match_truth(self, got, truth, metadata, label, atol):
        pt = 0
        for s, pruned_len in enumerate(metadata.extend_logprob_pruned_lens_cpu):
            if pruned_len <= 0:
                self.assertEqual(got.top_logprobs_val[s], [], label)
                continue
            k = metadata.top_logprobs_nums[s]
            probe_ids = metadata.token_ids_logprobs[s]
            for j in range(pruned_len):
                row_truth = truth[pt + j]
                vals = got.top_logprobs_val[s][j]
                idxs = got.top_logprobs_idx[s][j]
                self.assertEqual(len(vals), k, label)
                for v, i in zip(vals, idxs):
                    self.assertAlmostEqual(
                        v, row_truth[i].item(), delta=atol, msg=label
                    )
                if probe_ids is not None:
                    probe_vals = got.token_ids_logprobs_val[s][j]
                    for v, i in zip(probe_vals, probe_ids):
                        self.assertAlmostEqual(
                            v, row_truth[i].item(), delta=atol, msg=label
                        )
            pt += pruned_len

    def test_shift_invariant_large_offset(self):
        # Regression: a large common fp32 offset must not round the log-sum
        # term away. Uniform logits at 1e8 have true logprob -log(vocab).
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            logits = torch.full((4, 1000), 1e8, dtype=torch.float32, device=device)
            row_max, row_log_sum = compute_row_log_normalizer(logits)
            logprob = (logits[:, 0].float() - row_max) - row_log_sum
            expected = -torch.log(torch.tensor(1000.0))
            torch.testing.assert_close(
                logprob.cpu(), expected.expand(4), rtol=1e-5, atol=1e-5
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fast_path_run_to_run_deterministic(self):
        from sglang.srt.layers.logsumexp import row_logsumexp

        torch.manual_seed(0)
        logits = torch.randn(512, 151936, dtype=torch.bfloat16, device="cuda")
        m1, l1 = row_logsumexp(logits)
        m2, l2 = row_logsumexp(logits)
        self.assertTrue(torch.equal(m1, m2) and torch.equal(l1, l2))
        v1, i1 = _logits_topk(logits, 5)
        v2, i2 = _logits_topk(logits, 5)
        self.assertTrue(torch.equal(v1, v2) and torch.equal(i1, i2))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_logits_topk_matches_torch(self):
        torch.manual_seed(0)
        # fp32 randn is tie-free w.h.p., so indices must match torch exactly.
        logits = torch.randn(64, 151936, device="cuda")
        ref_v, ref_i = torch.topk(logits, k=5, dim=-1, sorted=True)
        got_v, got_i = _logits_topk(logits, 5)
        torch.testing.assert_close(ref_v, got_v.to(ref_v.dtype))
        self.assertTrue(torch.equal(ref_i, got_i.to(ref_i.dtype)))
        # bf16 has value ties; only the sorted values are contractual.
        logits = torch.randn(64, 151936, device="cuda", dtype=torch.bfloat16)
        ref_v, _ = torch.topk(logits, k=5, dim=-1, sorted=True)
        got_v, got_i = _logits_topk(logits, 5)
        self.assertTrue(torch.equal(ref_v, got_v.to(ref_v.dtype)))
        # Returned indices must point at the returned values.
        self.assertTrue(
            torch.equal(logits.gather(-1, got_i.long()).to(got_v.dtype), got_v)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_row_logsumexp_kernel_matches_reference(self):
        from sglang.srt.layers.logsumexp import row_logsumexp

        torch.manual_seed(0)
        for rows, cols in ((0, 128), (3, 0), (1, 1), (7, 1000), (64, 151936)):
            for dtype in (torch.bfloat16, torch.float32):
                logits = torch.randn(rows, cols, dtype=dtype, device="cuda") * 8
                got_max, got_log_sum = row_logsumexp(logits)
                self.assertEqual(got_max.dtype, torch.float32)
                self.assertEqual(got_log_sum.dtype, torch.float32)
                if not cols:
                    self.assertTrue((got_max == float("-inf")).all())
                    self.assertTrue((got_log_sum == 0).all())
                    continue
                self.assertTrue(torch.equal(got_max, logits.float().amax(-1)))
                ref_log_sum = torch.logsumexp(
                    logits.double() - got_max.double()[:, None], dim=-1
                ).float()
                torch.testing.assert_close(
                    ref_log_sum,
                    got_log_sum,
                    rtol=1e-4,
                    atol=1e-4,
                    msg=f"{rows}x{cols} {dtype}",
                )
        # Rows dominated by -inf (masked-vocab shapes) must stay nan-free.
        logits = torch.full((4, 1000), float("-inf"), device="cuda")
        logits[1, 3] = 2.5
        logits[2, :] = torch.randn(1000, device="cuda")
        got_max, got_log_sum = row_logsumexp(logits.bfloat16())
        self.assertEqual(got_max[0].item(), float("-inf"))
        self.assertAlmostEqual((got_max[1] + got_log_sum[1]).item(), 2.5, delta=1e-2)
        self.assertFalse(got_max.isnan().any() or got_log_sum.isnan().any())
        # Non-contiguous input (sliced rows) exercises the stride args.
        base = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
        view = base[::2]
        got_max, got_log_sum = row_logsumexp(view)
        torch.testing.assert_close(
            torch.logsumexp(view.double(), dim=-1).float(),
            got_max + got_log_sum,
            rtol=1e-4,
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
