"""Fast input-logprob path must match the log-softmax reference.

The fast path (SGLANG_ENABLE_FAST_INPUT_LOGPROBS) computes token / top-k /
token-ids logprobs directly from logits with a per-row logsumexp normalizer,
never materializing the full-vocab log-softmax. Same math, so results must
agree with the reference path to floating-point tolerance across chunk splits
and heterogeneous per-sequence params. For distributed Top-N, cross-shard
ties intentionally accept any valid equal-cutoff token IDs because
``torch.topk`` does not promise a monolithic tie order.
"""

import itertools
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.layers.logprob_processor import (
    DistributedLogprobContext,
    InputLogprobProcessor,
    compute_row_log_normalizer,
    get_distributed_topk,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.logprob_test_utils import coverage_cases
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

VOCAB = 11
# Heterogeneous per-sequence parameters; uniform ones hide misalignment.
TOPK_CYCLE = [2, 0, 3]
# [] is a valid probe set distinct from None (opt-out).
TOKEN_IDS_CYCLE = [[0, 3], None, [1], []]
# start == extend_len is the zero-logprob-row shape. Order determines the cyclic
# width-3 heterogeneous coverage cases.
SEQ_SPEC_MENU = ((1, 1), (3, 0), (4, 1), (5, 5), (6, 2))
# 5 singletons + 5*5 ordered pairs + 4*5 wide cases at width 3.
EXPECTED_CASES = 50


class _TorchDistributedGroup:
    def __init__(self):
        self.device_group = dist.group.WORLD

    def all_reduce(self, tensor):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.device_group)
        return tensor

    def all_gather(self, tensor, dim):
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor, group=self.device_group)
        return torch.cat(gathered, dim=dim)


def _distributed_input_logprob_worker(rank, world_size, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(17)
        rows, vocab, shard_width = 7, 11, 6
        full_logits = torch.randn(rows, vocab, dtype=torch.float32) * 3
        # Force a winner on the nonzero shard. The random remainder keeps the
        # test sensitive to ordinary cross-rank ordering as well.
        full_logits[0, 10] = 20.0
        if rank == 0:
            local_logits = full_logits[:, :shard_width].clone()
            vocab_start, vocab_end = 0, 6
        else:
            # A deliberately dominant padding value catches accidental
            # participation in the distributed normalizer.
            padding = torch.full((rows, 1), 1e4)
            local_logits = torch.cat((full_logits[:, 6:], padding), dim=1)
            vocab_start, vocab_end = 6, vocab

        group = _TorchDistributedGroup()

        def get_local_logits_fn(states, lm_head, logits_metadata):
            return states

        def gather_sampled_logits_fn(local_sampled):
            shards = [torch.empty_like(local_sampled) for _ in range(world_size)]
            dist.all_gather(shards, local_sampled, group=group.device_group)
            return torch.cat(shards, dim=-1)[:, :vocab]

        context = DistributedLogprobContext(
            tp_group=group,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_end,
            vocab_size=vocab,
            get_local_logits_fn=get_local_logits_fn,
            gather_sampled_logits_fn=gather_sampled_logits_fn,
        )
        sample_indices = torch.tensor([2, 5, 6], dtype=torch.long)
        input_indices = torch.arange(7, dtype=torch.long)
        target_ids = torch.tensor([0, 6, 10, 1, 7, 9, 4], dtype=torch.long)
        metadata = SimpleNamespace(
            extend_return_top_logprob=True,
            extend_token_ids_logprob=True,
            # Rank one has only five valid columns, exercising candidate
            # padding for k=7. The middle request confirms k=0 does not
            # disturb its row accounting.
            top_logprobs_nums=[7, 0, 3],
            extend_logprob_pruned_lens_cpu=[3, 3, 1],
            extend_input_logprob_token_ids_gpu=target_ids,
            token_ids_logprobs=[[0, 6, 10], [2, 7], []],
        )

        proc = InputLogprobProcessor()
        proc.enable_fast_input_logprobs = True
        proc.enable_logprobs_chunk = True
        proc.logprobs_chunk_size = 2

        def unused_gathered_logits(*args, **kwargs):
            raise AssertionError("distributed path unexpectedly gathered prompt rows")

        result, sampled_logits = proc.forward(
            pruned_states=local_logits,
            sample_indices=sample_indices,
            input_logprob_indices=input_indices,
            token_to_seq_idx=[0, 0, 0, 1, 1, 1, 2],
            lm_head=None,
            get_logits_fn=unused_gathered_logits,
            logits_metadata=metadata,
            distributed_context=context,
        )

        reference = torch.log_softmax(full_logits, dim=-1)
        expected_targets = reference[input_indices, target_ids]
        torch.testing.assert_close(
            result.token_logprobs, expected_targets, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            sampled_logits, full_logits[sample_indices], rtol=0, atol=0
        )

        expected_explicit = [
            reference[:3, [0, 6, 10]].tolist(),
            reference[3:6, [2, 7]].tolist(),
            [[]],
        ]
        _assert_nested_close(
            unittest.TestCase(),
            expected_explicit,
            result.token_ids_logprobs_val,
            "distributed explicit token IDs",
            rtol=1e-5,
            atol=1e-5,
        )
        assert result.token_ids_logprobs_idx == [
            [[0, 6, 10]] * 3,
            [[2, 7]] * 3,
            [[]],
        ]
        reference_top_vals, reference_top_idx = reference.topk(7, dim=-1)
        expected_top_vals = [
            reference_top_vals[:3].tolist(),
            [[]] * 3,
            [reference_top_vals[6, :3].tolist()],
        ]
        expected_top_idx = [
            reference_top_idx[:3].tolist(),
            [[]] * 3,
            [reference_top_idx[6, :3].tolist()],
        ]
        _assert_nested_close(
            unittest.TestCase(),
            expected_top_vals,
            result.top_logprobs_val,
            "distributed prompt top-k",
            rtol=1e-5,
            atol=1e-5,
        )
        assert result.top_logprobs_idx == expected_top_idx
        assert result.top_logprobs_idx[0][0][0] == 10

        # Cross-shard Nth-boundary ties are not required to select the same
        # IDs as one monolithic torch.topk. They must, however, retain the
        # exact target entry, N ranked scores, all strictly-above-cutoff IDs,
        # and only distinct equal-cutoff alternatives for the remainder.
        tie_full_logits = torch.tensor(
            [
                [10.0, 9.0, 8.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, -5.0, -6.0],
                [9.0, 8.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, -4.0, -5.0, -6.0],
            ]
        )
        if rank == 0:
            tie_local_logits = tie_full_logits[:, :shard_width].clone()
        else:
            tie_padding = torch.full((2, 1), 1e4)
            tie_local_logits = torch.cat((tie_full_logits[:, 6:], tie_padding), dim=1)

        tie_metadata = SimpleNamespace(
            extend_return_top_logprob=True,
            extend_token_ids_logprob=False,
            top_logprobs_nums=[5],
            extend_logprob_pruned_lens_cpu=[2],
            extend_input_logprob_token_ids_gpu=torch.tensor([8, 3]),
        )
        tie_proc = InputLogprobProcessor()
        tie_proc.enable_fast_input_logprobs = True
        tie_proc.enable_logprobs_chunk = True
        tie_proc.logprobs_chunk_size = 1
        tie_result, tie_sampled_logits = tie_proc.forward(
            pruned_states=tie_local_logits,
            sample_indices=torch.empty(0, dtype=torch.long),
            input_logprob_indices=torch.arange(2, dtype=torch.long),
            token_to_seq_idx=[0, 0],
            lm_head=None,
            get_logits_fn=unused_gathered_logits,
            logits_metadata=tie_metadata,
            distributed_context=context,
        )
        assert tie_sampled_logits.shape == (0, vocab)
        _assert_tied_distributed_topk(
            unittest.TestCase(), tie_full_logits, torch.tensor([8, 3]), tie_result
        )
    finally:
        dist.destroy_process_group()


def _build_batch(seq_specs, dtype, vocab=VOCAB):
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
        pruned_rows.append(torch.randn(rows, vocab).to(dtype))
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


def _assert_tied_distributed_topk(test, full_logits, target_ids, result):
    """Check the public distributed tie contract without assuming tie IDs."""
    reference = torch.log_softmax(full_logits, dim=-1)
    row_ids = torch.arange(full_logits.shape[0])
    torch.testing.assert_close(
        result.token_logprobs,
        reference[row_ids, target_ids],
        rtol=1e-6,
        atol=1e-6,
        msg="target-token logprobs remain exact across the tied cutoff",
    )

    test.assertEqual(len(result.top_logprobs_val), 1)
    test.assertEqual(len(result.top_logprobs_idx), 1)
    test.assertEqual(len(result.top_logprobs_val[0]), full_logits.shape[0])
    test.assertEqual(len(result.top_logprobs_idx[0]), full_logits.shape[0])
    for row, (values, indices) in enumerate(
        zip(result.top_logprobs_val[0], result.top_logprobs_idx[0])
    ):
        test.assertEqual(len(values), 5)
        test.assertEqual(len(indices), 5)
        test.assertEqual(len(set(indices)), 5, "Top-N IDs must be distinct")
        test.assertTrue(
            all(0 <= token_id < full_logits.shape[1] for token_id in indices)
        )
        test.assertTrue(
            all(values[pos] >= values[pos + 1] for pos in range(len(values) - 1)),
            "Top-N values must remain rank-descending",
        )

        returned = set(indices)
        expected_values = reference[row, torch.tensor(indices)]
        torch.testing.assert_close(
            torch.tensor(values), expected_values, rtol=1e-6, atol=1e-6
        )
        cutoff = torch.topk(full_logits[row], 5).values[-1]
        strictly_above = set(
            torch.nonzero(full_logits[row] > cutoff, as_tuple=False).flatten().tolist()
        )
        equal_cutoff = set(
            torch.nonzero(full_logits[row] == cutoff, as_tuple=False).flatten().tolist()
        )
        test.assertTrue(strictly_above <= returned)
        test.assertTrue(returned - strictly_above <= equal_cutoff)
        test.assertEqual(len(returned & equal_cutoff), 5 - len(strictly_above))


class TestFastInputLogprobs(CustomTestCase):
    def test_distributed_topk_rejects_over_vocab_k(self):
        with self.assertRaisesRegex(ValueError, "exceeds global vocabulary size"):
            get_distributed_topk(
                torch.empty((1, 3)),
                valid_vocab_size=3,
                max_k=4,
                vocab_start_index=0,
                vocab_size=3,
                tp_group=None,
            )

    def test_distributed_exact_input_logprobs_cpu(self):
        # Real collectives, but Gloo/CPU only: validates TP=2 normalization,
        # padding exclusion, local/remote token ownership, heterogeneous
        # prompt top-k, ragged explicit probes, chunk stitching, and
        # sampled-row-only vocab gathering.
        with tempfile.TemporaryDirectory() as tmp_dir:
            mp.spawn(
                _distributed_input_logprob_worker,
                args=(2, f"{tmp_dir}/init"),
                nprocs=2,
                join=True,
            )

    def _sweep(self, dtype, rtol, atol):
        torch.manual_seed(0)
        proc = InputLogprobProcessor()
        combos = list(coverage_cases(SEQ_SPEC_MENU, max_seqs=3))
        self.assertEqual(len(combos), EXPECTED_CASES)
        tried = 0
        for combo in combos:
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
        for combo in coverage_cases(SEQ_SPEC_MENU, max_seqs=3):
            batch = _build_batch(list(combo), torch.bfloat16)
            pruned_states, _, input_logprob_indices, _, metadata = batch
            truth = torch.log_softmax(pruned_states.double(), dim=-1)[
                input_logprob_indices
            ]
            for chunk_size in (None, 2, 5):
                got, _ = _run(proc, batch, True, chunk_size)
                label = f"specs={list(combo)} chunk={chunk_size}"
                self._assert_rows_match_truth(got, truth, metadata, label, atol=1e-4)

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

    def test_fast_path_emits_fp32_logprobs(self):
        # Chosen dtype policy: the fast path returns fp32 token logprobs
        # regardless of logits dtype (the normalizer is fp32, so fp32 is the
        # true precision of the result), while the log_softmax path keeps
        # the logits dtype. Runs on CPU CI so the policy is pinned even
        # where the CUDA kernels never execute.
        proc = InputLogprobProcessor()
        batch = _build_batch([(4, 1), (3, 0)], torch.bfloat16)
        got, _ = _run(proc, batch, True, None)
        self.assertEqual(got.token_logprobs.dtype, torch.float32)
        ref, _ = _run(proc, batch, False, None)
        self.assertEqual(ref.token_logprobs.dtype, torch.bfloat16)

    def test_logsumexp_module_imports(self):
        # Runs on CPU CI too: catches import rot in the CUDA-only kernel
        # module, whose imports otherwise only execute on GPU machines.
        import sglang.srt.layers.logsumexp  # noqa: F401

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_tiny_shapes(self):
        from sglang.srt.layers.logsumexp import row_logsumexp_topk

        for rows, cols, k in ((1, 1, 1), (3, 2, 2), (2, 3, 1), (2, 5, 5)):
            logits = torch.randn(rows, cols, device="cuda")
            got_m, got_ls, got_v, got_i = row_logsumexp_topk(logits, k)
            ref_v, ref_i = torch.topk(logits, k, dim=-1, sorted=True)
            self.assertTrue(torch.equal(got_v, ref_v.float()), (rows, cols, k))
            self.assertTrue(torch.equal(got_i, ref_i), (rows, cols, k))
            torch.testing.assert_close(
                got_m + got_ls,
                torch.logsumexp(logits.double(), dim=-1).float(),
                rtol=1e-5,
                atol=1e-5,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_run_to_run_deterministic(self):
        from sglang.srt.layers.logsumexp import row_logsumexp_topk

        torch.manual_seed(0)
        logits = torch.randn(512, 151936, dtype=torch.bfloat16, device="cuda")
        a = row_logsumexp_topk(logits, 5)
        b = row_logsumexp_topk(logits, 5)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a, b)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_logsumexp_topk_matches_torch(self):
        from sglang.srt.layers.logsumexp import row_logsumexp, row_logsumexp_topk

        torch.manual_seed(0)
        for k in (1, 2, 3, 5, 8):
            for dtype in (torch.float32, torch.bfloat16):
                logits = torch.randn(64, 151936, dtype=dtype, device="cuda")
                got_m, got_ls, got_v, got_i = row_logsumexp_topk(logits, k)
                # The (max, log_sum) pair is bitwise the non-fused kernel's.
                ref_m, ref_ls = row_logsumexp(logits)
                self.assertTrue(torch.equal(got_m, ref_m), (k, dtype))
                self.assertTrue(torch.equal(got_ls, ref_ls), (k, dtype))
                ref_v, ref_i = torch.topk(logits, k, dim=-1, sorted=True)
                self.assertTrue(torch.equal(got_v, ref_v.float()), (k, dtype))
                if dtype == torch.float32:
                    # fp32 randn is tie-free w.h.p.: indices match exactly.
                    self.assertTrue(torch.equal(got_i, ref_i), (k, dtype))
                else:
                    # bf16 has value ties; indices must point at the values.
                    self.assertTrue(
                        torch.equal(logits.gather(-1, got_i).float(), got_v),
                        (k, dtype),
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_topk_tie_break_is_lowest_index(self):
        from sglang.srt.layers.logsumexp import row_logsumexp_topk

        logits = torch.zeros(1, 1000, device="cuda")
        logits[0, [7, 3, 500]] = 5.0
        _, _, _, got_i = row_logsumexp_topk(logits, 3)
        self.assertEqual(got_i[0].tolist(), [3, 7, 500])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_topk_inf_rows(self):
        from sglang.srt.layers.logsumexp import row_logsumexp_topk

        logits = torch.full((3, 1000), float("-inf"), device="cuda")
        logits[1, 3] = 2.5
        _, _, got_v, got_i = row_logsumexp_topk(logits.bfloat16(), 2)
        self.assertEqual(got_i[0].tolist(), [0, 1])
        self.assertEqual(got_i[1, 0].item(), 3)
        self.assertFalse(got_v.isnan().any().item())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fast_path_end_to_end_on_cuda(self):
        # Exercises the fused-kernel integration inside _forward_by_chunk
        # (the CPU sweeps only cover the torch fallbacks), including the
        # k > FUSED_TOPK_MAX_K fallback.
        torch.manual_seed(0)
        proc = InputLogprobProcessor()
        for k_override in (None, 20):
            # k=20 exceeds FUSED_TOPK_MAX_K, exercising the torch fallback;
            # it needs a vocab that can supply 20 entries.
            batch = _build_batch([(4, 1), (6, 2), (3, 0)], torch.float32, vocab=64)
            pruned_states, sample_indices, lp_indices, t2s, metadata = batch
            if k_override is not None:
                metadata.top_logprobs_nums = [k_override] * 3
            batch = (
                pruned_states.cuda(),
                sample_indices.cuda(),
                lp_indices.cuda(),
                t2s,
                metadata,
            )
            metadata.extend_input_logprob_token_ids_gpu = (
                metadata.extend_input_logprob_token_ids_gpu.cuda()
            )
            for chunk_size in (None, 2, 5):
                ref, _ = _run(proc, batch, False, chunk_size)
                got, _ = _run(proc, batch, True, chunk_size)
                label = f"k_override={k_override} chunk={chunk_size}"
                self.assertEqual(ref.top_logprobs_idx, got.top_logprobs_idx, label)
                _assert_nested_close(
                    self, ref.top_logprobs_val, got.top_logprobs_val, label, 1e-5, 1e-5
                )
                torch.testing.assert_close(
                    ref.token_logprobs, got.token_logprobs, rtol=1e-5, atol=1e-5
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
