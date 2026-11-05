"""Chunked input-logprob processing must match the non-chunked reference.

Regression for the cross-chunk stitching accounting: zero-logprob-row
sequences (logprob opt-outs in mixed batches, mid-chunked-prefill segments)
were skipped or double-emitted, drifting the per-request entry counts that
the scheduler asserts on. Next-token top-k and token-ids logprobs of mixed
batches must also reach the host exactly as the per-row tensors deliver them.
"""

import json
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import (
    _LIST_INDEX_MAX_IDS,
    InputLogprobProcessor,
    OutputLogprobProcessor,
    compute_spec_logprobs,
    get_token_ids_logprobs,
    get_top_logprobs,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.utils.common import async_d2h
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.logprob_test_utils import coverage_cases
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

VOCAB = 11
# Heterogeneous per-sequence parameters; uniform ones hide misalignment.
TOPK_CYCLE = [2, 0, 3]
# [] is a valid probe set distinct from None (opt-out).
TOKEN_IDS_CYCLE = [[0, 3], None, [1], []]
# start == extend_len is the zero-logprob-row shape. Order determines the cyclic
# width-3/4 heterogeneous coverage cases.
SEQ_SPEC_MENU = ((1, 1), (2, 2), (3, 0), (4, 1), (5, 5), (2, 0), (6, 2))
# 7 singletons + 7*7 ordered pairs + 4*7 wide cases each at width 3 and 4.
EXPECTED_CASES = 112

# Next-token rows: k=0 opt-outs, k below and at the batch max, and probe sets
# that are None, [] or repeat an id.
NEXT_TOPK_CYCLE = [0, 1, 5, 3, 0, 5, 2]
NEXT_TOKEN_IDS_CYCLE = [[0, 3], None, [1], [], [7, 7, 2], None, [10]]
DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


class _HostCopyDone:
    def record(self):
        pass

    def synchronize(self):
        pass


def _cycle(values, rows):
    return [values[i % len(values)] for i in range(rows)]


def _per_row_output(logprobs, top_ks, token_ids):
    """One device tensor per row: the layout MLX and PP stages still deliver."""
    output = LogitsProcessorOutput(next_token_logits=None)
    if any(k > 0 for k in top_ks):
        output.next_token_top_logprobs_val, output.next_token_top_logprobs_idx = (
            get_top_logprobs(logprobs, top_ks, no_copy_to_cpu=True)
        )
    if any(ids is not None for ids in token_ids):
        (
            output.next_token_token_ids_logprobs_val,
            output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(logprobs, token_ids, no_copy_to_cpu=True)
    return output


def _sampler_output(logprobs, top_ks, token_ids):
    output = LogitsProcessorOutput(next_token_logits=None)
    OutputLogprobProcessor().compute_logprobs(
        logprobs=logprobs,
        top_logprobs_nums=top_ks,
        token_ids_logprobs=token_ids,
        batch_next_token_ids=torch.zeros(
            logprobs.shape[0], dtype=torch.long, device=logprobs.device
        ),
    ).write_output_to(output)
    return output


def _read_back(output, device):
    """The overlap scheduler's readback: copy_to_cpu on a side stream behind the
    producer, then finalize after copy_done. Returns the host rows as JSON."""
    result = GenerationBatchResult(
        logits_output=output,
        next_token_ids=torch.zeros(1, dtype=torch.long, device=device),
        copy_done=torch.cuda.Event() if device == "cuda" else _HostCopyDone(),
    )
    if device == "cuda":
        copy_stream = torch.cuda.Stream()
        copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            result.copy_to_cpu(return_logprob=True)
    else:
        result.copy_to_cpu(return_logprob=True)
    result.copy_done.synchronize()
    output.finalize_next_token_logprobs()
    # json.dumps rejects tensors and spells every float by its exact repr.
    return json.dumps(
        [
            output.next_token_top_logprobs_val,
            output.next_token_top_logprobs_idx,
            output.next_token_token_ids_logprobs_val,
            output.next_token_token_ids_logprobs_idx,
        ]
    )


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
        sample_indices_cpu=sample_indices,
        input_logprob_indices_cpu=input_logprob_indices,
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
        proc = InputLogprobProcessor(vocab_size=VOCAB)
        combos = list(coverage_cases(SEQ_SPEC_MENU, max_seqs=4))
        self.assertEqual(len(combos), EXPECTED_CASES)
        tried = 0
        for combo in combos:
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
        self.assertGreater(tried, 100)

    def test_top_logprobs_stitching(self):
        self._sweep(with_token_ids=False)

    def test_token_ids_logprobs_stitching(self):
        self._sweep(with_token_ids=True)

    def test_finalizing_input_logprobs_preserves_request_boundaries(self):
        rows = [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0]])]
        copy_done = Mock()
        output = LogitsProcessorOutput(
            next_token_logits=None,
            input_token_ids_logprobs_val=[[rows[0][:1], rows[0][1:]], [rows[1]]],
            input_logprobs_copy_done=copy_done,
        )
        output.finalize_input_logprobs()
        self.assertEqual(output.input_token_ids_logprobs_val, [[[1.0], [2.0]], [[3.0]]])
        copy_done.synchronize.assert_called_once_with()
        self.assertIsNone(output.input_logprobs_copy_done)

        # Multi-item scoring returns one tensor per request, with no borrow event.
        output.input_token_ids_logprobs_val = rows
        output.finalize_input_logprobs()
        self.assertIs(output.input_token_ids_logprobs_val, rows)

    def test_mixed_batch_token_ids_copy_to_cpu_like_the_eager_path(self):
        # One request with token ids, one without, and one with an empty probe set.
        token_ids = [[1, 3], None, []]
        logprobs = torch.log_softmax(torch.randn(len(token_ids), 5), dim=-1)
        vals, idxs = get_token_ids_logprobs(logprobs, token_ids, no_copy_to_cpu=True)
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_token_ids_logprobs_val=vals,
            next_token_token_ids_logprobs_idx=idxs,
        )
        SchedulerBatchResultProcessor.move_logprobs_to_cpu(
            None,
            batch=SimpleNamespace(return_logprob=True),
            logits_output=output,
        )
        # The empty entry must not be a view that keeps the full logprobs alive.
        self.assertIsNone(vals[1]._base)
        expected = get_token_ids_logprobs(logprobs, token_ids, no_copy_to_cpu=False)
        self.assertEqual(
            (
                output.next_token_token_ids_logprobs_val,
                output.next_token_token_ids_logprobs_idx,
            ),
            expected,
        )
        self.assertEqual(expected[0][1:], [[], []])
        self.assertEqual(expected[1], [[1, 3], [], []])


class TestNextTokenLogprobReadback(CustomTestCase):
    def test_packed_readback_matches_per_row_tensors(self):
        """Batch-level copies plus host slicing must rebuild bit-identical rows."""
        torch.manual_seed(0)
        cases = [
            (_cycle(NEXT_TOPK_CYCLE, 7), _cycle(NEXT_TOKEN_IDS_CYCLE, 7)),
            (_cycle(NEXT_TOPK_CYCLE, 64), _cycle(NEXT_TOKEN_IDS_CYCLE, 64)),
            ([4], [[2]]),
            # Every probe set empty: the gather has no element.
            ([0, 2, 0], [[], None, []]),
            ([3] * 5, [None] * 5),
        ]
        for device in DEVICES:
            for top_ks, token_ids in cases:
                logprobs = torch.log_softmax(
                    torch.randn(len(top_ks), VOCAB, device=device), dim=-1
                )
                with self.subTest(device=device, top_ks=top_ks, token_ids=token_ids):
                    packed = _sampler_output(logprobs, top_ks, token_ids)
                    per_row = _per_row_output(logprobs, top_ks, token_ids)
                    self.assertEqual(
                        _read_back(packed, device), _read_back(per_row, device)
                    )

    def test_large_probe_sets_match_per_row_tensors(self):
        """Probe sets of ~1k ids per row, with ids past the int16 range of a real
        vocabulary, must read back the same rows as the per-row tensors."""
        torch.manual_seed(0)
        vocab = 151936
        token_ids = [
            list(range(vocab - 1000, vocab)),
            None,
            torch.randint(0, vocab, (1024,)).tolist(),
            [],
            [vocab - 1] * 999,
            torch.randint(0, vocab, (1001,)).tolist(),
        ]
        top_ks = [0, 5, 0, 0, 20, 1]
        # Above the list threshold, so the numpy-built gather index runs.
        self.assertGreater(
            sum(len(ids) for ids in token_ids if ids is not None), _LIST_INDEX_MAX_IDS
        )
        for device in DEVICES:
            logprobs = torch.log_softmax(
                torch.randn(len(token_ids), vocab, device=device), dim=-1
            )
            with self.subTest(device=device):
                packed = _sampler_output(logprobs, top_ks, token_ids)
                per_row = _per_row_output(logprobs, top_ks, token_ids)
                self.assertEqual(
                    _read_back(packed, device), _read_back(per_row, device)
                )

    def test_spec_rows_match_per_row_tensors(self):
        """A speculative step packs bs * width rows; row b * width + j must carry
        request b's top-k and probe logprobs, in chain and tree layouts."""
        torch.manual_seed(0)
        top_ks = [0, 5, 2, 0, 3]
        token_ids = [[1, 4], None, [], [9, 9], None]
        bs = len(top_ks)
        for device, width, tree in (
            (device, width, tree)
            for device in DEVICES
            for width in (1, 3)
            for tree in (False, True)
        ):
            num_rows = bs * 4 if tree else bs * width
            logits = torch.randn(num_rows, VOCAB, device=device)
            predict = torch.randint(0, VOCAB, (num_rows,), device=device)
            batch = SimpleNamespace(
                seq_lens=torch.zeros(bs, device=device),
                sampling_info=SimpleNamespace(is_all_greedy=True),
                top_logprobs_nums=top_ks,
                token_ids_logprobs=token_ids,
            )
            output = LogitsProcessorOutput(next_token_logits=logits)
            if tree:
                accept_index = torch.randperm(num_rows, device=device)[: bs * width]
                accept_index = accept_index.view(bs, width)
                compute_spec_logprobs(
                    batch=batch,
                    logits_output=output,
                    predict=predict,
                    accept_index=accept_index,
                )
                accepted_logits = logits[accept_index.reshape(-1)]
            else:
                compute_spec_logprobs(
                    batch=batch,
                    logits_output=output,
                    predict=predict,
                    chain_stride=width,
                )
                accepted_logits = logits
            reference = torch.log_softmax(accepted_logits, dim=-1)
            reference.clamp_(min=torch.finfo(reference.dtype).min)
            expected = _per_row_output(
                reference,
                [k for k in top_ks for _ in range(width)],
                [ids for ids in token_ids for _ in range(width)],
            )
            with self.subTest(device=device, width=width, tree=tree):
                self.assertEqual(
                    _read_back(output, device), _read_back(expected, device)
                )

    def test_readback_copy_count_does_not_grow_with_rows(self):
        """A step costs the same number of device-to-host copies at any batch
        size; one copy per row dominates the scheduler's decode step."""
        counts = []
        for rows in (4, 64):
            logprobs = torch.log_softmax(torch.randn(rows, VOCAB), dim=-1)
            output = _sampler_output(
                logprobs,
                _cycle(NEXT_TOPK_CYCLE, rows),
                _cycle(NEXT_TOKEN_IDS_CYCLE, rows),
            )
            result = GenerationBatchResult(
                logits_output=output,
                next_token_ids=torch.zeros(rows, dtype=torch.long),
                copy_done=_HostCopyDone(),
            )
            with patch(
                "sglang.srt.managers.utils._async_d2h", side_effect=async_d2h
            ) as copy:
                result.copy_to_cpu(return_logprob=True)
            counts.append(copy.call_count)
        self.assertEqual(counts[0], counts[1])

    def test_pp_relay_finalizes_like_the_sending_rank(self):
        """PP stages relay per-row tensors as pickled metadata; the receiving
        rank must finalize them to the lists the sending rank would build."""
        torch.manual_seed(0)
        rows = 9
        logprobs = torch.log_softmax(torch.randn(rows, VOCAB), dim=-1)
        top_ks = _cycle(NEXT_TOPK_CYCLE, rows)
        token_ids = _cycle(NEXT_TOKEN_IDS_CYCLE, rows)
        payload = get_logprob_dict_from_result(
            GenerationBatchResult(
                logits_output=_sampler_output(logprobs, top_ks, token_ids)
            )
        )
        received, _, _ = get_logprob_from_pp_outputs(
            PPProxyTensors(pickle.loads(pickle.dumps(payload)))
        )
        self.assertEqual(
            _read_back(received, "cpu"),
            _read_back(_sampler_output(logprobs, top_ks, token_ids), "cpu"),
        )

    def test_pp_payload_size_scales_linearly_with_requested_logprobs(self):
        """Doubling rows must not pickle each row's shared backing tensor again.

        Per-row views of a packed probe tensor otherwise serialize quadratic
        storage, although their finalized response values are still correct.
        """
        sizes = []
        for rows in (32, 64):
            logprobs = torch.full((rows, 128), -0.5)
            output = _sampler_output(logprobs, [5] * rows, [list(range(128))] * rows)
            payload = get_logprob_dict_from_result(
                GenerationBatchResult(logits_output=output)
            )
            encoded = pickle.dumps(payload)
            sizes.append(len(encoded))
            received, _, _ = get_logprob_from_pp_outputs(
                PPProxyTensors(pickle.loads(encoded))
            )
            self.assertEqual(_read_back(received, "cpu"), _read_back(output, "cpu"))
        self.assertLess(sizes[1], 2.5 * sizes[0])


if __name__ == "__main__":
    unittest.main()
