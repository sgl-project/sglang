"""Input candidate selectors replace the top-k of the requests that set one.

A selector's rows must land on the positions they were computed for however
the prompt is chunked, and every other request in the batch must get the
values it would get without the selector request.

Every forward here goes through the carriers a scheduler uses: Req ->
ScheduleBatch -> ForwardBatch.init_new -> LogitsProcessor ->
LogitsProcessorOutput.
"""

import contextlib
import dataclasses
import itertools
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.logprob_processor import (
    InputLogprobProcessor,
    compute_row_log_normalizer,
)
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    compute_extend_logprob_start_len,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    input_candidate_selectors,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

VOCAB = 31
K = 5


class _StridedProcessor(InputLogprobProcessor):
    """A selector is a stride: candidate j of position p is token
    (p * stride + j) % vocab_size, with weight p + j / 16."""

    # True stands for a graph pool with room for the chunks: the results then
    # leave through pinned host copies.
    pinned_copy = False

    def _can_borrow_logprob_memory(self, chunk_size, borrow_logits):
        return self.pinned_copy

    def select_input_candidates(
        self, selector, logits, k, first_position, log_normalizer
    ):
        columns = torch.arange(k, device=logits.device)
        rows = torch.arange(logits.shape[0], device=logits.device)
        positions = first_position + rows[:, None]
        ids = (positions * selector + columns) % self.vocab_size
        values = logits.gather(1, ids)
        if log_normalizer is not None:
            row_max, row_log_sum = log_normalizer
            values = (values - row_max[:, None]) - row_log_sum[:, None]
        # fp64, as a selector may compute them; the result carries fp32.
        return values, ids, positions + columns.double() / 16


class _SelfNormalizingProcessor(_StridedProcessor):
    """Normalizes the rows it is handed itself, in fp32 whatever their dtype."""

    def select_input_candidates(
        self, selector, logits, k, first_position, log_normalizer
    ):
        _, ids, weights = super().select_input_candidates(
            selector, logits, k, first_position, None
        )
        return logits.float().log_softmax(-1).gather(1, ids), ids, weights


class _UnweightedProcessor(_StridedProcessor):
    """Its candidates come without weights."""

    def select_input_candidates(
        self, selector, logits, k, first_position, log_normalizer
    ):
        values, ids, _ = super().select_input_candidates(
            selector, logits, k, first_position, log_normalizer
        )
        return values, ids, None


class _IdentityHead(LogitsProcessor):
    """The hidden states of a forward are its logits."""

    def _get_logits(self, hidden_states, lm_head, logits_metadata, **kwargs):
        return hidden_states


class _StridedLogitsProcessor(_IdentityHead):
    input_logprob_processor_cls = _StridedProcessor


class _SelfNormalizingLogitsProcessor(_IdentityHead):
    input_logprob_processor_cls = _SelfNormalizingProcessor


class _UnweightedLogitsProcessor(_IdentityHead):
    input_logprob_processor_cls = _UnweightedProcessor


def _schedule_batch(reqs, mode):
    """The prefill batch of reqs: (logits rows of the extend, prefix_len,
    logprob_start_len, top_logprobs_num, selector[, token ids to probe]) per
    request, with the fields prepare_for_extend derives from the requests."""
    device = reqs[0][0].device
    batch_reqs = []
    for n, (rows, prefix_len, start, top_k, selector, *probes) in enumerate(reqs):
        req = Req(
            f"req-{n}",
            "",
            array("q", [0] * (prefix_len + len(rows))),
            SamplingParams(),
            return_logprob=True,
            top_logprobs_num=top_k,
            token_ids_logprob=probes[0] if probes else None,
        )
        req.logprob.input_candidate_selector = selector
        req.logprob_start_len = start
        req.refresh_fill_ids()
        req.prefix_len, req.extend_end = prefix_len, prefix_len + len(rows)
        batch_reqs.append(req)
    extend_lens = [req.extend_len for req in batch_reqs]
    starts = [
        compute_extend_logprob_start_len(
            logprob_start_len=req.logprob_start_len,
            prefix_len=req.prefix_len,
            extend_len=req.extend_len,
            full_untruncated_fill_len=len(req.full_untruncated_fill_ids),
        )
        for req in batch_reqs
    ]
    seq_lens = torch.tensor([req.extend_end for req in batch_reqs])
    num_tokens = sum(extend_lens)
    return ScheduleBatch(
        reqs=batch_reqs,
        device=device.type,
        model_config=SimpleNamespace(is_encoder_decoder=False, is_matryoshka=False),
        forward_mode=mode,
        input_ids=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        req_pool_indices=torch.arange(len(batch_reqs), device=device),
        seq_lens=seq_lens.to(device),
        seq_lens_cpu=seq_lens,
        seq_lens_sum=int(seq_lens.sum()),
        out_cache_loc=torch.arange(num_tokens, device=device),
        prefix_lens=[req.prefix_len for req in batch_reqs],
        extend_lens=extend_lens,
        extend_num_tokens=num_tokens,
        extend_logprob_start_lens=starts,
        return_logprob=True,
        top_logprobs_nums=[req.logprob.top_logprobs_num for req in batch_reqs],
        token_ids_logprobs=[req.logprob.token_ids_logprob for req in batch_reqs],
        # The token each logprob row is scored on: not the same one everywhere.
        extend_input_logprob_token_ids=torch.arange(num_tokens - sum(starts))
        % reqs[0][0].shape[1],
    )


def _run(processor, *reqs, mode=ForwardMode.EXTEND, **forward_fields):
    """One prefill forward of the reqs of _schedule_batch, with ``forward_fields``
    set on its forward batch: its output, as the scheduler reads it."""
    batch = _schedule_batch(reqs, mode)
    model_runner = SimpleNamespace(
        device=batch.device,
        is_draft_worker=False,
        model_config=SimpleNamespace(
            requires_mm_token_modalities=False, model_is_mrope=False
        ),
        runner_ids=mock.MagicMock(),
        prefill_attention_backend_str="torch_native",
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        lora_manager=None,
        attn_backend=mock.MagicMock(),
        attn_tp_sequence_sharded=lambda _: False,
    )
    forward_batch = ForwardBatch.init_new(
        batch,
        model_runner,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
    )
    forward_batch = dataclasses.replace(forward_batch, **forward_fields)
    output = processor(
        batch.input_ids, torch.cat([req[0] for req in reqs]), None, forward_batch
    )
    output.finalize_input_logprobs()
    return output


@contextlib.contextmanager
def _top_k_widths():
    """The widths of the torch top-k calls made inside the block."""
    widths = []
    top_k = torch.Tensor.topk

    def record_width(logits, k, *args, **kwargs):
        widths.append(k)
        return top_k(logits, k, *args, **kwargs)

    with mock.patch.object(torch.Tensor, "topk", record_width):
        yield widths


class TestInputCandidateSelector(CustomTestCase):
    def _processor(
        self, processor_cls=_StridedLogitsProcessor, vocab_size=VOCAB, device="cpu"
    ):
        """A logits processor and its input logprob hook, in a runtime context
        that lasts the test."""
        override = get_context().override_server_args(device=device)
        override.install()
        self.addCleanup(override.restore)
        processor = processor_cls(SimpleNamespace(vocab_size=vocab_size))
        hook = processor.input_logprob_processor
        hook.enable_logprobs_chunk = True
        return processor, hook

    def _check_selected(self, hook, passes, selector, k, logits, first):
        """The (output, sequence) passes of one request hold, in order, what
        one selector call returns for its rows from position `first` on."""
        rows = logits[first - 1 :]
        if hook.enable_fast_input_logprobs:
            normalizer = compute_row_log_normalizer(rows)
        else:
            rows, normalizer = rows.log_softmax(-1), None
        values, ids, weights = hook.select_input_candidates(
            selector, rows, k, first, normalizer
        )
        self.assertEqual(
            [row for r, n in passes for row in r.input_top_logprobs_idx[n]],
            ids.tolist(),
        )
        self.assertEqual(
            [row for r, n in passes for row in r.input_top_logprobs_val[n]],
            values.tolist(),
        )
        spans = [span for r, n in passes for span in r.input_top_logprobs_weights[n]]
        if weights is None:
            self.assertEqual(spans, [])
            return
        got = torch.cat(spans)
        self.assertEqual(got.dtype, torch.float32)
        self.assertTrue(torch.equal(got, weights.float().cpu()))

    def test_selector_rows_follow_positions_and_leave_other_requests_alone(self):
        generator = torch.Generator().manual_seed(0)
        a, b, c, d = (
            torch.randn(n, VOCAB, generator=generator) * 3 for n in (17, 6, 9, 12)
        )
        processor, hook = self._processor()
        for fast, chunk_size, start in itertools.product(
            (False, True), (1, 2, 3, 5, 4096), (0, 1, 4, 9)
        ):
            hook.enable_fast_input_logprobs, hook.logprobs_chunk_size = fast, chunk_size
            # B (top-3) and C (top-9, starting later) are plain top-k requests.
            plain = [(b, 0, 0, 3, None), (c, 0, 2, 9, None)]
            alone = _run(processor, *plain)
            # A's prompt arrives in three prefill forwards: between B and C, by
            # itself, and after a request without logprobs and D, a selector
            # request with another stride and K behind a cached prefix of its
            # own.
            first = _run(processor, plain[0], (a[:6], 0, start, K, 3), plain[1])
            second = _run(processor, (a[6:13], 6, start, K, 3))
            third = _run(
                processor,
                (b, 0, 6, 0, None),
                (d[4:], 4, start, 7, 2),
                (a[13:], 13, start, K, 3),
            )
            with self.subTest(fast=fast, chunk_size=chunk_size, start=start):
                self._check_selected(
                    hook, [(first, 1), (second, 0), (third, 2)], 3, K, a, start + 1
                )
                self._check_selected(hook, [(third, 1)], 2, 7, d, max(start, 4) + 1)
                # B and C get the bits they get without A, and no weights.
                self.assertEqual(
                    first.input_top_logprobs_val[::2], alone.input_top_logprobs_val
                )
                self.assertEqual(
                    first.input_top_logprobs_idx[::2], alone.input_top_logprobs_idx
                )
                self.assertEqual(first.input_top_logprobs_weights[::2], [[], []])
                self.assertIsNone(alone.input_top_logprobs_weights)

    def test_a_selector_request_changes_nothing_else_in_any_batch_order(self):
        generator = torch.Generator().manual_seed(1)
        a, b, c = (torch.randn(n, VOCAB, generator=generator) * 3 for n in (11, 6, 9))
        processor, hook = self._processor()
        # A and B also probe token ids.
        requests = [(a, 0, 0, K, 3, [1, 2, 30]), (b, 0, 0, 3, None, [0, 3])]
        requests.append((c, 0, 2, 9, None))
        without_selector = (a, 0, 0, K, None, [1, 2, 30])
        for fast, chunk_size, order in itertools.product(
            (False, True), (3, 4096), itertools.permutations(range(3))
        ):
            hook.enable_fast_input_logprobs, hook.logprobs_chunk_size = fast, chunk_size
            selected = _run(processor, *(requests[n] for n in order))
            plain = _run(
                processor, *(requests[n] if n else without_selector for n in order)
            )
            alone = _run(processor, *(requests[n] for n in order if n))
            others = [at for at, n in enumerate(order) if n]
            with self.subTest(fast=fast, chunk_size=chunk_size, order=order):
                self._check_selected(hook, [(selected, order.index(0))], 3, K, a, 1)
                # What does not come from a selector is what the same batch
                # computes without one: A's rows are read, not written.
                self.assertTrue(
                    torch.equal(
                        selected.input_token_logprobs, plain.input_token_logprobs
                    )
                )
                self.assertEqual(
                    selected.input_token_ids_logprobs_val,
                    plain.input_token_ids_logprobs_val,
                )
                self.assertEqual(
                    selected.input_token_ids_logprobs_idx,
                    plain.input_token_ids_logprobs_idx,
                )
                self.assertTrue(
                    torch.equal(selected.next_token_logits, plain.next_token_logits)
                )
                # B and C get the bits they get in a batch without A.
                self.assertEqual(
                    [selected.input_top_logprobs_val[at] for at in others],
                    alone.input_top_logprobs_val,
                )
                self.assertEqual(
                    [selected.input_top_logprobs_idx[at] for at in others],
                    alone.input_top_logprobs_idx,
                )
                self.assertEqual(
                    [selected.input_top_logprobs_weights[at] for at in others],
                    [[], []],
                )

    def test_selectors_are_read_for_prefill_batches_only(self):
        generator = torch.Generator().manual_seed(2)
        a, b = (torch.randn(n, VOCAB, generator=generator) * 3 for n in (7, 6))
        processor, hook = self._processor()
        hook.enable_fast_input_logprobs, hook.logprobs_chunk_size = True, 4
        requests = [(b, 0, 0, 3, None), (a, 0, 0, K, 3)]
        for mode, return_logprob, carried in (
            (ForwardMode.EXTEND, True, True),
            # A prefill chunk that runs with the decoding requests.
            (ForwardMode.MIXED, True, True),
            (ForwardMode.EXTEND, False, False),
            (ForwardMode.DECODE, True, False),
            (ForwardMode.IDLE, True, False),
            (ForwardMode.TARGET_VERIFY, True, False),
        ):
            with self.subTest(mode=mode, return_logprob=return_logprob):
                batch = _schedule_batch(requests, mode)
                batch.return_logprob = return_logprob
                self.assertEqual(
                    input_candidate_selectors(batch), [None, 3] if carried else None
                )
        self._check_selected(
            hook, [(_run(processor, *requests, mode=ForwardMode.MIXED), 1)], 3, K, a, 1
        )

    def test_selector_requests_take_no_part_in_the_top_k(self):
        generator = torch.Generator().manual_seed(3)
        a, b = (torch.randn(n, VOCAB, generator=generator) * 3 for n in (9, 6))
        processor, hook = self._processor()
        # One chunk each, then chunks of 6 rows: B's, and four of A's alone.
        for fast, (chunk_size, expected) in itertools.product(
            (False, True), ((4096, [3, 0]), (6, [3, 0, 0, 0, 0]))
        ):
            hook.enable_fast_input_logprobs, hook.logprobs_chunk_size = fast, chunk_size
            with _top_k_widths() as widths:
                _run(processor, (b, 0, 0, 3, None), (a, 0, 0, K, 3))
                _run(processor, (a, 0, 0, K, 3))
            with self.subTest(fast=fast, chunk_size=chunk_size):
                # A chunk's vocabulary-wide top-k is as wide as its plain
                # requests ask, whatever K the selector request has.
                self.assertEqual(widths, expected)

    def test_a_selector_request_without_a_logprob_row_selects_nothing(self):
        generator = torch.Generator().manual_seed(4)
        a, b, c = (torch.randn(n, VOCAB, generator=generator) * 3 for n in (6, 6, 9))
        processor, hook = self._processor()
        # A's logprobs start past this prefill chunk: it keeps one row, to
        # sample from, and none to score.
        requests = [(a, 0, 9, K, 3), (b, 0, 0, 3, None), (c, 0, 2, 9, None)]
        for fast, chunk_size, order in itertools.product(
            (False, True), (1, 3, 4096), itertools.permutations(range(3))
        ):
            hook.enable_fast_input_logprobs, hook.logprobs_chunk_size = fast, chunk_size
            with mock.patch.object(
                hook, "select_input_candidates", wraps=hook.select_input_candidates
            ) as select:
                output = _run(processor, *(requests[n] for n in order))
            with self.subTest(fast=fast, chunk_size=chunk_size, order=order):
                select.assert_not_called()
                self.assertEqual(output.input_top_logprobs_weights, [[], [], []])
                self.assertEqual(output.input_top_logprobs_val[order.index(0)], [])
                self.assertEqual(output.input_top_logprobs_idx[order.index(0)], [])

    def test_selector_values_stay_fp32_beside_half_precision_logits(self):
        generator = torch.Generator().manual_seed(5)
        a, b = (
            (torch.randn(n, VOCAB, generator=generator) * 3).to(torch.bfloat16)
            for n in (9, 6)
        )
        processor, hook = self._processor(_SelfNormalizingLogitsProcessor)
        hook.logprobs_chunk_size = 4
        neighbour = (b, 0, 0, 3, None)
        for fast, neighbours in itertools.product((False, True), ([], [neighbour])):
            hook.enable_fast_input_logprobs = fast
            output = _run(processor, *neighbours, (a, 0, 0, K, 3))
            rows = a if fast else a.log_softmax(-1)
            values, _, _ = hook.select_input_candidates(3, rows, K, 1, None)
            with self.subTest(fast=fast, neighbours=len(neighbours)):
                # Half precision cannot hold them, or this proves nothing.
                self.assertFalse(torch.equal(values, values.to(torch.bfloat16).float()))
                self._check_selected(hook, [(output, len(neighbours))], 3, K, a, 1)
                if neighbours:
                    # Widening the neighbour's half-precision values is exact.
                    alone = _run(processor, neighbour)
                    self.assertEqual(
                        output.input_top_logprobs_val[0],
                        alone.input_top_logprobs_val[0],
                    )

    def test_a_selector_without_weights_replaces_the_candidates_only(self):
        generator = torch.Generator().manual_seed(8)
        a, b = (torch.randn(n, VOCAB, generator=generator) * 3 for n in (9, 6))
        processor, hook = self._processor(_UnweightedLogitsProcessor)
        hook.logprobs_chunk_size = 4
        for fast in (False, True):
            hook.enable_fast_input_logprobs = fast
            output = _run(processor, (b, 0, 0, 3, None), (a, 0, 0, K, 3))
            with self.subTest(fast=fast):
                self._check_selected(hook, [(output, 1)], 3, K, a, 1)
                self.assertEqual(output.input_top_logprobs_weights, [[], []])

    def test_selector_without_an_implementation_fails(self):
        processor, _ = self._processor(_IdentityHead)
        with self.assertRaisesRegex(
            NotImplementedError, "serves no input candidate selector"
        ):
            _run(processor, (torch.randn(5, VOCAB), 0, 0, K, 3))

    def test_position_scoring_refuses_a_selector(self):
        processor, _ = self._processor()
        rows = torch.randn(5, VOCAB)
        positions = [torch.tensor([1, 3])]
        for scoring in (
            dict(token_indices_to_pool=positions),
            dict(multi_item_delimiter_indices=positions),
        ):
            with self.subTest(scoring=list(scoring)):
                # Those forwards return the top-k of the positions they score.
                plain = _run(
                    processor, (rows, 0, 0, K, None), is_prefill_only=True, **scoring
                )
                self.assertIsNone(plain.next_token_logits)
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "Position scoring serves no input candidate selector",
                ):
                    _run(processor, (rows, 0, 0, K, 3), is_prefill_only=True, **scoring)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_weights_reach_the_host_as_fp32_spans(self):
        # Confidential computing has no pinned memory to copy through.
        self.enterContext(envs.SGLANG_FORCE_CONFIDENTIAL_COMPUTE.override(False))
        generator = torch.Generator().manual_seed(6)
        plain, selected = (
            (torch.randn(n, VOCAB, generator=generator) * 3).cuda() for n in (10, 23)
        )
        processor, hook = self._processor(device="cuda")
        hook.logprobs_chunk_size = 7
        for fast, pinned_copy in itertools.product((False, True), repeat=2):
            hook.enable_fast_input_logprobs, hook.pinned_copy = fast, pinned_copy
            output = _run(processor, (plain, 0, 0, 3, None), (selected, 0, 0, K, 3))
            with self.subTest(fast=fast, pinned_copy=pinned_copy):
                # The scheduler concatenates the spans and reads them as numpy.
                for span in output.input_top_logprobs_weights[1]:
                    self.assertEqual(
                        (span.device.type, span.dtype, span.is_pinned()),
                        ("cpu", torch.float32, pinned_copy),
                    )
                self._check_selected(hook, [(output, 1)], 3, K, selected, 1)
                self.assertEqual(output.input_top_logprobs_weights[0], [])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_a_selector_request_leaves_its_neighbour_on_the_fused_top_k(self):
        from sglang.srt.layers import logsumexp

        vocab = 4099
        generator = torch.Generator().manual_seed(7)
        plain, selected = (
            (torch.randn(n, vocab, generator=generator) * 3).cuda() for n in (10, 23)
        )
        # Ties tell the fused kernel, which puts the lowest index first, from
        # torch.topk.
        plain[3] = 0.0
        plain[7, 100:120] = 50.0
        processor, hook = self._processor(vocab_size=vocab, device="cuda")
        hook.enable_fast_input_logprobs, hook.logprobs_chunk_size = True, 4096
        # The selector's K is past what the fused kernel serves, the
        # neighbour's 5 is not.
        wide = 4 * logsumexp.FUSED_TOPK_MAX_K
        requests = [(plain, 0, 0, 5, None), (selected, 0, 0, wide, 3)]
        alone = _run(processor, requests[0])
        for order in ((0, 1), (1, 0)):
            with (
                mock.patch.object(
                    logsumexp, "row_logsumexp_topk", wraps=logsumexp.row_logsumexp_topk
                ) as fused,
                _top_k_widths() as widths,
            ):
                output = _run(processor, *(requests[n] for n in order))
            at = order.index(0)
            with self.subTest(order=order):
                # One chunk, read once: its top-k is the neighbour's top-5.
                self.assertEqual([call.args[1] for call in fused.call_args_list], [5])
                self.assertEqual(widths, [])
                self.assertEqual(
                    output.input_top_logprobs_idx[at], alone.input_top_logprobs_idx[0]
                )
                self.assertEqual(
                    output.input_top_logprobs_val[at], alone.input_top_logprobs_val[0]
                )
                self.assertEqual(output.input_top_logprobs_idx[at][3], [0, 1, 2, 3, 4])
                self.assertEqual(
                    output.input_top_logprobs_idx[at][7], [100, 101, 102, 103, 104]
                )
                self._check_selected(hook, [(output, 1 - at)], 3, wide, selected, 1)
        # And those are log-probabilities: (logit - max) - log_sum in fp32.
        # log_sum adds 4099 terms, off by at most 4099 * 2**-24 whatever the
        # order of the sum; its logarithm and the two subtractions each round
        # by at most an ulp of a number below 64, 2**-18.
        torch.testing.assert_close(
            torch.tensor(alone.input_top_logprobs_val[0], dtype=torch.float64),
            plain.double().log_softmax(-1).topk(5).values.cpu(),
            rtol=0,
            atol=4099 * 2.0**-24 + 3 * 2.0**-18,
        )


if __name__ == "__main__":
    unittest.main()
