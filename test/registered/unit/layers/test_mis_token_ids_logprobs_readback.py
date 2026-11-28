import math
import pickle
import time
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers import logits_processor
from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    LogitsProcessorOutput,
    _multi_item_token_ids_logprobs_to_host,
)
from sglang.srt.layers.logprob_processor import LogprobStage, get_token_ids_logprobs_raw
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def _reference(logprobs, token_ids, rows_per_seq):
    ref_vals, ref_idxs = get_token_ids_logprobs_raw(
        logprobs,
        token_ids,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=rows_per_seq,
        no_copy_to_cpu=True,
    )
    return [v.tolist() if torch.is_tensor(v) else v for v in ref_vals], ref_idxs


def _assert_matches(case, output, ref_vals, ref_idxs):
    case.assertEqual(output.input_token_ids_logprobs_idx, ref_idxs)
    case.assertEqual(len(output.input_token_ids_logprobs_val), len(ref_vals))
    for got, ref in zip(output.input_token_ids_logprobs_val, ref_vals):
        case.assertEqual(got, ref)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMultiItemTokenIdsLogprobsReadback(CustomTestCase):
    def _check(self, token_ids, rows_per_seq, h2d, d2h, vocab=50):
        """Check the values and that the readback issued ``h2d`` id uploads and
        ``d2h`` host copies, which tells the gather strategies apart."""
        torch.manual_seed(0)
        logprobs = torch.randn(sum(rows_per_seq), vocab, device="cuda").log_softmax(-1)
        ref_vals, ref_idxs = _reference(logprobs, token_ids, rows_per_seq)

        with (
            mock.patch.object(
                logits_processor, "async_h2d", wraps=logits_processor.async_h2d
            ) as up,
            mock.patch.object(
                logits_processor, "async_d2h", wraps=logits_processor.async_d2h
            ) as down,
        ):
            vals, idxs, copy_done = _multi_item_token_ids_logprobs_to_host(
                logprobs, token_ids, rows_per_seq
            )
        self.assertEqual((up.call_count, down.call_count), (h2d, d2h))
        for spans in vals:
            for span in spans:
                self.assertFalse(span.is_cuda)
        output = LogitsProcessorOutput(
            next_token_logits=None,
            input_token_ids_logprobs_val=vals,
            input_token_ids_logprobs_idx=idxs,
            input_logprobs_copy_done=copy_done,
        )
        output.finalize_input_logprobs()
        _assert_matches(self, output, ref_vals, ref_idxs)

    def test_shared_labels(self):
        self._check([[3, 7], [3, 7], [3, 7]], [4, 1, 6], h2d=1, d2h=1)

    def test_sequence_without_scoring_positions(self):
        self._check([[3, 7], [3, 7], [3, 7]], [4, 0, 2], h2d=1, d2h=1)

    def test_shared_labels_with_unscored_none_sequence(self):
        self._check([[3, 7], None, [3, 7]], [4, 0, 2], h2d=1, d2h=1)

    def test_none_labels_on_scored_sequence_skips_single_gather(self):
        self._check([[3, 7], None, [3, 7]], [4, 1, 2], h2d=2, d2h=1)

    def test_same_count_different_labels(self):
        self._check([[3, 7], [1, 2], [9, 4]], [2, 3, 1], h2d=3, d2h=1)

    def test_per_sequence_labels(self):
        self._check([[3, 7], [1, 2], None, [9, 4]], [2, 3, 2, 1], h2d=3, d2h=1)

    def test_ragged_label_counts(self):
        self._check([[3, 7, 11], [5], [2, 8]], [2, 3, 1], h2d=3, d2h=3)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMultiItemReadbackOutstandingCopy(CustomTestCase):
    """The copy is queued behind ~1s of work on a side stream, so a reader that
    skips the event wait returns while the pinned buffer is still unwritten."""

    BUSY_SECONDS = 1.0

    @classmethod
    def setUpClass(cls):
        cls.stream = torch.cuda.Stream()
        cls.a = torch.randn(4096, 4096, device="cuda")
        cls.out = torch.empty_like(cls.a)
        torch.mm(cls.a, cls.a, out=cls.out)
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.mm(cls.a, cls.a, out=cls.out)
        torch.cuda.synchronize()
        cls.repeats = math.ceil(cls.BUSY_SECONDS / (time.perf_counter() - start))

    def _produce_delayed(self, produce):
        # The first allocation on a stream synchronizes implicitly, which would
        # finish the copy before the reader runs.
        with torch.cuda.stream(self.stream):
            produce()
        torch.cuda.synchronize()
        with torch.cuda.stream(self.stream):
            for _ in range(self.repeats):
                torch.mm(self.a, self.a, out=self.out)
            return produce()

    def _read_and_check(self, output, read):
        event = output.input_logprobs_copy_done
        self.assertFalse(
            event.query(),
            "copy already finished: workload too short, or event recorded on the wrong stream",
        )
        read(output)
        self.assertTrue(event.query(), "reader returned before the copy finished")

    def _scoring_inputs(self, token_ids, rows_per_seq, vocab=50):
        torch.manual_seed(0)
        logprobs = torch.randn(sum(rows_per_seq), vocab, device="cuda").log_softmax(-1)
        return logprobs, _reference(logprobs, token_ids, rows_per_seq)

    def _readback_output(self, logprobs, token_ids, rows_per_seq):
        def produce():
            vals, idxs, done = _multi_item_token_ids_logprobs_to_host(
                logprobs, token_ids, rows_per_seq
            )
            return LogitsProcessorOutput(
                next_token_logits=None,
                input_token_ids_logprobs_val=vals,
                input_token_ids_logprobs_idx=idxs,
                input_logprobs_copy_done=done,
            )

        return self._produce_delayed(produce)

    def _scheduler_read(self, output):
        SchedulerBatchResultProcessor.move_logprobs_to_cpu(
            None, batch=types.SimpleNamespace(return_logprob=True), logits_output=output
        )

    def _check_scheduler_reader(self, token_ids, rows_per_seq):
        logprobs, (ref_vals, ref_idxs) = self._scoring_inputs(token_ids, rows_per_seq)
        output = self._readback_output(logprobs, token_ids, rows_per_seq)
        self._read_and_check(output, self._scheduler_read)
        _assert_matches(self, output, ref_vals, ref_idxs)

    def test_scheduler_reader_shared_labels(self):
        self._check_scheduler_reader([[3, 7], [3, 7], [3, 7]], [4, 1, 6])

    def test_scheduler_reader_ragged_label_counts(self):
        self._check_scheduler_reader([[3, 7, 11], [5], [2, 8]], [2, 3, 1])

    def test_pipeline_parallel_reader(self):
        token_ids, rows_per_seq = [[3, 7], [3, 7], [3, 7]], [4, 1, 6]
        logprobs, (ref_vals, ref_idxs) = self._scoring_inputs(token_ids, rows_per_seq)
        output = self._readback_output(logprobs, token_ids, rows_per_seq)
        result = GenerationBatchResult(
            logits_output=output,
            extend_input_len_per_req=[5, 2, 7],
            extend_logprob_start_len_per_req=[0, 0, 0],
        )

        rebuilt = []

        def read(_):
            sent = pickle.loads(pickle.dumps(get_logprob_dict_from_result(result)))
            rebuilt.append(get_logprob_from_pp_outputs(PPProxyTensors(sent)))

        self._read_and_check(output, read)
        received, _, _ = rebuilt[0]
        _assert_matches(self, received, ref_vals, ref_idxs)

    def test_event_survives_scoring_and_trim(self):
        token_ids, rows_per_seq = [[3, 7], [3, 7]], [3, 2]
        torch.manual_seed(0)
        logits = torch.randn(sum(rows_per_seq), 50, device="cuda")
        ref_vals, ref_idxs = _reference(logits.log_softmax(-1), token_ids, rows_per_seq)

        scorer = types.SimpleNamespace(_get_logits=lambda *_: logits)
        metadata = types.SimpleNamespace(
            extend_seq_lens_cpu=[10, 8],
            token_ids_logprobs=token_ids,
            extend_token_ids_logprob=True,
            extend_return_top_logprob=False,
            mm_input_embeds=None,
        )
        delimiters = [torch.tensor([3, 6, 9]), torch.tensor([2, 5])]
        input_ids = torch.zeros(18, dtype=torch.long, device="cuda")
        hidden = torch.zeros(18, 8, device="cuda")
        trimmer = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(
                spec_algorithm=types.SimpleNamespace(is_speculative=lambda: False)
            ),
            raw_bs=2,
            _is_full_backend=True,
        )

        def produce():
            scored = LogitsProcessor.compute_logprobs_for_multi_item_scoring(
                scorer, input_ids, hidden, None, metadata, delimiters
            )
            return PrefillCudaGraphRunner._trim_logits_output(trimmer, scored)

        output = self._produce_delayed(produce)
        self._read_and_check(output, self._scheduler_read)
        _assert_matches(self, output, ref_vals, ref_idxs)


if __name__ == "__main__":
    unittest.main()
