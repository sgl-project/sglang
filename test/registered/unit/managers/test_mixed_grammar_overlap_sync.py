"""Mixed batches whose decode rows carry a grammar must see the previous result."""

import unittest
from collections import deque
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(grammar):
    return SimpleNamespace(grammar=object() if grammar else None)


def _batch(mode, decoding_reqs=None, algo=SpeculativeAlgorithm.DFLASH):
    batch = SimpleNamespace(
        forward_mode=mode,
        spec_algorithm=algo,
        decoding_reqs=decoding_reqs,
        has_grammar=True,
    )
    batch.grammar_needs_sync = lambda: ScheduleBatch.grammar_needs_sync(batch)
    batch.mixed_decode_rows_have_grammar = lambda: (
        ScheduleBatch.mixed_decode_rows_have_grammar(batch)
    )
    return batch


def _disable(batch, queued=1):
    scheduler = SimpleNamespace(
        require_mlp_sync=False, result_queue=deque([object()] * queued)
    )
    return Scheduler.is_disable_overlap_for_batch(scheduler, batch, last_batch=None)


class TestMixedDecodeRowsHaveGrammar(unittest.TestCase):
    def test_only_mixed_batches_with_grammar_decode_rows(self):
        def has(mode, reqs):
            return _batch(mode, reqs).mixed_decode_rows_have_grammar()

        self.assertTrue(has(ForwardMode.MIXED, [_req(False), _req(True)]))
        self.assertFalse(has(ForwardMode.MIXED, [_req(False)]))
        self.assertFalse(has(ForwardMode.MIXED, None))
        self.assertFalse(has(ForwardMode.EXTEND, [_req(True)]))


class TestMixedGrammarOverlapSync(unittest.TestCase):
    def test_worker_without_mixed_barrier_syncs(self):
        batch = _batch(ForwardMode.MIXED, [_req(True)], SpeculativeAlgorithm.EAGLE)
        self.assertTrue(_disable(batch))

    def test_worker_with_mixed_barrier_overlaps(self):
        # DFLASH runs the barrier between the target extend and its sampling.
        self.assertTrue(SpeculativeAlgorithm.DFLASH.supports_mixed_grammar_barrier())
        self.assertFalse(_disable(_batch(ForwardMode.MIXED, [_req(True)])))

    def test_no_grammar_decode_rows_overlaps(self):
        batch = _batch(ForwardMode.MIXED, [_req(False)], SpeculativeAlgorithm.EAGLE)
        self.assertFalse(_disable(batch))

    def test_nothing_queued_overlaps(self):
        batch = _batch(ForwardMode.MIXED, [_req(True)], SpeculativeAlgorithm.EAGLE)
        self.assertFalse(_disable(batch, queued=0))

    def test_grammar_overlap_decode_keeps_overlapping(self):
        self.assertFalse(_disable(_batch(ForwardMode.DECODE)))


if __name__ == "__main__":
    unittest.main()
