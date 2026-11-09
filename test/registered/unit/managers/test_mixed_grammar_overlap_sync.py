"""Mixed batches whose decode rows carry a grammar must see the previous result."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402
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


def _early(last_batch, mixed_chunk=True):
    scheduler = SimpleNamespace(require_mlp_sync=False, is_mixed_chunk=mixed_chunk)
    return Scheduler.should_process_last_batch_early(scheduler, last_batch)


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
        self.assertTrue(
            _early(_batch(ForwardMode.DECODE, algo=SpeculativeAlgorithm.EAGLE))
        )

    def test_worker_with_mixed_barrier_overlaps(self):
        # DFLASH runs the barrier between the target extend and its sampling.
        self.assertTrue(SpeculativeAlgorithm.DFLASH.supports_mixed_grammar_barrier())
        self.assertFalse(_early(_batch(ForwardMode.DECODE)))

    def test_dspark_syncs(self):
        # DSPARK's mixed-batch prefill does not run the barrier.
        self.assertFalse(SpeculativeAlgorithm.DSPARK.supports_mixed_grammar_barrier())
        self.assertTrue(
            _early(_batch(ForwardMode.DECODE, algo=SpeculativeAlgorithm.DSPARK))
        )

    def test_without_mixed_chunk_overlaps(self):
        batch = _batch(ForwardMode.DECODE, algo=SpeculativeAlgorithm.EAGLE)
        self.assertFalse(_early(batch, mixed_chunk=False))

    def test_no_grammar_overlaps(self):
        batch = _batch(ForwardMode.DECODE, algo=SpeculativeAlgorithm.EAGLE)
        batch.has_grammar = False
        self.assertFalse(_early(batch))

    def test_no_last_batch_overlaps(self):
        self.assertFalse(_early(None))


if __name__ == "__main__":
    unittest.main()
