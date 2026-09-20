import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestMtpPhaseBoundaryOverlap(unittest.TestCase):
    @staticmethod
    def _batch(
        *,
        is_extend: bool,
        is_mixed: bool = False,
        is_speculative: bool = True,
        grammar_needs_sync: bool = False,
    ):
        return SimpleNamespace(
            is_extend_in_batch=is_extend,
            forward_mode=SimpleNamespace(
                is_extend=lambda: is_extend,
                is_decode=lambda: not is_extend,
                is_mixed=lambda: is_mixed,
            ),
            spec_algorithm=SimpleNamespace(is_none=lambda: not is_speculative),
            grammar_needs_sync=lambda: grammar_needs_sync,
        )

    def _scheduler(self, *, require_mlp_sync: bool):
        scheduler = object.__new__(Scheduler)
        scheduler.require_mlp_sync = require_mlp_sync
        scheduler.result_queue = [object()]
        return scheduler

    @patch(
        "sglang.srt.managers.scheduler.envs."
        "SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get",
        return_value=False,
    )
    def test_mtp_phase_crossing_keeps_overlap(self, _disable_consecutive_prefill):
        extend = self._batch(is_extend=True)
        decode = self._batch(is_extend=False)

        for require_mlp_sync in (False, True):
            scheduler = self._scheduler(require_mlp_sync=require_mlp_sync)
            self.assertFalse(scheduler.is_disable_overlap_for_batch(decode, extend))
            self.assertFalse(scheduler.is_disable_overlap_for_batch(extend, decode))

    @patch(
        "sglang.srt.managers.scheduler.envs."
        "SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get",
        return_value=False,
    )
    def test_mixed_grammar_decode_tail_forces_sync(self, _disable_consecutive_prefill):
        mixed = self._batch(
            is_extend=True,
            is_mixed=True,
            grammar_needs_sync=True,
        )
        scheduler = self._scheduler(require_mlp_sync=False)

        self.assertTrue(scheduler.is_disable_overlap_for_batch(mixed, None))


class TestMixedGrammarSyncDecision(unittest.TestCase):
    @staticmethod
    def _batch(*, prefill_grammar, decode_grammar):
        prefill_req = SimpleNamespace(grammar=prefill_grammar)
        decode_req = SimpleNamespace(grammar=decode_grammar)
        batch = ScheduleBatch(reqs=[prefill_req, decode_req])
        batch.forward_mode = ForwardMode.MIXED
        batch.has_grammar = prefill_grammar is not None or decode_grammar is not None
        batch.decoding_reqs = [decode_req]
        batch.spec_algorithm = SimpleNamespace(supports_grammar_overlap=lambda: True)
        return batch

    def test_syncs_when_decode_tail_has_grammar(self):
        batch = self._batch(prefill_grammar=None, decode_grammar=object())

        self.assertTrue(batch.grammar_needs_sync())

    def test_does_not_sync_for_prefill_only_grammar(self):
        batch = self._batch(prefill_grammar=object(), decode_grammar=None)

        self.assertFalse(batch.grammar_needs_sync())

    def test_regular_decode_keeps_worker_grammar_overlap(self):
        batch = self._batch(prefill_grammar=None, decode_grammar=object())
        batch.forward_mode = ForwardMode.DECODE

        self.assertFalse(batch.grammar_needs_sync())


if __name__ == "__main__":
    unittest.main()
