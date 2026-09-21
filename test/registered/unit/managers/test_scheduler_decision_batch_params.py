import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestMtpPhaseBoundaryOverlap(unittest.TestCase):
    @staticmethod
    def _batch(*, is_extend: bool, is_speculative: bool = True):
        return SimpleNamespace(
            is_extend_in_batch=is_extend,
            forward_mode=SimpleNamespace(
                is_extend=lambda: is_extend,
                is_decode=lambda: not is_extend,
            ),
            spec_algorithm=SimpleNamespace(is_none=lambda: not is_speculative),
            grammar_needs_sync=lambda: False,
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


if __name__ == "__main__":
    unittest.main()
