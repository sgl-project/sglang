"""A mixed step holds decode rows back when they do not fit the KV pool."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSelectDecodeRowsHoldMode(CustomTestCase):
    def _scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.enable_skip_finishing_decode = False
        s.new_token_ratio_tracker = Mock()
        return s

    def _batch(self, fits: bool) -> Mock:
        batch = Mock()
        batch.reqs = [SimpleNamespace(rid="req-0"), SimpleNamespace(rid="req-1")]
        batch.batch_size.return_value = 2
        batch.is_empty.return_value = False
        batch.check_decode_mem.return_value = fits
        batch.batch_is_full = True
        return batch

    def test_kv_full_holds_rows_without_retracting(self):
        s = self._scheduler()
        batch = self._batch(fits=False)

        self.assertFalse(s.select_decode_rows(batch, retract=False))
        batch.check_decode_mem.assert_called_once_with()
        batch.retract_decode.assert_not_called()
        batch.prepare_for_decode.assert_not_called()
        s.new_token_ratio_tracker.decay_step.assert_not_called()
        self.assertTrue(batch.batch_is_full)

    def test_rows_that_fit_are_prepared_without_ratio_decay(self):
        s = self._scheduler()
        batch = self._batch(fits=True)

        self.assertTrue(s.select_decode_rows(batch, retract=False))
        batch.prepare_for_decode.assert_called_once_with()
        s.new_token_ratio_tracker.decay_step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
