"""A mixed step holds decode rows back when they do not fit the KV pool."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.decoding_batch import DecodingBatch

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
        return batch

    def test_kv_full_holds_rows_without_retracting(self):
        s = self._scheduler()
        batch = self._batch(fits=False)

        self.assertFalse(s.select_decode_rows(DecodingBatch(batch), retract=False))
        batch.check_decode_mem.assert_called_once_with()
        batch.retract_decode.assert_not_called()
        batch.prepare_for_decode.assert_not_called()
        s.new_token_ratio_tracker.decay_step.assert_not_called()

    def test_rows_that_fit_are_prepared_without_ratio_decay(self):
        s = self._scheduler()
        batch = self._batch(fits=True)

        self.assertTrue(s.select_decode_rows(DecodingBatch(batch), retract=False))
        batch.prepare_for_decode.assert_called_once_with()
        s.new_token_ratio_tracker.decay_step.assert_not_called()


class _AdmitAllAdder:
    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None

    def add_one_req(self, req, **kwargs):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class TestMixedStepHoldsDecodeRows(CustomTestCase):
    def test_kv_full_running_rows_sit_out_without_retracting(self):
        s = MagicMock()
        s.select_decode_rows = MethodType(Scheduler.select_decode_rows, s)
        s.admission_skip_reason = MethodType(Scheduler.admission_skip_reason, s)
        s._reqs_finishing_inflight.return_value = []
        s.grammar_manager.has_waiting_grammars.return_value = False
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.min_free_slots_delayer = None
        s.chunked_req = None
        s.enable_lora = False
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.enable_hierarchical_cache = False
        s.enable_unified_cache_external_linker = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.get_num_allocatable_reqs.return_value = 8
        s.is_mixed_chunk = True
        s.tp_worker.model_runner.prefill_aware_swa = False
        # Lets a mutant that retracts run to completion and reach the asserts.
        s.decode_offload_manager = None
        s.token_to_kv_pool_allocator.available_size.return_value = 0
        s.new_token_ratio_tracker.current = 0.5
        s.metrics_reporter.enable_metrics = False
        s.waiting_queue = [MagicMock(token_indices_to_pool=None, beam_group=None)]

        running_batch = MagicMock(
            reqs=[SimpleNamespace(beam_group=None)],
            return_logprob=False,
        )
        running_batch.is_empty.return_value = False
        running_batch.batch_size.return_value = 1
        running_batch.check_decode_mem.return_value = False
        running_batch.retract_decode.return_value = ([], 0.5, [])
        new_batch = MagicMock(return_logprob=False, input_embeds=None)

        with (
            patch.object(scheduler_module, "PrefillAdder", _AdmitAllAdder),
            patch.object(scheduler_module, "PrefillStats"),
            patch.object(scheduler_module, "get_schedule"),
            patch.object(scheduler_module, "set_time_batch"),
            patch.object(
                scheduler_module.ScheduleBatch, "init_new", return_value=new_batch
            ),
        ):
            plan = Scheduler._get_new_batch_prefill_raw(
                s,
                prefill_delayer_single_pass=None,
                decoding_batch=DecodingBatch(running_batch),
            )

        self.assertIs(plan.batch_to_run, new_batch)
        self.assertIs(plan.running_batch, running_batch)
        running_batch.retract_decode.assert_not_called()
        running_batch.prepare_for_decode.assert_not_called()
        new_batch.mix_with_running.assert_not_called()


if __name__ == "__main__":
    unittest.main()
