"""KV accounting of the dLLM scheduler path between rounds (CPU).

Under long-context x concurrency the dLLM scheduler crashed in its idle leak
check (``pool memory leak detected! ... available=1, uncached=0``): admitted
requests wait in the dLLM queue between rounds holding their rows, a round
that admits nothing returned no batch, and ``is_fully_idle`` ignored the dLLM
queue, so the held KV looked leaked. Covers: parked requests keep the
scheduler busy, a prefill round that admits nothing falls back to a denoise
round, and a round stuck on KV retracts the most recently admitted holder.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.srt.dllm.mixin import scheduler as dllm_scheduler
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import ListKind
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(rid, *, prefill=False, holds_kv=True):
    return FakeReq(
        rid=rid,
        dllm_phase=(
            DllmReqPhase.STAGING_PREFILL if prefill else DllmReqPhase.STAGING_DECODE
        ),
        kv=SimpleNamespace(holds_kv=holds_kv),
        finished=lambda: False,
        time_stats=MagicMock(),
        to_finish=None,
    )


class _Batch:
    def __init__(self, reqs=()):
        self.reqs = list(reqs)
        self.batch_is_full = False

    def is_empty(self):
        return not self.reqs


class TestParkedDllmRequestsAreNotIdle(CustomTestCase):
    def _scheduler(self, dllm_reqs):
        lists = {
            ListKind.WAITING: [],
            ListKind.DECODING: [],
            ListKind.DLLM: dllm_reqs,
        }
        return SimpleNamespace(
            running_batch=_Batch(),
            last_batch=None,
            enable_overlap=False,
            result_queue=[],
            _pp_microbatches_drained=lambda: True,
            request_lists=lambda: lists,
            pause_state=SimpleNamespace(paused=False),
            request_receiver=SimpleNamespace(has_pending=lambda: False),
            uocr_runtime=None,
            disaggregation_mode=None,
            decode_offload_manager=None,
            enable_hierarchical_cache=False,
            enable_lmcache=False,
            enable_unified_cache_external_linker=False,
        )

    def test_parked_requests_keep_the_scheduler_busy(self):
        # A round that schedules nothing leaves last_batch empty; the requests
        # in the dLLM queue still hold KV, so the idle leak check must not run.
        s = self._scheduler([_req("p0")])
        self.assertFalse(Scheduler.is_fully_idle(s))

    def test_empty_dllm_queue_is_idle(self):
        self.assertTrue(Scheduler.is_fully_idle(self._scheduler([])))


class _Sched(SchedulerDllmMixin):
    def __init__(self, reqs, prefill_result):
        self.enable_priority_preemption = False
        self.waiting_queue = SimpleNamespace(reorder=lambda order: None)
        self.policy = SimpleNamespace(calc_priority=lambda q: None)
        self.dllm_manager = SimpleNamespace(
            waiting_queue=list(reqs),
            last_add_result=AddReqResult.CONTINUE,
            get_prefill_requests=lambda: [
                r
                for r in self.dllm_manager.waiting_queue
                if r.dllm_phase.endswith("prefill")
            ],
            get_decode_requests=lambda: [
                r
                for r in self.dllm_manager.waiting_queue
                if not r.dllm_phase.endswith("prefill")
            ],
        )
        self.dllm_config = SimpleNamespace(requires_separate_context_encoding=True)
        self.prefill_result = prefill_result
        self.adders = []
        self._should_skip_prefill = lambda running_batch: False
        self._ack_waiting_health_checks = lambda: None
        self._fetch_waiting_reqs = lambda: None
        self._update_state_for_batch = lambda adder: None
        self._create_dllm_batch = Mock(return_value="batch")
        self._add_request_to_queue = Mock()
        self.tree_cache = object()
        self.request_retirer = MagicMock()

    def _create_dllm_prefill_adder(self, running_bs, running_batch):
        adder = SimpleNamespace(can_run_list=[])
        self.adders.append(adder)
        return adder

    def _process_batch_by_phase(
        self, adder, batch, staging_phase, incoming_phase, running_batch
    ):
        if staging_phase == DllmReqPhase.STAGING_PREFILL:
            # The prefill candidates do not fit (KV) or are deferred.
            return self.prefill_result
        adder.can_run_list.extend(batch)
        return AddReqResult.CONTINUE


class TestEmptyPrefillRound(CustomTestCase):
    def test_prefill_round_admitting_nothing_runs_the_denoise_blocks(self):
        decode = _req("d0")
        s = _Sched([_req("p0", prefill=True), decode], AddReqResult.NO_TOKEN)
        batch = s.get_new_batch_dllm(_Batch(), staged=[])
        self.assertEqual(batch, "batch")
        can_run_list, forward_mode = s._create_dllm_batch.call_args.args
        self.assertEqual(can_run_list, [decode])
        self.assertEqual(forward_mode, ForwardMode.DLLM_EXTEND)
        s._add_request_to_queue.assert_not_called()


class TestKvDeadlockRetraction(CustomTestCase):
    def _run(self, reqs, result=AddReqResult.NO_TOKEN):
        s = _Sched(reqs, result)
        with patch.object(dllm_scheduler, "retract_all", return_value=[]) as ra:
            batch = s.get_new_batch_dllm(_Batch(), staged=[])
        self.assertIsNone(batch)
        return s, ra

    def test_stuck_round_retracts_the_latest_holder(self):
        reqs = [
            _req("a", prefill=True),
            _req("b", prefill=True),
            _req("c", prefill=True, holds_kv=False),
        ]
        s, ra = self._run(reqs)
        ra.assert_called_once()
        self.assertEqual(ra.call_args.kwargs["reqs"], [reqs[1]])
        s._add_request_to_queue.assert_called_once_with(reqs[1], is_retracted=True)

    def test_sole_holder_is_not_retracted(self):
        s, ra = self._run(
            [_req("a", prefill=True), _req("b", prefill=True, holds_kv=False)]
        )
        ra.assert_not_called()

    def test_non_kv_stall_does_not_retract(self):
        s, ra = self._run(
            [_req("a", prefill=True), _req("b", prefill=True)],
            result=AddReqResult.OTHER,
        )
        ra.assert_not_called()

    def test_unresumable_victim_is_aborted(self):
        reqs = [_req("a", prefill=True), _req("b", prefill=True)]
        s = _Sched(reqs, AddReqResult.NO_TOKEN)
        with patch.object(
            dllm_scheduler, "retract_all", side_effect=lambda **kw: kw["reqs"]
        ):
            s.get_new_batch_dllm(_Batch(), staged=[])
        s.request_retirer.abort_retracted.assert_called_once()
        s._add_request_to_queue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
