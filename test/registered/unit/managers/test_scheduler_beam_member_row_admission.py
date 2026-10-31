"""Admission reserves member rows for beam leaders admitted earlier in the same prefill.

A beam_width=k leader takes one req_to_token row at admission and spawns k-1
member rows at its prefill relay. The gate subtracted the member rows of
running groups but not those of leaders already in the prefill being built, so
a later candidate could take a promised row and the spawn asserted
("Beam member spawn needs ... req-to-token slots"), killing the scheduler.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MAX_RUNNING = 8


class _Req:
    def __init__(self, beam_width=None):
        self.kv = SimpleNamespace(holds_kv=False)
        self.beam_group = (
            SimpleNamespace(beam_width=beam_width, member_rows=None, retired=False)
            if beam_width is not None
            else None
        )
        self.token_indices_to_pool = None
        self.inflight_middle_chunks = 0

    def finished(self):
        return False

    def init_next_round_input(self, tree_cache=None):
        pass


def _pending_member_rows(reqs):
    # BeamCoordinator.pending_member_rows over a request list.
    return sum(
        r.beam_group.beam_width - 1
        for r in reqs
        if r.beam_group is not None
        and r.beam_group.member_rows is None
        and not r.beam_group.retired
        and not r.finished()
    )


class _Adder:
    """Admits whatever the gate lets through."""

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _BatchBuilt(Exception):
    pass


class TestBeamMemberRowAdmission(CustomTestCase):
    def _scheduler(self, free_rows, waiting):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = waiting
        s.chunked_req = None
        s.min_free_slots_delayer = None
        s.policy = Mock()
        s.policy.shortest_prefill_chunk_limit.return_value = None
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = 8192
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: _Adder
        s.page_size = 1
        s.tree_cache = SimpleNamespace(
            buffer_pipeline=None, storage_prefetch_retries=None
        )
        s.token_to_kv_pool_allocator = None
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = MAX_RUNNING
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: free_rows)
        s.beam_coordinator = SimpleNamespace(
            pending_member_rows=lambda batch: _pending_member_rows(batch.reqs),
            pending_member_rows_of=_pending_member_rows,
        )
        return s

    def _admit(self, free_rows, waiting):
        running_batch = ScheduleBatch(reqs=[])
        s = self._scheduler(free_rows, list(waiting))

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            get_context().override_server_args(pp_max_micro_batch_size=MAX_RUNNING),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt) as built:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None, running_batch=running_batch
                )
        return built.exception.args[0], running_batch.batch_is_full

    def test_request_after_a_leader_leaves_its_member_rows_free(self):
        leader, other = _Req(beam_width=3), _Req()
        # Three free rows: the leader and its two members take all of them.
        can_run_list, batch_is_full = self._admit(3, [leader, other])
        self.assertEqual(can_run_list, [leader])
        self.assertTrue(batch_is_full)

    def test_request_that_fits_next_to_a_leader_is_still_admitted(self):
        leader, other = _Req(beam_width=3), _Req()
        can_run_list, batch_is_full = self._admit(4, [leader, other])
        self.assertEqual(can_run_list, [leader, other])
        self.assertFalse(batch_is_full)


if __name__ == "__main__":
    unittest.main()
