"""A request that already holds a req_to_token row is not counted twice at admission (#38118, #35537)."""

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

NUM_ROWS = 4


class _Req:
    def __init__(self, holds_row, beam_width=1):
        self.kv = SimpleNamespace(holds_kv=holds_row)
        self.beam_group = (
            SimpleNamespace(beam_width=beam_width, member_rows=None)
            if beam_width > 1
            else None
        )
        self.beam_width = beam_width
        self.token_indices_to_pool = None
        self.inflight_middle_chunks = 0

    def init_next_round_input(self, tree_cache=None):
        pass


class _Adder:
    """Admits whatever the gate lets through; the final chunk joins first."""

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return None

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _BatchBuilt(Exception):
    pass


class TestHeldRowAdmission(CustomTestCase):
    def _scheduler(
        self, disaggregation_mode, running, final_chunk, waiting, num_rows=NUM_ROWS
    ):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = waiting
        s.chunked_req = final_chunk
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
        s.max_running_requests = num_rows
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = disaggregation_mode
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        # Continuing chunks already hold a leader row; newly staged requests do not.
        s.req_to_token_pool = SimpleNamespace(
            available_size=lambda: (
                num_rows - len(running) - int(final_chunk is not None)
            )
        )
        s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        return s

    def _admitted(self, disaggregation_mode):
        running = [_Req(holds_row=True), _Req(holds_row=True)]
        final_chunk = _Req(holds_row=True)
        waiting = _Req(holds_row=False)
        running_batch = ScheduleBatch(reqs=running)
        s = self._scheduler(disaggregation_mode, running, final_chunk, [waiting])

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            get_context().override_server_args(pp_max_micro_batch_size=NUM_ROWS),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt) as built:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None, running_batch=running_batch
                )
        return (
            built.exception.args[0],
            running_batch.batch_is_full,
            final_chunk,
            waiting,
        )

    def test_final_chunk_leaves_the_free_row_to_a_waiting_request(self):
        for mode in (DisaggregationMode.NULL, DisaggregationMode.PREFILL):
            with self.subTest(mode=mode):
                can_run_list, batch_is_full, final_chunk, waiting = self._admitted(mode)
                self.assertEqual(can_run_list, [final_chunk, waiting])
                self.assertFalse(batch_is_full)

    def _admitted_widths(self, widths, num_rows, *, running=(), final_chunk=None):
        waiting = [_Req(holds_row=False, beam_width=w) for w in widths]
        running_batch = ScheduleBatch(reqs=list(running))
        scheduler = self._scheduler(
            DisaggregationMode.NULL,
            running,
            final_chunk,
            waiting,
            num_rows=num_rows,
        )

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            get_context().override_server_args(pp_max_micro_batch_size=num_rows),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt) as built:
                scheduler._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None, running_batch=running_batch
                )
        return [r.beam_width for r in built.exception.args[0]]

    def test_mixed_widths_reserve_the_actual_number_of_member_rows(self):
        for num_rows, widths, expected in [
            (4, [4, 1], [4]),
            (4, [2, 2, 1], [2, 2]),
            (4, [1, 4], [1]),
            (8, [2, 1, 4], [2, 1, 4]),
            (8, [4, 2, 1, 1], [4, 2, 1, 1]),
            (4, [2, 2, 2], [2, 2]),
        ]:
            with self.subTest(num_rows=num_rows, widths=widths):
                self.assertEqual(self._admitted_widths(widths, num_rows), expected)

    def test_held_beam_leader_still_reserves_its_unspawned_members(self):
        self.assertEqual(
            self._admitted_widths(
                [1, 1],
                4,
                running=[_Req(holds_row=True)],
                final_chunk=_Req(holds_row=True, beam_width=2),
            ),
            [2, 1],
        )

    def test_running_member_reservations_and_pp_budget_are_preserved(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 7)
        scheduler.beam_coordinator = SimpleNamespace(
            pending_member_rows=lambda batch: 3
        )
        with get_context().override_server_args(pp_max_micro_batch_size=8):
            # Four physical rows remain; the staged width-2 beam and incoming
            # width-2 beam reserve one member each, leaving two leader slots.
            self.assertEqual(
                scheduler.get_num_allocatable_reqs(
                    2, beam_width=2, num_staged_beam_rows=1
                ),
                2,
            )
        with get_context().override_server_args(pp_max_micro_batch_size=3):
            self.assertEqual(scheduler.get_num_allocatable_reqs(2), 1)


if __name__ == "__main__":
    unittest.main()
