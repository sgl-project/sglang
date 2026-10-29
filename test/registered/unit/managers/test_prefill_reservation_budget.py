import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Budget:
    """No memory pressure; exercise the scheduler's compute-budget path."""

    def available_chunk_tokens(self, chunk_limit):
        return chunk_limit

    def fit_chunk(self, *, extend_input_len, max_new_tokens, chunk_limit):
        return chunk_limit

    def reserve(self, *args, **kwargs):
        pass

    def has_capacity(self):
        return True

    def state_slots_exhausted(self):
        return False


class _Req:
    def __init__(self, prefix, total):
        self.prefix_len = prefix
        self.full_untruncated_fill_ids = range(total)
        self.origin_input_ids = range(total)
        self.output_ids = []
        self.num_matched_prefix_tokens = prefix
        self.extend_end = None
        self.retracted_stain = False
        self.sampling_params = SimpleNamespace(max_new_tokens=0)

    @property
    def extend_len(self):
        return self.extend_end - self.prefix_len

    def init_next_round_input(self):
        pass


class TestPrefillReservationBudget(unittest.TestCase):
    def make_adder(self, *, page, input_budget, chunk_budget, granule=0):
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.page_size = page
        adder.logical_page_size = page
        adder.mamba_chunk_grid = 1
        adder.dllm_config = None
        adder.rem_input_tokens = input_budget
        adder.rem_chunk_tokens = chunk_budget
        adder.chunked_req_limit = None
        adder.kv_shard_granule = granule
        adder.prefill_delayer_single_pass = None
        adder.memory_budget = _Budget()
        adder.exact_chunk_fill = False
        adder.per_req_token_overhead = page
        adder.can_run_list = []
        adder.log_hit_tokens = 0
        adder.log_input_tokens = 0
        adder.log_replay_tokens = 0
        adder.reprocessed_log_hit_tokens = 0
        adder.reprocessed_log_input_tokens = 0
        adder._swa_new_tokens = lambda req: 0
        adder._request_slot_reserve_for_req = lambda req, extend_input_len: 0
        adder._state_slot_reserve_for_req = lambda req: 0
        adder._kv_shard_reserve_scratch = lambda **kwargs: True
        return adder

    def schedule_continuation(
        self,
        adder,
        continuation,
        waiting,
        policy_kind=CacheAwarePolicy.SHORTEST_PREFILL_FIRST,
        dcp_enabled=False,
        cp_padded=False,
    ):
        policy = SchedulePolicy.__new__(SchedulePolicy)
        policy.policy = policy_kind
        scheduler = SimpleNamespace(
            chunked_req=continuation,
            waiting_queue=waiting,
            page_size=adder.page_size,
            policy=policy,
        )
        with (
            patch.dict(
                Scheduler._admit_chunked_req.__globals__,
                get_parallel=lambda: SimpleNamespace(dcp_enabled=dcp_enabled),
                is_cp_sharded_padded=lambda: cp_padded,
            ),
            # Scheduler and PrefillAdder read the CP strategy from different
            # modules. Keep unrelated process-global CP state out of this test.
            patch.dict(
                PrefillAdder._update_prefill_budget.__globals__,
                is_cp_sharded_padded=lambda: False,
            ),
        ):
            Scheduler._admit_chunked_req(scheduler, adder)
        return scheduler

    def test_input_budget_preserves_waiter_budget(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        waiter = _Req(0, 256)
        self.schedule_continuation(adder, continuation, [waiter])
        self.assertEqual(continuation.extend_len, 1792)
        self.assertEqual(adder.rem_input_tokens, 256)
        self.assertEqual(adder.budget_state(), AddReqResult.CONTINUE)

    def test_sharded_reservation_keeps_existing_behavior(self):
        adder = self.make_adder(page=4, input_budget=12, chunk_budget=16, granule=4)
        continuation = _Req(1, 33)
        waiter = _Req(0, 4)
        self.schedule_continuation(adder, continuation, [waiter])
        # Verify the existing chunk-budget input to the reservation policy;
        # do not make an assertion about the resulting shard boundary.
        self.assertEqual(adder.chunked_req_limit, 12)

    def test_dcp_without_page_sharding_keeps_existing_behavior(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        self.schedule_continuation(
            adder, continuation, [_Req(0, 256)], dcp_enabled=True
        )
        self.assertEqual(adder.chunked_req_limit, 3840)

    def test_padded_cp_keeps_existing_reservation_budget(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        # This case checks the policy cap only; CP allocation is not simulated.
        adder.add_chunked_req = lambda req: req
        self.schedule_continuation(adder, continuation, [_Req(0, 256)], cp_padded=True)
        self.assertEqual(adder.chunked_req_limit, 3840)

    def test_no_eligible_waiter_keeps_native_continuation(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        self.schedule_continuation(adder, continuation, [])
        self.assertEqual(continuation.extend_len, 4096)

    def test_fcfs_keeps_native_continuation(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        self.schedule_continuation(
            adder, continuation, [_Req(0, 256)], CacheAgnosticPolicy.FCFS
        )
        self.assertEqual(continuation.extend_len, 4096)


if __name__ == "__main__":
    unittest.main()
