import unittest
from types import MethodType, SimpleNamespace
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
        self.kv = SimpleNamespace(holds_kv=False)
        self.beam_group = None
        self.session = None
        self.warm_match = 0
        self.match_calls = 0

    @property
    def extend_len(self):
        return self.extend_end - self.prefix_len

    def init_next_round_input(self, tree_cache=None, cow_mamba=None):
        if tree_cache is not None:
            self.match_calls += 1
            self.num_matched_prefix_tokens = self.warm_match

    def is_dllm(self):
        return False


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
        adder.prefill_max_requests = None
        adder.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
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
        warm_reserve=0,
        available_slots=2,
        encoder_replay=False,
        external_linker=False,
        flexkv=False,
        custom_backend=None,
        hisparse=False,
        unified_memory=False,
    ):
        policy = SchedulePolicy.__new__(SchedulePolicy)
        policy.policy = policy_kind
        scheduler = SimpleNamespace(
            chunked_req=continuation,
            waiting_queue=waiting,
            page_size=adder.page_size,
            policy=policy,
            enable_hierarchical_cache=False,
            enable_lmcache=False,
            enable_unified_cache_external_linker=external_linker,
            enable_hisparse=hisparse,
            enable_unified_memory=unified_memory,
            enable_lora=False,
            tree_cache=SimpleNamespace(
                supports_prefix_sharing=lambda: True,
                supports_mamba=lambda: False,
            ),
            get_num_allocatable_reqs=lambda *args, **kwargs: available_slots,
        )
        scheduler._fcfs_warm_chunk_limit = MethodType(
            Scheduler._fcfs_warm_chunk_limit, scheduler
        )
        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: False)
        with (
            patch.dict(
                Scheduler._admit_chunked_req.__globals__,
                get_parallel=lambda: SimpleNamespace(dcp_enabled=dcp_enabled),
                is_cp_sharded_padded=lambda: cp_padded,
                get_exec=lambda: SimpleNamespace(
                    features=SimpleNamespace(
                        enable_encoder_swa_bounded_replay=encoder_replay
                    )
                ),
                get_memory=lambda: SimpleNamespace(
                    enable_flexkv=flexkv, radix_cache_backend=custom_backend
                ),
                envs=SimpleNamespace(
                    SGLANG_FCFS_WARM_PREFILL_RESERVE_TOKENS=SimpleNamespace(
                        get=lambda: warm_reserve
                    )
                ),
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

    def test_fcfs_warm_head_reserves_with_opt_in(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        waiter = _Req(0, 1024)
        waiter.warm_match = 768
        self.schedule_continuation(
            adder,
            continuation,
            [waiter],
            CacheAgnosticPolicy.FCFS,
            warm_reserve=512,
        )
        self.assertEqual(waiter.match_calls, 1)
        self.assertEqual(continuation.extend_len, 1792)
        self.assertEqual(adder.rem_input_tokens, 256)

    def test_fcfs_default_does_not_match(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        waiter = _Req(0, 1024)
        waiter.warm_match = 768
        self.schedule_continuation(
            adder, continuation, [waiter], CacheAgnosticPolicy.FCFS
        )
        self.assertEqual(waiter.match_calls, 0)
        self.assertEqual(continuation.extend_len, 4096)

    def test_fcfs_no_free_waiter_slot_does_not_match(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        waiter = _Req(0, 1024)
        waiter.warm_match = 768
        self.schedule_continuation(
            adder,
            continuation,
            [waiter],
            CacheAgnosticPolicy.FCFS,
            warm_reserve=512,
            available_slots=1,
        )
        self.assertEqual(waiter.match_calls, 0)
        self.assertEqual(continuation.extend_len, 4096)

    def test_fcfs_dcp_does_not_match_with_opt_in(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        waiter = _Req(0, 1024)
        waiter.warm_match = 768
        self.schedule_continuation(
            adder,
            continuation,
            [waiter],
            CacheAgnosticPolicy.FCFS,
            warm_reserve=512,
            dcp_enabled=True,
        )
        self.assertEqual(waiter.match_calls, 0)
        self.assertEqual(continuation.extend_len, 4096)

    def test_fcfs_beam_waiter_does_not_match_with_opt_in(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        waiter = _Req(0, 1024)
        waiter.warm_match = 768
        waiter.beam_group = SimpleNamespace(beam_width=2)
        self.schedule_continuation(
            adder,
            continuation,
            [waiter],
            CacheAgnosticPolicy.FCFS,
            warm_reserve=512,
        )
        self.assertEqual(waiter.match_calls, 0)
        self.assertEqual(continuation.extend_len, 4096)

    def test_fcfs_encoder_replay_does_not_match_with_opt_in(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
        continuation = _Req(0, 8192)
        waiter = _Req(0, 1024)
        waiter.warm_match = 768
        self.schedule_continuation(
            adder,
            continuation,
            [waiter],
            CacheAgnosticPolicy.FCFS,
            warm_reserve=512,
            encoder_replay=True,
        )
        self.assertEqual(waiter.match_calls, 0)
        self.assertEqual(continuation.extend_len, 4096)

    def test_fcfs_external_cache_does_not_match_with_opt_in(self):
        for cache, value in (
            ("external_linker", True),
            ("flexkv", True),
            ("custom_backend", "other"),
            ("hisparse", True),
            ("unified_memory", True),
        ):
            with self.subTest(cache=cache):
                adder = self.make_adder(page=256, input_budget=2048, chunk_budget=4096)
                continuation = _Req(0, 8192)
                waiter = _Req(0, 1024)
                waiter.warm_match = 768
                self.schedule_continuation(
                    adder,
                    continuation,
                    [waiter],
                    CacheAgnosticPolicy.FCFS,
                    warm_reserve=512,
                    **{cache: value},
                )
                self.assertEqual(waiter.match_calls, 0)
                self.assertEqual(continuation.extend_len, 4096)

    def test_stale_prefix_does_not_replace_continuation(self):
        adder = self.make_adder(page=256, input_budget=2048, chunk_budget=256)
        adder.chunked_req_limit = 1792
        adder._replay_len = lambda req, prefix: 0
        adder._check_prefill_budget = lambda *args, **kwargs: (True, 256)
        adder._check_prefill_tile_budget = lambda *args: None
        req = _Req(0, 1024)
        with patch.dict(
            PrefillAdder._select_prefill_admission.__globals__,
            get_schedule=lambda: SimpleNamespace(schedule_policy="fcfs"),
        ):
            result = adder._select_prefill_admission(
                req,
                total_tokens=1024,
                host_hit_length=0,
                swa_host_hit_length=0,
                truncation_align_size=None,
                has_chunked_req=True,
            )
        self.assertEqual(result, AddReqResult.OTHER)


if __name__ == "__main__":
    unittest.main()
