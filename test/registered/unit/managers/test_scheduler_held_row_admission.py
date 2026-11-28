"""A request that already holds a req_to_token row is not counted twice at admission (#38118, #35537),
a parked chunked request lets no one be admitted ahead of it, and the shortest-prefill-first
load scaling reaches admission."""

import inspect
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.shortest_prefill_load import ShortestPrefillLoadScaling
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

NUM_ROWS = 4


class _Req:
    owner_list = None

    def __init__(self, holds_row, rid=None, input_len=0, prefix_len=0):
        self.rid = rid
        self.origin_input_ids = [0] * input_len
        self.output_ids = []
        self.full_untruncated_fill_ids = self.origin_input_ids
        self.num_matched_prefix_tokens = prefix_len
        self.prefix_len = prefix_len
        self.arrival_processed_tokens = 0
        self.cache_request_handle = None
        self.time_stats = SimpleNamespace(wait_queue_entry_time=0.0)
        self.kv = SimpleNamespace(holds_kv=holds_row)
        self.beam_group = None
        self.token_indices_to_pool = None

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    def init_next_round_input(self, tree_cache=None, **_):
        pass


class _Adder(PrefillAdder):
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


class _ParkingAdder(_Adder):
    """The chunked request finds no budget and stays out of the batch."""

    def add_chunked_req(self, req):
        return req


class _LimitsAdder(_Adder):
    """Records the pass limits it gets; the chunked request may continue."""

    def __init__(self, chunk_continues, *args, **kwargs):
        super().__init__()
        limits = inspect.signature(PrefillAdder).bind(*args, **kwargs).arguments
        self.chunk_continues = chunk_continues
        self.rem_input_tokens = limits["rem_input_tokens"]
        self.rem_chunk_tokens = limits["rem_chunk_tokens"]
        self.chunked_req_limit = None

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return req if self.chunk_continues else None


class _BatchBuilt(Exception):
    pass


class TestHeldRowAdmission(CustomTestCase):
    def _scheduler(
        self, disaggregation_mode, running, final_chunk, waiting, adder_class=_Adder
    ):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = IndexedReqList(ListKind.WAITING)
        move(waiting, s.waiting_queue)
        s.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        if final_chunk is not None:
            move([final_chunk], s.chunked_reqs)
        s.min_free_slots_delayer = None
        s.invariant_checker = Mock()
        s.policy = Mock(
            _bounded_active=False,
            calc_priority=lambda waiting, *args, **kwargs: list(waiting),
            admission_candidates=lambda waiting: iter(waiting),
        )
        s.policy.shortest_prefill_chunk_limit.return_value = None
        s.policy.prefill_token_budget.return_value = None
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = 8192
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: adder_class
        s.request_retirer = None
        s.page_size = 1
        s.tree_cache = SimpleNamespace(
            buffer_pipeline=None, storage_prefetch_retries=None
        )
        s.token_to_kv_pool_allocator = SimpleNamespace(
            state_slots=None, pools_share_memory=False
        )
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = NUM_ROWS
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = disaggregation_mode
        s.disagg_prefill_bootstrap_queue = None
        s.disagg_prefill_transfer_queue = []
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.enable_unified_cache_external_linker = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        # The final chunk and the running requests hold rows; one row is free.
        s.req_to_token_pool = SimpleNamespace(
            available_size=lambda: (
                NUM_ROWS - len(running) - int(final_chunk is not None)
            )
        )
        s.beam_coordinator = SimpleNamespace(
            pending_member_rows=lambda batch: 0, pending_member_rows_of=lambda reqs: 0
        )
        s.running_reqs = KeyedReqList(ListKind.DECODING)
        s.prefill_launched_reqs = KeyedReqList(ListKind.PREFILL_LAUNCHED)
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

    def test_parked_chunk_admits_no_waiting_request(self):
        running = [_Req(holds_row=True)]
        parked = _Req(holds_row=True)
        waiting = _Req(holds_row=False)
        running_batch = ScheduleBatch(reqs=running)
        s = self._scheduler(
            DisaggregationMode.NULL, running, parked, [waiting], _ParkingAdder
        )
        with (
            get_context().override_server_args(pp_max_micro_batch_size=NUM_ROWS),
            patch.object(ScheduleBatch, "init_new", side_effect=_BatchBuilt),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            new_batch, _ = s._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None, running_batch=running_batch
            )
        self.assertIsNone(new_batch)
        self.assertIs(s.chunked_req, parked)
        self.assertEqual(list(s.waiting_queue), [waiting])

    def _build_batch(self, s, running_batch, chunk_continues):
        adders = []

        def make_adder(*args, **kwargs):
            adders.append(_LimitsAdder(chunk_continues, *args, **kwargs))
            return adders[-1]

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        s.get_prefill_adder_class = lambda: make_adder
        with (
            get_context().override_server_args(pp_max_micro_batch_size=NUM_ROWS),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt) as built:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None, running_batch=running_batch
                )
        return built.exception.args[0], adders[0]

    def _admit_beside_chunk(self, chunk_continues):
        # 4096 tokens left in the chunk and 4096 waiting: the class boundary
        # halves twice to 2048, and the budget doubles once to 2048.
        policy = SchedulePolicy(
            policy="shortest-prefill-first",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            shortest_prefill_load_scaling=ShortestPrefillLoadScaling.resolve(
                chunked_prefill_size=8192,
                halving_chunks=0.5,
                decay_start_chunks=0,
                budget_min_ratio=0.125,
                budget_doubling_chunks=1,
            ),
        )
        chunk = _Req(holds_row=True, rid="chunk", input_len=6096, prefix_len=2000)
        small = _Req(holds_row=False, rid="small", input_len=500)
        mid = _Req(holds_row=False, rid="mid", input_len=3596)
        s = self._scheduler(DisaggregationMode.NULL, [], chunk, [mid, small])
        s.policy = policy
        with patch.object(policy, "_compute_prefix_matches", return_value=set()):
            can_run_list, adder = self._build_batch(
                s, ScheduleBatch(reqs=[]), chunk_continues
            )
        return [req.rid for req in can_run_list], adder

    def test_adder_gets_the_load_scaled_budget(self):
        _, adder = self._admit_beside_chunk(chunk_continues=True)
        self.assertEqual(adder.rem_input_tokens, 2048)
        self.assertEqual(adder.rem_chunk_tokens, 2048)
        # The small request joins out of the same budget.
        self.assertEqual(adder.chunked_req_limit, 2048 - 500)

    def test_arrival_ordered_request_waits_only_beside_a_continuing_chunk(self):
        for chunk_continues, admitted in (
            (True, ["chunk", "small"]),
            (False, ["chunk", "small", "mid"]),
        ):
            with self.subTest(chunk_continues=chunk_continues):
                self.assertEqual(self._admit_beside_chunk(chunk_continues)[0], admitted)

    def test_mixed_decode_rows_do_not_come_out_of_the_budget(self):
        """With mixed chunk the adder takes the decode rows out of its limits, so a
        token budget must reach it with those rows added back."""
        running = [_Req(holds_row=True), _Req(holds_row=True)]
        s = self._scheduler(
            DisaggregationMode.NULL, running, _Req(holds_row=True), [_Req(False)]
        )
        s.is_mixed_chunk = True
        s.policy.prefill_token_budget.return_value = 1024
        _, adder = self._build_batch(s, ScheduleBatch(reqs=running), False)
        self.assertEqual(adder.rem_input_tokens, 1024 + len(running))
        self.assertEqual(adder.rem_chunk_tokens, 1024 + len(running))


class TestPrefillCapacityRecovery(CustomTestCase):
    """A backup's KV lock can outlive its request and a NO_TOKEN decision.

    Use real cache locks, token allocation and PrefillAdder admission. Only the
    transfer's completion and construction of a model batch are simulated.
    """

    _scheduler = TestHeldRowAdmission._scheduler

    def setUp(self):
        self.enterContext(published_topology(pp_max_micro_batch_size=NUM_ROWS))
        self.allocator = TokenToKVPoolAllocator(
            size=1024, dtype=torch.float32, device="cpu", kvcache=None, need_sort=False
        )
        self.cache = RadixCache.create_simulated(mock_allocator=self.allocator)
        key = RadixKey(array("q", [1] * 512))
        cached = self.allocator.alloc(512)
        self.cache.insert(InsertParams(key=key, value=cached))
        node = self.cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        self.backup_lock = self.cache.lock(node)
        self.addCleanup(self._finish_backup)

        running = [self._req(f"running-{i}", 1) for i in range(2)]
        for i, req in enumerate(running):
            req.kv.req_pool_idx = i
        self.waiting = self._req("waiting", 600)
        self.batch = ScheduleBatch(reqs=running)
        self.scheduler = self._scheduler(
            DisaggregationMode.NULL, running, None, [self.waiting], PrefillAdder
        )
        self.scheduler.tree_cache = self.cache
        self.scheduler.token_to_kv_pool_allocator = self.allocator
        self.scheduler.attn_cp_group = SimpleNamespace(world_size=1)
        self.scheduler.attn_tp_group = SimpleNamespace(world_size=1)

    @staticmethod
    def _req(rid, prompt_len):
        return Req(rid, None, array("q", [2] * prompt_len), SamplingParams())

    def _finish_backup(self):
        self.cache.unlock(self.backup_lock)
        self.backup_lock = None

    def _try_prefill(self):
        def build(reqs, *args, **kwargs):
            raise _BatchBuilt(list(reqs))

        with (
            patch.object(ScheduleBatch, "init_new", side_effect=build),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            return self.scheduler._get_new_batch_prefill_raw(None, self.batch)

    def _reject_for_capacity(self):
        new_batch, running_batch = self._try_prefill()
        self.assertIsNone(new_batch)
        self.assertIs(running_batch, self.batch)
        self.assertTrue(self.batch.batch_is_full)

    def test_backup_unlock_retries_waiting_prefill_with_running_requests(self):
        self._reject_for_capacity()
        self.assertEqual(self.cache.evictable_size(), 0)
        self._finish_backup()
        self.assertEqual(self.cache.evictable_size(), 512)

        with self.assertRaises(_BatchBuilt) as built:
            self._try_prefill()
        self.assertEqual(built.exception.args[0], [self.waiting])
        self.assertFalse(self.batch.batch_is_full)
        self.assertIsNone(self.batch.prefill_no_token_capacity)

    def test_unchanged_total_capacity_does_not_repeat_admission(self):
        self._reject_for_capacity()
        # Free -> evictable changes both counters, but restores no capacity.
        extra = self.allocator.alloc(128)
        self.cache.insert(
            InsertParams(key=RadixKey(array("q", [3] * 128)), value=extra)
        )
        self.assertEqual(self.allocator.available_size(), 384)
        self.assertEqual(self.cache.evictable_size(), 128)
        with patch.object(
            self.scheduler,
            "get_prefill_adder_class",
            side_effect=AssertionError("unchanged capacity must remain gated"),
        ):
            self._try_prefill()
        self.assertTrue(self.batch.batch_is_full)
        self.assertEqual(self.batch.prefill_no_token_capacity, 512)

    def test_recovery_still_runs_admission_and_updates_failed_snapshot(self):
        held = self.allocator.alloc(64)
        self._try_prefill()
        self.assertEqual(self.batch.prefill_no_token_capacity, 448)
        self.allocator.free(held)

        # 512 tokens still cannot admit the 600-token prompt plus generation.
        self._try_prefill()
        self.assertTrue(self.batch.batch_is_full)
        self.assertEqual(self.batch.prefill_no_token_capacity, 512)
        self.assertEqual(list(self.scheduler.waiting_queue), [self.waiting])

    def test_request_slot_limit_does_not_inherit_old_capacity_failure(self):
        self._reject_for_capacity()
        self.batch.batch_is_full = False
        self.scheduler.get_num_allocatable_reqs = lambda *args, **kwargs: 0
        self._try_prefill()
        self.assertTrue(self.batch.batch_is_full)
        self.assertIsNone(self.batch.prefill_no_token_capacity)

        self._finish_backup()
        self._try_prefill()
        self.assertTrue(self.batch.batch_is_full)
        self.assertIsNone(self.batch.prefill_no_token_capacity)

    def test_multi_resource_budgets_are_not_compared_as_full_tokens(self):
        class OtherBudget(PrefillBudget):
            pass

        self.batch.batch_is_full = True
        for mamba, factory in (
            (True, None),
            (False, lambda cache: OtherBudget(self.allocator, cache)),
        ):
            with (
                self.subTest(mamba=mamba),
                patch.object(self.cache, "supports_mamba", return_value=mamba),
                patch.object(self.allocator, "create_prefill_budget", factory),
            ):
                self.scheduler._record_prefill_no_token(self.batch)
                self.assertIsNone(self.batch.prefill_no_token_capacity)

    def test_peer_capacity_recovery_reopens_admission_on_all_ranks(self):
        self._reject_for_capacity()
        cp, tp = object(), object()
        self.scheduler.attn_cp_group = SimpleNamespace(world_size=2, cpu_group=cp)
        self.scheduler.attn_tp_group = SimpleNamespace(world_size=2, cpu_group=tp)
        for peer_recovered in (False, True):
            with self.subTest(peer_recovered=peer_recovered):
                self.batch.batch_is_full = True
                self.batch.prefill_no_token_capacity = 512
                votes = []

                def reduce(vote, *, op, group):
                    self.assertEqual(vote.device.type, "cpu")
                    votes.append((group, vote.item(), op))
                    if group is cp and peer_recovered:
                        vote.fill_(1)

                with (
                    torch.device("meta"),
                    patch("torch.distributed.all_reduce", side_effect=reduce),
                ):
                    self.scheduler._retry_prefill_after_capacity_recovery(self.batch)
                self.assertEqual(self.batch.batch_is_full, not peer_recovered)
                self.assertEqual(
                    self.batch.prefill_no_token_capacity,
                    None if peer_recovered else 512,
                )
                self.assertEqual(
                    votes,
                    [
                        (cp, 0, torch.distributed.ReduceOp.MAX),
                        (tp, int(peer_recovered), torch.distributed.ReduceOp.MAX),
                    ],
                )


class TestBoundedV2SkippedAdmission(TestHeldRowAdmission):
    def test_each_temporary_skip_can_use_fixed_next_batch(self):
        from sglang.srt.managers.schedule_policy import SchedulePolicy
        from sglang.srt.mem_cache.radix_cache import RadixCache

        for reason in ("lora", "pooling", "prefetch", "device_hit_loss", "staging"):
            with self.subTest(reason=reason):
                running = [_Req(True), _Req(True)]
                final_chunk = _Req(True)
                waiting = [_Req(False) for _ in range(130)]
                blocked = set(waiting[:64])
                for i, req in enumerate(running + [final_chunk] + waiting):
                    req.rid = str(i)
                    req.num_matched_prefix_tokens = 0
                    req.lora_id = "loaded"
                    req.finished = lambda: False
                    req.cache_request_handle = req
                    req.time_stats = Mock()
                s = self._scheduler(
                    DisaggregationMode.NULL, running, final_chunk, waiting
                )
                s.policy = SchedulePolicy(
                    "lpm", RadixCache.create_simulated(), False, False, False
                )
                s.policy._should_start_bounded_batch = Mock(return_value=True)
                s.policy._compute_prefix_matches = Mock(return_value=set())
                s.policy._touch_waiting_prefixes = Mock()
                s.policy.shortest_prefill_chunk_limit = Mock(return_value=None)
                if reason == "lora":
                    s.enable_lora = True
                    s.lora_drainer = None
                    s.can_schedule_lora_req = lambda req, _: req not in blocked
                elif reason == "pooling":
                    for req in blocked:
                        req.token_indices_to_pool = [0]
                else:
                    s.enable_hicache_storage = True
                    s.tree_cache.check_prefetch_progress = lambda req: (
                        reason != "prefetch" or req not in blocked
                    )
                    s.tree_cache.pop_prefetch_loaded_span = lambda _: (0, 0)
                    s._prefetch_after_device_hit_loss = lambda req: (
                        reason == "device_hit_loss" and req in blocked
                    )
                    if reason == "staging":
                        s.tree_cache.buffer_pipeline = SimpleNamespace(
                            prepare_staged_prefetch=lambda req: req not in blocked
                        )

                def capture(can_run_list, *args, **kwargs):
                    raise _BatchBuilt(list(can_run_list))

                with (
                    get_context().override_server_args(
                        pp_max_micro_batch_size=NUM_ROWS
                    ),
                    patch.object(ScheduleBatch, "init_new", side_effect=capture),
                    patch("sglang.srt.managers.scheduler.set_time_batch"),
                ):
                    with self.assertRaises(_BatchBuilt) as built:
                        s._get_new_batch_prefill_raw(None, ScheduleBatch(reqs=running))
                self.assertEqual(built.exception.args[0], [final_chunk, waiting[64]])
                self.assertTrue(s.policy._bounded_next_selected)
                self.assertTrue(
                    all(req not in blocked for req in built.exception.args[0])
                )


if __name__ == "__main__":
    unittest.main()
