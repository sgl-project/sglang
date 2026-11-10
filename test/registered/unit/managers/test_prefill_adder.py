import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.managers.schedule_policy as schedule_policy
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    max_bootstrap_input_len,
)
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
    estimate_prefill_extend_tile_metrics,
)
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefParams,
    DecLockRefResult,
    IncLockRefResult,
    TreeLock,
)
from sglang.srt.mem_cache.kv.alloc.demand import WindowRetirement
from sglang.srt.mem_cache.page_interleave import PageShardSpec, make_page_shard_spec
from sglang.srt.mem_cache.prefill_budget import (
    ChainPrefillBudget,
    PrefillBudget,
    SWAPrefillBudget,
    estimate_swa_kv_tokens,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import build_test_unified_pools

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# Window rows retire at once here; the scheduler's window retirement is
# priced in test_request_demand.py.
RETIRE_AT_ONCE = dict(prefill_retire_lag=0, eviction_interval=1)


def setUpModule():
    for target in (
        "sglang.srt.mem_cache.prefill_budget.window_retirement",
        "sglang.srt.managers.scheduler.window_retirement",
    ):
        patcher = patch(target, return_value=WindowRetirement(0, 0, 1))
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


class _RecordingDelayer:
    """Duck-typed stand-in for PrefillDelayerSinglePassExecutor that records
    the local_prefillable value of every negotiate call and returns a fixed
    verdict."""

    def __init__(self, allow: bool):
        self.allow = allow
        self.calls = []

    def negotiate_should_allow_prefill(self, local_prefillable, **kwargs):
        self.calls.append(local_prefillable)
        return self.allow


class TestPrefillAdder(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.mock_tree_cache = self.create_tree_cache()
        self.mock_token_allocator = self.create_token_allocator()

    def create_tree_cache(
        self,
        *,
        full_evictable_size: int = 0,
        swa_evictable_size: int = 0,
        evictable_size: int = 0,
    ) -> MagicMock:
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = full_evictable_size
        tree_cache.window_evictable_size.return_value = swa_evictable_size
        tree_cache.evictable_size.return_value = evictable_size
        tree_cache.disable = False
        tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        # Route lock/unlock through inc/dec_lock_ref so tests can hook those.
        tree_cache.lock.side_effect = lambda node: TreeLock(
            node, tree_cache.inc_lock_ref(node).to_dec_params()
        )
        tree_cache.unlock.side_effect = lambda lock: (
            lock is not None and tree_cache.dec_lock_ref(lock.node, lock.receipt)
        )
        tree_cache.buffer_pipeline = None
        tree_cache.window_checkpoint_interval.return_value = 0
        return tree_cache

    def create_token_allocator(
        self,
        *,
        full_available_size: int = 0,
        swa_available_size: int = 0,
        available_size: int = 0,
        size_swa: int = 1_000_000,
    ) -> MagicMock:
        allocator = MagicMock()
        # Separate pools: the allocator's own hooks reclaim from the cache.
        allocator.pools_share_memory = False
        allocator.has_window_pool = False
        allocator.full_available_size.return_value = full_available_size
        allocator.swa_available_size.return_value = swa_available_size
        allocator.available_size.return_value = available_size
        allocator.size_swa = size_swa
        allocator.swa_req_ring = False
        allocator.request_slot_reserve.return_value = 0
        allocator.create_prefill_budget.side_effect = lambda tree_cache, **kwargs: (
            PrefillBudget(allocator, tree_cache, **kwargs)
        )
        return allocator

    def create_running_batch(self, reqs=None) -> MagicMock:
        batch = MagicMock()
        batch.reqs = list(reqs or [])
        batch.release_req.return_value = None
        batch.filter_batch.return_value = None
        return batch

    def scheduling_order(self, *, schedule_low_priority_values_first: bool):
        """State the policy on the context, which is where the scheduler reads
        it: `preempt_to_schedule` takes no record to state it on."""
        override = get_context().override_server_args(
            schedule_low_priority_values_first=schedule_low_priority_values_first
        )
        override.install()
        self.addCleanup(override.restore)

    def create_mock_req(self, rid, priority, max_new_tokens, output_len=0, wait_time=0):
        req = MagicMock(spec=Req)
        req.kv = ReqKvInfo()
        req.rid = str(rid)
        req.cache_request_handle = CacheRequestHandle(req.rid, 0)
        req.priority = priority
        req.prefix_len = 0
        # A spec=Req mock gets a per-instance class; derive extend_len like Req.
        type(req).extend_len = Req.extend_len
        req.extend_end = None
        req.full_untruncated_fill_ids = []
        req.output_ids = [0] * output_len
        req.sampling_params = SimpleNamespace(
            max_new_tokens=max_new_tokens, ignore_eos=False
        )
        req.time_stats = SimpleNamespace(wait_queue_entry_time=wait_time)
        req.retracted_stain = False
        req.host_hit_length = 0
        req.window_host_hit_lengths = {}
        req.mamba_host_hit_length = 0
        req.storage_hit_length = 0
        req.storage_hit_start = None
        req.host_hit_is_storage = False
        req.host_loaded_length = 0
        req.materialized_host_hit_len.return_value = 0
        req.fulfilled_storage_hit_len.return_value = 0
        req.finished.return_value = False
        req.needs_host_load_back.return_value = False
        return req

    def create_adder(self, running_batch, **kwargs):
        defaults = dict(
            page_size=1,
            tree_cache=self.mock_tree_cache,
            token_to_kv_pool_allocator=self.mock_token_allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=None,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )
        defaults.update(kwargs)
        defaults["token_to_kv_pool_allocator"].page_size = defaults["page_size"]
        return PrefillAdder(**defaults)

    def create_shared_adder(self, *, num_mixed_decode_tokens=0, total_bytes=2048):
        self.mock_tree_cache.supports_mamba.return_value = False
        self.mock_tree_cache.sliding_window_size = 8
        # The tree's target window slot retains its sliding window.
        self.mock_tree_cache.window_retain_size.side_effect = lambda component: 8
        self.mock_tree_cache.supports_prefix_sharing.return_value = False
        self.mock_tree_cache.transfers_in_flight.return_value = False
        allocator = build_test_unified_pools(
            device="cpu",
            kv_cache_dtype=torch.float16,
            head_num=1,
            head_dim=8,
            v_head_dim=8,
            swa_head_num=1,
            swa_head_dim=8,
            swa_v_head_dim=8,
            page_size=4,
            start_layer=0,
            end_layer=2,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            total_bytes=total_bytes,
            enable_memory_saver=False,
            need_sort=False,
            lazy_compaction=True,
        ).token_to_kv_pool_allocator
        return self.create_adder(
            self.create_running_batch(),
            page_size=4,
            rem_chunk_tokens=16,
            num_mixed_decode_tokens=num_mixed_decode_tokens,
            token_to_kv_pool_allocator=allocator,
        )

    def create_shared_req(self, rid, max_new_tokens=4):
        req = self.create_mock_req(rid, priority=0, max_new_tokens=max_new_tokens)
        req.sampling_params.ignore_eos = False
        req.window_host_hit_lengths = {}
        req.last_node = MagicMock()
        req.full_untruncated_fill_ids = list(range(12))
        return req

    def create_shortest_prefill_adder(self, *, chunk_tokens=4096):
        override = get_context().override_server_args(
            schedule_policy="shortest-prefill-first"
        )
        override.install()
        self.addCleanup(override.restore)
        self.mock_tree_cache.supports_mamba.return_value = False
        self.mock_tree_cache.supports_prefix_sharing.return_value = False
        self.mock_tree_cache.transfers_in_flight.return_value = False
        self.mock_token_allocator.available_size.return_value = 32768
        return self.create_adder(
            self.create_running_batch(), page_size=256, rem_chunk_tokens=chunk_tokens
        )

    def test_shortest_prefill_reserves_space_for_complete_waiting_requests(self):
        adder = self.create_shortest_prefill_adder()
        policy = SchedulePolicy(
            policy="shortest-prefill-first",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        continuation = self.create_shared_req("continuation")
        continuation.full_untruncated_fill_ids = list(range(16384))
        waiting = [self.create_shared_req("a"), self.create_shared_req("b")]
        for req, length in zip(waiting, [512, 1024]):
            req.origin_input_ids = list(range(length))
            req.full_untruncated_fill_ids = list(range(length))
            req.num_matched_prefix_tokens = 0
        adder.chunked_req_limit = policy.shortest_prefill_chunk_limit(
            continuation, waiting, adder.rem_chunk_tokens, adder.page_size
        )
        self.assertIs(adder.add_chunked_req(continuation), continuation)
        self.assertEqual(continuation.extend_len, 2560)
        for req in waiting:
            adder.add_one_req(req, has_chunked_req=True, truncation_align_size=None)
        self.assertEqual(adder.can_run_list, [continuation, *waiting])
        self.assertIsNone(adder.new_chunked_req)
        self.assertEqual(adder.rem_chunk_tokens, 0)
        self.assertGreaterEqual(adder.rem_total_tokens, 0)

    def test_shortest_prefill_rejects_second_unfinished_chunk(self):
        adder = self.create_shortest_prefill_adder(chunk_tokens=512)
        req = self.create_shared_req("second-chunk")
        req.full_untruncated_fill_ids = list(range(1024))
        self.assertEqual(
            adder.add_one_req(req, has_chunked_req=True, truncation_align_size=None),
            AddReqResult.OTHER,
        )
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(adder.new_chunked_req)
        self.assertIsNone(req.extend_end)
        self.mock_tree_cache.init_load_back.assert_not_called()

    def test_shortest_prefill_rechecks_chunk_limit_after_host_miss(self):
        adder = self.create_shortest_prefill_adder(chunk_tokens=512)
        req = self.create_shared_req("host-miss")
        req.full_untruncated_fill_ids = list(range(1024))
        req.prefix_len = 0
        req.host_hit_length = 768
        req.best_match_node = req.last_node
        req.needs_host_load_back.return_value = True
        self.mock_tree_cache.init_load_back.return_value = (0, req.last_node)
        self.assertEqual(
            adder.add_one_req(req, has_chunked_req=True, truncation_align_size=None),
            AddReqResult.OTHER,
        )
        self.mock_tree_cache.init_load_back.assert_called_once()
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(req.extend_end)

    def test_shortest_prefill_preserves_memory_admission(self):
        adder = self.create_shortest_prefill_adder()
        self.mock_token_allocator.available_size.return_value = 256
        req = self.create_shared_req("no-memory")
        req.full_untruncated_fill_ids = list(range(512))
        self.assertEqual(
            adder.add_one_req(req, has_chunked_req=True, truncation_align_size=None),
            AddReqResult.NO_TOKEN,
        )
        self.assertEqual(adder.can_run_list, [])

    def test_hisparse_short_prompts_reserve_full_buffers_and_scratch(self):
        """Short prompts and chunks must fit persistent buffers plus one scratch."""
        for available, chunk, prefix, ignore_eos, admitted in (
            (10496, None, 0, False, 2),
            (10496, None, 0, True, 2),
            (5248, 64, 0, False, 1),
            (4096, 64, 64, False, 0),
            (4160, 64, 64, False, 1),
        ):
            with self.subTest(available=available, chunk=chunk, prefix=prefix):
                self.mock_tree_cache.supports_mamba.return_value = False
                self.mock_tree_cache.disable = ignore_eos
                allocator = HiSparseTokenToKVPoolAllocator(
                    size=available,
                    page_size=64,
                    dtype=torch.float32,
                    device="cpu",
                    kvcache=SimpleNamespace(register_mapping=lambda mapping: None),
                    need_sort=False,
                )
                allocator.configure_spec_scratch(1024, device_buffer_size=4160)
                adder = self.create_adder(
                    self.create_running_batch(),
                    page_size=64,
                    rem_chunk_tokens=chunk,
                    token_to_kv_pool_allocator=allocator,
                )
                reqs = []
                for i in range(3):
                    req = self.create_shared_req(f"hisparse-{i}", max_new_tokens=0)
                    req.sampling_params.ignore_eos = ignore_eos
                    req.full_untruncated_fill_ids = list(range(100))
                    req.origin_input_ids = req.full_untruncated_fill_ids
                    req.prefix_len = prefix
                    req.kv = SimpleNamespace(req_pool_idx=i if prefix else None)
                    reqs.append(req)
                    if prefix:
                        adder.add_chunked_req(req)
                    else:
                        adder.add_one_req(
                            req, has_chunked_req=False, truncation_align_size=None
                        )
                self.assertEqual(adder.can_run_list, reqs[:admitted])
                if admitted:
                    self.assertEqual(adder.cur_rem_tokens, 0)
                    self.assertEqual(
                        reqs[0].extend_len,
                        64 if chunk and not prefix else 100 - prefix,
                    )

    def test_continuation_without_limit_keeps_normal_chunk_size(self):
        adder = self.create_shortest_prefill_adder()
        req = self.create_shared_req("continuation")
        req.full_untruncated_fill_ids = list(range(8192))
        self.assertIs(adder.add_chunked_req(req), req)
        self.assertEqual(req.extend_len, 4096)

    def test_exact_chunk_fill_keeps_mamba_chunks_page_aligned(self):
        # A Mamba checkpoint only lands on a page-aligned chunk end, so an
        # off-grid chunk leaves the rest of the prompt uncacheable. Without
        # prefix sharing nothing is cached, so the chunk stays exact.
        self.mock_token_allocator.available_size.return_value = 32768
        cases = ((False, True, 100), (True, True, 64), (True, False, 100))
        for supports_mamba, supports_prefix_sharing, expected in cases:
            with (
                self.subTest(
                    supports_mamba=supports_mamba,
                    supports_prefix_sharing=supports_prefix_sharing,
                ),
                patch.object(
                    schedule_policy, "_use_exact_chunk_fill", return_value=True
                ),
            ):
                self.mock_tree_cache.supports_mamba.return_value = supports_mamba
                self.mock_tree_cache.supports_prefix_sharing.return_value = (
                    supports_prefix_sharing
                )
                adder = self.create_adder(
                    self.create_running_batch(), page_size=64, rem_chunk_tokens=100
                )
                req = self.create_shared_req("chunked")
                req.full_untruncated_fill_ids = list(range(300))
                adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )
                self.assertEqual(req.extend_len, expected)

    def test_encoder_replay_rows_share_the_chunk(self):
        # A hit's replay rows run in the same step as its extend, so both must
        # fit the chunk: the MegaMoE token cap is sized from it.
        override = get_context().override_server_args(
            enable_encoder_swa_bounded_replay=True
        )
        override.install()
        self.addCleanup(override.restore)
        self.mock_tree_cache.supports_mamba.return_value = False
        self.mock_tree_cache.supports_prefix_sharing.return_value = False
        self.mock_tree_cache.transfers_in_flight.return_value = False
        self.mock_token_allocator.available_size.return_value = 32768
        for prefix_len, rem_chunk, extend_len in (
            (0, 1024, 1024),
            (64, 1024, 960),
            (4096, 1024, 896),
            (4096, 192, 64),  # the startup floor: replay rows plus one page
            (4096, 128, None),
        ):
            with self.subTest(prefix_len=prefix_len, rem_chunk=rem_chunk):
                adder = self.create_adder(
                    self.create_running_batch(),
                    page_size=64,
                    rem_chunk_tokens=rem_chunk,
                )
                req = self.create_shared_req(f"hit-{prefix_len}-{rem_chunk}")
                req.prefix_len = prefix_len
                req.full_untruncated_fill_ids = list(range(prefix_len + 4096))
                result = adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )
                if extend_len is None:
                    self.assertEqual(result, AddReqResult.OTHER)
                    self.assertEqual(adder.can_run_list, [])
                    continue
                self.assertEqual(req.extend_len, extend_len)
                self.assertEqual(adder.log_replay_tokens, min(prefix_len, 128))
                self.assertEqual(adder.rem_chunk_tokens, 0)

    def test_shared_admission_reserves_all_pending_requests(self):
        adder = self.create_shared_adder()
        first, second = (
            self.create_shared_req("first"),
            self.create_shared_req("second"),
        )
        adder.add_one_req(first, has_chunked_req=False, truncation_align_size=None)
        self.assertEqual(adder.can_run_list, [first])
        # The second request is priced beside the first one's reservation: it
        # does not fit whole there, but would fit an empty pool, so it waits
        # rather than being cut to the chunk that fits beside the first.
        self.assertEqual(
            adder.add_one_req(
                second, has_chunked_req=False, truncation_align_size=None
            ),
            AddReqResult.NO_TOKEN,
        )
        self.assertEqual(adder.can_run_list, [first])
        self.assertIsNone(adder.new_chunked_req)

    def test_shared_admission_rechecks_after_prefix_lock(self):
        adder = self.create_shared_adder()
        self.assertIsNotNone(adder.token_to_kv_pool_allocator.alloc(24))
        # Running work holds a row: the pressure it leaves is transient.
        allocator = adder.token_to_kv_pool_allocator
        self.assertIsNotNone(kv_storage_of(None, allocator).take_rows(1))
        self.mock_tree_cache.full_evictable_size.return_value = 24
        self.mock_tree_cache.window_evictable_size.return_value = 24

        def lock_prefix(_):
            self.mock_tree_cache.full_evictable_size.return_value = 0
            self.mock_tree_cache.window_evictable_size.return_value = 0
            return IncLockRefResult()

        self.mock_tree_cache.inc_lock_ref.side_effect = lock_prefix
        req = self.create_shared_req("locked-prefix")
        self.assertEqual(
            adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None),
            AddReqResult.NO_TOKEN,
        )
        self.mock_tree_cache.inc_lock_ref.assert_called_once()
        self.assertEqual(adder.can_run_list, [])

    def test_shared_continuation_defers_when_decode_consumes_chunk_budget(self):
        """Mixed decode must not commit an empty or negatively sliced prompt."""
        for decode_tokens in (16, 28):
            with self.subTest(decode_tokens=decode_tokens):
                adder = self.create_shared_adder(num_mixed_decode_tokens=decode_tokens)
                req = self.create_shared_req("continuation")
                before = (
                    adder.memory_budget.total_offset,
                    dict(adder.memory_budget.window_offsets),
                )
                self.assertIs(adder.add_chunked_req(req), req)
                self.assertEqual(adder.can_run_list, [])
                self.assertIsNone(req.extend_end)
                self.assertEqual(
                    (
                        adder.memory_budget.total_offset,
                        adder.memory_budget.window_offsets,
                    ),
                    before,
                )

    def test_shared_continuation_takes_its_rows_and_commits_its_decode(self):
        """A finishing continuation takes the whole remainder now; its decode,
        committed at admission, is charged as pending, not as rows taken now."""
        adder = self.create_shared_adder()
        req = self.create_shared_req("continuation", max_new_tokens=80)
        self.assertIsNone(adder.add_chunked_req(req))
        self.assertEqual(req.extend_len, 12)
        budget = adder.memory_budget
        # The prompt, one page of rounding, and the clipped decode.
        self.assertEqual(budget.total_offset, 12 + 4 + 80)
        self.assertEqual(budget.current_offset, 12 + 4)
        # The window commits the request's own demand, priced alike.
        self.assertEqual(
            list(budget.window_offsets.values()),
            list(budget.window_tokens(12, 80).values()),
        )
        # This pass takes the window's tail beyond the window, the window, a page.
        self.assertEqual(list(budget.window_current_offsets.values()), [4 + 8 + 4])

    def test_storage_prefetch_fulfillment_resolves_at_admission(self):
        adder = self.create_adder(self.create_running_batch())
        req = self.create_mock_req("storage-hit", priority=0, max_new_tokens=1)
        req.host_hit_length = 12
        req.host_loaded_length = 4
        req.storage_hit_length = 8
        req.storage_hit_start = 4
        req.materialized_host_hit_len.return_value = 4
        req.fulfilled_storage_hit_len.return_value = 8
        req.needs_host_load_back.return_value = True

        adder._account_prefill_cache_admission(req, prefix_len=12)

        self.mock_tree_cache.finish_storage_prefetch_admission.assert_called_once_with(
            req.cache_request_handle,
            fulfilled_tokens=8,
            reason=None,
        )
        self.assertEqual(adder.log_device_hit_tokens, 4)
        self.assertEqual(adder.log_host_hit_tokens, 0)
        self.assertEqual(adder.log_storage_hit_tokens, 8)

        self.mock_tree_cache.finish_storage_prefetch_admission.reset_mock()
        req.host_loaded_length = 0
        req.materialized_host_hit_len.return_value = 0
        req.fulfilled_storage_hit_len.return_value = 0
        adder._account_prefill_cache_admission(req, prefix_len=0)
        self.mock_tree_cache.finish_storage_prefetch_admission.assert_called_once_with(
            req.cache_request_handle, fulfilled_tokens=0, reason="device_capacity"
        )

        self.mock_tree_cache.finish_storage_prefetch_admission.reset_mock()
        req.host_hit_length = 4
        req.host_loaded_length = 4
        req.storage_hit_length = 8
        req.storage_hit_start = 4
        req.materialized_host_hit_len.return_value = 4
        req.fulfilled_storage_hit_len.return_value = 4
        req.needs_host_load_back.return_value = True
        adder._account_prefill_cache_admission(req, prefix_len=8)
        self.mock_tree_cache.finish_storage_prefetch_admission.assert_called_once_with(
            req.cache_request_handle,
            fulfilled_tokens=4,
            reason="cache_admission_shortfall",
        )
        self.assertEqual(adder.log_device_hit_tokens, 8)
        self.assertEqual(adder.log_host_hit_tokens, 0)
        self.assertEqual(adder.log_storage_hit_tokens, 12)

    def test_retracted_storage_prefetch_accounting_is_omitted(self):
        adder = self.create_adder(self.create_running_batch())
        req = self.create_mock_req(
            "retracted-storage-hit", priority=0, max_new_tokens=1
        )
        req.retracted_stain = True

        adder._account_prefill_cache_admission(req, prefix_len=8)

        self.mock_tree_cache.discard_storage_prefetch_accounting.assert_called_once_with(
            req.cache_request_handle
        )
        self.mock_tree_cache.finish_storage_prefetch_admission.assert_called_once_with(
            req.cache_request_handle, fulfilled_tokens=0, reason=None
        )

    def test_preempt_success_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=False)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.memory_budget.total_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req)

        self.assertTrue(success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(
            adder.memory_budget.total_offset, 175
        )  # 50 + 75 + 100 - 50 = 175
        running_batch.release_req.assert_called_once()

    def test_preempt_prices_a_unified_request_in_owner_tokens(self):
        """A unified chain's scalar preemption budget counts owner tokens, not
        `available_size` (here a quarter of it)."""
        running = self.create_mock_req("run", priority=0, max_new_tokens=100)
        self.scheduling_order(schedule_low_priority_values_first=False)
        allocator = self.mock_token_allocator
        allocator.full_available_size.return_value = 1600
        allocator.available_size.return_value = 400
        allocator.mamba_allocator = None
        allocator.window_pool_names.return_value = ("swa",)
        self.mock_tree_cache.window_retain_size.side_effect = lambda name: 32
        self.mock_tree_cache.supports_prefix_sharing.return_value = True
        allocator.create_prefill_budget.side_effect = lambda tree_cache, **kwargs: (
            ChainPrefillBudget(allocator, tree_cache, **kwargs)
        )
        running_batch = self.create_running_batch([running])
        adder = self.create_adder(running_batch, page_size=4)
        self.assertEqual(adder.memory_budget.total_offset, 100)
        self.assertEqual(adder.rem_total_tokens, 1500)

        new_req = self.create_mock_req("new", priority=1, max_new_tokens=32)
        new_req.full_untruncated_fill_ids = [0] * 800
        self.assertTrue(adder.preempt_to_schedule(new_req))
        self.assertIn(running, adder.preempt_list)

    def test_preempt_success_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=True)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.memory_budget.total_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req)

        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(
            adder.memory_budget.total_offset, 125
        )  # 50 + 75 + 100 - 100 = 125
        running_batch.release_req.assert_called_once()

    def test_preempt_fail_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=True)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.memory_budget.total_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=2, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_fail_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=False)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.memory_budget.total_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=0, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=-1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_skip_already_preempted_request(self):
        params = [
            ("req_prio_0", 0, 50),
            ("req_prio_1", 1, 75),
            ("req_prio_2", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=False)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.memory_budget.total_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = 225
        self.mock_token_allocator.available_size.return_value = 225

        # New request preempts req_prio_0
        first_req = self.create_mock_req(
            "new_req_prio_1", priority=1, max_new_tokens=49
        )
        first_success = adder.preempt_to_schedule(first_req)
        self.assertTrue(first_success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.memory_budget.total_offset, 175)
        running_batch.release_req.assert_called_once()

        # Second call needs more tokens than currently free, so it would need to
        # preempt req_prio_0 again if already-preempted requests were not filtered out.
        second_req = self.create_mock_req(
            "second_new_req_prio_1", priority=1, max_new_tokens=76
        )
        second_success = adder.preempt_to_schedule(second_req)

        self.assertFalse(second_success)
        self.assertEqual(adder.memory_budget.total_offset, 175)
        self.assertEqual(adder.preempt_list.count(running_reqs[0]), 1)
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_once(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=True)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.memory_budget.total_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=75)

        success = adder.preempt_to_schedule(new_req)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(
            adder.memory_budget.total_offset, 375
        )  # 50 + 75 + 100 + 125 + 125 - 100 = 375
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_twice(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=True)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.memory_budget.total_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=200)

        success = adder.preempt_to_schedule(new_req)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertIn(running_reqs[3], adder.preempt_list)
        self.assertEqual(
            adder.memory_budget.total_offset, 250
        )  # 50 + 75 + 100 + 125 + 125 - 100 - 125 = 250
        self.assertEqual(running_batch.release_req.call_count, 2)

    def test_mixed_chunk_prefill_budgets(self):
        self.mock_token_allocator.available_size.return_value = 1000

        decode_reqs = [
            self.create_mock_req(f"decode_{i}", priority=0, max_new_tokens=50)
            for i in range(8)
        ]
        running_batch = self.create_running_batch(decode_reqs)

        adder = self.create_adder(
            running_batch,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            num_mixed_decode_tokens=len(decode_reqs),
        )

        self.assertEqual(adder.rem_input_tokens, 192)  # 200 - 8
        self.assertEqual(adder.rem_chunk_tokens, 56)  # 64 - 8
        self.assertEqual(adder.memory_budget.total_offset, 408)  # 8 + 8 * 50
        self.assertEqual(adder.memory_budget.current_offset, 8)
        self.assertEqual(adder.budget_state(), AddReqResult.CONTINUE)

        # Add a prefill that exactly consumes the chunk budget
        req1 = self._create_delayer_req(56, rid="req1", max_new_tokens=64)

        result1 = adder.add_one_req(
            req1, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder.can_run_list), 1)
        self.assertEqual(adder.rem_chunk_tokens, 0)  # 56 - 56
        self.assertEqual(adder.rem_input_tokens, 136)  # 192 - 56
        self.assertEqual(result1, AddReqResult.OTHER)

        # 3 decode requests finished
        remaining_decode_reqs = decode_reqs[3:]
        running_batch2 = self.create_running_batch(remaining_decode_reqs)

        adder2 = self.create_adder(
            running_batch2,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            num_mixed_decode_tokens=len(remaining_decode_reqs),
        )

        self.assertEqual(adder2.rem_input_tokens, 195)  # 200 - 5
        self.assertEqual(adder2.rem_chunk_tokens, 59)  # 64 - 5
        self.assertEqual(adder2.memory_budget.total_offset, 255)  # 5 + 5 * 50
        self.assertEqual(adder2.budget_state(), AddReqResult.CONTINUE)

        # Same prefill no longer exhausts the chunk budget
        req2 = self._create_delayer_req(56, rid="req2", max_new_tokens=64)

        result2 = adder2.add_one_req(
            req2, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 1)
        self.assertEqual(adder2.rem_chunk_tokens, 3)  # 59 - 56 = 3 remaining
        self.assertEqual(result2, AddReqResult.CONTINUE)

        # Fit last small prefill request
        req3 = self._create_delayer_req(3, rid="req3", max_new_tokens=16)

        result3 = adder2.add_one_req(
            req3, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 2)
        self.assertEqual(adder2.rem_chunk_tokens, 0)  # 3 - 3 = 0
        self.assertEqual(result3, AddReqResult.OTHER)

    @patch(
        "sglang.srt.managers.schedule_policy.page_interleave_shard_size",
        return_value=4,
    )
    def test_ignore_eos_reserves_all_shard_class_pages(self, _shard_size):
        """Disabled-radix admission must charge one page per shard class."""
        self.mock_tree_cache.disable = True
        self.mock_token_allocator.page_size = 16
        self.mock_token_allocator.shard_spec = SimpleNamespace(
            max_prefix_tokens=1024, chunk_tokens=1024
        )
        self.mock_token_allocator.available_size.return_value = 64
        adder = self.create_adder(self.create_running_batch(), page_size=16)

        req = self.create_mock_req("ignore_eos", priority=0, max_new_tokens=1)
        req.sampling_params.ignore_eos = True
        req.origin_input_ids = list(range(16))
        req.full_untruncated_fill_ids = list(range(16))
        req.last_node = MagicMock()

        result = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(req.extend_end)

    def create_sharded_adder(
        self,
        *,
        chunk_tokens=16,
        max_prefix_tokens=16,
        shard_spec=None,
        allocator_size=1024,
        **kwargs,
    ):
        self.mock_tree_cache.supports_mamba.return_value = False
        self.mock_tree_cache.supports_prefix_sharing.return_value = False
        self.mock_tree_cache.transfers_in_flight.return_value = False
        if shard_spec is None:
            shard_spec = PageShardSpec(
                shard_rank=0,
                shard_size=2,
                page_size=4,
                max_prefix_tokens=max_prefix_tokens,
                chunk_tokens=chunk_tokens,
            )
        allocator = PageInterleavePoolAllocator(
            size=allocator_size,
            physical_page_size=shard_spec.page_size,
            shard_size=shard_spec.shard_size,
            dtype=torch.int64,
            device="cpu",
            kvcache=None,
            need_sort=False,
            shard_spec=shard_spec,
        )
        return self.create_adder(
            self.create_running_batch(),
            page_size=shard_spec.page_size,
            rem_chunk_tokens=shard_spec.chunk_tokens,
            token_to_kv_pool_allocator=allocator,
            **kwargs,
        )

    def create_sharded_req(self, rid, *, prefix_len=12, extend_len=4):
        req = self.create_shared_req(rid, max_new_tokens=1)
        req.prefix_len = prefix_len
        req.full_untruncated_fill_ids = list(range(prefix_len + extend_len))
        return req

    def test_sharded_admission_defers_prefix_scratch_overflow(self):
        adder = self.create_sharded_adder()
        first = self.create_sharded_req("first")
        second = self.create_sharded_req("second")
        self.assertEqual(
            adder.add_one_req(first, has_chunked_req=False, truncation_align_size=None),
            AddReqResult.CONTINUE,
        )
        remaining = (adder.rem_total_tokens, adder.rem_chunk_tokens)

        # Both requests fit KV capacity, but prefix scratch fits only one.
        self.assertEqual(
            adder.add_one_req(
                second, has_chunked_req=False, truncation_align_size=None
            ),
            AddReqResult.OTHER,
        )
        self.assertEqual(adder.can_run_list, [first])
        self.assertIsNone(second.extend_end)
        self.assertEqual((adder.rem_total_tokens, adder.rem_chunk_tokens), remaining)
        self.assertEqual(adder.kv_shard_block_bound_pages, 2)
        self.assertEqual(adder.kv_shard_chunk_pages, 1)

        # Rejection leaves the remaining chunk scratch usable.
        third = self.create_sharded_req("no-prefix", prefix_len=0)
        self.assertEqual(
            adder.add_one_req(third, has_chunked_req=False, truncation_align_size=None),
            AddReqResult.CONTINUE,
        )
        self.assertEqual(adder.can_run_list, [first, third])
        self.assertEqual(adder.kv_shard_block_bound_pages, 2)
        self.assertEqual(adder.kv_shard_chunk_pages, 2)

    def test_eight_long_prefixes_fit_an_eight_k_chunk(self):
        context_len = 16384
        extend_len = 1024
        prefix_len = context_len - extend_len
        kvc = SimpleNamespace(
            is_draft_worker=False,
            page_size=4,
            model_config=SimpleNamespace(context_len=context_len),
        )
        for chunk_tokens in (8192, 16384):
            with self.subTest(chunk_tokens=chunk_tokens):
                with (
                    patch(
                        "sglang.srt.mem_cache.page_interleave.get_kv_shard_group_info",
                        return_value=(0, 2),
                    ),
                    patch(
                        "sglang.srt.mem_cache.page_interleave.get_schedule",
                        return_value=SimpleNamespace(chunked_prefill_size=chunk_tokens),
                    ),
                ):
                    spec = make_page_shard_spec(kvc)
                self.assertEqual(spec.max_prefix_tokens, 8 * context_len)
                adder = self.create_sharded_adder(
                    shard_spec=spec,
                    allocator_size=8 * context_len,
                    rem_input_tokens=32768,
                )
                reqs = [
                    self.create_sharded_req(
                        str(i), prefix_len=prefix_len, extend_len=extend_len
                    )
                    for i in range(9)
                ]
                for i, req in enumerate(reqs[:8]):
                    result = adder.add_one_req(
                        req, has_chunked_req=False, truncation_align_size=None
                    )
                    # The eighth request commits even if it exhausts the budget.
                    expected = (
                        AddReqResult.OTHER
                        if i == 7 and chunk_tokens == 8192
                        else AddReqResult.CONTINUE
                    )
                    self.assertEqual(result, expected)
                self.assertEqual(adder.can_run_list, reqs[:8])
                self.assertEqual(adder.rem_chunk_tokens, chunk_tokens - 8192)
                self.assertEqual(adder.kv_shard_chunk_pages * spec.page_size, 8192)
                self.assertEqual(
                    adder.kv_shard_block_bound_pages * spec.logical_page_size,
                    8 * prefix_len,
                )
                remaining = (
                    adder.rem_total_tokens,
                    adder.rem_chunk_tokens,
                    adder.kv_shard_block_bound_pages,
                    adder.kv_shard_chunk_pages,
                )
                self.assertEqual(
                    adder.add_one_req(
                        reqs[8], has_chunked_req=False, truncation_align_size=None
                    ),
                    AddReqResult.OTHER,
                )
                self.assertEqual(adder.can_run_list, reqs[:8])
                self.assertIsNone(reqs[8].extend_end)
                self.assertEqual(
                    (
                        adder.rem_total_tokens,
                        adder.rem_chunk_tokens,
                        adder.kv_shard_block_bound_pages,
                        adder.kv_shard_chunk_pages,
                    ),
                    remaining,
                )
                if chunk_tokens == 16384:
                    # Prefix scratch is full; a request without a prefix still fits.
                    no_prefix = self.create_sharded_req(
                        "no-prefix", prefix_len=0, extend_len=extend_len
                    )
                    self.assertEqual(
                        adder.add_one_req(
                            no_prefix,
                            has_chunked_req=False,
                            truncation_align_size=None,
                        ),
                        AddReqResult.CONTINUE,
                    )
                    self.assertEqual(adder.can_run_list, reqs[:8] + [no_prefix])

    def test_sharded_admission_reserves_only_selected_chunk(self):
        # Use normal page rounding without a sharding alignment override.
        for remaining, selected, verdict in (
            (8, 8, AddReqResult.OTHER),
            (7, 4, AddReqResult.CONTINUE),
        ):
            with self.subTest(remaining=remaining):
                adder = self.create_sharded_adder(chunk_tokens=8)
                adder.exact_chunk_fill = False
                adder.rem_chunk_tokens = remaining
                req = self.create_sharded_req("chunked", extend_len=28)

                self.assertEqual(
                    adder.add_one_req(
                        req, has_chunked_req=False, truncation_align_size=None
                    ),
                    verdict,
                )
                self.assertEqual(adder.can_run_list, [req])
                self.assertIs(adder.new_chunked_req, req)
                self.assertEqual((req.prefix_len, req.extend_end), (12, 12 + selected))
                self.assertEqual(adder.kv_shard_block_bound_pages, 2)
                self.assertEqual(adder.kv_shard_chunk_pages, selected // 4)

    def test_sharded_admission_reserves_nothing_when_delayed(self):
        delayer = _RecordingDelayer(allow=False)
        adder = self.create_sharded_adder(prefill_delayer_single_pass=delayer)
        req = self.create_sharded_req("delayed")

        self.assertEqual(
            adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None),
            AddReqResult.OTHER,
        )
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(req.extend_end)
        self.assertEqual(adder.kv_shard_block_bound_pages, 0)
        self.assertEqual(adder.kv_shard_chunk_pages, 0)

        delayer.allow = True
        self.assertEqual(
            adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None),
            AddReqResult.CONTINUE,
        )
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(adder.kv_shard_block_bound_pages, 2)
        self.assertEqual(adder.kv_shard_chunk_pages, 1)

    def test_bootstrap_limit_is_the_sharded_admission_boundary(self):
        """The P/D bootstrap check rejects every prompt that sharded admission
        can never fit, so none of them waits in the queue forever."""
        page_size, tokens_per_rank = 64, 16384
        self.assertEqual(
            max_bootstrap_input_len(capacity=32768, page_size=64, shard_size=2), 32576
        )
        self.assertEqual(
            max_bootstrap_input_len(capacity=16384, page_size=64, shard_size=1), 16384
        )
        for shard_size in (2, 4):
            capacity = shard_size * tokens_per_rank
            queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
            queue.scheduler = SimpleNamespace(
                request_retirer=SimpleNamespace(retire_unadmitted=MagicMock())
            )
            queue.max_input_len = max_bootstrap_input_len(
                capacity=capacity, page_size=page_size, shard_size=shard_size
            )
            limit = queue.max_input_len
            # Both admission paths fit the limit; the stricter ignore_eos path
            # (radix cache disabled) refuses one token more.
            for prompt_len, ignore_eos, fits in (
                (limit, False, True),
                (limit, True, True),
                (limit + 1, True, False),
            ):
                with self.subTest(
                    shard_size=shard_size, prompt_len=prompt_len, ignore_eos=ignore_eos
                ):
                    self.mock_tree_cache.disable = ignore_eos
                    spec = PageShardSpec(
                        shard_rank=0,
                        shard_size=shard_size,
                        page_size=page_size,
                        max_prefix_tokens=capacity,
                        chunk_tokens=8192,
                    )
                    adder = self.create_sharded_adder(
                        shard_spec=spec,
                        allocator_size=tokens_per_rank,
                        rem_input_tokens=capacity,
                    )
                    self.assertEqual(adder.rem_total_tokens, capacity)  # empty pool
                    # PD prefill runs every request with max_new_tokens=1.
                    req = self.create_sharded_req(
                        "boundary", prefix_len=0, extend_len=prompt_len
                    )
                    req.origin_input_ids = req.full_untruncated_fill_ids
                    req.sampling_params.ignore_eos = ignore_eos
                    req.return_logprob = False
                    req.time_stats.trace_ctx = MagicMock()

                    result = adder.add_one_req(
                        req, has_chunked_req=False, truncation_align_size=None
                    )
                    self.assertEqual(req in adder.can_run_list, fits)
                    if not fits:
                        self.assertEqual(result, AddReqResult.NO_TOKEN)
                    self.assertEqual(
                        queue._check_if_req_exceed_kv_capacity(req), not fits
                    )

    def test_sharded_admission_reserves_scratch_before_load_back(self):
        """A queued H2D must be consumed by its own batch, whose prefix gather
        waits on the load: a host hit the scratch cannot hold never loads."""
        adder = self.create_sharded_adder()
        first = self.create_sharded_req("first")
        self.assertEqual(
            adder.add_one_req(first, has_chunked_req=False, truncation_align_size=None),
            AddReqResult.CONTINUE,
        )
        req = self.create_sharded_req("host-hit", prefix_len=0, extend_len=16)
        req.host_hit_length = 12
        req.needs_host_load_back = MagicMock(return_value=True)
        req.best_match_node = req.last_node

        self.assertEqual(
            adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None),
            AddReqResult.OTHER,
        )
        self.mock_tree_cache.init_load_back.assert_not_called()
        self.assertEqual(adder.can_run_list, [first])
        self.assertEqual(adder.kv_shard_block_bound_pages, 2)

    def _build_hybrid_swa_chunked_req(
        self,
        *,
        page_size,
        rem_swa,
        rem_chunk=2048,
        extend_input_len=500,
        is_hybrid_swa=True,
        full_available=100_000,
    ):
        self.mock_token_allocator.swa_available_size.return_value = rem_swa
        self.mock_token_allocator.full_available_size.return_value = full_available
        self.mock_token_allocator.available_size.return_value = full_available
        self.mock_tree_cache.sliding_window_size = 128
        adder = self.create_adder(
            self.create_running_batch(),
            page_size=page_size,
            rem_chunk_tokens=rem_chunk,
        )
        adder.is_hybrid_swa = is_hybrid_swa
        if is_hybrid_swa:
            adder.memory_budget = SWAPrefillBudget(
                self.mock_token_allocator, self.mock_tree_cache
            )

        req = self.create_mock_req("chunked", priority=0, max_new_tokens=128)
        req.prefix_len = 0
        req.full_untruncated_fill_ids = list(range(extend_input_len))
        return adder, req

    def test_add_chunked_req_hybrid_swa_reserves_page_for_alloc_extend(self):
        # alloc_extend needs extend_num_tokens + page_size per request. If the
        # scheduler hands out all of rem_swa_tokens, alloc_extend cannot get its
        # extra page and OOMs. With the fix, extend_input_len must cap at
        # rem_swa_tokens - page_size so the page is reserved.
        PAGE_SIZE = 64
        REM_SWA = 100
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE, rem_swa=REM_SWA
        )

        result = adder.add_chunked_req(req)

        self.assertIs(result, req)  # truncated → chunked prefill continues
        start, end = req.prefix_len, req.extend_end
        new_len = end - start
        self.assertLessEqual(new_len + PAGE_SIZE, REM_SWA)
        self.assertEqual(new_len, REM_SWA - PAGE_SIZE)

    def test_add_chunked_req_hybrid_swa_defers_when_swa_below_page(self):
        # When rem_swa_tokens <= page_size there is no room to serve even the
        # reservation, so the chunked req must be deferred (returned unchanged)
        # instead of falling back to rem_chunk_tokens and bypassing SWA budget.
        PAGE_SIZE = 64
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE, rem_swa=PAGE_SIZE
        )

        result = adder.add_chunked_req(req)

        self.assertIs(result, req)
        self.assertIsNone(req.extend_end)
        self.assertEqual(len(adder.can_run_list), 0)

    def test_a_continuation_below_the_checkpoint_grid_waits(self):
        """REGRESSION: a continuation whose memory fits less than one
        checkpoint grid unit waits instead of running an empty chunk."""
        adder = self.create_adder(
            self.create_running_batch(), page_size=16, rem_chunk_tokens=64
        )
        budget = object.__new__(ChainPrefillBudget)
        budget.page_size = 16
        budget.windows = ()
        budget.retirement = SimpleNamespace(
            prefill_retire_lag=0, decode_retire_lag=0, eviction_interval=1
        )
        budget.checkpoint_interval = 0
        # 48 owner tokens free; nothing else holds memory to give back.
        budget._fits = lambda full, windows, state_slots, **kwargs: full <= 48
        budget._others_hold_memory = lambda req: False
        adder.memory_budget = budget
        adder.mamba_chunk_grid = 64
        adder.strict_chunk_grid = True
        req = self.create_mock_req("continuing", priority=0, max_new_tokens=1)
        # Fitted on the page grid alone, the chunk is 32 tokens.
        self.assertEqual(
            budget.fit_chunk(
                req, extend_input_len=3072, max_new_tokens=0, chunk_limit=64
            ),
            32,
        )
        req.prefix_len = 1024
        req.full_untruncated_fill_ids = list(range(4096))
        req.extend_end = None

        self.assertIs(adder.add_chunked_req(req), req)
        self.assertIsNone(req.extend_end)
        self.assertEqual(adder.can_run_list, [])

    def test_swa_budget_for_req(self):
        # budget = max(alloc - window, 0) + min(extend + max_new, window) + page,
        # where alloc = min(extend, rem_chunk). The decode headroom is the SWA the
        # request adds to its own sliding window (extend + decode), capped at the
        # window -- a cached prefix funds the rest -- rather than a constant
        # window, which over-charged short requests into an admission livelock.
        cases = [
            # (extend, max_new, rem_chunk, window, page, expected, label)
            (64, 512, None, 128, 16, 128 + 16, "long_decode_hits_window_floor"),
            (64, 32, None, 128, 16, 96 + 16, "short_req_reserves_below_window"),
            (10, 20, None, 512, 8, 30 + 8, "short_resume_tiny_budget"),
            (300, 512, None, 128, 16, 300 + 16, "big_extend_over_window"),
            (200, 512, 50, 64, 8, 64 + 8, "chunk_capped_alloc_hits_floor"),
            # Multi-chunk: alloc = rem_chunk (1024) caps a huge extend, and the
            # chunk itself exceeds the window, so term1 = alloc - window is driven
            # by the chunk, not the full extend -> budget = chunk + page.
            (2000, 256, 1024, 512, 16, 1024 + 16, "multichunk_alloc_over_window"),
            (0, 512, None, 128, 16, 128 + 16, "extend_zero_long_decode"),
            (0, 40, None, 128, 16, 40 + 16, "extend_zero_short_decode"),
        ]
        for extend, max_new, rem_chunk, window, page, expected, label in cases:
            with self.subTest(label=label):
                self.mock_tree_cache.sliding_window_size = window
                adder = self.create_adder(
                    self.create_running_batch(),
                    page_size=page,
                    rem_chunk_tokens=rem_chunk,
                )
                self.assertEqual(
                    estimate_swa_kv_tokens(
                        extend,
                        max_new,
                        sliding_window_size=window,
                        page_size=page,
                        allocation_limit=rem_chunk,
                    ),
                    expected,
                )

    def test_swa_admission_admits_short_cached_resume_at_two_window_pool(self):
        # Livelock regression (real incident). At an SWA pool ~= 2 sliding
        # windows, a cached-prefix resume matches >= 1 window (locked, excluded
        # from rem_swa) and has only a short uncached tail + a little decode
        # left. The pre-fix constant-window reservation charged a second full
        # window, so the admission gate (swa_needed >= rem_swa_tokens) rejected
        # it every scheduler iteration while LPM kept it at the queue head --
        # 100% scheduler CPU, idle GPU. Capping the reservation at
        # min(extend + decode, window) admits it.
        WINDOW, PAGE, REM_SWA = 128, 8, 100
        PREFIX, EXTEND = 200, 16  # cached prefix > window; short uncached tail
        self.mock_token_allocator.swa_available_size.return_value = REM_SWA
        self.mock_token_allocator.full_available_size.return_value = 100_000
        self.mock_token_allocator.available_size.return_value = 100_000
        self.mock_tree_cache.sliding_window_size = WINDOW
        self.mock_tree_cache.supports_prefix_sharing.return_value = False
        self.mock_tree_cache.transfers_in_flight.return_value = False
        adder = self.create_adder(self.create_running_batch(), page_size=PAGE)
        adder.is_hybrid_swa = True
        adder.memory_budget = SWAPrefillBudget(
            self.mock_token_allocator, self.mock_tree_cache
        )

        req = self.create_mock_req(
            "resume", priority=0, max_new_tokens=40, output_len=10
        )
        req.prefix_len = PREFIX
        req.full_untruncated_fill_ids = list(range(PREFIX + EXTEND))
        req.host_hit_length = 0
        req.window_host_hit_lengths = {}
        req.last_node = MagicMock()
        req.sampling_params = SimpleNamespace(max_new_tokens=40, ignore_eos=False)

        # Pre-fix: a constant sliding-window reservation rejects the resume.
        with patch.object(
            adder.memory_budget, "swa_tokens", return_value=WINDOW + PAGE
        ):
            self.assertIs(
                adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                ),
                AddReqResult.NO_TOKEN,
            )
        self.assertEqual(len(adder.can_run_list), 0)

        # Fix: min(extend + decode, window) reservation admits it.
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIn(req, adder.can_run_list)

    def test_load_back_delivery_mismatch_reselects_the_prefill_shape(self):
        # Two incidents: a load that delivers nothing left the SWA gate sized
        # for the tail and the allocator OOMed; a cache-mode load that also
        # surfaces FULL device tokens behind a host-only SWA window tripped a
        # strict promised==loaded check and crashed the scheduler.
        WINDOW, PAGE = 128, 8
        SPAN, HOST_HIT = 1024, 1016
        self.mock_token_allocator.swa_available_size.return_value = 400
        self.mock_token_allocator.full_available_size.return_value = 100_000
        self.mock_token_allocator.available_size.return_value = 100_000
        self.mock_tree_cache.sliding_window_size = WINDOW
        self.mock_tree_cache.supports_prefix_sharing.return_value = False
        self.mock_tree_cache.transfers_in_flight.return_value = False

        def run(delivered: int, remaining_after_load: int = 100_000):
            self.mock_token_allocator.full_available_size.return_value = 100_000
            self.mock_token_allocator.swa_available_size.return_value = 400
            adder = self.create_adder(self.create_running_batch(), page_size=PAGE)
            adder.is_hybrid_swa = True
            adder.memory_budget = SWAPrefillBudget(
                self.mock_token_allocator, self.mock_tree_cache
            )
            req = self.create_mock_req("dropped-fetch", priority=0, max_new_tokens=8)
            req.prefix_len = 0
            req.full_untruncated_fill_ids = list(range(SPAN))
            req.host_hit_length = HOST_HIT
            req.window_host_hit_lengths = {"swa": WINDOW}
            req.needs_host_load_back.return_value = True
            req.last_node = MagicMock()
            req.best_match_node = MagicMock()
            req.kv.cache_protected_len = 0

            req.sampling_params = SimpleNamespace(max_new_tokens=8, ignore_eos=False)

            def load_back(params):
                self.mock_token_allocator.full_available_size.return_value = (
                    remaining_after_load
                )
                if remaining_after_load == 0:
                    self.mock_token_allocator.swa_available_size.return_value = 0
                return delivered, req.last_node

            self.mock_tree_cache.init_load_back.side_effect = load_back
            with patch.object(
                adder, "_kv_shard_reserve_scratch", side_effect=AssertionError
            ):
                verdict = adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )
            return verdict, list(adder.can_run_list), req

        # Promise kept: only the 8-token tail is prefilled, which fits.
        _, admitted, _ = run(HOST_HIT)
        self.assertEqual(len(admitted), 1)
        # Nothing delivered: the whole span is prefilled and no longer fits, so
        # admission must decline rather than OOM the pool.
        verdict, admitted, _ = run(0)
        self.assertIs(verdict, AddReqResult.NO_TOKEN)
        self.assertEqual(admitted, [])
        # Over-delivery: admitted with the loaded prefix, not the promise.
        # The loaded prefix is now pinned and no longer part of the evictable
        # budget. A successful load must not run admission gates again.
        _, admitted, req = run(HOST_HIT + 4, remaining_after_load=0)
        self.assertEqual(len(admitted), 1)
        self.assertEqual(req.prefix_len, HOST_HIT + 4)
        self.assertEqual(req.kv.cache_protected_len, HOST_HIT + 4)
        self.assertEqual((req.prefix_len, req.extend_end), (HOST_HIT + 4, SPAN))
        # A partial FULL load stays fatal.
        with self.assertRaisesRegex(RuntimeError, "promised"):
            run(HOST_HIT // 2)

    def _build_hicache_restore_admission(
        self,
        *,
        restored_tokens=0,
        restored_swa_tokens=0,
        available_swa=4096,
        chunk_tokens=16384,
        size_swa=16384,
        ring=False,
    ):
        self.mock_token_allocator = self.create_token_allocator(
            full_available_size=100_000,
            swa_available_size=available_swa,
            size_swa=size_swa,
        )
        self.mock_token_allocator.mock_add_spec(SWATokenToKVPoolAllocator)
        self.mock_token_allocator.pools_share_memory = False
        self.mock_token_allocator.has_window_pool = True
        self.mock_token_allocator.swa_req_ring = ring
        self.mock_token_allocator.swa_ring_cost_tokens = 4096
        self.mock_token_allocator.create_prefill_budget.side_effect = (
            lambda tree_cache, **kwargs: SWAPrefillBudget(
                self.mock_token_allocator, tree_cache, **kwargs
            )
        )
        self.mock_tree_cache = self.create_tree_cache()
        self.mock_tree_cache.sliding_window_size = 1024
        self.mock_tree_cache.supports_mamba.return_value = False
        adder = self.create_adder(
            self.create_running_batch(), page_size=128, rem_chunk_tokens=chunk_tokens
        )
        req = self._create_host_hit_req(host_hit=6144, tail=2048)
        req.sampling_params.max_new_tokens = 1024
        req.window_host_hit_lengths = {"swa": 1024}
        req.best_match_node = MagicMock()
        req.needs_host_load_back.return_value = True

        def load_back(params):
            self.assertIs(params.req, req)
            self.mock_token_allocator.swa_available_size.return_value -= (
                restored_swa_tokens
            )
            return restored_tokens, req.last_node

        self.mock_tree_cache.init_load_back.side_effect = load_back
        return adder, req

    def test_swa_failed_host_restore_defers_before_committing_request(self):
        for chunk_tokens, has_prior_req in (
            (None, False),
            (4096, False),
            (16384, False),
            (16384, True),
        ):
            with self.subTest(chunk_tokens=chunk_tokens, has_prior_req=has_prior_req):
                adder, req = self._build_hicache_restore_admission(
                    chunk_tokens=chunk_tokens
                )
                if has_prior_req:
                    first = self._create_delayer_req(
                        512, rid="first", max_new_tokens=16
                    )
                    first.window_host_hit_lengths = {}
                    adder.add_one_req(
                        first, has_chunked_req=False, truncation_align_size=None
                    )
                admitted_before = [first] if has_prior_req else []
                budget_before = (
                    adder.memory_budget.remaining_window_tokens,
                    adder.rem_total_tokens,
                    adder.rem_chunk_tokens,
                )
                original_node = req.last_node = 42
                original_receipt = req.lock_receipt = DecLockRefParams(node_id=7)
                acquired = IncLockRefResult(
                    node_id=42, skipped_lock_components=(ComponentType.SWA,)
                )
                self.mock_tree_cache.inc_lock_ref.return_value = acquired
                self.mock_tree_cache.reset_mock()

                result = adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )

                self.assertIs(result, AddReqResult.NO_TOKEN)
                self.assertEqual(adder.can_run_list, admitted_before)
                self.assertIsNone(adder.new_chunked_req)
                self.assertIsNone(req.extend_end)
                self.assertEqual(
                    (
                        adder.memory_budget.remaining_window_tokens,
                        adder.rem_total_tokens,
                        adder.rem_chunk_tokens,
                    ),
                    budget_before,
                )
                self.assertIs(req.lock_receipt, original_receipt)
                self.mock_tree_cache.init_load_back.assert_called_once()
                self.mock_tree_cache.inc_lock_ref.assert_called_once_with(original_node)
                self.mock_tree_cache.dec_lock_ref.assert_called_once_with(
                    original_node, acquired.to_dec_params()
                )

    def test_swa_failed_host_restore_retries_when_capacity_recovers(self):
        adder, req = self._build_hicache_restore_admission()
        self.assertIs(
            adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None),
            AddReqResult.NO_TOKEN,
        )
        self.assertEqual(adder.can_run_list, [])

        self.mock_token_allocator.swa_available_size.return_value = 10_240
        result = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertIs(result, AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual((req.prefix_len, req.extend_end), (0, 8192))
        self.assertGreaterEqual(adder.memory_budget.remaining_window_tokens, 0)
        self.assertEqual(self.mock_tree_cache.inc_lock_ref.call_count, 3)
        self.assertEqual(self.mock_tree_cache.dec_lock_ref.call_count, 2)

    def test_swa_host_restore_admitted(self):
        for label, options, expected_range, remaining_swa, expected_result in (
            (
                "restored",
                dict(
                    restored_tokens=6144, restored_swa_tokens=1024, available_swa=3328
                ),
                (6144, 8192),
                128,
                AddReqResult.CONTINUE,
            ),
            (
                "exact_fit_ring",
                dict(ring=True),
                (0, 8192),
                0,
                AddReqResult.NO_TOKEN,
            ),
            (
                "fitting_chunk",
                dict(chunk_tokens=2048),
                (0, 2048),
                1920,
                AddReqResult.OTHER,
            ),
            (
                "never_fits",
                dict(size_swa=4096),
                (0, 2944),
                1024,
                AddReqResult.CONTINUE,
            ),
        ):
            with self.subTest(label=label):
                adder, req = self._build_hicache_restore_admission(**options)

                result = adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )

                self.assertIs(result, expected_result)
                self.mock_tree_cache.init_load_back.assert_called_once()
                self.assertEqual(adder.can_run_list, [req])
                expected_start, expected_end = expected_range
                self.assertEqual((req.prefix_len, req.extend_end), expected_range)
                self.assertEqual(req.host_loaded_length, expected_start)
                self.assertIs(
                    adder.new_chunked_req, req if expected_end < 8192 else None
                )
                self.assertEqual(
                    adder.memory_budget.remaining_window_tokens, remaining_swa
                )

    def test_swa_partial_host_restore_fails_loudly_before_request_commit(self):
        adder, req = self._build_hicache_restore_admission(
            restored_tokens=3072, restored_swa_tokens=1024
        )

        with self.assertRaisesRegex(RuntimeError, "HiCache load-back"):
            adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(req.extend_end)

    def _create_host_hit_req(self, *, prefix_len=0, host_hit=8192, tail=1024):
        req = self._create_delayer_req(prefix_len + host_hit + tail)
        req.prefix_len = prefix_len
        req.host_hit_length = host_hit
        req.needs_host_load_back.return_value = True
        req.best_match_node = req.last_node
        req.kv.cache_protected_len = prefix_len
        return req

    def _build_mamba_restore_admission(self, *, holds_state, available_slots):
        self.mock_token_allocator = self.create_token_allocator(available_size=100_000)
        self.mock_tree_cache = self.create_tree_cache()
        self.mock_tree_cache.mock_add_spec(schedule_policy.UnifiedRadixCache)
        self.mock_tree_cache.buffer_pipeline = MagicMock()
        allocator = self.mock_token_allocator
        allocator.window_pool_names.return_value = ()
        allocator.sub_pool_roles.return_value = ("full", (), None)
        # One shared memory: the cache evicts, and the pool says it fits.
        allocator.pools_share_memory = True
        allocator.prealloc_shortfall.return_value = None
        # A state slot takes free slot ids and 64 tokens' worth of shared bytes.
        allocator.admits.side_effect = lambda demand, **kwargs: (
            demand.state_slots <= allocator.mamba_allocator.schedulable_available_size()
            and demand.full_tokens + demand.state_slots * 64
            <= allocator.available_size()
        )
        allocator.create_prefill_budget.side_effect = lambda tree_cache, **kwargs: (
            ChainPrefillBudget(allocator, tree_cache, **kwargs)
        )
        adder = self.create_adder(self.create_running_batch())
        allocator.mamba_allocator.schedulable_available_size.return_value = (
            available_slots
        )
        self.mock_tree_cache.supports_mamba.return_value = True
        self.mock_tree_cache.mamba_evictable_size.return_value = 0
        req = self._create_host_hit_req(host_hit=32, tail=8)
        req.mamba_host_hit_length = 1
        req.kv.mamba_pool_idx = torch.tensor([0]) if holds_state else None
        allocator.storage.pending_state_slots.side_effect = lambda req: int(
            not req.kv.holds_mamba
        )

        def load_back(params):
            allocator.mamba_allocator.schedulable_available_size.return_value -= (
                int(not req.kv.holds_mamba) + req.mamba_host_hit_length
            )
            req.kv.mamba_pool_idx = torch.tensor([0])
            return req.host_hit_length, req.last_node

        self.mock_tree_cache.init_load_back.side_effect = load_back
        return adder, req

    def _unreserved_state_slots(self, adder):
        """Free state slots not yet promised to admitted requests."""
        return (
            self.mock_token_allocator.mamba_allocator.schedulable_available_size()
            - adder.memory_budget.pending_state_slots
        )

    def test_mamba_restore_admission_counts_request_and_checkpoint(self):
        for holds_state, slots, capacity, admitted in (
            (False, 1, 100_000, False),
            (True, 0, 100_000, False),
            (False, 2, 100_000, True),
            (True, 1, 100_000, True),
            (False, 2, 150, False),  # Enough shared bytes for one state, not two.
        ):
            with self.subTest(holds_state=holds_state, slots=slots, capacity=capacity):
                adder, req = self._build_mamba_restore_admission(
                    holds_state=holds_state, available_slots=slots
                )
                self.mock_token_allocator.available_size.return_value = capacity
                result = adder.add_one_req(req, False, None)
                self.assertEqual(adder.can_run_list, [req] if admitted else [])
                self.assertEqual(
                    self._unreserved_state_slots(adder), 0 if admitted else slots
                )
                if admitted:
                    self.mock_tree_cache.init_load_back.assert_called_once()
                    self.assertEqual(adder.memory_budget.total_offset, 8 + 8 + 1)
                else:
                    self.assertIs(result, AddReqResult.NO_TOKEN)
                    self.mock_tree_cache.init_load_back.assert_not_called()
                    self.assertEqual(adder.memory_budget.total_offset, 0)
                    if capacity == 100_000:
                        self.mock_tree_cache.buffer_pipeline.defer_staged_admission.assert_called_once_with(
                            req, "mamba"
                        )

    def test_idle_budget_failure_defers_staged_prefix_unless_kv_awaits_transfer(self):
        for kv_awaiting_transfer in (False, True):
            with self.subTest(kv_awaiting_transfer=kv_awaiting_transfer):
                self.mock_token_allocator = self.create_token_allocator(
                    available_size=100
                )
                self.mock_tree_cache = self.create_tree_cache()
                self.mock_tree_cache.buffer_pipeline = MagicMock()
                adder = self.create_adder(
                    self.create_running_batch(),
                    kv_awaiting_transfer=kv_awaiting_transfer,
                )
                req = self._create_delayer_req(1000)

                result = adder.add_one_req(req, False, None)

                self.assertIs(result, AddReqResult.NO_TOKEN)
                defer = self.mock_tree_cache.buffer_pipeline.defer_staged_admission
                if kv_awaiting_transfer:
                    defer.assert_not_called()
                else:
                    defer.assert_called_once_with(req, "prefill_budget")

    def test_refused_chunk_continuation_admits_no_other_request(self):
        """A waiting chunked continuation closes the pass: another request
        admitted beside it would take the room it waits for."""
        probe, allocator = self._pressured_shared_adder(running=0, held_tokens=0)
        adder, _ = self._pressured_shared_adder(
            running=1, held_tokens=allocator.available_size() - 4
        )
        continuation = self.create_shared_req("continuation", max_new_tokens=4096)
        continuation.prefix_len = 4
        self.assertIs(adder.add_chunked_req(continuation), continuation)
        self.assertEqual(adder.can_run_list, [])

        waiting = self.create_shared_req("waiting")
        waiting.full_untruncated_fill_ids = list(range(40))
        result = adder.add_one_req(
            waiting, has_chunked_req=True, truncation_align_size=None
        )
        self.assertIs(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(adder.new_chunked_req)
        self.assertIs(adder.budget_state(), AddReqResult.NO_TOKEN)

    def test_admitted_chunk_continuation_lets_others_follow(self):
        """A continuation that makes progress leaves the pass open."""
        adder = self.create_shared_adder()
        continuation = self.create_shared_req("continuation")
        continuation.prefix_len = 11
        self.assertIsNone(adder.add_chunked_req(continuation))
        self.assertEqual(adder.can_run_list, [continuation])
        waiting = self.create_shared_req("waiting")
        waiting.full_untruncated_fill_ids = list(range(4))
        adder.add_one_req(waiting, has_chunked_req=False, truncation_align_size=None)
        self.assertEqual(adder.can_run_list, [continuation, waiting])

    def test_memory_cut_continuation_admits_no_other_request(self):
        """A continuation cut short by memory, with chunk tokens to spare, admits
        no waiting request beside it, whole or as a second chunked request."""
        probe, allocator = self._pressured_shared_adder(running=0, held_tokens=0)
        # Rows held by no running request (a transfer, a load): with nothing
        # running, the continuation shrinks to the rows there are.
        adder, _ = self._pressured_shared_adder(
            running=0, held_tokens=allocator.available_size() - 12
        )
        continuation = self.create_shared_req("continuation", max_new_tokens=4096)
        continuation.prefix_len = 0
        self.assertIs(adder.add_chunked_req(continuation), continuation)
        self.assertEqual(adder.can_run_list, [continuation])
        self.assertLess(continuation.extend_end, 12)
        self.assertGreater(adder.rem_chunk_tokens, 0)

        for length in (4, 40):
            waiting = self.create_shared_req("waiting")
            waiting.full_untruncated_fill_ids = list(range(length))
            result = adder.add_one_req(
                waiting, has_chunked_req=True, truncation_align_size=None
            )
            self.assertIs(result, AddReqResult.NO_TOKEN)
            self.assertIsNone(adder.new_chunked_req)
            self.assertEqual(adder.can_run_list, [continuation])

    def _pressured_shared_adder(self, *, running, held_tokens):
        """A shared-pool adder holding `held_tokens` ids and `running` row holders
        outside its (empty) batch, as for PD sends or PP microbatches."""
        adder = self.create_shared_adder()
        allocator = adder.token_to_kv_pool_allocator
        if held_tokens:
            self.assertIsNotNone(allocator.alloc(held_tokens))
        if running:
            self.assertIsNotNone(kv_storage_of(None, allocator).take_rows(running))
        return adder, allocator

    def test_transient_pressure_waits_instead_of_shrinking_a_new_chunk(self):
        """A request that fits an empty pool but not now waits while running work
        will give memory back; with nothing to give back it is unschedulable."""
        probe, allocator = self._pressured_shared_adder(running=0, held_tokens=0)
        capacity = allocator.available_size()
        for running, held, expected in (
            (1, capacity - 8, AddReqResult.NO_TOKEN),
            (0, capacity - 8, AddReqResult.UNSCHEDULABLE),
        ):
            with self.subTest(running=running):
                adder, _ = self._pressured_shared_adder(
                    running=running, held_tokens=held
                )
                req = self.create_shared_req("transient")
                req.full_untruncated_fill_ids = list(range(capacity // 2))
                result = adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )
                self.assertIs(result, expected)
                self.assertEqual(adder.can_run_list, [])
                self.assertIsNone(adder.new_chunked_req)

    def test_permanent_shortfall_still_shrinks_a_new_chunk(self):
        """A request that could not fit even an empty pool is cut to a chunk,
        so it can make progress at all."""
        adder, allocator = self._pressured_shared_adder(running=1, held_tokens=0)
        req = self.create_shared_req("long", max_new_tokens=4096)
        req.full_untruncated_fill_ids = list(range(allocator.available_size() * 2))
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertEqual(adder.can_run_list, [req])
        self.assertIs(adder.new_chunked_req, req)
        self.assertGreater(req.extend_end, 0)

    def test_continuation_waits_for_running_requests_under_transient_pressure(self):
        """A continuation waits, closing the pass, while other row holders will
        free memory; with no other holder it shrinks to make progress."""
        probe, allocator = self._pressured_shared_adder(running=0, held_tokens=0)
        capacity = allocator.available_size()
        held = capacity - 8
        adder, _ = self._pressured_shared_adder(running=1, held_tokens=held)
        continuation = self.create_shared_req("continuation")
        continuation.full_untruncated_fill_ids = list(range(64))
        continuation.prefix_len = 4
        self.assertIs(adder.add_chunked_req(continuation), continuation)
        self.assertEqual(adder.can_run_list, [])
        self.assertIs(adder.budget_state(), AddReqResult.NO_TOKEN)

        alone, _ = self._pressured_shared_adder(running=0, held_tokens=held)
        continuation = self.create_shared_req("continuation")
        continuation.full_untruncated_fill_ids = list(range(64))
        continuation.prefix_len = 4
        self.assertIs(alone.add_chunked_req(continuation), continuation)
        self.assertEqual(alone.can_run_list, [continuation])
        self.assertGreater(continuation.extend_end, 4)

    def test_an_idle_session_row_does_not_hold_back_a_continuation(self):
        """An idle streaming session's parked row gives no memory back, so a
        continuation shrinks to fit rather than wait for the session to close."""
        probe, _ = self._pressured_shared_adder(running=0, held_tokens=0)
        held = probe.token_to_kv_pool_allocator.available_size() - 8
        adder, allocator = self._pressured_shared_adder(running=0, held_tokens=held)
        storage = kv_storage_of(None, allocator)
        session = StreamingSession(
            SimpleNamespace(
                req_to_token_pool=storage.requests,
                token_to_kv_pool_allocator=allocator,
                page_size=4,
                logical_page_size=4,
            )
        )
        turn = SimpleNamespace(
            session=SimpleNamespace(
                session_id="idle", streaming=True, finish_req=lambda req: None
            ),
            kv=ReqKvInfo(),
            origin_input_ids=list(range(4)),
            output_ids=[],
            finished_reason=FINISH_LENGTH(length=0),
            finished_len=0,
            swa_branching_seqlen=None,
        )
        turn.detach_kv = lambda: setattr(turn, "kv", ReqKvInfo())
        self.assertIsNotNone(storage.requests.alloc([turn]))
        turn.kv.kv_committed_len = turn.kv.kv_allocated_len = 4
        session.slots["idle"] = SessionSlot(kv=turn.kv)
        self.assertTrue(session.try_cache_finished_req(turn))
        self.assertTrue(session.slots["idle"].kv.holds_kv)

        continuation = self.create_shared_req("continuation")
        continuation.full_untruncated_fill_ids = list(range(64))
        continuation.prefix_len = 4
        self.assertIs(adder.add_chunked_req(continuation), continuation)
        self.assertEqual(adder.can_run_list, [continuation])
        self.assertGreater(continuation.extend_end, 4)

        # BUG REGRESSION. Matching the session's next turn, before admission,
        # must not mark the row in use, or the continuation and turn deadlock.
        next_turn = SimpleNamespace(
            session=turn.session, kv=ReqKvInfo(), to_finish=None
        )
        self.assertIsNotNone(
            session.try_match_prefix(SimpleNamespace(req=next_turn, key=list(range(4))))
        )
        self.assertIs(next_turn.kv, session.slots["idle"].kv)
        self.assertFalse(storage.others_hold_rows(None))
        continuation = self.create_shared_req("continuation")
        continuation.full_untruncated_fill_ids = list(range(64))
        continuation.prefix_len = 4
        again = self._next_pass(allocator)
        self.assertIs(again.add_chunked_req(continuation), continuation)
        self.assertEqual(again.can_run_list, [continuation])
        # Allocating the admitted turn's rows puts the row in use again.
        session.take(next_turn)
        self.assertTrue(storage.others_hold_rows(None))

    def test_parked_session_memory_lets_a_new_request_take_a_smaller_chunk(self):
        """BUG REGRESSION. With only an idle session's parked rows held, a new
        request is admitted at a smaller chunk instead of waiting for the session."""
        prompt = 80

        def adder_after(held, *, park):
            # ``held`` tokens held by a request whose row is live (running
            # work, which gives them back) or parked (an idle session).
            adder = self.create_shared_adder(total_bytes=8192)
            allocator = adder.token_to_kv_pool_allocator
            storage = kv_storage_of(None, allocator)
            if held:
                self.assertIsNotNone(allocator.alloc(held))
            holder = SimpleNamespace(kv=ReqKvInfo())
            self.assertIsNotNone(storage.requests.alloc([holder]))
            if park:
                storage.park_row(holder)
            return adder

        def new_req():
            req = self.create_shared_req("after-session", max_new_tokens=4)
            req.full_untruncated_fill_ids = list(range(prompt))
            return req

        capacity = adder_after(0, park=False).token_to_kv_pool_allocator
        capacity = capacity.available_size()
        found = False
        for held in range(0, capacity, 4):
            waiting = adder_after(held, park=False)
            req = new_req()
            result = waiting.add_one_req(
                req, has_chunked_req=False, truncation_align_size=None
            )
            if waiting.can_run_list:
                continue
            parked = adder_after(held, park=True)
            req = new_req()
            parked.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
            if parked.can_run_list != [req]:
                continue
            found = True
            # Held by running work the same pressure is transient: it waits.
            self.assertIs(result, AddReqResult.NO_TOKEN)
            # Admitted whole, with a chunk limit below the configured 16.
            self.assertIs(parked.new_chunked_req, req)
            self.assertGreater(req.extend_end, 0)
            self.assertLess(req.extend_end, 16)
            break
        self.assertTrue(found, "no pool state where only a smaller chunk fits")

    def _next_pass(self, allocator, *, running=0):
        adder = self.create_adder(
            self.create_running_batch(),
            page_size=4,
            rem_chunk_tokens=16,
            token_to_kv_pool_allocator=allocator,
        )
        if running:
            self.assertIsNotNone(kv_storage_of(None, allocator).take_rows(running))
        return adder

    def _write_chunk(self, allocator, req, held):
        """What the chunk's forward and its cache insert leave held: its rows,
        less the window rows that fell out of the window."""
        rows = allocator.alloc(req.extend_end - req.prefix_len)
        self.assertIsNotNone(rows)
        held = torch.cat([held, rows])
        (window,) = allocator.window_pool_names()
        out_of_window = len(held) - 8
        if out_of_window > 0:
            allocator.free_window(window, held[:out_of_window])
        req.prefix_len = req.extend_end
        return held

    def test_lone_chunked_request_reaches_its_last_chunk(self):
        """A lone clipped request runs every chunk, including an unaligned last
        chunk shorter than a page: the clip, admission and chunks price alike."""
        for prompt in (16 * 2 + 3, 16 * 2 + 1, 16 * 3):
            with self.subTest(prompt=prompt):
                allocator = self.create_shared_adder(
                    total_bytes=8192
                ).token_to_kv_pool_allocator
                max_new = allocator.max_new_tokens_for_memory(
                    prompt,
                    4096,
                    token_capacity=allocator.available_size(),
                    sliding_window_size=8,
                    chunk_size=16,
                    **RETIRE_AT_ONCE,
                )
                self.assertGreater(max_new, 0)
                req = self.create_shared_req("lone", max_new_tokens=max_new)
                req.full_untruncated_fill_ids = list(range(prompt))
                adder = self._next_pass(allocator)
                adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )
                self.assertIs(adder.new_chunked_req, req)
                held = self._write_chunk(
                    allocator, req, torch.empty(0, dtype=torch.int64)
                )
                for _ in range(prompt):
                    if self._next_pass(allocator).add_chunked_req(req) is None:
                        break
                    self.assertGreater(req.extend_end, req.prefix_len)
                    held = self._write_chunk(allocator, req, held)
                self.assertEqual(req.extend_end, prompt)

    def test_fresh_chunked_request_commits_its_window_peak(self):
        """A fresh chunked request's first commitment covers its window peak (the
        window plus a chunk), so later chunks never take room promised to others."""
        for prompt in (40, 16 * 2 + 3, 16 * 4):
            with self.subTest(prompt=prompt):
                adder = self.create_shared_adder(total_bytes=8192)
                allocator = adder.token_to_kv_pool_allocator
                window = adder.memory_budget.windows[0][0]
                req = self.create_shared_req("fresh", max_new_tokens=4)
                req.full_untruncated_fill_ids = list(range(prompt))
                adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )
                self.assertIs(adder.new_chunked_req, req)
                committed = adder.memory_budget.window_offsets[window]
                held = torch.empty(0, dtype=torch.int64)
                peak = 0
                chunks = []
                while True:
                    rows = -(-(req.extend_end - req.prefix_len) // 4) * 4
                    chunks.append((req.prefix_len, req.extend_end, len(held)))
                    taken = allocator.alloc(rows)
                    self.assertIsNotNone(taken)
                    held = torch.cat([held, taken])
                    peak = max(peak, len(held))
                    if len(held) > 8:
                        allocator.free_swa(held[: len(held) - 8])
                        held = held[len(held) - 8 :]
                    req.prefix_len = req.extend_end
                    if req.extend_end >= prompt:
                        break
                    self._next_pass(allocator).add_chunked_req(req)
                    self.assertGreater(req.extend_end, req.prefix_len)
                self.assertLessEqual(peak, committed, chunks)

    def test_chunked_request_commits_its_rest_at_admission(self):
        """Each chunk of a chunked request commits the rest of its prompt and
        decode, so a request admitted beside it is priced after that."""
        adder = self.create_shared_adder(total_bytes=8192)
        allocator = adder.token_to_kv_pool_allocator
        req = self.create_shared_req("long", max_new_tokens=4)
        req.full_untruncated_fill_ids = list(range(40))
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIs(adder.new_chunked_req, req)
        self.assertEqual(req.extend_end, 16)
        budget = adder.memory_budget
        # Taken now: the chunk and its rounding page. Pending: all 40 tokens,
        # the page, and the decode.
        self.assertEqual(budget.current_offset, 16 + 4)
        self.assertEqual(budget.total_offset, 40 + 4 + 4)

        held = self._write_chunk(allocator, req, torch.empty(0, dtype=torch.int64))
        passed = self._next_pass(allocator)
        self.assertIs(passed.add_chunked_req(req), req)
        self.assertEqual(passed.memory_budget.total_offset, 24 + 4 + 4)
        self.assertEqual(passed.memory_budget.current_offset, 16 + 4)

        # The rest of the continuation is held for it: a pass beside it admits
        # a smaller request than its chunk alone would leave room for.
        def admitted(prompt, committed):
            probe = self._next_pass(allocator)
            if committed:
                self.assertIs(probe.add_chunked_req(req), req)
            else:
                probe.memory_budget.reserve(16, 0, chunk_limit=16)
            req.extend_end = None
            other = self.create_shared_req("other", max_new_tokens=4)
            other.full_untruncated_fill_ids = list(range(prompt))
            probe.add_one_req(other, has_chunked_req=True, truncation_align_size=None)
            return other in probe.can_run_list

        largest = {
            committed: max(
                (n for n in range(4, 400, 4) if admitted(n, committed)), default=0
            )
            for committed in (False, True)
        }
        self.assertLess(largest[True], largest[False])
        self.assertIsNotNone(held)

    def test_waiting_chunk_stops_priority_preemption(self):
        """While the unfinished chunked request waits, the pass admits no one,
        so no running request is preempted for a waiting one."""
        self.scheduling_order(schedule_low_priority_values_first=False)
        self.mock_token_allocator.full_available_size.return_value = 225
        self.mock_token_allocator.available_size.return_value = 225
        for waits in (False, True):
            with self.subTest(waits=waits):
                running = [
                    self.create_mock_req(rid, priority, max_new_tokens)
                    for rid, priority, max_new_tokens in (
                        ("run1", 0, 50),
                        ("run2", 1, 75),
                        ("run3", 2, 100),
                    )
                ]
                running_batch = self.create_running_batch(running)
                adder = self.create_adder(running_batch)
                adder.chunked_req_waits = waits
                new_req = self.create_mock_req("new", priority=1, max_new_tokens=49)
                # Without a waiting chunk the same request preempts run1.
                self.assertEqual(adder.preempt_to_schedule(new_req), not waits)
                self.assertEqual(adder.preempt_list, [] if waits else [running[0]])

    def test_mamba_chunk_continuation_does_not_charge_restored_state_again(self):
        adder, req = self._build_mamba_restore_admission(
            holds_state=False, available_slots=2
        )
        adder.rem_chunk_tokens = 4
        adder.add_one_req(req, False, None)
        self.assertIs(adder.new_chunked_req, req)
        req.prefix_len = req.extend_end
        # Continuing chunks retain the original host-hit metadata.
        self.assertEqual(req.mamba_host_hit_length, 1)
        next_adder = self.create_adder(self.create_running_batch(), rem_chunk_tokens=4)
        self.mock_token_allocator.mamba_allocator.schedulable_available_size.return_value = 0
        self.assertIsNone(next_adder.add_chunked_req(req))
        self.assertEqual(self._unreserved_state_slots(next_adder), 0)
        self.assertEqual(next_adder.memory_budget.total_offset, 4 + 8 + 1)

    def test_successful_load_back_commits_the_selected_shape_once(self):
        cases = (
            ("full", 0, 24, None, None, 8, 8),
            ("full_unaligned", 0, 24, None, None, 7, 8),
            ("retracted_unaligned", 0, 24, None, None, 7, 8),
            ("chunk", 0, 24, 4, None, 4, 0),
            ("aux_only", 24, 0, None, None, 8, 8),
            ("overdelivery_full", 0, 24, None, None, 8, 8),
            ("overdelivery_chunk", 0, 24, 4, None, 4, 0),
            ("overdelivery_chunk_end", 0, 24, 8, None, 8, 8),
            (
                "dllm",
                0,
                24,
                None,
                SimpleNamespace(
                    block_size=4,
                    max_running_requests=2,
                    requires_separate_context_encoding=False,
                ),
                4,
                0,
            ),
            (
                "overdelivery_dllm",
                0,
                24,
                None,
                SimpleNamespace(
                    block_size=4,
                    max_running_requests=2,
                    requires_separate_context_encoding=False,
                ),
                4,
                0,
            ),
        )
        for name, prefix_len, host_hit, chunk, dllm, extend, decode in cases:
            with self.subTest(mode=name):
                self.mock_tree_cache.reset_mock()
                adder = self._create_delayer_adder(
                    available_tokens=100_000,
                    delayer=None,
                    page_size=2,
                    rem_chunk_tokens=chunk,
                    dllm_config=dllm,
                )
                req = self._create_host_hit_req(
                    prefix_len=prefix_len,
                    host_hit=host_hit,
                    tail=extend if chunk is None and dllm is None else 8,
                )
                req.retracted_stain = name == "retracted_unaligned"
                if name.startswith("overdelivery"):
                    req.host_hit_length -= 4
                old_node, restored_node = req.last_node, object()
                if name == "aux_only":
                    req.window_host_hit_lengths = {"swa": 8}

                def load_back(params):
                    self.assertIs(params.req, req)
                    tile_gate.assert_called_once()
                    tile_gate.return_value = AddReqResult.OTHER
                    return host_hit, restored_node

                self.mock_tree_cache.init_load_back.side_effect = load_back
                with patch.object(
                    adder, "_check_prefill_tile_budget", return_value=None
                ) as tile_gate:
                    adder.add_one_req(req, False, None)
                    tile_gate.assert_called_once()
                self.mock_tree_cache.init_load_back.assert_called_once()
                self.assertEqual(adder.can_run_list, [req])
                self.assertEqual((req.prefix_len, req.extend_end), (24, 24 + extend))
                self.mock_tree_cache.inc_lock_ref.assert_any_call(restored_node)
                self.assertIs(
                    self.mock_tree_cache.dec_lock_ref.call_args.args[0], old_node
                )
                self.assertEqual(adder.log_hit_tokens, 24)
                self.assertEqual(adder.log_input_tokens, extend)
                self.assertEqual(
                    adder.reprocessed_log_input_tokens,
                    extend if req.retracted_stain else 0,
                )
                self.assertEqual(
                    adder.memory_budget.total_offset,
                    adder.ceil_paged_tokens(extend) + decode + 2,
                )
                self.assertEqual(
                    adder.new_chunked_req is req,
                    name in ("chunk", "overdelivery_chunk"),
                )
                self.mock_tree_cache.init_load_back.side_effect = None

    def test_swa_new_tokens_clamps_remaining_not_total(self):
        # Remaining decode headroom must be min(max_new - generated, CLIP)
        # (subtract-then-clip). The reversed order (clip-then-subtract) zeroes
        # out a request that has already generated >= CLIP tokens but still has a
        # long decode ahead, under-reserving its SWA window -> OOM risk on resume
        # of a long-generation request.
        from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS as CLIP

        adder = self.create_adder(self.create_running_batch())
        cases = [
            # (max_new, generated, expected, label)
            (100, 10, 90, "below_clip_normal"),
            (40, 100, 0, "already_finished"),
            # clip-then-subtract would give 3996; subtract-then-clip caps at CLIP.
            (CLIP + 6000, 100, CLIP, "long_gen_small_output_caps_at_clip"),
            # clip-then-subtract would give 0; the long decode still needs CLIP.
            (CLIP + 6000, CLIP + 100, CLIP, "long_gen_output_over_clip"),
        ]
        for max_new, generated, expected, label in cases:
            with self.subTest(label=label):
                req = self.create_mock_req(
                    label, priority=0, max_new_tokens=max_new, output_len=generated
                )
                self.assertEqual(adder._swa_new_tokens(req), expected)

    def test_delayer_not_consulted_when_kv_budget_rejects(self):
        """A rank whose first candidate fails the KV-budget gate must NOT
        report local_prefillable=True: add_one_req returns NO_TOKEN before
        negotiating, and finalize() later reports the rank as not
        prefillable. Regression guard: the negotiate used to run at the top
        of add_one_req, so under KV pressure a full rank still claimed
        prefillable=True, the delayer saw "all prefillable" and allowed, the
        full ranks then NO_TOKEN'ed out, and the DP-synced forward mixed
        prefill and decode — the exact pattern the delayer exists to
        prevent."""
        delayer = _RecordingDelayer(allow=True)
        adder = self._create_delayer_adder(available_tokens=10, delayer=delayer)

        result = adder.add_one_req(
            self._create_delayer_req(50),
            has_chunked_req=False,
            truncation_align_size=None,
        )

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(delayer.calls, [])
        self.assertEqual(adder.can_run_list, [])

    def test_delayer_not_consulted_when_post_lock_recheck_rejects(self):
        """Locking the request's own prefix converts evictable tokens into
        protected ones, so a request can pass the pre-lock KV gate yet fail
        the post-lock recheck — precisely the high-utilization regime the
        delayer targets. The negotiate must sit after that recheck too, or
        the rank claims prefillable=True and then runs decode."""
        delayer = _RecordingDelayer(allow=True)
        adder = self._create_delayer_adder(available_tokens=10, delayer=delayer)
        # Pre-lock budget is covered by evictable tokens; inc_lock_ref pins
        # them (evictable -> protected), shrinking the budget below demand.
        self.mock_tree_cache.evictable_size.return_value = 1000
        self.mock_tree_cache.full_evictable_size.return_value = 1000

        def _pin_prefix(node):
            self.mock_tree_cache.evictable_size.return_value = 0
            self.mock_tree_cache.full_evictable_size.return_value = 0
            return IncLockRefResult()

        self.mock_tree_cache.inc_lock_ref.side_effect = _pin_prefix

        result = adder.add_one_req(
            self._create_delayer_req(50),
            has_chunked_req=False,
            truncation_align_size=None,
        )

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(delayer.calls, [])
        self.assertEqual(adder.can_run_list, [])

    def test_delay_verdict_blocks_admissible_request(self):
        """An admissible request must still be gated by the (relocated)
        negotiate: on a delay verdict it is not admitted, and the rank
        reported prefillable=True exactly once."""
        delayer = _RecordingDelayer(allow=False)
        adder = self._create_delayer_adder(available_tokens=100_000, delayer=delayer)

        result = adder.add_one_req(
            self._create_delayer_req(50),
            has_chunked_req=False,
            truncation_align_size=None,
        )

        self.assertEqual(result, AddReqResult.OTHER)
        self.assertEqual(delayer.calls, [True])
        self.assertEqual(adder.can_run_list, [])

    def test_allow_verdict_admits_request(self):
        """On an allow verdict the request proceeds through admission — the
        relocated negotiate must not block the commit path."""
        delayer = _RecordingDelayer(allow=True)
        adder = self._create_delayer_adder(available_tokens=100_000, delayer=delayer)
        req = self._create_delayer_req(50)

        result = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(delayer.calls, [True])
        self.assertIn(req, adder.can_run_list)

    def test_chunked_req_negotiates_prefillable_and_proceeds(self):
        """A rank resuming a chunked prefill runs it this pass regardless of
        the verdict, so add_chunked_req must report prefillable=True (else a
        rank with an empty waiting queue reports False via finalize() and
        peers delay while it prefills alone) and must not drop the chunk on
        a delay verdict (that would leak memory)."""
        delayer = _RecordingDelayer(allow=False)
        adder = self._create_delayer_adder(
            available_tokens=100_000, delayer=delayer, rem_chunk_tokens=500
        )
        req = self._create_delayer_req(200)

        result = adder.add_chunked_req(req)

        self.assertIsNone(result)  # chunk fully admitted, not truncated
        self.assertEqual(delayer.calls, [True])
        self.assertIn(req, adder.can_run_list)

    def _create_delayer_adder(self, *, available_tokens, delayer, **kwargs):
        self.mock_token_allocator.available_size.return_value = available_tokens
        self.mock_token_allocator.full_available_size.return_value = available_tokens
        return self.create_adder(
            self.create_running_batch(),
            prefill_delayer_single_pass=delayer,
            **kwargs,
        )

    def _create_delayer_req(
        self, num_tokens: int, *, rid="delayer_req", max_new_tokens=8
    ):
        req = self.create_mock_req(rid, priority=0, max_new_tokens=max_new_tokens)
        req.full_untruncated_fill_ids = list(range(num_tokens))
        req.host_hit_length = 0
        req.last_node = MagicMock()
        req.sampling_params.ignore_eos = False
        return req

    def test_add_chunked_req_non_hybrid_no_swa_reservation(self):
        # Non-hybrid path: the SWA-pool reservation must NOT apply, otherwise
        # the fix would regress non-SWA models.
        PAGE_SIZE = 16
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE,
            rem_swa=10,
            rem_chunk=500,
            extend_input_len=200,
            is_hybrid_swa=False,
            full_available=300,
        )

        result = adder.add_chunked_req(req)
        self.assertIsNone(result)
        self.assertEqual((req.prefix_len, req.extend_end), (0, 200))
        self.assertIn(req, adder.can_run_list)

    def _adder_with_extend_lens(self, extend_lens):
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.can_run_list = [
            SimpleNamespace(extend_input_len=length) for length in extend_lens
        ]
        # BLOCK_M is auto-detected from the attention backend in production; the
        # __new__ helper bypasses __init__, so set it explicitly. 64 matches the
        # block_m the tile-count assertions below are computed against.
        adder.prefill_tile_block_m = 64
        return adder

    def test_estimate_prefill_extend_tile_metrics(self):
        metrics = estimate_prefill_extend_tile_metrics([1, 7, 13, 129], block_m=64)

        self.assertEqual(metrics["q_tiles_per_request"], [1, 1, 1, 3])
        self.assertEqual(metrics["legacy_q_tiles_per_head"], 12)
        self.assertEqual(metrics["compact_q_tiles_per_head"], 6)
        self.assertEqual(metrics["saved_q_tiles_per_head"], 6)
        self.assertEqual(metrics["saved_q_tile_ratio"], 0.5)

    def test_compact_prefill_tile_budget_admits_more_than_legacy(self):
        adder = self._adder_with_extend_lens([1, 7, 13])

        # The tile-budget admission is gated on HIP in production; force the gate
        # on so this vendor-neutral admission-math check runs on any CI runner.
        with (
            patch.object(schedule_policy, "_IS_HIP", True),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET", 6),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET_MODE", "compact"),
        ):
            self.assertIsNone(adder._check_prefill_tile_budget(129))

        with (
            patch.object(schedule_policy, "_IS_HIP", True),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET", 6),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET_MODE", "legacy"),
        ):
            self.assertEqual(adder._check_prefill_tile_budget(129), AddReqResult.OTHER)

    def test_prefill_tile_budget_always_allows_first_request(self):
        adder = self._adder_with_extend_lens([])

        with (
            patch.object(schedule_policy, "_IS_HIP", True),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET", 1),
        ):
            self.assertIsNone(adder._check_prefill_tile_budget(4096))

    def test_prefill_tile_budget_disabled_on_non_hip(self):
        # AMD-only: on non-HIP vendors the tile-budget admission must be a no-op
        # even when the budget env is set, so scheduler behavior is unchanged.
        adder = self._adder_with_extend_lens([1, 7, 13])

        with (
            patch.object(schedule_policy, "_IS_HIP", False),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET", 6),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET_MODE", "legacy"),
        ):
            self.assertIsNone(adder._check_prefill_tile_budget(129))

    # ---- _swa_req_never_fits: the gate on the _swa_chunk_cap escape hatch ----
    # The hatch shrinks a chunk and admits it instead of waiting; it must fire
    # ONLY for a request whose SWA budget can never fit the drained pool (true
    # head-of-line livelock). Admitting transient-pressure requests instead
    # collapses the SWA evictable cushion and causes a retraction storm.

    def create_swa_adder(
        self, *, size_swa: int, sliding_window: int, page_size: int = 16
    ) -> PrefillAdder:
        self.mock_tree_cache.sliding_window_size = sliding_window
        self.mock_token_allocator = self.create_token_allocator(size_swa=size_swa)
        self.mock_token_allocator.create_prefill_budget.side_effect = (
            lambda tree_cache, **kwargs: SWAPrefillBudget(
                self.mock_token_allocator, tree_cache, **kwargs
            )
        )
        return self.create_adder(
            self.create_running_batch(),
            page_size=page_size,
            rem_chunk_tokens=512,
        )

    def test_swa_never_fits_false_under_transient_pressure(self):
        # Small request against an ample pool: budget << pool, so it would fit
        # once running decodes drain -> must wait, not take the hatch.
        adder = self.create_swa_adder(size_swa=1024, sliding_window=128)
        self.assertFalse(
            adder.memory_budget.swa_never_fits(extend_input_len=256, max_new_tokens=64)
        )

    def test_swa_never_fits_true_when_budget_exceeds_whole_pool(self):
        # A large host-hit load-back charge pushes the budget past the entire
        # pool: it can never fit however far the pool drains -> hatch.
        adder = self.create_swa_adder(size_swa=1024, sliding_window=128)
        self.assertTrue(
            adder.memory_budget.swa_never_fits(
                extend_input_len=256,
                max_new_tokens=64,
                window_host_hit_lengths={"swa": 4096},
            )
        )

    def test_swa_never_fits_is_gated_by_pool_capacity(self):
        # Same request; only the pool size changes. Proves the check compares
        # the budget against size_swa (guards against a wrong-accessor bug).
        req = dict(
            extend_input_len=256,
            max_new_tokens=64,
            window_host_hit_lengths={"swa": 600},
        )
        self.assertTrue(
            self.create_swa_adder(
                size_swa=512, sliding_window=128
            ).memory_budget.swa_never_fits(**req)
        )
        self.assertFalse(
            self.create_swa_adder(
                size_swa=4096, sliding_window=128
            ).memory_budget.swa_never_fits(**req)
        )


if __name__ == "__main__":
    unittest.main()
