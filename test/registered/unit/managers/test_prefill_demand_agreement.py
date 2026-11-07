"""Every site that prices a request under unified memory builds its demand the
same way: the memory clip of an arriving request, prefill admission, the
final shape check, the commitment a chunked request keeps, load-back and the
decode node's ceiling. They differ only where the purpose table says so, and
admission, its shape check and its commitment price the chunk admission
chose."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import (
    PrefillAdder,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefResult,
    IncLockRefResult,
    TreeLock,
)
from sglang.srt.mem_cache.kv.alloc.demand import (
    DEMAND_RULES,
    DemandPurpose,
    KVDemand,
    StateSlots,
    decode_estimate_tokens,
    demand_inputs,
    largest_fitting_chunk,
    smallest_admissible_chunk,
    window_retirement,
)
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    MHAGeometry,
    StateGeometry,
    kv_layer_decls,
)
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftBand,
    FusedDraftPlacement,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_test_unified_pools,
    setup_allocator_context,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# Window rows retire at once in these tests, which price chunks and
# admission; the scheduler's window retirement is priced in
# test_request_demand.py.
RETIRE_AT_ONCE = dict(prefill_retire_lag=0, eviction_interval=1)


def setUpModule():
    for target in (
        "sglang.srt.mem_cache.prefill_budget.window_retirement",
        "sglang.srt.managers.scheduler.window_retirement",
    ):
        patcher = patch(target, return_value=(0, 1))
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


_DENSE = MHAGeometry(
    head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
)
_STATE = StateGeometry(
    conv_state_shapes=((3, 8),),
    conv_dtype=torch.bfloat16,
    temporal_state_shape=(1, 4, 8),
    temporal_dtype=torch.float32,
)


def _window_pool(total_bytes):
    """A FULL and one window pool of page 4 whose window the cache builder
    binds to 8 tokens, sized by bytes."""
    setup_allocator_context()
    return build_test_unified_pools(
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


def _draft_window(window, name="draft_swa"):
    """A fused draft whose window layers get a band of their own."""
    return FusedDraftPlacement.from_counts(
        counts={name: [1]},
        regions={
            name: DenseDraftRegion(
                lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16
            )
        },
        new_bands={name: DraftBand(kind="window", window=window)},
    )


def _storage(
    *, windows=(), state=False, full_tokens=64, window_tokens=64, fused_draft=None
):
    """A storage of page 4: one FULL layer, at most one window layer (its
    length in ``windows``, None for one the model does not state), with
    ``state`` two state layers, and a ``fused_draft``'s bands."""
    setup_allocator_context()
    assert len(windows) <= 1, "a target declares at most one window"
    layer = 1 + len(windows)
    state_layers = (layer, layer + 1) if state else ()
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=_DENSE,
        window_layer_ids=(1,) if windows else (),
        window_geometry=_DENSE if windows else None,
        window=windows[0] if windows else None,
        state_layer_ids=state_layers,
        state_geometry=_STATE if state else None,
        checkpoint_interval=4 if state else None,
    )
    plan = plan_kv_storage(decls, page_size=4, fused_draft=fused_draft)
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=full_tokens,
            window_tokens=window_tokens,
            state_slots=4 if state else 0,
            max_running_requests=3,
            context_len=256,
            alignment=1,
        ),
    )
    return build_kv_storage(
        plan,
        sized,
        UnifiedBuildContext(
            device="cpu",
            start_layer=0,
            end_layer=layer + (2 if state else 0),
            enable_memory_saver=False,
            need_sort=False,
            max_num_reqs=3,
            model_context_len=256,
            lazy_compaction=True,
            state_cache_params=SimpleNamespace(layers=state_layers),
        ),
    )


def _tree_cache(window=8):
    tree = MagicMock()
    tree.full_evictable_size.return_value = 0
    tree.window_evictable_size.return_value = 0
    tree.evictable_size.return_value = 0
    tree.supports_mamba.return_value = False
    tree.supports_prefix_sharing.return_value = False
    tree.window_checkpoint_interval.return_value = 0
    tree.sliding_window_size = window
    tree.window_retain_size.side_effect = lambda component: window
    tree.disable = False
    tree.inc_lock_ref.return_value = IncLockRefResult()
    tree.dec_lock_ref.return_value = DecLockRefResult()
    tree.lock.side_effect = lambda node: TreeLock(
        node, tree.inc_lock_ref(node).to_dec_params()
    )
    tree.unlock.side_effect = lambda lock: (
        lock is not None and tree.dec_lock_ref(lock.node, lock.receipt)
    )
    tree.buffer_pipeline = None
    return tree


def _req(rid, prompt, max_new_tokens=4):
    req = MagicMock(spec=Req)
    req.kv = ReqKvInfo()
    req.rid = rid
    req.cache_request_handle = CacheRequestHandle(rid, 0)
    req.priority = 0
    req.prefix_len = 0
    type(req).extend_len = Req.extend_len
    req.extend_end = None
    req.full_untruncated_fill_ids = list(range(prompt))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(
        max_new_tokens=max_new_tokens, ignore_eos=False
    )
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
    req.last_node = MagicMock()
    return req


class _AdderCase(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.tree = _tree_cache()

    def _adder(self, allocator, *, chunk=16):
        batch = MagicMock()
        batch.reqs = []
        return PrefillAdder(
            page_size=4,
            tree_cache=self.tree,
            token_to_kv_pool_allocator=allocator,
            running_batch=batch,
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=chunk,
        )

    def _write_chunk(self, allocator, req, held):
        """What the chunk's forward and its cache insert leave held: its rows,
        less the window rows that fell out of the 8-token window."""
        rows = allocator.alloc(req.extend_end - req.prefix_len)
        self.assertIsNotNone(rows)
        held = torch.cat([held, rows])
        (window,) = allocator.window_pool_names()
        out_of_window = len(held) - 8
        if out_of_window > 0:
            allocator.free_window(window, held[:out_of_window])
        req.prefix_len = req.extend_end
        return held

    def _runs_alone(self, allocator, prompt, max_new):
        """Whether a lone request runs every chunk of its prompt."""
        req = _req("lone", prompt, max_new_tokens=max_new)
        adder = self._adder(allocator)
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        if adder.can_run_list != [req]:
            return False
        held = self._write_chunk(allocator, req, torch.empty(0, dtype=torch.int64))
        if adder.new_chunked_req is not req:
            return True
        for _ in range(prompt):
            if self._adder(allocator).add_chunked_req(req) is None:
                self._write_chunk(allocator, req, held)
                return True
            if req.extend_end <= req.prefix_len:
                return False
            held = self._write_chunk(allocator, req, held)
        return False


class TestPurposeTable(CustomTestCase):
    def test_every_purpose_has_its_intended_differences(self):
        self.assertEqual(set(DEMAND_RULES), set(DemandPurpose))
        ceiling = DEMAND_RULES[DemandPurpose.CEILING]
        # A ceiling prices the whole generation of a fresh request in an empty
        # pool; every other purpose prices the decode estimate and what the
        # request still lacks.
        self.assertFalse(ceiling.estimates_decode)
        self.assertIs(ceiling.state_slots, StateSlots.FRESH)
        self.assertFalse(ceiling.credits_evictable)
        for purpose in (
            DemandPurpose.ADMIT,
            DemandPurpose.LOAD_BACK,
            DemandPurpose.DECODE_PREALLOC,
        ):
            self.assertTrue(DEMAND_RULES[purpose].estimates_decode)
            self.assertIs(DEMAND_RULES[purpose].state_slots, StateSlots.PENDING)
        # A load-back reclaims before it asks, so it credits no eviction.
        self.assertTrue(DEMAND_RULES[DemandPurpose.ADMIT].credits_evictable)
        self.assertFalse(DEMAND_RULES[DemandPurpose.LOAD_BACK].credits_evictable)
        self.assertTrue(DEMAND_RULES[DemandPurpose.DECODE_PREALLOC].credits_evictable)

    def test_inputs_differ_only_by_the_table(self):
        long = 10**7
        common = dict(
            windows={"window_8": 8}, page_size=4, chunk_limit=16, **RETIRE_AT_ONCE
        )
        ceiling = demand_inputs(DemandPurpose.CEILING, 40, long, **common)
        admit = demand_inputs(DemandPurpose.ADMIT, 40, long, **common)
        self.assertEqual(ceiling.max_new_tokens, long)
        self.assertEqual(admit.max_new_tokens, decode_estimate_tokens(long))
        self.assertLess(admit.max_new_tokens, long)
        # Same chunk, windows, rounding page and new window otherwise.
        self.assertEqual(
            demand_inputs(DemandPurpose.CEILING, 40, 4, **common).demand(),
            demand_inputs(DemandPurpose.ADMIT, 40, 4, **common).demand(),
        )
        # The token sub-pool takes the ceiled prompt, the decode and one page.
        self.assertEqual(admit.demand().full_tokens, 40 + admit.max_new_tokens + 4)
        with self.assertRaises(ValueError):
            demand_inputs(DemandPurpose.CEILING, 40, 4, cached_tokens=8, **common)

    def test_one_new_window_definition(self):
        common = dict(
            windows={"window_8": 8}, page_size=4, chunk_limit=16, **RETIRE_AT_ONCE
        )

        def window(**kwargs):
            demand = demand_inputs(DemandPurpose.ADMIT, 64, 0, **common, **kwargs)
            return demand.demand().window_tokens["window_8"]

        # Between its chunks a request with nothing cached keeps a window of
        # its own beside the next chunk; a cached prefix already holds part
        # of it, and a later chunk all of it.
        self.assertEqual(window(), 8 + 16 + 4)
        self.assertEqual(window(cached_tokens=4), 8 - 4 + 16 + 4)
        self.assertLess(window(continuation=True), window(cached_tokens=4))

    def test_chunk_search_helpers(self):
        self.assertEqual(smallest_admissible_chunk(None, 4), None)
        self.assertEqual(smallest_admissible_chunk(16, 4), 4)
        self.assertEqual(smallest_admissible_chunk(64, 4, 6), 12)
        self.assertEqual(smallest_admissible_chunk(8, 4, 64), 0)
        self.assertEqual(largest_fitting_chunk(lambda c: c <= 10, 16, 4), 8)
        self.assertEqual(largest_fitting_chunk(lambda c: False, 16, 4), 0)


class TestWindowRetirement(CustomTestCase):
    def test_the_scheduler_states_when_it_retires_window_rows(self):
        """Without prefix sharing the overlap scheduler retires a prefill's
        rows a configured chunk behind its frontier
        (`ScheduleBatch.maybe_evict_swa`); decode retires every
        SGLANG_SWA_EVICTION_INTERVAL rows."""
        for shares, no_overlap, lag in (
            (False, False, 16),
            (True, False, 0),
            (False, True, 0),
        ):
            schedule = SimpleNamespace(
                chunked_prefill_size=16, disable_overlap_schedule=no_overlap
            )
            with (
                self.subTest(shares=shares, no_overlap=no_overlap),
                patch("sglang.srt.runtime_context.get_schedule", return_value=schedule),
                envs.SGLANG_SWA_EVICTION_INTERVAL.override(32),
            ):
                self.assertEqual(window_retirement(shares_prefixes=shares), (lag, 32))


class TestAdmission(_AdderCase):
    # 144 token rows, a window row costing as much as a token row.
    POOL_BYTES = 4736

    def _finish(self, allocator, req, held, *, chunk):
        """Run ``req``'s chunks after its first alone; whether it finishes."""
        held = self._write_chunk(allocator, req, held)
        if self.adder.new_chunked_req is not req:
            return True
        for _ in range(len(req.full_untruncated_fill_ids)):
            if self._adder(allocator, chunk=chunk).add_chunked_req(req) is None:
                self._write_chunk(allocator, req, held)
                return True
            if req.extend_end <= req.prefix_len:
                return False
            held = self._write_chunk(allocator, req, held)
        return False

    def test_the_empty_pool_check_prices_the_cached_prefix(self):
        """BUG REGRESSION. The empty-pool check credited a request its cached
        prefix, which the pool still holds for it (admission locks it), and
        so saw room at the configured chunk the request never has. With
        nothing else running it waited on its own prefix forever, though a
        smaller chunk fits."""
        allocator = _window_pool(self.POOL_BYTES)
        self.assertEqual(allocator.size_full, 144)
        (window,) = allocator.window_pool_names()
        prefix = allocator.alloc(64)
        allocator.free_window(window, prefix[:-8])
        req = _req("cached", 104, max_new_tokens=4)
        req.prefix_len = 64
        self.adder = self._adder(allocator, chunk=32)

        self.adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(self.adder.can_run_list, [req])
        self.assertEqual(req.extend_end - req.prefix_len, 20)
        self.assertTrue(self._finish(allocator, req, prefix[-8:], chunk=32))

    def test_ignore_eos_admission_selects_the_chunk_as_every_admission(self):
        """BUG REGRESSION. An ignore-EOS request without a prefix cache was
        admitted only at the configured chunk, where its whole demand does
        not fit an empty pool, so it waited forever on an idle server. It
        selects its chunk, runs it and commits it as any admission does."""
        commitments = {}
        for ignore_eos in (False, True):
            with self.subTest(ignore_eos=ignore_eos):
                allocator = _window_pool(self.POOL_BYTES)
                self.tree.disable = ignore_eos
                req = _req("lone", 100, max_new_tokens=4)
                req.sampling_params.ignore_eos = ignore_eos
                req.origin_input_ids = list(range(100))
                self.adder = self._adder(allocator, chunk=32)

                self.adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )

                self.assertEqual(self.adder.can_run_list, [req])
                self.assertEqual(req.extend_end, 24)
                budget = self.adder.memory_budget
                commitments[ignore_eos] = (
                    budget.total_offset,
                    dict(budget.window_offsets),
                )
                self.assertTrue(
                    self._finish(
                        allocator, req, torch.empty(0, dtype=torch.int64), chunk=32
                    )
                )
        self.assertEqual(commitments[True], commitments[False])


class TestMemoryClip(_AdderCase):
    def test_clip_accepts_prompts_adaptive_chunking_runs(self):
        """BUG REGRESSION. The clip priced a prompt at the configured chunk and
        rejected it when that chunk's window peak did not fit, though
        admission shrinks the chunk and runs it. It now prices the smallest
        chunk admission may cut to."""
        for total_bytes, prompt in ((2048, 32), (4096, 96), (8192, 224)):
            with self.subTest(total_bytes=total_bytes, prompt=prompt):
                allocator = _window_pool(total_bytes)
                limit = allocator.max_new_tokens_for_memory(
                    prompt,
                    4,
                    token_capacity=allocator.size_full,
                    sliding_window_size=8,
                    chunk_size=16,
                    **RETIRE_AT_ONCE,
                )
                self.assertEqual(limit, 4)
                self.assertTrue(self._runs_alone(allocator, prompt, limit))

    def test_clip_agrees_with_a_lone_request(self):
        """The clip lets a prompt through exactly when admission, at the
        smallest chunk it may cut to, fits all of it in an empty pool; and a
        lone request it lets through runs to its last chunk. Both price the
        window retirement the scheduler states, whatever it is."""
        for lag, interval in ((0, 1), (16, 8)):
            accepted = refused = 0
            with patch(
                "sglang.srt.mem_cache.prefill_budget.window_retirement",
                return_value=(lag, interval),
            ):
                for prompt in range(8, 400, 12):
                    allocator = _window_pool(4096)
                    limit = allocator.max_new_tokens_for_memory(
                        prompt,
                        64,
                        token_capacity=allocator.size_full,
                        sliding_window_size=8,
                        chunk_size=16,
                        prefill_retire_lag=lag,
                        eviction_interval=interval,
                    )
                    adder = self._adder(allocator)
                    admitted_whole = adder.memory_budget._admits(
                        adder.memory_budget.request_demand(
                            prompt,
                            0,
                            chunk_limit=4,
                            extra_tokens=adder.per_req_token_overhead - 4,
                        ),
                        empty_pool=True,
                    )
                    with self.subTest(lag=lag, prompt=prompt, limit=limit):
                        self.assertEqual(limit is not None, admitted_whole)
                        if limit is None:
                            refused += 1
                        else:
                            accepted += 1
                            self.assertTrue(self._runs_alone(allocator, prompt, limit))
            self.assertGreater(accepted, 0)
            self.assertGreater(refused, 0)

    def test_clip_applies_the_scheduler_token_capacity(self):
        """BUG REGRESSION. The clip ignored the token capacity the scheduler
        passes (shrunk for CP padding, widened for KV sharding), so it could
        grant decode the scheduler's own capacity does not hold."""
        allocator = _window_pool(8192)
        kwargs = dict(sliding_window_size=8, chunk_size=16)
        free = allocator.max_new_tokens_for_memory(
            12, 80, token_capacity=allocator.size_full, **kwargs, **RETIRE_AT_ONCE
        )
        self.assertEqual(free, 80)
        for capacity in (40, 64, 90):
            with self.subTest(capacity=capacity):
                self.assertEqual(
                    allocator.max_new_tokens_for_memory(
                        12, 80, token_capacity=capacity, **kwargs, **RETIRE_AT_ONCE
                    ),
                    min(80, max(0, capacity - 12 - 4 - 1)),
                )

    def test_clip_cuts_no_chunk_below_the_alignment(self):
        """A truncated chunk is cut to the scheduler's alignment, so the clip
        prices no smaller chunk than that."""
        allocator = _window_pool(4096)
        kwargs = dict(token_capacity=allocator.size_full, sliding_window_size=8)
        with patch.object(allocator, "admits", wraps=allocator.admits) as admits:
            allocator.max_new_tokens_for_memory(
                96, 4, chunk_size=64, chunk_align=16, **kwargs, **RETIRE_AT_ONCE
            )
        (window,) = allocator.window_pool_names()
        # The window peak of a first chunk of 16: the window it keeps, the
        # chunk and a page.
        peaks = {call.args[0].window_tokens[window] for call in admits.call_args_list}
        self.assertEqual(peaks, {8 + 16 + 4})

    def test_clip_prices_stated_windows_before_retention_binds(self):
        """BUG REGRESSION. Before the storage binds retention, every window
        was priced at the target model's window, a window that states its own
        length (a draft's, say) included. Only a window the model does not
        state takes the scheduler's window, as binding gives it."""
        storage = _storage(
            windows=(None,),
            full_tokens=128,
            window_tokens=128,
            fused_draft=_draft_window(8, name="window_8"),
        )
        allocator = storage.allocator
        self.assertIsNone(allocator.retained_windows)
        self.assertEqual(allocator._ceiling_windows(32), {"window": 32, "window_8": 8})
        kwargs = dict(token_capacity=10**6, sliding_window_size=32)
        before = [
            allocator.max_new_tokens_for_memory(
                prompt, 200, chunk_size=chunk, **kwargs, **RETIRE_AT_ONCE
            )
            for prompt, chunk in itertools.product((8, 40, 96), (None, 16))
        ]
        storage.bind_window_retention(32, None)
        after = [
            allocator.max_new_tokens_for_memory(
                prompt, 200, chunk_size=chunk, **kwargs, **RETIRE_AT_ONCE
            )
            for prompt, chunk in itertools.product((8, 40, 96), (None, 16))
        ]
        self.assertEqual(before, after)

    def test_clip_prices_a_draft_window_by_its_own_length(self):
        """A draft's window is priced by its own length, never the target
        model's window parameter: where it is the only window (a target of
        FULL and state layers), and before retention binds beside a target
        window the model does not state."""
        only = _storage(state=True, fused_draft=_draft_window(8)).allocator
        self.assertEqual(only._ceiling_windows(63), {"draft_swa": 8})
        kwargs = dict(token_capacity=64, sliding_window_size=63, chunk_size=None)
        own = only.max_new_tokens_for_memory(16, 46, **kwargs, **RETIRE_AT_ONCE)
        with patch.object(only, "retained_windows", {"draft_swa": 63}):
            target = only.max_new_tokens_for_memory(16, 46, **kwargs, **RETIRE_AT_ONCE)
        self.assertIsNotNone(own)
        self.assertGreaterEqual(own, target or 0)
        beside = _storage(
            windows=(None,), window_tokens=512, fused_draft=_draft_window(8)
        )
        self.assertIsNone(beside.allocator.retained_windows)
        self.assertEqual(
            beside.allocator._ceiling_windows(63), {"window": 63, "draft_swa": 8}
        )


class TestDecodeCeiling(CustomTestCase):
    def _queue(self, storage):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool_allocator = storage.allocator
        queue.req_to_token_pool = storage.requests
        queue.num_reserved_decode_tokens = 0
        queue.max_total_num_tokens = 10**6
        queue._uses_swa_tail_prealloc = lambda: False
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False, output_streamer=MagicMock()
        )
        return queue

    def test_ceiling_prices_the_state_slots_a_request_takes(self):
        """BUG REGRESSION. The decode node's ceiling priced no state slots, so
        it accepted a request no empty pool can give its tokens and its state
        together: it waited in the queue forever. It now prices the slots a
        fresh request acquires."""
        storage = _storage(state=True, full_tokens=64)
        allocator = storage.allocator
        queue = self._queue(storage)
        fresh = allocator.state.fresh_request_slots(storage.requests)
        self.assertGreater(fresh, 0)
        self.assertEqual(queue._state_slots(None, DemandPurpose.CEILING), fresh)
        owner_pages = allocator.full_attn_allocator.ids.num_virtual_ids
        tight = next(
            (
                tokens
                for tokens in range(4, owner_pages * 4 + 4, 4)
                if allocator.prealloc_ceiling_fits(KVDemand(tokens, {}, 0))
                and not allocator.prealloc_ceiling_fits(KVDemand(tokens, {}, fresh))
            ),
            None,
        )
        self.assertIsNotNone(tight, "no length fits without its state but not with")
        req = SimpleNamespace(
            rid="tight",
            origin_input_ids=[0] * tight,
            output_ids=[],
            return_logprob=False,
        )
        with patch("sglang.srt.disaggregation.decode.prepare_abort") as abort:
            self.assertTrue(queue._check_if_req_exceed_kv_capacity(req))
        self.assertIn("state_slots", abort.call_args.args[1])
        req.origin_input_ids = [0] * (tight - 4)
        with patch("sglang.srt.disaggregation.decode.prepare_abort") as abort:
            self.assertFalse(queue._check_if_req_exceed_kv_capacity(req))
        abort.assert_not_called()


if __name__ == "__main__":
    unittest.main()
