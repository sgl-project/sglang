"""Every site that prices a request under unified memory (clip, admission, shape
check, commitment, load-back, decode ceiling) builds its demand the same way."""

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
    DemandPurpose,
    KVDemand,
    WindowRetirement,
    decode_estimate_tokens,
    demand_inputs,
    largest_fitting_chunk,
    window_retirement,
)
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    MHAGeometry,
    StateGeometry,
    WindowLayers,
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
    UnifiedGeometry,
    build_test_unified_pools,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

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


def _draft_window(window):
    """A fused draft whose window layers get a band of their own."""
    return FusedDraftPlacement.from_counts(
        counts={"draft_swa": [1]},
        regions={
            "draft_swa": DenseDraftRegion(
                lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16
            )
        },
        new_bands={"draft_swa": DraftBand(kind="window", window=window)},
    )


def _storage(
    *, windows=(), state=False, full_tokens=64, window_tokens=64, fused_draft=None
):
    """A storage of page 4: one FULL layer, a layer per window length (None
    for one the model does not state), with ``state`` two state layers, and
    a ``fused_draft``'s bands."""
    setup_allocator_context()
    layer = 1
    groups = []
    for window in windows:
        groups.append(WindowLayers(layer_ids=(layer,), geometry=_DENSE, window=window))
        layer += 1
    state_layers = (layer, layer + 1) if state else ()
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=_DENSE,
        windows=groups,
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
    tree.transfers_in_flight.return_value = False
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
        # of it, which it is not charged again.
        self.assertGreaterEqual(window(), 8 + 16)
        self.assertEqual(window() - window(cached_tokens=4), 4)

    def test_chunk_search_helpers(self):
        self.assertEqual(largest_fitting_chunk(lambda c: c <= 10, 16, 4), 8)
        self.assertEqual(largest_fitting_chunk(lambda c: False, 16, 4), 0)


class TestWindowRetirement(CustomTestCase):
    def test_the_scheduler_states_when_it_retires_window_rows(self):
        """Must match `ScheduleBatch.maybe_evict_swa`: without prefix sharing an
        overlapped prefill retires a chunk behind its frontier."""
        for shares, no_overlap, lag in (
            (False, False, 16),
            (True, False, 0),
            (False, True, 0),
        ):
            schedule = SimpleNamespace(
                chunked_prefill_size=16, disable_overlap_schedule=no_overlap
            )
            spec = SimpleNamespace(speculative_algorithm=None)
            with (
                self.subTest(shares=shares, no_overlap=no_overlap),
                patch("sglang.srt.runtime_context.get_schedule", return_value=schedule),
                patch("sglang.srt.runtime_context.get_spec", return_value=spec),
                envs.SGLANG_SWA_EVICTION_INTERVAL.override(32),
            ):
                self.assertEqual(
                    window_retirement(shares_prefixes=shares),
                    (lag, 0 if no_overlap else 1, 32),
                )

    def test_a_decode_step_allocates_ahead_of_its_processed_length(self):
        """Decode allocates ahead of retirement by the overlapped step, or by the
        speculative reserve, which covers the overlap lag too."""
        for algorithm, no_overlap, lag in (
            (None, False, 1),
            (None, True, 0),
            ("EAGLE", False, 10),
            ("EAGLE", True, 10),
        ):
            schedule = SimpleNamespace(
                chunked_prefill_size=16, disable_overlap_schedule=no_overlap
            )
            spec = SimpleNamespace(speculative_algorithm=algorithm)
            with (
                self.subTest(algorithm=algorithm, no_overlap=no_overlap),
                patch("sglang.srt.runtime_context.get_schedule", return_value=schedule),
                patch("sglang.srt.runtime_context.get_spec", return_value=spec),
                patch(
                    "sglang.srt.mem_cache.allocation_sizing.get_alloc_reserve_per_decode",
                    return_value=10,
                ),
            ):
                retirement = window_retirement(shares_prefixes=True)
                self.assertEqual(retirement.decode_retire_lag, lag)


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
        """BUG REGRESSION. A lone request with a locked cached prefix must be
        admitted at a smaller chunk, not wait forever on its own prefix."""
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
        self.assertEqual(req.extend_end - req.prefix_len, 8)
        self.assertTrue(self._finish(allocator, req, prefix[-8:], chunk=32))

    def test_ignore_eos_admission_selects_the_chunk_as_every_admission(self):
        """BUG REGRESSION. An ignore-EOS request selects, runs and commits its
        chunk as any admission does, instead of waiting forever when idle."""
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
                self.assertEqual(req.extend_end, 12)
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
        """BUG REGRESSION. The clip must price the smallest chunk admission may
        cut to, not reject a prompt whose configured chunk does not fit."""
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
        """The clip passes a prompt iff an empty pool admits it at the smallest
        chunk, and a lone request it passes runs to its last chunk."""
        for lag, interval in ((0, 1), (16, 8)):
            accepted = refused = 0
            with patch(
                "sglang.srt.mem_cache.prefill_budget.window_retirement",
                return_value=WindowRetirement(lag, 0, interval),
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
        """BUG REGRESSION. The clip must not grant decode beyond the token
        capacity the scheduler passes (CP padding, KV sharding)."""
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
        # chunk, the tail a branch early in the chunk keeps past it (a whole
        # chunk) and a page.
        peaks = {call.args[0].window_tokens[window] for call in admits.call_args_list}
        self.assertEqual(peaks, {8 + 16 + 16 + 4})

    def test_clip_prices_stated_windows_before_retention_binds(self):
        """BUG REGRESSION. Before retention binds, a window that states its own
        length (a draft's) is priced by it, as after binding."""
        storage = _storage(windows=(None, 8), full_tokens=128, window_tokens=128)
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
        """A draft's only window (a target of FULL and state layers) is priced by
        its own length, never the target model's window parameter."""
        only = _storage(state=True, fused_draft=_draft_window(8)).allocator
        self.assertEqual(only._ceiling_windows(63), {"draft_swa": 8})
        kwargs = dict(token_capacity=64, sliding_window_size=63, chunk_size=None)
        own = only.max_new_tokens_for_memory(16, 46, **kwargs, **RETIRE_AT_ONCE)
        with patch.object(only, "retained_windows", {"draft_swa": 63}):
            target = only.max_new_tokens_for_memory(16, 46, **kwargs, **RETIRE_AT_ONCE)
        self.assertIsNotNone(own)
        self.assertGreaterEqual(own, target or 0)


class TestHostHitChargedAlike(CustomTestCase):
    """BUG REGRESSION. Admission and the shape check charge host-hit rows alike
    (page-ceiled with the extend): admission passes nothing the check refuses."""

    def test_both_checks_agree_on_every_fill(self):
        from sglang.srt.mem_cache.capacity import create_prefill_budget

        setup_allocator_context()
        allocator = build_unified_bundle(
            UnifiedGeometry(
                chain="full",
                full_layer_ids=(0,),
                page_size=16,
                full_tokens=256,
                context_len=256,
            )
        ).token_to_kv_pool_allocator
        cache = SimpleNamespace(
            sliding_window_size=None,
            full_evictable_size=lambda: 0,
            supports_mamba=lambda: False,
            supports_prefix_sharing=lambda: True,
            window_checkpoint_interval=lambda: 0,
            transfers_in_flight=lambda: False,
        )
        page, extend, host_hit, max_new = 16, 15, 2, 4
        overhead = page
        verdicts = set()
        held = []
        while True:
            budget = create_prefill_budget(allocator, cache)
            admitted, _ = budget.check_prefill(
                extend_input_len=extend,
                total_tokens=extend + host_hit + max_new + overhead,
                max_new_tokens=max_new,
                input_tokens=extend + host_hit,
                chunk_limit=None,
                host_hit_tokens=host_hit,
            )
            fits = budget.can_allocate_prefill(
                paged_input=budget.ceil_paged_tokens(host_hit + extend) + overhead,
                extend_input_len=extend,
                max_new_tokens=max_new,
                chunk_limit=None,
                cached_tokens=host_hit,
            )
            with self.subTest(available=allocator.full_available_size()):
                self.assertEqual(admitted, fits)
            verdicts.add(admitted)
            ids = allocator.alloc(page)
            if ids is None:
                break
            held.append(ids)
        self.assertEqual(verdicts, {True, False})


class TestDecodeCeiling(CustomTestCase):
    def _queue(self, storage):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool_allocator = storage.allocator
        queue.req_to_token_pool = storage.requests
        queue.num_reserved_decode_tokens = 0
        queue.max_total_num_tokens = 10**6
        queue._uses_swa_tail_prealloc = lambda: False
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False,
            output_streamer=MagicMock(),
            request_retirer=MagicMock(),
        )
        return queue

    def test_ceiling_prices_the_state_slots_a_request_takes(self):
        """BUG REGRESSION. The decode ceiling prices a fresh request's state slots,
        so a request no empty pool can hold is refused, not queued forever."""
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
