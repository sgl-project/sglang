"""FULL identity limits do not charge recurrent state as token identities."""

import unittest
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.mem_cache.capacity import create_prefill_budget
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, StateGeometry, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.token_demand import demand_windows
from sglang.test.unified_allocator_fixtures import (
    hold_band,
    largest_take,
    setup_allocator_context,
)

# A request that holds no request row: no other holder of memory unless a
# test takes rows for one.
_NO_ROW_REQ = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=None))

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _build(*, window, lazy):
    setup_allocator_context()
    kv = MHAGeometry(
        head_num=1, head_dim=128, v_head_dim=128, kv_cache_dtype=torch.bfloat16
    )
    plan = plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=kv,
            window_layer_ids=(1,) if window else (),
            window_geometry=kv,
            window=16,
            state_layer_ids=(2,),
            state_geometry=StateGeometry(
                conv_state_shapes=((3, 3072),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(16, 128, 128),
                temporal_dtype=torch.float32,
            ),
            checkpoint_interval=64,
        ),
        page_size=1,
    )
    # A FULL cap (a pipeline stage's synchronized share) of 512 tokens.
    sized = msgspec.structs.replace(
        size_plan(
            plan,
            KVSizingInputs(
                budget_bytes=64 << 20,
                max_total_tokens=512,
                max_running_requests=2,
                context_len=512,
                alignment=1,
            ),
        ),
        full_cap_tokens=512,
    )
    return build_kv_storage(
        plan,
        sized,
        UnifiedBuildContext(
            device="cpu",
            start_layer=0,
            end_layer=3,
            enable_memory_saver=False,
            need_sort=False,
            lazy_compaction=lazy,
            max_num_reqs=2,
            model_context_len=512,
            state_cache_params=SimpleNamespace(layers=(2,)),
        ),
    )


def _cache():
    return SimpleNamespace(
        sliding_window_size=16,
        window_retain_size=lambda component: 16,
        full_evictable_size=lambda: 0,
        window_evictable_size=lambda component: 0,
        mamba_evictable_size=lambda: 0,
        supports_mamba=lambda: True,
        supports_prefix_sharing=lambda: True,
        window_checkpoint_interval=lambda: 0,
        transfers_in_flight=lambda: False,
    )


class TestStateAdmissionLimits(unittest.TestCase):
    def test_pending_state_competes_for_bytes_not_full_identities(self):
        for window in (False, True):
            for lazy in (True,):
                with self.subTest(window=window, lazy=lazy):
                    storage = _build(window=window, lazy=lazy)
                    allocator = storage.allocator
                    budget = create_prefill_budget(allocator, _cache())
                    self.assertEqual(allocator.full_attn_allocator.max_live_tokens, 512)
                    self.assertGreater(
                        allocator.mamba_allocator.entry_bytes,
                        512 * allocator.full_attn_allocator.entry_bytes,
                    )
                    request = dict(
                        extend_input_len=1,
                        total_tokens=3,
                        max_new_tokens=1,
                        input_tokens=1,
                        chunk_limit=None,
                        state_slots=1,
                    )
                    self.assertEqual(budget.check_prefill(**request), (True, None))
                    budget.reserve(1, 1, state_slots=1)
                    self.assertEqual(budget.total_offset, 3)
                    self.assertEqual(budget.current_offset, 2)
                    self.assertEqual(budget.check_prefill(**request), (True, None))
                    self.assertFalse(
                        budget.check_prefill(
                            **(request | {"total_tokens": 513, "max_new_tokens": 511})
                        )[0]
                    )
                    for _ in range(2):
                        self.assertIsNotNone(allocator.alloc_state_slots(1))
                        self.assertIsNotNone(allocator.alloc(3))
                    self.assertFalse(allocator.verify_byte_accounting())
                    self.assertEqual(allocator.full_attn_allocator.allocated_count(), 6)
                    self.assertFalse(
                        allocator.admits(
                            KVDemand(507, demand_windows(allocator, 507, 1))
                        )
                    )

    def test_fractional_pending_tokens_round_up_at_the_physical_boundary(self):
        allocator = _build(window=False, lazy=True).allocator
        allocator.full_attn_allocator.max_live_tokens = None
        low, high = (
            0,
            allocator.arena.total_bytes // allocator.full_attn_allocator.entry_bytes
            + 1,
        )
        while low < high:
            mid = (low + high + 1) // 2
            if allocator.admits(KVDemand(mid, {}, 1)):
                low = mid
            else:
                high = mid - 1
        self.assertGreater(low, 0)
        self.assertTrue(allocator.admits(KVDemand(low, {}, 1)))
        self.assertFalse(allocator.admits(KVDemand(low + 0.5, {}, 1)))

    def test_held_window_admission_is_its_take(self):
        from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
        from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
        from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
        from sglang.test.unified_allocator_fixtures import (
            FakeKVCache,
            FakeUnifiedSWAKVPool,
            arena_of,
            stand_in_views,
        )

        setup_allocator_context()
        specs = [
            MHASubPoolSpec(
                name=name,
                layer_num=1,
                head_num=1,
                head_dim=16,
                store_dtype=torch.bfloat16,
                grow_direction=direction,
            )
            for name, direction in (("full", "down"), ("swa", "float"))
        ]
        specs.append(
            MambaSubPoolSpec(
                name="mamba",
                layer_num=1,
                grow_direction="up",
                conv_state_shapes=((1, 32),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(0, 0, 0),
                temporal_dtype=torch.float32,
            )
        )
        arena = arena_of(specs, total_bytes=101 * 64, device="cpu")
        cache = FakeUnifiedSWAKVPool(arena)
        roles = ChainRoles(owner="full", window="swa", state="mamba")
        allocator = ChainAllocator(
            roles=roles,
            arena=arena,
            kvcache=cache,
            views=stand_in_views(roles, cache, FakeKVCache(arena, "mamba")),
            device="cpu",
            page_size=1,
            need_sort=False,
            full_max_total_num_tokens=100,
            swa_max_total_num_tokens=100,
            lazy_compaction=True,
        )
        state, owner, window = (
            allocator.mamba_allocator,
            allocator.full_attn_allocator,
            allocator.swa_attn_allocator,
        )
        self.assertIsNotNone(state.alloc(20))
        full = owner.alloc(20)
        self.assertIsNotNone(full)
        window.alloc_with_virtual(full)
        hold_band(window)
        self.assertEqual(state.available_size(), 20)
        self.assertEqual(owner.available_size(), 20)
        self.assertEqual(window.available_size(), 20)
        budget = create_prefill_budget(allocator, _cache())
        self.assertTrue(budget.state_slots_fit(12))
        # Each side alone takes 12 tokens; together they need not.
        self.assertTrue(allocator.units_fit({owner.sub_pool_name: 12}))
        self.assertTrue(allocator.units_fit({window.sub_pool_name: 12}))
        held = {p: int(window.ids.physical_to_virtual[p]) for p in window.fixed_pages()}
        fits = budget.check_prefill(
            extend_input_len=11,
            total_tokens=12,
            max_new_tokens=0,
            input_tokens=11,
            chunk_limit=None,
            state_slots=12,
        )[0]
        # The held window's pages stay; the admission answer is the takes'.
        slots = state.alloc(12)
        tokens = allocator.alloc(12) if slots is not None else None
        self.assertEqual(fits, tokens is not None)
        for page, virtual in held.items():
            self.assertEqual(int(window.ids.physical_to_virtual[page]), virtual)
        self.assertFalse(allocator.verify_byte_accounting())

    def test_pending_state_still_consumes_shared_memory(self):
        for window in (False, True):
            with self.subTest(window=window):
                storage = _build(window=window, lazy=True)
                allocator = storage.allocator
                budget = create_prefill_budget(allocator, _cache())
                slots = largest_take(allocator, allocator.sub_pool_roles()[2]) + 1
                self.assertFalse(
                    budget.check_prefill(
                        extend_input_len=1,
                        total_tokens=3,
                        max_new_tokens=1,
                        input_tokens=1,
                        chunk_limit=None,
                        state_slots=slots,
                    )[0]
                )


class TestStateChunkProgress(unittest.TestCase):
    def test_pending_state_chunk_bound_uses_physical_pressure(self):
        for window in (False, True):
            with self.subTest(window=window):
                storage = _build(window=window, lazy=True)
                allocator = storage.allocator
                allocator.full_attn_allocator.max_live_tokens = None
                budget = create_prefill_budget(allocator, _cache())
                budget.reserve(0, 0, state_slots=1)
                limit = int(budget.available_chunk_tokens(1 << 20))
                selected = budget.fit_chunk(
                    _NO_ROW_REQ,
                    extend_input_len=limit + 1,
                    max_new_tokens=0,
                    chunk_limit=limit,
                    state_slots=0,
                )
                self.assertIsNotNone(selected)
                self.assertGreater(selected, 0)
                # The pending state's bytes bound the chunk, with or without a
                # window sub-pool: one admission rule prices both.
                self.assertLess(selected, limit)
                self.assertTrue(
                    budget.can_allocate_prefill(
                        paged_input=selected + allocator.page_size,
                        extend_input_len=selected,
                        max_new_tokens=0,
                        chunk_limit=selected,
                    )
                )
                self.assertIsNotNone(allocator.alloc_state_slots(1))
                self.assertIsNotNone(allocator.alloc(selected))
                self.assertEqual(storage.accounting_violations(), [])


def _build_token_heavy(
    *,
    budget_bytes=40 << 20,
    context_len=2048,
    head_num=16,
    window=64,
    page_size=1,
):
    """A FULL/window/state pool whose KV rows, not its state slots, take most
    of the bytes, so a long cached prefix fills it."""
    setup_allocator_context()
    kv = MHAGeometry(
        head_num=head_num, head_dim=128, v_head_dim=128, kv_cache_dtype=torch.bfloat16
    )
    plan = plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=kv,
            window_layer_ids=(1,),
            window_geometry=kv,
            window=window,
            state_layer_ids=(2,),
            state_geometry=StateGeometry(
                conv_state_shapes=((3, 3072),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(16, 128, 128),
                temporal_dtype=torch.float32,
            ),
            checkpoint_interval=64,
        ),
        page_size=page_size,
    )
    sized = size_plan(
        plan,
        KVSizingInputs(
            budget_bytes=budget_bytes,
            max_running_requests=2,
            context_len=context_len,
            alignment=page_size,
        ),
    )
    return build_kv_storage(
        plan,
        sized,
        UnifiedBuildContext(
            device="cpu",
            start_layer=0,
            end_layer=3,
            enable_memory_saver=False,
            need_sort=False,
            lazy_compaction=True,
            max_num_reqs=2,
            model_context_len=context_len,
            state_cache_params=SimpleNamespace(layers=(2,)),
        ),
    )


class TestCachedStateHitMakesProgress(unittest.TestCase):
    """Regression: a lone cached-prefix hit whose held state (working copy and
    pinned checkpoint) crowds its chunk is cut to a chunk that runs, not left
    waiting on its own state forever."""

    PREFIX, CHUNK, ALIGN, NEW = 1024, 256, 64, 4

    def _admission(self, extend):
        storage = _build_token_heavy()
        allocator = storage.allocator
        window = allocator.window_pool_names()[0]
        # The cached prefix: its rows, the window behind its end, its state's
        # working copy and the checkpoint that copy came from. Nothing else
        # holds memory.
        ids = allocator.alloc(self.PREFIX)
        self.assertIsNotNone(ids)
        allocator.free_window(window, ids[: self.PREFIX - 64])
        working, checkpoint = allocator.alloc_state_slots(2).tolist()
        req = SimpleNamespace(
            kv=SimpleNamespace(
                req_pool_idx=None,
                holds_mamba=True,
                mamba_ping_pong_track_buffer=None,
                mamba_cow_src_index=torch.tensor([checkpoint]),
            )
        )
        budget = create_prefill_budget(allocator, _cache_with_window(64))
        pending = budget.request_state_slots(req)
        held = budget.request_held_state_slots(req)
        kwargs = dict(
            extend_input_len=extend,
            total_tokens=extend + self.NEW + allocator.page_size,
            max_new_tokens=self.NEW,
            input_tokens=self.PREFIX + extend,
            chunk_limit=self.CHUNK,
            state_slots=pending,
            chunk_align=self.ALIGN,
        )
        return storage, budget, held, kwargs

    def test_a_lone_cached_hit_is_cut_to_a_chunk_that_runs(self):
        stalled = []
        # Prompts the pool fits only once the request's chunk is cut, those
        # whose cut chunk also needs the cached window rows counted once.
        for extend in range(3120, 3504, 8):
            storage, budget, held, kwargs = self._admission(extend)
            self.assertEqual(held, 2)  # the working copy and the checkpoint
            # As if other work held memory: pressure that subsides waits, so
            # only the empty-pool check decides whether the chunk is cut.
            budget._waiting_frees_memory = lambda: True
            if budget.check_prefill(**kwargs) != (False, None):
                continue
            # Left out, the held state made the request wait on itself.
            stalled.append(extend)
            admitted, limit = budget.check_prefill(held_state_slots=held, **kwargs)
            self.assertTrue(admitted, extend)
            self.assertLess(limit, self.CHUNK)
            self.assertEqual(limit % self.ALIGN, 0)
            # The chunk it is cut to passes the shape check and is taken.
            self.assertTrue(
                budget.can_allocate_prefill(
                    paged_input=limit + storage.allocator.page_size,
                    extend_input_len=limit,
                    max_new_tokens=self.NEW,
                    chunk_limit=limit,
                    admitted_chunk_limit=limit,
                    cached_tokens=self.PREFIX,
                )
            )
            self.assertIsNotNone(storage.allocator.alloc(limit))
        self.assertTrue(stalled, "no configuration reproduced the stall")

    def test_nothing_freeing_memory_aborts_rather_than_waits(self):
        """A request that fits only an empty pool waits while something will give
        memory back; with nothing to wait for it is unschedulable, not queued."""
        storage, budget, held, kwargs = self._admission(3000)
        # Memory nothing will give back: no request row holds it.
        self.assertIsNotNone(storage.allocator.alloc(600))
        refused = budget.check_prefill(held_state_slots=held, **kwargs)
        self.assertEqual(refused, (False, None))
        self.assertTrue(budget.unschedulable)
        # Work admitted earlier in the pass frees memory once it ends.
        budget.reserve(64, 0)
        self.assertEqual(
            budget.check_prefill(held_state_slots=held, **kwargs), (False, None)
        )
        self.assertFalse(budget.unschedulable)
        # A forward in flight returns its freed pages once processed.
        storage, budget, held, kwargs = self._admission(3000)
        self.assertIsNotNone(storage.allocator.alloc(600))
        storage.allocator.forward_steps.forward_launched()
        storage.allocator.forward_steps.forward_recorded(object())
        self.assertEqual(
            budget.check_prefill(held_state_slots=held, **kwargs), (False, None)
        )
        self.assertFalse(budget.unschedulable)

    def test_buffer_mode_copies_count_as_memory_coming_back(self):
        """A buffer-mode host copy (storage without a host pool) in flight
        frees device entries when it ends: the cache reports it, so the idle
        backstop waits for it rather than abort."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        for idle in (True, False):
            cache = SimpleNamespace(
                ongoing_write_through={},
                ongoing_load_back={},
                ongoing_backup={},
                ongoing_prefetch={},
                buffer_pipeline=SimpleNamespace(is_idle=lambda idle=idle: idle),
            )
            self.assertIs(UnifiedRadixCache.transfers_in_flight(cache), not idle)

    def test_an_unschedulable_refusal_reaches_the_scheduler(self):
        from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

        for unschedulable, result in (
            (True, AddReqResult.UNSCHEDULABLE),
            (False, AddReqResult.NO_TOKEN),
        ):
            adder = SimpleNamespace(
                memory_budget=SimpleNamespace(unschedulable=unschedulable)
            )
            self.assertIs(PrefillAdder._refused(adder), result)


class TestCommitmentIsTheAdmittedDemand(unittest.TestCase):
    """Regression: a cached-prefix request's commitment is the demand admission
    priced, its prefix credited in each window, so later requests that fit join."""

    PAGE, WINDOW, PREFIX, EXTEND, NEW, CHUNK = 128, 1024, 8192, 4096, 2048, 1024

    def _pass(self, held_tokens, *, credit=True):
        storage = _build_token_heavy(
            budget_bytes=96 << 20,
            context_len=16384,
            head_num=1,
            window=self.WINDOW,
            page_size=self.PAGE,
        )
        allocator = storage.allocator
        window = allocator.window_pool_names()[0]
        ids = allocator.alloc(self.PREFIX)
        allocator.free_window(window, ids[: self.PREFIX - self.WINDOW])
        if held_tokens:
            self.assertIsNotNone(allocator.alloc(held_tokens))
        cache = _cache_with_window(self.WINDOW)
        cache.window_checkpoint_interval = lambda: self.PAGE
        budget = create_prefill_budget(allocator, cache)
        first = dict(
            extend_input_len=self.EXTEND,
            total_tokens=self.EXTEND + self.NEW + self.PAGE,
            max_new_tokens=self.NEW,
            input_tokens=self.PREFIX + self.EXTEND,
            chunk_limit=self.CHUNK,
        )
        admitted, limit = budget.check_prefill(**first)
        if not admitted:
            return None
        priced = budget.request_demand(
            self.EXTEND,
            self.NEW,
            chunk_limit=limit,
            extra_tokens=0,
            cached_tokens=self.PREFIX,
        )
        budget.reserve(
            self.EXTEND,
            self.NEW,
            admitted_chunk_limit=limit,
            cached_tokens=self.PREFIX if credit else 0,
        )
        second = dict(
            extend_input_len=512,
            total_tokens=512 + 16 + self.PAGE,
            max_new_tokens=16,
            input_tokens=512,
            chunk_limit=self.CHUNK,
        )
        return budget, window, priced, budget.check_prefill(**second)[0]

    def test_the_commitment_credits_the_cached_prefix(self):
        budget, window, priced, _ = self._pass(0)
        self.assertEqual(budget.window_offsets[window], priced.window_tokens[window])
        self.assertEqual(budget.total_offset, priced.full_tokens)
        self.assertLess(
            priced.window_tokens[window],
            self._pass(0, credit=False)[0].window_offsets[window],
        )

    def test_a_request_the_pool_has_room_for_joins_the_pass(self):
        # Fill the pool until the second request no longer fits beside the
        # first one's commitment; just before that, it joins the pass, which
        # the uncredited commitment would have refused.
        joined = None
        for held in range(0, 1 << 20, self.PAGE):
            outcome = self._pass(held)
            if outcome is None or not outcome[3]:
                break
            joined = held
        self.assertIsNotNone(joined)
        self.assertTrue(self._pass(joined)[3])
        self.assertFalse(self._pass(joined, credit=False)[3])


def _cache_with_window(window):
    cache = _cache()
    cache.sliding_window_size = window
    cache.window_retain_size = lambda component: window
    return cache


if __name__ == "__main__":
    unittest.main()
