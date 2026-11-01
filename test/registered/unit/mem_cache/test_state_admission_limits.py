"""FULL identity limits do not charge recurrent state as token identities."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, StateGeometry, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.token_demand import demand_windows
from sglang.test.unified_allocator_fixtures import (
    hold_band,
    setup_allocator_context,
)

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
    sized = size_plan(
        plan,
        KVSizingInputs(
            budget_bytes=64 << 20,
            max_total_tokens=512,
            max_running_requests=2,
            context_len=512,
            alignment=1,
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
        swa_evictable_size=lambda: 0,
        window_evictable_size=lambda component: 0,
        mamba_evictable_size=lambda: 0,
        supports_mamba=lambda: True,
        supports_prefix_sharing=lambda: True,
    )


class TestStateAdmissionLimits(unittest.TestCase):
    def test_pending_state_competes_for_bytes_not_full_identities(self):
        for window in (False, True):
            for lazy in (False, True):
                with self.subTest(window=window, lazy=lazy):
                    storage = _build(window=window, lazy=lazy)
                    allocator = storage.allocator
                    budget = allocator.create_prefill_budget(_cache())
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
                        allocator.admits(507, demand_windows(allocator, 507, 1))
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
            if allocator.admits(mid, {}, 1):
                low = mid
            else:
                high = mid - 1
        self.assertGreater(low, 0)
        self.assertTrue(allocator.admits(low, {}, 1))
        self.assertFalse(allocator.admits(low + 0.5, {}, 1))

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
        budget = allocator.create_prefill_budget(_cache())
        self.assertTrue(budget.state_slots_fit(12))
        self.assertGreaterEqual(budget.remaining_total, 12)
        self.assertGreaterEqual(budget.remaining_swa, 12)
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
                budget = allocator.create_prefill_budget(_cache())
                slots = allocator.state_slots_available() + 1
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
                budget = allocator.create_prefill_budget(_cache())
                budget.reserve(0, 0, state_slots=1)
                limit = int(budget.available_chunk_tokens(1 << 20))
                selected = budget.fit_chunk(
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


if __name__ == "__main__":
    unittest.main()
