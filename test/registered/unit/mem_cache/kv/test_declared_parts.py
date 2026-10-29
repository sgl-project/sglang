"""Declared payloads preserve identity through reset, relocation and checkpointing."""

import unittest
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    DerivedPartDecl,
    MHAGeometry,
    RequestPartDecl,
    StateGeometry,
    StateReplay,
    TensorPartDecl,
    kv_layer_decls,
    with_state_ple,
    with_state_replay,
)
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _RequestKV:
    def __init__(self):
        self.req_pool_idx = None
        self.kv_allocated_len = 1

    @property
    def holds_kv(self):
        return self.req_pool_idx is not None


def _build(
    *,
    extra=(),
    requests=(),
    derived=(),
    state=False,
    gamma=None,
    replay=None,
    ple=False,
    fused_draft=None,
    window_with_state=False,
    setup=True,
):
    if setup:
        setup_allocator_context()
    dense = MHAGeometry(
        head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
    )
    geometry = StateGeometry(
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(1, 4, 8),
        temporal_dtype=torch.float32,
        extra_parts=extra,
    )
    if replay is not None:
        geometry = with_state_replay(
            geometry, replay, layer_ids=(1, 2), speculative_num_draft_tokens=gamma
        )
    if ple:
        geometry = with_state_ple(
            geometry,
            short_conv_layer_ids=(1, 2),
            short_conv_state_shape=(3, 4),
            ngram_context_len=3,
            ngram_eos_token_id=17,
            speculative_num_draft_tokens=gamma,
        )
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=dense,
        window_layer_ids=()
        if state and not window_with_state
        else (3 if state else 1,),
        window_geometry=dense,
        window=16,
        state_layer_ids=(1, 2) if state else (),
        state_geometry=geometry if state else None,
    )
    plan = plan_kv_storage(
        decls,
        page_size=4,
        speculative_num_draft_tokens=gamma,
        request_parts=requests,
        derived_parts=derived,
        fused_draft=fused_draft,
    )
    inputs = KVSizingInputs(
        full_tokens=256,
        window_tokens=128,
        state_slots=16,
        max_running_requests=3,
        context_len=64,
        alignment=1,
    )
    sized = size_plan(plan, inputs)
    ctx = UnifiedBuildContext(
        device="cpu",
        start_layer=0,
        end_layer=(4 if window_with_state else 3) if state else 2,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=3,
        model_context_len=64,
        state_cache_params=SimpleNamespace(layers=(1, 2)),
        speculative_num_draft_tokens=gamma,
    )
    return build_kv_storage(plan, sized, ctx), plan, sized


class TestDeclaredParts(CustomTestCase):
    def test_reserved_request_backing_is_not_allocatable_capacity(self):
        """Capped concurrency must satisfy idle accounting even with larger backing."""
        from sglang.srt.mem_cache.kv.requests import RequestTable

        for prealloc in (0, 2):
            table = RequestTable(
                size=3,
                pre_alloc_size=prealloc,
                reserved_rows=12,
                max_context_len=64,
                state_sub_pool=None,
                tracking=None,
                device="cpu",
                enable_memory_saver=False,
            )
            pointer = table.req_to_token.data_ptr()
            for _ in range(2):
                self.assertEqual(table.available_size(), 3 + prealloc)
                rows = table.alloc_rows(3 + prealloc)
                self.assertEqual(set(rows), set(range(1, 4 + prealloc)))
                self.assertIsNone(table.alloc_rows(1))
                table.free_rows(rows)
                table.clear()
            self.assertEqual(table.req_to_token.shape[0], 12)
            self.assertEqual(table.req_to_token.data_ptr(), pointer)

    def test_initial_scratch_and_concurrency_fit_together(self):
        """An oversized initial request estimate must not reject a feasible pool."""
        setup_allocator_context()
        dense = MHAGeometry(
            head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
        )
        state = StateGeometry(
            conv_state_shapes=((1,),),
            conv_dtype=torch.float32,
            temporal_state_shape=(1024, 1024),
            temporal_dtype=torch.float32,
            scratch_parts=(
                TensorPartDecl(
                    name="intermediate", shape=(4 * 1024 * 1024,), dtype=torch.float32
                ),
            ),
        )
        plan = plan_kv_storage(
            kv_layer_decls(
                full_layer_ids=(0,),
                full_geometry=dense,
                state_layer_ids=(1,),
                state_geometry=state,
            ),
            page_size=4,
        )
        inp = KVSizingInputs(
            budget_bytes=512 * 1024 * 1024,
            context_len=1024,
            max_running_requests=48,
            state_share=0.1,
            alignment=4096,
        )
        sized = size_plan(plan, inp)
        smaller_estimate = size_plan(
            plan, msgspec.structs.replace(inp, max_running_requests=8)
        )
        self.assertEqual(sized, smaller_estimate)
        self.assertGreaterEqual(sized.max_running_requests, 1)
        self.assertLess(sized.max_running_requests, 8)
        self.assertEqual(sized.scratch_rows, sized.max_running_requests + 1)
        self.assertEqual(sized.request_rows, sized.scratch_rows)
        self.assertLessEqual(sized.fixed_bytes + sized.total_bytes, inp.budget_bytes)
        frozen = size_plan(
            plan,
            msgspec.structs.replace(inp, budget_bytes=400 * 1024 * 1024),
            frozen=sized,
        )
        self.assertEqual(frozen.fixed, sized.fixed)
        self.assertEqual(frozen.scratch_rows, sized.scratch_rows)
        self.assertLessEqual(frozen.max_running_requests, sized.max_running_requests)
        from sglang.srt.mem_cache.kv.state_policy import StateSlotStrategy

        strategy = StateSlotStrategy(radix_cache=False, extra_buffer=True)
        uncached = size_plan(
            plan,
            msgspec.structs.replace(
                inp,
                radix_cache=False,
                state_strategy=strategy,
            ),
        )
        self.assertGreater(uncached.max_running_requests, 0)
        self.assertGreaterEqual(
            uncached.units("mamba"),
            uncached.max_running_requests * strategy.slots_per_request(),
        )
        with self.assertRaises(RuntimeError):
            size_plan(plan, msgspec.structs.replace(inp, budget_bytes=16 * 1024 * 1024))

    def test_each_request_part_resets_only_on_fresh_acquisition(self):
        declarations = tuple(
            RequestPartDecl(part=p)
            for p in (
                TensorPartDecl(name="cursor", shape=(), dtype=torch.int32),
                TensorPartDecl(
                    name="ring", shape=(4, 8), dtype=torch.float32, layer_ids=(1, 3)
                ),
                TensorPartDecl(
                    name="context", shape=(3,), dtype=torch.int64, reset_value=7
                ),
            )
        )
        storage, _, sized = _build(requests=declarations)
        memory = storage._request_memory
        self.assertEqual(
            memory.nbytes,
            sum(t.bytes for t in sized.fixed if t.name != "request_table"),
        )
        req = SimpleNamespace(kv=_RequestKV())
        row = storage.acquire([req])[0]
        addresses = {n: t.data_ptr() for n, t in memory.tensors.items()}
        for part in memory.parts:
            tensor = memory.tensors[part.name].select(part.row_axis, row)
            self.assertTrue(torch.all(tensor == part.reset_value))
            tensor.fill_(19)
        self.assertEqual(storage.acquire([req]), [row])
        for part in memory.parts:
            self.assertTrue(
                torch.all(memory.tensors[part.name].select(part.row_axis, row) == 19)
            )
        storage.release_row(req)
        other = SimpleNamespace(kv=_RequestKV())
        self.assertEqual(storage.acquire([other]), [row])
        for part in memory.parts:
            self.assertTrue(
                torch.all(
                    memory.tensors[part.name].select(part.row_axis, row)
                    == part.reset_value
                )
            )
        self.assertEqual(
            addresses, {n: t.data_ptr() for n, t in memory.tensors.items()}
        )

    def test_total_budget_charges_fixed_storage_once(self):
        storage, plan, _ = _build(state=True, gamma=3)
        inp = KVSizingInputs(
            budget_bytes=1 << 20,
            context_len=64,
            max_running_requests=3,
            state_share=0.2,
            alignment=256,
        )
        sized = size_plan(plan, inp)
        self.assertLessEqual(sized.total_bytes + sized.fixed_bytes, inp.budget_bytes)
        self.assertLess(inp.budget_bytes - sized.total_bytes - sized.fixed_bytes, 256)
        expected_table = sized.request_rows * (64 * 4 + 4)
        self.assertEqual(sized.fixed[0].bytes, expected_table)
        self.assertEqual(
            sized.units("mamba"),
            int((sized.total_bytes - plan.sink_bytes()) * 0.2)
            // plan.sub_pool("mamba").spec.entry_bytes(),
        )
        smaller = size_plan(
            plan, msgspec.structs.replace(inp, budget_bytes=700000), frozen=sized
        )
        self.assertEqual(smaller.fixed, sized.fixed)
        self.assertEqual(smaller.request_rows, sized.request_rows)
        self.assertLess(smaller.total_bytes, sized.total_bytes)
        self.assertLessEqual(smaller.full_tokens, sized.full_tokens)
        self.assertLessEqual(smaller.max_running_requests, sized.max_running_requests)

    def test_derived_extent_is_in_the_total_budget(self):
        _, plan, _ = _build(
            derived=(
                DerivedPartDecl(
                    sub_pool="full",
                    part=TensorPartDecl(
                        name="compressed", shape=(32, 8), dtype=torch.float32
                    ),
                ),
            )
        )
        inp = KVSizingInputs(
            budget_bytes=1 << 20, context_len=64, max_running_requests=3, alignment=256
        )
        sized = size_plan(plan, inp)
        self.assertLessEqual(sized.total_bytes + sized.fixed_bytes, inp.budget_bytes)
        page_bytes = plan.owner().unit * plan.owner().spec.entry_bytes()
        blob = plan.derived_parts[0].part.row_bytes
        self.assertEqual(sized.fixed[-1].bytes, sized.total_bytes // page_bytes * blob)
        next_cost = (
            sized.total_bytes
            + 256
            + sized.fixed[0].bytes
            + (sized.total_bytes + 256) // page_bytes * blob
        )
        self.assertGreater(next_cost, inp.budget_bytes)

    def test_state_target_does_not_cap_the_state_band(self):
        storage, _, sized = _build(state=True)
        count = sized.units("mamba") + 1
        slots = storage.allocator.alloc_state_slots(count)
        self.assertIsNotNone(slots)
        self.assertEqual(slots.numel(), count)

    def test_full_cap_is_enforced_on_live_allocations(self):
        storage, _, _ = _build()
        owner = storage.allocator.full_attn_allocator
        owner.max_live_tokens = 16
        first = storage.allocator.alloc(12)
        self.assertIsNotNone(first)
        self.assertIsNone(storage.allocator.alloc(8))
        self.assertEqual(owner.allocated_count(), 12)
        self.assertIsNotNone(storage.allocator.alloc(4))
        self.assertIsNone(storage.allocator.alloc(4))

    def test_ple_views_reset_and_checkpoint_the_same_entry(self):
        storage, plan, _ = _build(state=True, gamma=3, ple=True)
        state = storage.allocator.state
        conv, ngram = state.short_conv_pool, state.ngram_pool
        self.assertTrue(conv.enabled and ngram.enabled)
        self.assertIs(conv.conv_state, state.pool.extra_views["short_conv"])
        self.assertIs(ngram.context, state.pool.extra_views["ngram"])
        ids = torch.tensor([3, 5])
        state.pool.clear_slots(ids)
        self.assertTrue(torch.all(ngram.get_context(ids) == 17))
        conv.layer_cache(2)[3].fill_(29)
        ngram.set_context(torch.tensor([3]), torch.tensor([[1, 2, 3]]))
        state.pool.copy_from(torch.tensor([3]), torch.tensor([7]))
        saved = state.pool.get_cpu_copy(torch.tensor([7]))
        state.pool.clear_slots(torch.tensor([7]))
        state.pool.load_cpu_copy(saved, torch.tensor([7]))
        self.assertTrue(torch.all(conv.layer_cache(2)[7] == 29))
        torch.testing.assert_close(
            ngram.get_context(torch.tensor([7])), torch.tensor([[1, 2, 3]])
        )
        self.assertEqual(conv.layer_intermediate_cache(2).shape, (4, 3, 3, 4))
        self.assertTrue(torch.all(ngram.intermediate_context == 17))
        self.assertEqual(len(plan.scratch), 4)

    def test_gdn_replay_request_parts_reset_on_row_reuse(self):
        storage, plan, _ = _build(
            state=True,
            gamma=3,
            replay=StateReplay(
                decode=False,
                speculative=True,
                is_kda=False,
                cache_len=8,
                key_heads=1,
            ),
        )
        view = storage.view("mamba")
        self.assertIsNone(view.mamba_cache.intermediate_ssm)
        self.assertEqual(len(plan.request_parts), 8)
        row = storage.take_rows(1)[0]
        for part in storage._request_memory.parts:
            with self.subTest(part=part.name):
                storage._request_memory.tensors[part.name].select(
                    part.row_axis, row
                ).fill_(23)
        storage.release_rows([row])
        self.assertEqual(storage.take_rows(1), [row])
        for part in storage._request_memory.parts:
            with self.subTest(part=part.name):
                self.assertTrue(
                    torch.all(
                        storage._request_memory.tensors[part.name].select(
                            part.row_axis, row
                        )
                        == part.reset_value
                    )
                )
        self.assertIs(
            view.replayssm_spec_write_pos,
            storage._request_memory.tensors["mamba.replayssm_spec_write_pos"],
        )
        self.assertEqual(view._checkpoint_extra_buffers, ())

    def test_replay_state_records_and_cursor_move_together(self):
        for speculative in (False, True):
            with self.subTest(speculative=speculative):
                storage, plan, _ = _build(
                    state=True,
                    gamma=3 if speculative else None,
                    replay=StateReplay(
                        decode=True,
                        speculative=speculative,
                        is_kda=speculative,
                        cache_len=8,
                        key_heads=1,
                    ),
                )
                view = storage.view("mamba")
                self.assertFalse(plan.request_parts)
                self.assertEqual(len(view.extra_parts), 7 if speculative else 4)
                for part in view.extra_parts:
                    view.extra_views[part.name].select(part.row_axis, 3).fill_(5)
                view.copy_from(torch.tensor([3]), torch.tensor([7]))
                saved = view.get_cpu_copy(torch.tensor([7]))
                view.clear_slots(torch.tensor([7]))
                view.load_cpu_copy(saved, torch.tensor([7]))
                for part in view.extra_parts:
                    self.assertTrue(
                        torch.all(
                            view.extra_views[part.name].select(part.row_axis, 7) == 5
                        )
                    )
                self.assertIs(
                    view.replayssm_write_pos, view.extra_views["replayssm_write_pos"]
                )

    def test_window_resume_keeps_the_handoff_boundary(self):
        storage, _, _ = _build()
        req = SimpleNamespace(kv=_RequestKV())
        row = storage.acquire([req])[0]
        tokens = storage.allocator.alloc(24)
        storage.requests.req_to_token[row, :24] = tokens.to(torch.int32)
        expected = (
            storage.allocator.translate_swa_indices_for_transfer(tokens[4:23])[::4] // 4
        )
        resume = storage.resume_units("swa", req, token_start=0, token_boundary=23)
        self.assertEqual((resume.token_start, resume.token_boundary), (4, 23))
        torch.testing.assert_close(resume.unit_ids, expected)
        resumed = storage.resume_units("swa", req, token_start=12, token_boundary=23)
        self.assertEqual((resumed.token_start, resumed.token_boundary), (12, 23))
        torch.testing.assert_close(resumed.unit_ids, expected[2:])

    def test_scratch_is_allocated_and_priced_from_the_plan(self):
        storage, plan, sized = _build(state=True, gamma=3)
        state = storage.view("mamba")
        scratch = storage.scratch
        self.assertEqual(
            scratch.nbytes,
            sum(t.bytes for t in sized.fixed if t.name != "request_table"),
        )
        self.assertEqual(set(scratch.tensors), {d.part.name for d in plan.scratch})
        self.assertIs(
            state.mamba_cache.intermediate_ssm,
            scratch.tensors["mamba.intermediate_ssm"],
        )
        self.assertEqual(state.mamba_cache.intermediate_ssm.shape, (2, 4, 3, 1, 4, 8))
        self.assertIs(
            state.mamba_cache.intermediate_conv_window[0],
            scratch.tensors["mamba.intermediate_conv_window.0"],
        )
        self.assertEqual(
            state.get_device_pool_infos()[0].buffer_info.extra_checkpoint_buffers, ()
        )

    def test_owner_only_plan_allocates_reuses_and_checkpoints_pages(self):
        setup_allocator_context()
        plan = plan_kv_storage(
            kv_layer_decls(
                full_layer_ids=(0, 1),
                full_geometry=MHAGeometry(
                    head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
                ),
            ),
            page_size=4,
        )
        sized = size_plan(
            plan,
            KVSizingInputs(
                budget_bytes=8192, max_running_requests=2, context_len=32, alignment=1
            ),
        )
        storage = build_kv_storage(
            plan,
            sized,
            UnifiedBuildContext(
                device="cpu",
                start_layer=0,
                end_layer=2,
                enable_memory_saver=False,
                need_sort=False,
                max_num_reqs=2,
                model_context_len=32,
            ),
        )
        allocator = storage.allocator
        available = allocator.available_size()
        ids = allocator.alloc(available)
        self.assertIsNotNone(ids)
        self.assertIsNone(allocator.alloc(4))
        physical = allocator.full_attn_allocator.ids.translate(ids)
        view = storage.view("full")
        view.k_buffer[0][physical] = 7
        saved = allocator.get_cpu_copy(ids)
        view.k_buffer[0][physical] = 0
        allocator.load_cpu_copy(saved, ids)
        self.assertTrue(torch.all(view.k_buffer[0][physical] == 7))
        allocator.free(ids)
        self.assertEqual(allocator.available_size(), available)
        self.assertEqual(allocator.verify_byte_accounting(), [])
        self.assertEqual(
            storage.plan.tree_slots()[0].component,
            storage.plan.owner().external.tree_component,
        )

    def test_full_cap_uses_widened_dcp_units(self):
        _, plan, _ = _build(state=True)
        sized = size_plan(
            plan,
            KVSizingInputs(
                budget_bytes=65536,
                state_slots=16,
                max_running_requests=3,
                context_len=64,
                max_total_tokens=32,
                alignment=1,
            ),
        )
        for dcp in (2, 4):
            with self.subTest(dcp=dcp), get_parallel().override(attn_dcp_size=dcp):
                storage = build_kv_storage(
                    plan,
                    sized,
                    UnifiedBuildContext(
                        device="cpu",
                        start_layer=0,
                        end_layer=3,
                        enable_memory_saver=False,
                        need_sort=False,
                        max_num_reqs=3,
                        model_context_len=64,
                        state_cache_params=SimpleNamespace(layers=(1, 2)),
                    ),
                )
                alloc = storage.allocator
                self.assertEqual(alloc.available_size(), 32 * dcp)
                first = alloc.alloc(16 * dcp)
                self.assertIsNotNone(first)
                self.assertEqual(alloc.available_size(), 16 * dcp)
                self.assertFalse(alloc.can_reserve(20 * dcp, 0))
                second = alloc.alloc(16 * dcp)
                self.assertIsNotNone(second)
                self.assertIsNone(alloc.alloc(4 * dcp))
                alloc.free(first)
                self.assertEqual(alloc.available_size(), 16 * dcp)
                alloc.free(second)
                self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_entry_parts_follow_copy_move_and_nonzero_reset(self):
        parts = (
            TensorPartDecl(name="cursor", shape=(), dtype=torch.int32),
            TensorPartDecl(
                name="ring", shape=(4, 8), dtype=torch.bfloat16, layer_ids=(1, 2)
            ),
            TensorPartDecl(name="ngram", shape=(3,), dtype=torch.int64, reset_value=17),
        )
        storage, plan, _ = _build(state=True, extra=parts)
        view = storage.view("mamba")
        for p in parts:
            view.extra_views[p.name].select(p.row_axis, 3).fill_(23)
        view.copy_from(torch.tensor([3]), torch.tensor([5]))
        view._arena.move_units("mamba", torch.tensor([7]), torch.tensor([5]))
        for p in parts:
            self.assertTrue(
                torch.all(view.extra_views[p.name].select(p.row_axis, 7) == 23)
            )
        saved = view.get_cpu_copy(torch.tensor([7]))
        view.clear_slots(torch.tensor([7]))
        for p in parts:
            self.assertTrue(
                torch.all(
                    view.extra_views[p.name].select(p.row_axis, 7) == p.reset_value
                )
            )
        view.load_cpu_copy(saved, torch.tensor([7]))
        for p in parts:
            self.assertTrue(
                torch.all(view.extra_views[p.name].select(p.row_axis, 7) == 23)
            )
        info = view.get_device_pool_infos()[0].buffer_info
        self.assertEqual(len(info.extra_checkpoint_buffers), len(parts))
        info.validate(layer_ids=tuple(view.mamba_layer_ids))
        for part, tensor in zip(parts, info.extra_checkpoint_buffers):
            self.assertEqual(
                tensor.stride(0) * tensor.itemsize,
                plan.sub_pool("mamba").spec.entry_bytes(),
            )

    def test_draft_windows_allocate_and_retire_independently(self):
        from unittest.mock import patch

        from sglang.srt.mem_cache.layout.fused_draft import (
            DraftKVGeometry,
            DraftKVProfile,
            DraftLayerSet,
            PlacementContext,
            place_fused_draft,
        )

        for target_state in (False, True):
            for draft_window in (8, 32):
                with self.subTest(state=target_state, draft_window=draft_window):
                    placement = place_fused_draft(
                        profile=DraftKVProfile(
                            num_layers=1,
                            kinds={
                                "window": DraftLayerSet(
                                    layer_ids=(0,),
                                    window=draft_window,
                                    geometry=DraftKVGeometry(
                                        head_num=1, head_dim=3, v_head_dim=5
                                    ),
                                )
                            },
                        ),
                        num_runners=1,
                        ctx=PlacementContext(
                            host_names=("full", "swa", "mamba")
                            if target_state
                            else ("full", "swa"),
                            target_window=16,
                        ),
                        store_dtype=torch.bfloat16,
                    ).placement
                    self.assertEqual(set(placement.new_bands), {"draft_swa"})
                    storage, plan, _ = _build(
                        state=target_state,
                        window_with_state=True,
                        fused_draft=placement,
                    )
                    allocator = storage.allocator
                    self.assertEqual(len(allocator.bands), 4 if target_state else 3)
                    ids = allocator.alloc(8)
                    self.assertIsNotNone(ids)
                    draft = allocator.window_allocators["draft_swa"]
                    physical = draft.ids.translate(ids).long()
                    k, v = plan.sub_pool("draft_swa").spec.draft_views(
                        storage.allocator.arena
                    )
                    k[0][physical] = 11
                    v[0][physical] = 13
                    allocator.free_swa_segment(ids[:4], start_pos=0)
                    self.assertTrue(
                        torch.all(draft.ids.translate(ids).long() == physical)
                    )
                    torch.testing.assert_close(
                        k[0][physical], torch.full_like(k[0][physical], 11)
                    )
                    # Failure in the last physical reservation leaves every id
                    # and live page untouched, including the earlier window.
                    before = [
                        (b, b.allocated_count(), b.ids.virtual_to_physical.clone())
                        for b in allocator.bands
                    ]
                    free_ids = allocator.full_attn_allocator.free_virtual_ids.clone()
                    with patch.object(
                        allocator._windows[-1], "take_physical", return_value=None
                    ):
                        self.assertIsNone(allocator.alloc(4))
                    for band, count, mapping in before:
                        self.assertEqual(band.allocated_count(), count)
                        torch.testing.assert_close(
                            band.ids.virtual_to_physical, mapping
                        )
                    torch.testing.assert_close(
                        allocator.full_attn_allocator.free_virtual_ids, free_ids
                    )
                    allocator.free(ids)
                    self.assertEqual(allocator.verify_byte_accounting(), [])
                    self.assertTrue(
                        all(b.allocated_count() == 0 for b in allocator.bands)
                    )

    def test_request_cleanup_releases_independently_retained_windows(self):
        """Returning owner ids must not strand a longer window's physical pages."""
        from sglang.srt.mem_cache.common import free_kv_row_segments
        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            DraftBand,
            FusedDraftPlacement,
        )

        for window in (8, 32):
            placement = FusedDraftPlacement.from_counts(
                counts={"draft_swa": [1]},
                regions={
                    "draft_swa": DenseDraftRegion(
                        lane_num=1,
                        head_num=1,
                        head_dim=8,
                        store_dtype=torch.bfloat16,
                    )
                },
                new_bands={"draft_swa": DraftBand(kind="window", window=window)},
            )
            for grouped in (False, True):
                for direct in (False, True):
                    with self.subTest(window=window, grouped=grouped, direct=direct):
                        storage, _, _ = _build(fused_draft=placement)
                        a = storage.allocator
                        ids = a.alloc(32)
                        self.assertIsNotNone(ids)
                        a.free_swa_segment(ids[:16], start_pos=0)
                        if window < 16:
                            a.free_window_segment("draft_swa", ids[:24], start_pos=0)
                        if grouped:
                            a.free_group_begin()
                        if direct:
                            a.free_full(ids[:16])
                            a.free(ids[16:])
                        else:
                            free_kv_row_segments(a, [(ids, 0)], swa_evicted_seqlen=16)
                        if grouped:
                            a.free_group_end()
                        self.assertEqual(
                            {b.sub_pool_name: b.allocated_count() for b in a.bands},
                            {b.sub_pool_name: 0 for b in a.bands},
                        )
                        self.assertEqual(storage.accounting_violations(), [])
                        reused = a.alloc(32)
                        self.assertIsNotNone(reused)
                        for band in a._windows:
                            self.assertTrue(torch.all(band.ids.translate(reused) > 0))
                        a.free(reused)
                        self.assertEqual(storage.accounting_violations(), [])

    def test_multiwindow_admission_reclaims_cached_prefixes(self):
        """Evictable prefixes must not leave an idle allocator refusing new work."""
        from array import array

        from sglang.srt.mem_cache.base_prefix_cache import (
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            DraftBand,
            FusedDraftPlacement,
        )
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        placement = FusedDraftPlacement.from_counts(
            counts={"draft_swa": [1]},
            regions={
                "draft_swa": DenseDraftRegion(
                    lane_num=1,
                    head_num=1,
                    head_dim=8,
                    store_dtype=torch.bfloat16,
                )
            },
            new_bands={"draft_swa": DraftBand(kind="window", window=32)},
        )
        storage, _, _ = _build(fused_draft=placement)
        a = storage.allocator
        empty_demands = [n for n in range(8, 128, 4) if a.can_reserve(n, n)]
        ids = []
        while a.can_reserve(4, 4):
            allocation = a.alloc(4)
            self.assertIsNotNone(allocation)
            ids.append(allocation)
        ids = torch.cat(ids)
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=storage.requests,
                token_to_kv_pool_allocator=a,
                page_size=4,
                sliding_window_size=16,
                tree_slots=storage.tree_slots(),
                tree_core_backend="python",
            )
        )
        key = RadixKey(array("q", range(ids.numel())))
        cache.insert(InsertParams(key=key, value=ids.clone()))
        need = next(n for n in empty_demands if not a.can_reserve(n, n))
        budget = a.create_prefill_budget(cache)
        before = a.pricing_snapshot()
        self.assertTrue(budget._fits(need, need, empty_pool=True))
        self.assertTrue(budget._fits(need, need))
        self.assertEqual(a.pricing_snapshot(), before)
        node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        receipt = cache.inc_lock_ref(node)
        try:
            self.assertFalse(budget._fits(need, need))
            self.assertTrue(budget._fits(need, need, empty_pool=True))
        finally:
            cache.dec_lock_ref(node, receipt.to_dec_params())
        self.assertEqual(
            budget.check_prefill(
                extend_input_len=need - 4,
                total_tokens=need,
                max_new_tokens=0,
                input_tokens=need - 4,
                swa_host_hit_length=0,
                chunk_limit=None,
            ),
            (True, None),
        )
        self.assertIsNone(a.reclaim_for_prealloc(cache, need, need))
        fresh = a.alloc(need)
        self.assertIsNotNone(fresh)
        a.free(fresh)
        self.assertEqual(storage.accounting_violations(), [])

    def test_state_only_draft_owns_a_planned_band(self):
        from sglang.srt.mem_cache.layout.fused_draft import (
            DraftKVProfile,
            DraftLayerSet,
            DraftStateGeometry,
            PlacementContext,
            place_fused_draft,
        )

        placement = place_fused_draft(
            profile=DraftKVProfile(
                num_layers=1,
                kinds={
                    "state": DraftLayerSet(
                        layer_ids=(0,),
                        geometry=DraftStateGeometry(
                            conv_state_shapes=((1, 3),),
                            conv_dtype=torch.bfloat16,
                            temporal_state_shape=(3,),
                            temporal_dtype=torch.float32,
                        ),
                    )
                },
            ),
            num_runners=1,
            ctx=PlacementContext(host_names=("full", "swa"), target_window=16),
            store_dtype=torch.bfloat16,
        ).placement
        storage, plan, _ = _build(fused_draft=placement)
        spec = plan.sub_pool("mamba").spec
        self.assertEqual(spec.host_entry_bytes(), 0)
        slots = storage.allocator.alloc_state_slots(2)
        physical = storage.allocator.mamba_allocator.ids.translate(slots).long()
        conv, temporal = spec.draft_views(storage.allocator.arena)
        conv[0][:, physical[0]] = 7
        temporal[:, physical[0]] = 9
        storage.allocator.state.pool.copy_from(physical[:1], physical[1:])
        torch.testing.assert_close(temporal[:, physical[0]], temporal[:, physical[1]])
        storage.allocator.mamba_allocator.free(slots)
        self.assertEqual(storage.accounting_violations(), [])

    def test_two_window_tree_restores_only_the_evicted_component(self):
        from array import array

        from sglang.srt.mem_cache.base_prefix_cache import (
            EvictParams,
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            DraftBand,
            FusedDraftPlacement,
        )
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        placement = FusedDraftPlacement.from_counts(
            counts={"draft_swa": [1]},
            regions={
                "draft_swa": DenseDraftRegion(
                    lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16
                )
            },
            new_bands={"draft_swa": DraftBand(kind="window", window=32)},
        )
        storage, _, _ = _build(fused_draft=placement)
        alloc = storage.allocator
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=storage.requests,
                token_to_kv_pool_allocator=alloc,
                page_size=4,
                sliding_window_size=16,
                tree_slots=storage.tree_slots(),
                tree_core_backend="python",
            )
        )
        key = RadixKey(array("q", range(32)))
        ids = alloc.alloc(32)
        cache.insert(InsertParams(key=key, value=ids.clone()))
        cache.evict(EvictParams(auxiliary_swa_num_tokens=4))
        old_full = alloc.full_attn_allocator.ids.translate(ids).clone()
        incoming = alloc.alloc(32)
        self.assertIsNotNone(incoming)
        cache.insert(InsertParams(key=key, value=incoming.clone()))
        torch.testing.assert_close(
            alloc.full_attn_allocator.ids.translate(ids), old_full
        )
        matched = cache.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(matched.device_prefix_len, 32)
        self.assertEqual(storage.accounting_violations(), [])
        cache.evict(
            EvictParams(num_tokens=64, swa_num_tokens=64, auxiliary_swa_num_tokens=64)
        )
        self.assertEqual(
            {b.sub_pool_name: b.allocated_count() for b in alloc.bands},
            {b.sub_pool_name: 0 for b in alloc.bands},
        )

    def test_each_window_prices_its_own_minimum_retention(self):
        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            DraftBand,
            FusedDraftPlacement,
        )

        for window in (8, 32):
            placement = FusedDraftPlacement.from_counts(
                counts={"draft_swa": [1]},
                regions={
                    "draft_swa": DenseDraftRegion(
                        lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16
                    )
                },
                new_bands={"draft_swa": DraftBand(kind="window", window=window)},
            )
            _, plan, _ = _build(fused_draft=placement)
            sized = size_plan(
                plan,
                KVSizingInputs(
                    full_tokens=256,
                    window_tokens=128,
                    context_len=64,
                    floor_window=16,
                    alignment=1,
                ),
            )
            floors = dict(sized.bs1_floor_terms)
            self.assertEqual(
                floors["swa_window_kv"], 20 * plan.sub_pool("swa").spec.entry_bytes()
            )
            self.assertEqual(
                floors["draft_swa_window_kv"],
                (window + 4) * plan.sub_pool("draft_swa").spec.entry_bytes(),
            )

    def test_storage_namespace_distinguishes_draft_weights_and_layout(self):
        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            FusedDraftPlacement,
        )

        region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16
        )
        placement = FusedDraftPlacement.from_counts(
            counts={"full": [1]}, regions={"full": region}
        )
        tags = []
        for source, width in (("draft-a", 8), ("draft-b", 8), ("draft-a", 16)):
            candidate = msgspec.structs.replace(
                placement,
                source=(source, "revision"),
                regions={"full": msgspec.structs.replace(region, head_dim=width)},
            )
            _, plan, _ = _build(fused_draft=candidate)
            tags.append(plan.storage_format_tag())
        self.assertEqual(len(set(tags)), 3)

    def test_file_storage_roundtrip_preserves_fused_page_parts(self):
        from tempfile import TemporaryDirectory

        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
            PoolName,
            PoolTransfer,
        )
        from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions
        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            FusedDraftPlacement,
        )

        region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=3, v_head_dim=5, store_dtype=torch.bfloat16
        )
        placement = FusedDraftPlacement.from_counts(
            counts={"full": [1], "swa": [1]},
            regions={"full": region, "swa": region},
        )
        storage, _, _ = _build(fused_draft=placement)
        with TemporaryDirectory() as directory:
            backend = HiCacheFile(
                HiCacheStorageConfig(
                    tp_rank=0,
                    tp_size=1,
                    pp_rank=0,
                    pp_size=1,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    is_mla_model=False,
                    enable_storage_metrics=False,
                    is_page_first_layout=False,
                    model_name="fused-entry-test",
                    extra_config={"file_storage_path": directory},
                )
            )
            for name, wire in (("full", PoolName.KV), ("swa", PoolName.SWA)):
                host = storage.host_layout(name).build(
                    HostPoolOptions(
                        ratio=1,
                        size=0,
                        page_size=4,
                        layout="layer_first",
                        allocator_type="default",
                        pin_memory=False,
                        pool_label=name,
                    )
                )
                try:
                    ids = host.alloc(4)
                    page = torch.arange(host.item_bytes, dtype=torch.int64).to(
                        torch.uint8
                    )
                    host.set_from_flat_data_page(int(ids[0]), page)
                    backend.register_mem_host_pool_v2(host, wire)
                    transfer = PoolTransfer(
                        name=wire, host_indices=ids, keys=["prefix"]
                    )
                    self.assertEqual(backend.batch_set_v2([transfer])[wire], [True])
                    host.set_from_flat_data_page(int(ids[0]), torch.zeros_like(page))
                    self.assertEqual(backend.batch_get_v2([transfer])[wire], [True])
                    torch.testing.assert_close(host.get_data_page(int(ids[0])), page)
                    host.free(ids)
                finally:
                    host.destroy()

    def test_dcp_draft_window_keeps_all_tokens_and_releases_its_band(self):
        from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
        from sglang.srt.mem_cache.layout.fused_draft import (
            DraftKVGeometry,
            DraftKVProfile,
            DraftLayerSet,
            PlacementContext,
            place_fused_draft,
        )
        from sglang.srt.mem_cache.unified_draft_pool import routed_draft_pool

        setup_allocator_context()
        for factor in (2, 4):
            with (
                self.subTest(dcp=factor),
                get_parallel().override(attn_dcp_size=factor),
            ):
                placement = place_fused_draft(
                    profile=DraftKVProfile(
                        num_layers=1,
                        kinds={
                            "window": DraftLayerSet(
                                layer_ids=(0,),
                                window=16,
                                geometry=DraftKVGeometry(
                                    head_num=1, head_dim=3, v_head_dim=5
                                ),
                            ),
                        },
                    ),
                    num_runners=1,
                    ctx=PlacementContext(host_names=("full", "mamba"), dcp_size=factor),
                    store_dtype=torch.bfloat16,
                ).placement
                storage, plan, _ = _build(
                    state=True, fused_draft=placement, setup=False
                )
                alloc = storage.allocator
                ids = alloc.alloc(8 * factor)
                self.assertEqual(alloc.page_size, 4 * factor)
                pool = routed_draft_pool(
                    arena=alloc.arena,
                    plan=plan,
                    owner="full",
                    page_size=4,
                    lanes_by_host={"swa": {0: 0}},
                )
                runner = RunnerIds(
                    req_to_token=storage.requests.req_to_token,
                    token_to_kv_pool_allocator=alloc,
                    token_to_kv_pool=pool,
                    page_size=4,
                    device="cpu",
                    is_draft_worker=True,
                )
                self.assertEqual(
                    runner.capture_token_capacity(0),
                    alloc.full_attn_allocator.ids.virtual_to_physical.numel()
                    * alloc.page_size,
                )
                owner_mapping = runner.space("full").read_mapping()
                self.assertIsNotNone(owner_mapping)
                owner_physical = runner.space("full").ids(ids)
                torch.testing.assert_close(
                    owner_mapping[ids[::4] // 4], owner_physical[::4] // 4
                )
                space = runner.space("swa")
                physical = space.ids(ids).long()
                self.assertTrue(torch.all(physical > 0))
                self.assertEqual(physical.unique().numel(), ids.numel())
                mapping = space.read_mapping()
                torch.testing.assert_close(mapping[ids[::4] // 4], physical[::4] // 4)
                from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions

                for name in ("full", "swa"):
                    layout = storage.host_layout(name)
                    self.assertEqual(layout.page_size, alloc.page_size)
                    host = layout.build(
                        HostPoolOptions(
                            ratio=1,
                            size=0,
                            page_size=alloc.page_size,
                            layout="layer_first",
                            allocator_type="default",
                            pool_label=name,
                            pin_memory=False,
                        )
                    )
                    host_ids = host.alloc(ids.numel())
                    transfer_ids = storage.host_ids(name)
                    device_ids = (
                        transfer_ids(ids)
                        if transfer_ids is not None
                        else storage.window_ids(name, ids)
                    )
                    page_ids = device_ids[:: alloc.page_size] // alloc.page_size
                    buffer = layout.get_page_envelope_buffer()
                    buffer[page_ids] = 37
                    expected = buffer[page_ids].clone()
                    host.backup_from_device_all_layer(
                        layout, host_ids, device_ids, "kernel"
                    )
                    buffer[page_ids] = 0
                    host.load_to_device_per_layer(
                        layout, host_ids, device_ids, 0, "kernel"
                    )
                    torch.testing.assert_close(buffer[page_ids], expected)
                    host.free(host_ids)
                alloc.free(ids)
                self.assertEqual(alloc.verify_byte_accounting(), [])
                self.assertTrue(all(b.allocated_count() == 0 for b in alloc.bands))

    def test_latent_draft_parts_preserve_target_rows(self):
        from sglang.srt.mem_cache.kv.layout.mla import DraftMLAView
        from sglang.srt.mem_cache.layout.fused_draft import (
            DraftKVProfile,
            DraftLayerSet,
            DraftMLAGeometry,
            PlacementContext,
            place_fused_draft,
        )
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        placement = place_fused_draft(
            profile=DraftKVProfile(
                num_layers=2,
                kinds={
                    "full": DraftLayerSet(
                        layer_ids=(0, 1),
                        geometry=DraftMLAGeometry(
                            kv_lora_rank=3,
                            qk_rope_head_dim=5,
                        ),
                    ),
                },
            ),
            num_runners=1,
            ctx=PlacementContext(host_names=("full", "swa")),
            store_dtype=torch.bfloat16,
        ).placement
        storage, plan, _ = _build(fused_draft=placement)
        alloc = storage.allocator
        ids = alloc.alloc(8)
        loc = alloc.full_attn_allocator.ids.translate(ids).long()
        host = alloc.views["full"]
        host.k_buffer[0].fill_(17)
        host.v_buffer[0].fill_(19)
        view = DraftMLAView(
            arena=alloc.arena,
            host_spec=plan.owner().spec,
            layer_lanes={0: 0, 1: 1},
            page_size=4,
        )
        self.assertEqual(len(plan.owner().spec.draft_region.parts(0)), 1)
        for layer in (0, 1):
            data = torch.arange(64, dtype=torch.bfloat16).reshape(8, 1, 8) + layer
            view.set_kv_buffer(
                SimpleNamespace(layer_id=layer),
                KVWriteLoc(loc, physical=True),
                data,
                None,
            )
            torch.testing.assert_close(view.get_key_buffer(layer)[loc], data)
            torch.testing.assert_close(view.get_value_buffer(layer)[loc], data[..., :3])
        self.assertTrue(torch.all(host.k_buffer[0] == 17))
        self.assertTrue(torch.all(host.v_buffer[0] == 19))
        alloc.free(ids)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_replicated_draft_uses_all_dcp_tokens_and_its_own_coordinates(self):
        from sglang.srt.mem_cache.kv.alloc.iteration_ids import IterationIds
        from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
        from sglang.srt.mem_cache.kv.layout.mha import DraftMHAView
        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            FusedDraftPlacement,
        )

        setup_allocator_context()
        for factor in (2, 4):
            for rank in range(factor):
                with (
                    self.subTest(factor=factor, rank=rank),
                    get_parallel().override(attn_dcp_size=factor, attn_dcp_rank=rank),
                ):
                    placement = FusedDraftPlacement.from_counts(
                        counts={"full": [1]},
                        regions={
                            "full": DenseDraftRegion(
                                lane_num=1,
                                head_num=1,
                                head_dim=3,
                                v_head_dim=5,
                                store_dtype=torch.bfloat16,
                                token_replicas=factor,
                            )
                        },
                    )
                    plan = plan_kv_storage(
                        kv_layer_decls(
                            full_layer_ids=(0, 1, 2, 3),
                            full_geometry=MHAGeometry(
                                head_num=1,
                                head_dim=8,
                                v_head_dim=8,
                                kv_cache_dtype=torch.bfloat16,
                            ),
                        ),
                        page_size=4,
                        fused_draft=placement,
                    )
                    sized = size_plan(
                        plan,
                        KVSizingInputs(
                            budget_bytes=65536,
                            max_running_requests=2,
                            context_len=64,
                            alignment=1,
                        ),
                    )
                    storage = build_kv_storage(
                        plan,
                        sized,
                        UnifiedBuildContext(
                            device="cpu",
                            start_layer=0,
                            end_layer=4,
                            enable_memory_saver=False,
                            need_sort=False,
                            max_num_reqs=2,
                            model_context_len=64,
                        ),
                    )
                    alloc = storage.allocator
                    virtual = alloc.alloc(8 * factor)
                    table = storage.requests.req_to_token
                    table[1, : len(virtual)] = virtual.int()
                    target = RunnerIds(
                        req_to_token=table,
                        token_to_kv_pool_allocator=alloc,
                        token_to_kv_pool=alloc.get_kvcache(),
                        page_size=4,
                        device="cpu",
                    )
                    draft_pool = DraftMHAView(
                        arena=alloc.arena,
                        host_spec=plan.owner().spec,
                        layer_lanes={0: 0},
                        page_size=4,
                    )
                    draft = RunnerIds(
                        req_to_token=table,
                        token_to_kv_pool_allocator=alloc,
                        token_to_kv_pool=draft_pool,
                        page_size=4,
                        device="cpu",
                        is_draft_worker=True,
                    )
                    iteration = IterationIds(
                        source=target,
                        req_pool_indices=torch.tensor([1]),
                        seq_lens=torch.tensor([len(virtual)]),
                        seq_lens_cpu=torch.tensor([len(virtual)]),
                        write_virtual=virtual,
                    )
                    physical = iteration.write_ids(draft).long()
                    self.assertEqual(physical.unique().numel(), virtual.numel())
                    self.assertTrue(torch.all(physical > 0))
                    self.assertTrue(iteration.is_read_by(draft))
                    self.assertNotEqual(
                        target.space("full").key, draft.space("full").key
                    )
                    target_view = storage.view("full")
                    for k in target_view.k_buffer:
                        k.fill_(17)
                    draft_pool.k_buffer[0][physical] = torch.arange(
                        len(physical), dtype=torch.bfloat16
                    )[:, None, None]
                    for k in target_view.k_buffer:
                        self.assertTrue(torch.all(k == 17))
                    reads = iteration.read_table(reader=draft)
                    self.assertEqual(
                        reads.ids[0].tolist(), (physical[::4] // 4).tolist()
                    )
                    self.assertEqual(
                        (iteration.write_ids(target) > 0).sum().item(),
                        len(virtual) // factor,
                    )
                    mapping_ptr = draft.space("full").table.data_ptr()
                    alloc.free(virtual[: 4 * factor])
                    refreshed = draft.space("full").read_mapping()
                    self.assertEqual(refreshed.data_ptr(), mapping_ptr)
                    self.assertTrue(
                        torch.all(refreshed[virtual[: 4 * factor : 4] // 4] == 0)
                    )
                    self.assertEqual(storage.accounting_violations(), [])

    def test_host_roundtrip_preserves_every_fused_part(self):
        from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions
        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            DraftBand,
            FusedDraftPlacement,
        )

        region = DenseDraftRegion(
            lane_num=2, head_num=1, head_dim=3, v_head_dim=5, store_dtype=torch.bfloat16
        )
        placement = FusedDraftPlacement.from_counts(
            counts={"full": [2], "draft_swa": [2]},
            regions={"full": region, "draft_swa": region},
            new_bands={"draft_swa": DraftBand(kind="window", window=32)},
        )
        storage, plan, _ = _build(fused_draft=placement)
        allocator = storage.allocator
        ids = allocator.alloc(8)
        try:
            for name in ("full", "draft_swa"):
                band = next(b for b in allocator.bands if b.sub_pool_name == name)
                physical = band.ids.translate(ids).long()
                entries = allocator.arena.entries(name)
                entries[physical] = 37 if name == "full" else 93
                expected = entries[physical].clone()
                host = storage.host_layout(name).build(
                    HostPoolOptions(
                        ratio=1,
                        size=0,
                        page_size=4,
                        layout="page_first_direct",
                        allocator_type="default",
                        pool_label=name,
                        pin_memory=False,
                    )
                )
                try:
                    host_ids = host.alloc(8)
                    host.backup_from_device_all_layer(
                        storage.view(name), host_ids, physical, "direct"
                    )
                    entries[physical] = 0
                    host.load_to_device_per_layer(
                        storage.view(name), host_ids, physical, 0, "direct"
                    )
                    torch.testing.assert_close(entries[physical], expected)
                    host.free(host_ids)
                finally:
                    host.destroy()
        finally:
            allocator.free(ids)

    def test_runtime_budget_adapter_returns_the_declared_plan_sizes(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        _, plan, _ = _build(state=True)
        cfg = SimpleNamespace(
            _unified_plan=plan,
            _unified_sized=None,
            _unified_window_and_state=lambda: (False, True),
            _unified_request_rows=lambda **kwargs: (4, 0),
            _apply_token_constraints=lambda tokens: tokens,
            model_config=SimpleNamespace(context_len=64, sliding_window_size=None),
            attn_dp_size=1,
        )
        from sglang.srt.runtime_context import get_schedule

        with get_schedule().override(max_running_requests=3):
            config = KVCacheConfigurator._unified_config_from_budget(cfg, 1 << 20)
        self.assertEqual(config.max_total_num_tokens, cfg._unified_sized.full_tokens)
        self.assertEqual(config.unified_total_bytes, cfg._unified_sized.total_bytes)
        self.assertGreater(config.max_running_requests, 0)

    def test_state_slot_strategy_sizes_floor_and_acquisition_together(self):
        from sglang.srt.mem_cache.kv.state_policy import StateSlotStrategy

        _, plan, _ = _build(state=True)
        cases = (
            (StateSlotStrategy(), 3, 0),
            (StateSlotStrategy(extra_buffer=True, overlap=True), 5, 2),
            (
                StateSlotStrategy(
                    extra_buffer=True, overlap=True, lazy=True, skip_decode_lock=True
                ),
                4,
                1,
            ),
            (StateSlotStrategy(extra_buffer=True, decode_only=True), 2, 1),
            (StateSlotStrategy(radix_cache=False), 1, 0),
        )
        for strategy, slots, initial in cases:
            with self.subTest(strategy=strategy):
                sized = size_plan(
                    plan,
                    KVSizingInputs(
                        full_tokens=256,
                        state_slots=10,
                        context_len=64,
                        max_running_requests=8,
                        alignment=1,
                        state_strategy=strategy,
                        radix_cache=strategy.radix_cache,
                    ),
                )
                self.assertEqual(sized.max_running_requests, min(8, 10 // slots))
                self.assertEqual(
                    dict(sized.bs1_floor_terms)["mamba_bs1_state_slots"],
                    slots * plan.sub_pool("mamba").spec.entry_bytes(),
                )
                storage = build_kv_storage(
                    plan,
                    sized,
                    UnifiedBuildContext(
                        device="cpu",
                        start_layer=0,
                        end_layer=3,
                        enable_memory_saver=False,
                        need_sort=False,
                        max_num_reqs=8,
                        model_context_len=64,
                        state_cache_params=SimpleNamespace(layers=(1, 2)),
                    ),
                )
                self.assertEqual(storage.requests.mamba_initial_tracking_slots, initial)
                self.assertEqual(
                    storage.requests.mamba_ping_pong_track_buffer_size,
                    strategy.tracking_slots,
                )
                self.assertEqual(storage.requests.size, sized.max_running_requests)

    def test_retraction_carries_whole_entries_derived_pages_and_request_parts(self):
        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            DraftStateGeometry,
            DraftStateRegion,
            FusedDraftPlacement,
        )

        for with_state in (False, True):
            with self.subTest(state=with_state):
                regions = {
                    "full": DenseDraftRegion(
                        lane_num=2,
                        head_num=1,
                        head_dim=3,
                        v_head_dim=5,
                        store_dtype=torch.bfloat16,
                    )
                }
                if with_state:
                    regions["mamba"] = DraftStateRegion(
                        lane_num=1,
                        state=DraftStateGeometry(
                            conv_state_shapes=((1, 3),),
                            conv_dtype=torch.bfloat16,
                            temporal_state_shape=(3,),
                            temporal_dtype=torch.float32,
                        ),
                    )
                placement = FusedDraftPlacement.from_counts(
                    counts={
                        name: [region.lane_num] for name, region in regions.items()
                    },
                    regions=regions,
                )
                storage, _, _ = _build(
                    state=with_state,
                    fused_draft=placement,
                    derived=(
                        DerivedPartDecl(
                            sub_pool="full",
                            part=TensorPartDecl(
                                name="index",
                                shape=(528,),
                                dtype=torch.uint8,
                                layer_ids=(0,),
                            ),
                        ),
                    ),
                    requests=(
                        RequestPartDecl(
                            part=TensorPartDecl(
                                name="pending", shape=(3,), dtype=torch.int32
                            )
                        ),
                    ),
                )
                allocator = storage.allocator
                source = allocator.alloc(16)
                state_ids = allocator.alloc_state_slots(1) if with_state else None
                (request_row,) = storage.take_rows(1)
                storage.request_memory.tensors["pending"][request_row].fill_(97)
                original = {}
                for n, band in enumerate(allocator.members):
                    ids = (
                        state_ids
                        if band.sub_pool_name == allocator.roles.state
                        else source
                    )
                    physical = band.ids.translate(ids.to(torch.int64))
                    value = n + 17
                    allocator.arena.entries(band.sub_pool_name)[physical] = value
                    original[band.sub_pool_name] = value
                blob = allocator.arena.page_parts["full"].tensors["index"]
                src_pages = allocator.full_attn_allocator.ids.virtual_to_physical[
                    source[::4] // 4
                ]
                blob[:, src_pages] = 83
                saved = allocator.get_cpu_copy(
                    source, mamba_indices=state_ids, req_pool_index=request_row
                )
                # A distinct allocation changes both virtual and physical rows.
                destination = allocator.alloc(16)
                new_state = allocator.alloc_state_slots(1) if with_state else None
                (new_row,) = storage.take_rows(1)
                if not with_state:
                    allocator.free_swa_segment(destination[:4], start_pos=0)
                allocator.load_cpu_copy(
                    saved, destination, mamba_indices=new_state, req_pool_index=new_row
                )
                for band in allocator.members:
                    ids = (
                        new_state
                        if band.sub_pool_name == allocator.roles.state
                        else destination
                    )
                    physical = band.ids.translate(ids.to(torch.int64))
                    physical = physical[physical > 0]
                    self.assertTrue(
                        torch.all(
                            allocator.arena.entries(band.sub_pool_name)[physical]
                            == original[band.sub_pool_name]
                        )
                    )
                pages = allocator.full_attn_allocator.ids.virtual_to_physical[
                    destination[::4] // 4
                ]
                self.assertTrue(torch.all(blob[:, pages] == 83))
                self.assertTrue(
                    torch.all(storage.request_memory.tensors["pending"][new_row] == 97)
                )

    def test_sparse_page_regions_and_request_ring_reuse(self):
        # Packed DSA pages put all 128-byte keys before the 4-byte scales;
        # QSA pages contain two compressed K rows. Neither is token-interleaved.
        dsa = TensorPartDecl(
            name="index", shape=(4 * (128 + 4),), dtype=torch.uint8, layer_ids=(0, 4)
        )
        qsa = TensorPartDecl(
            name="compressed",
            shape=(2, 2, 8),
            dtype=torch.bfloat16,
            layer_ids=(0, 4),
            slice_axis=1,
        )
        tails = tuple(
            RequestPartDecl(
                part=TensorPartDecl(
                    name=name, shape=(7, 128), dtype=torch.bfloat16, layer_ids=(0, 4)
                ),
                wire="dsa_tail",
            )
            for name in ("tail_k", "tail_score")
        )
        pending = RequestPartDecl(
            part=TensorPartDecl(
                name="pending", shape=(2, 2, 8), dtype=torch.bfloat16, layer_ids=(0, 4)
            ),
            wire="qsa_pending",
        )
        rope = RequestPartDecl(
            part=TensorPartDecl(name="rope", shape=(2, 3), dtype=torch.int64),
            wire="qsa_pending",
        )
        storage, plan, sized = _build(
            derived=(
                DerivedPartDecl(sub_pool="full", part=dsa, wire="dsa"),
                DerivedPartDecl(sub_pool="full", part=qsa, wire="qsa_compressed"),
            ),
            requests=(*tails, pending, rope),
        )
        wires = plan.wire_entries()
        self.assertEqual(
            [e.state_type for e in wires],
            ["swa", "dsa", "dsa_tail", "qsa_pending", "qsa_compressed"],
        )
        self.assertEqual(storage.regions("full", part="dsa").item_lens, (528, 528))
        region = storage.regions("full", part="qsa_compressed")
        self.assertEqual(region.item_lens, (64, 64))
        self.assertEqual(region.layer_ids, (0, 4))
        self.assertEqual(region.dim_per_tensor, (2, 2))
        self.assertEqual(region.slice_outer_counts, (2, 2))
        self.assertEqual(
            storage.regions("full", part="dsa_tail").item_lens, (1792,) * 4
        )
        self.assertEqual(
            storage.regions("full", part="qsa_pending").item_lens, (64, 64, 48)
        )
        arena = storage.allocator.arena
        for tensor in arena.page_parts["full"].tensors.values():
            tensor[:, 3].fill_(7)
        arena.move_units("full", torch.tensor([9]), torch.tensor([3]))
        for tensor in arena.page_parts["full"].tensors.values():
            self.assertTrue(torch.all(tensor[:, 9] == 7))
        for declared in plan.request_parts:
            storage._request_memory.tensors[declared.part.name].fill_(9)
        row = storage.take_rows(1)[0]
        for declared in plan.request_parts:
            part = declared.part
            tensor = storage._request_memory.tensors[part.name]
            self.assertTrue(
                torch.all(tensor.select(part.row_axis, row) == part.reset_value)
            )
        storage.release_rows([row])
        for declared in plan.request_parts:
            part = declared.part
            storage._request_memory.tensors[part.name].select(part.row_axis, row).fill_(
                11
            )
        self.assertEqual(storage.take_rows(1), [row])
        for declared in plan.request_parts:
            part = declared.part
            self.assertTrue(
                torch.all(
                    storage._request_memory.tensors[part.name].select(
                        part.row_axis, row
                    )
                    == part.reset_value
                )
            )

    def test_derived_page_blobs_move_with_physical_pages(self):
        part = TensorPartDecl(
            name="compressed_k", shape=(2, 8), dtype=torch.float32, layer_ids=(0,)
        )
        storage, plan, sized = _build(
            derived=(DerivedPartDecl(sub_pool="full", part=part),)
        )
        arena = storage.allocator.arena
        memory = arena.page_parts["full"]
        blob = memory.tensors[part.name]
        self.assertEqual(blob[0, 0].stride(), (8, 1))
        self.assertEqual(memory.rows, arena.extent_entries("full") // plan.page_size)
        self.assertEqual(
            memory.nbytes,
            sum(t.bytes for t in sized.fixed if t.name != "request_table"),
        )
        blob[:, 3].fill_(29)
        arena.move_units("full", torch.tensor([7]), torch.tensor([3]))
        self.assertTrue(torch.all(blob[:, 7] == 29))
        arena.zero_units("full", torch.tensor([7]))
        self.assertTrue(torch.all(blob[:, 7] == 0))
        self.assertTrue(torch.all(blob[:, 3] == 29))


if __name__ == "__main__":
    unittest.main()
