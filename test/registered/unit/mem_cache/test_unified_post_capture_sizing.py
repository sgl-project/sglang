# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Post-capture KV sizing on the unified hybrid-SWA byte pool.

With post-capture sizing the shared arena is reserved as CUDA virtual memory
before graph capture: only the slot-0 sink is physically backed, the graphs
capture the final tensor address, and ``finalize_backing`` backs the measured
byte budget in place. ``ChainAllocator.resize`` then
re-derives every capacity the sub-allocators and sub-pools expose.

The oracle is the eagerly-allocated pool of the same final size: a resized pool
must hand out exactly the capacity it would have had if it had been built at
that size, without moving the tensor the graphs captured.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.layout.mha import DraftMHAView
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.unified_allocator_fixtures import (
    build_test_unified_pools,
    setup_allocator_context,
)

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")

_RESERVED_BYTES = 64 << 20
_FINAL_BYTES = 16 << 20
_PAGE_SIZE = 4


def _bundle(
    *, total_bytes: int, post_capture_active: bool, chain="swa", fused_draft=None
):
    has_state = chain in ("state", "tri")
    has_window = chain in ("swa", "tri")
    return build_test_unified_pools(
        device="cuda",
        kv_cache_dtype=torch.float16,
        head_num=2,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=2,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=_PAGE_SIZE,
        start_layer=0,
        end_layer=4,
        swa_attention_layer_ids=[1, 3] if has_window else [],
        mamba_layer_ids=[1, 3] if has_state else [],
        mamba2_cache_params=SimpleNamespace(
            layers=[1, 3],
            shape=SimpleNamespace(conv=[(3, 8)], temporal=(1, 4, 8)),
            dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
        )
        if has_state
        else None,
        max_mamba_cache_size=16 if has_state else 0,
        speculative_num_draft_tokens=2 if has_state else None,
        full_attention_layer_ids=[0, 2],
        total_bytes=total_bytes,
        fused_draft=fused_draft,
        enable_memory_saver=False,
        post_capture_active=post_capture_active,
        need_sort=False,
        model_context_len=256,
        sliding_window_size=64,
    )


def _final_config(bundle):
    storage = bundle.token_to_kv_pool_allocator.storage
    sized = size_plan(
        storage.plan,
        KVSizingInputs(
            budget_bytes=_FINAL_BYTES,
            context_len=256,
            floor_window=64,
            max_running_requests=1,
            state_slots=16,
            alignment=1,
        ),
        frozen=storage.sized,
    )
    return SimpleNamespace(
        unified_memory_pool_bytes=sized.total_bytes,
        full_max_total_num_tokens=sized.full_tokens,
        swa_max_total_num_tokens=sized.window_tokens,
    )


def _capacities(bundle) -> dict:
    """Every capacity-derived value a resize must keep consistent."""
    shared = bundle.arena
    kvcache = bundle.token_to_kv_pool
    allocator = bundle.token_to_kv_pool_allocator
    out = {
        "total_bytes": shared.total_bytes,
        "active_allocation_bytes": shared.accessible_bytes,
        "buf_lens": kvcache.get_contiguous_buf_infos()[1],
        "kvcache.size": kvcache.size,
        "allocator.size": allocator.size,
        "allocator.available_size": allocator.available_size(),
        "allocator.full_available_size": allocator.full_available_size(),
        "allocator.swa_available_size": allocator.swa_available_size(),
    }
    for name, sub, pool in (
        ("full", allocator.full_attn_allocator, kvcache.layer_router.view("full")),
        ("swa", allocator.swa_attn_allocator, kvcache.layer_router.view("swa")),
    ):
        out[f"{name}.max_slots"] = shared.num_entries(name)
        out[f"{name}.sub.max_slots"] = sub.max_slots
        out[f"{name}.sub.num_pages"] = sub.num_pages
        out[f"{name}.sub.ids.num_virtual_ids"] = sub.ids.num_virtual_ids
        out[f"{name}.sub.available_size"] = sub.available_size()
        out[f"{name}.pool.size"] = pool.size
        out[f"{name}.pool.host_capacity_tokens"] = pool.host_capacity_tokens
        out[f"{name}.pool.host_capacity_bytes"] = pool.host_capacity_bytes
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestUnifiedPostCaptureSizing(unittest.TestCase):
    def setUp(self):
        setup_allocator_context()

    def _post_capture_bundle(self, total_bytes=_RESERVED_BYTES):
        return _bundle(total_bytes=total_bytes, post_capture_active=True)

    def test_reservation_backs_only_the_sink_before_capture(self):
        """Before sizing, the arena is address space: the sink is resident and
        only it may be handed to transfer engines, not the reserved bound."""
        bundle = self._post_capture_bundle()
        shared = bundle.arena
        self.assertTrue(bundle.token_to_kv_pool.post_capture_active)
        self.assertEqual(
            shared.total_bytes
            + bundle.token_to_kv_pool_allocator.storage.sized.fixed_bytes,
            _RESERVED_BYTES,
        )
        self.assertGreater(shared.accessible_bytes, 0)
        self.assertLessEqual(shared.accessible_bytes, shared.backed_bytes)
        self.assertLess(shared.backed_bytes, _RESERVED_BYTES)
        _, buf_lens, _ = bundle.token_to_kv_pool.get_contiguous_buf_infos()
        self.assertEqual(buf_lens, [shared.accessible_bytes])

    def test_finalize_and_resize_match_the_eager_pool_of_the_final_size(self):
        bundle = self._post_capture_bundle()
        shared = bundle.arena
        allocator = bundle.token_to_kv_pool_allocator
        captured_ptr = shared.raw.data_ptr()
        tables = [
            table
            for sub in (allocator.full_attn_allocator, allocator.swa_attn_allocator)
            for table in (sub.ids.virtual_to_physical, sub.ids.physical_to_virtual)
        ]
        table_ptrs = [table.data_ptr() for table in tables]

        config = _final_config(bundle)
        bundle.token_to_kv_pool.finalize_backing(config)
        allocator.resize(config)

        # Graphs captured these addresses; sizing may not move them.
        self.assertEqual(shared.raw.data_ptr(), captured_ptr)
        self.assertEqual([table.data_ptr() for table in tables], table_ptrs)
        self.assertGreaterEqual(shared.backed_bytes, shared.accessible_bytes)
        # The whole advertised span is resident, not just the sink.
        last = shared.raw[shared.accessible_bytes - 1]
        last.fill_(7)
        torch.cuda.synchronize()
        self.assertEqual(int(last.item()), 7)

        eager = _bundle(total_bytes=_FINAL_BYTES, post_capture_active=False)
        self.assertEqual(_capacities(bundle), _capacities(eager))
        self.assertEqual(allocator.verify_byte_accounting(), [])

        # The resized capacity is fully allocatable and accounted.
        n = allocator.available_size()
        self.assertGreater(n, 0)
        self.assertIsNotNone(allocator.alloc(n))
        self.assertEqual(allocator.available_size(), 0)

    def test_retraction_restores_complete_entries_at_new_dcp_coordinates(self):
        placement = FusedDraftPlacement.from_counts(
            counts={"full": [2]},
            regions={
                "full": DenseDraftRegion(
                    lane_num=2,
                    head_num=1,
                    head_dim=3,
                    v_head_dim=5,
                    store_dtype=torch.bfloat16,
                )
            },
        )
        for dcp in (1, 2, 4):
            for rank in range(dcp):
                with (
                    self.subTest(dcp=dcp, rank=rank),
                    get_parallel().override(attn_dcp_size=dcp, attn_dcp_rank=rank),
                ):
                    bundle = _bundle(
                        total_bytes=_FINAL_BYTES,
                        post_capture_active=False,
                        chain="state",
                        fused_draft=placement,
                    )
                    allocator = bundle.token_to_kv_pool_allocator
                    source = allocator.alloc(16 * dcp)
                    old_state = allocator.alloc_state_slots(1)
                    target = allocator.alloc(16 * dcp)
                    new_state = allocator.alloc_state_slots(1)
                    for band in allocator.members:
                        virtual = old_state if band.sub_pool_name == "mamba" else source
                        physical = band.ids.translate_write(virtual.to(torch.int64))
                        physical = physical[physical > 0]
                        allocator.arena.entries(band.sub_pool_name)[physical] = 79
                    saved = allocator.get_cpu_copy(source, mamba_indices=old_state)
                    allocator.load_cpu_copy(saved, target, mamba_indices=new_state)
                    for band in allocator.members:
                        virtual = new_state if band.sub_pool_name == "mamba" else target
                        physical = band.ids.translate_write(virtual.to(torch.int64))
                        physical = physical[physical > 0]
                        self.assertTrue(
                            torch.all(
                                allocator.arena.entries(band.sub_pool_name)[physical]
                                == 79
                            )
                        )

    def test_every_chain_retains_captured_and_fixed_addresses(self):
        for chain, dcp in (
            ("full", 1),
            ("swa", 1),
            ("state", 1),
            ("tri", 1),
            ("state", 2),
            ("state", 4),
        ):
            with (
                self.subTest(chain=chain, dcp=dcp),
                get_parallel().override(attn_dcp_size=dcp),
            ):
                bundle = _bundle(
                    total_bytes=_RESERVED_BYTES, post_capture_active=True, chain=chain
                )
                allocator = bundle.token_to_kv_pool_allocator
                storage = allocator.storage
                tensors = [bundle.arena.raw, storage.requests.req_to_token]
                tensors.extend(storage.scratch.tensors.values())
                for band in allocator.members:
                    tensors.extend(
                        (band.ids.virtual_to_physical, band.ids.physical_to_virtual)
                    )
                pointers = [t.data_ptr() for t in tensors]
                shapes = [tuple(t.shape) for t in tensors]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    bundle.arena.raw[:1].fill_(9)
                config = _final_config(bundle)
                bundle.token_to_kv_pool.finalize_backing(config)
                allocator.resize(config)
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(bundle.arena.raw[0].item(), 9)
                self.assertEqual([t.data_ptr() for t in tensors], pointers)
                self.assertEqual([tuple(t.shape) for t in tensors], shapes)
                eager = _bundle(
                    total_bytes=_FINAL_BYTES, post_capture_active=False, chain=chain
                )
                self.assertEqual(
                    allocator.available_size(),
                    eager.token_to_kv_pool_allocator.available_size(),
                )
                self.assertEqual(allocator.verify_byte_accounting(), [])
                for band in allocator.members:
                    self.assertEqual(
                        band.max_slots, bundle.arena.num_entries(band.sub_pool_name)
                    )
                    view = allocator.views[band.sub_pool_name]
                    self.assertEqual(view.size, band.max_slots - band.pool_page_size)

    def test_short_draft_rows_store_and_replay_without_touching_host(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                region = DenseDraftRegion(
                    lane_num=2, head_num=1, head_dim=3, v_head_dim=5, store_dtype=dtype
                )
                bundle = _bundle(
                    total_bytes=1 << 20,
                    post_capture_active=False,
                    fused_draft=FusedDraftPlacement.from_counts(
                        counts={"full": [2]}, regions={"full": region}
                    ),
                )
                arena = bundle.arena
                spec = bundle.storage.plan.owner().spec
                view = DraftMHAView(
                    arena=arena,
                    host_spec=spec,
                    layer_lanes={0: 0, 1: 1},
                    page_size=_PAGE_SIZE,
                )
                arena.raw.fill_(11)
                before = arena.raw.clone()
                loc = torch.tensor([8, 13], device="cuda", dtype=torch.int64)
                k = (
                    torch.arange(6, device="cuda", dtype=torch.float32)
                    .view(2, 1, 3)
                    .to(dtype)
                )
                v = (
                    torch.arange(10, device="cuda", dtype=torch.float32)
                    .view(2, 1, 5)
                    .to(dtype)
                )
                layer = SimpleNamespace(layer_id=1)
                view.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k, v)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    view.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k, v)
                k.add_(3)
                v.add_(5)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(view.k_buffer[1][loc], k)
                torch.testing.assert_close(view.v_buffer[1][loc], v)
                changed = torch.zeros_like(arena.raw, dtype=torch.bool)
                for part in region.parts(spec.draft_offset_in_entry()):
                    for row in (8, 13):
                        lo = row * spec.entry_bytes() + part.layer_offset_bytes(1)
                        changed[lo : lo + part.row_bytes()] = True
                self.assertTrue(torch.equal(arena.raw[~changed], before[~changed]))

    def test_sizing_contract_is_enforced(self):
        bundle = self._post_capture_bundle()
        kvcache = bundle.token_to_kv_pool
        allocator = bundle.token_to_kv_pool_allocator
        with self.subTest("resize before finalize_backing"):
            with self.assertRaisesRegex(
                RuntimeError, "must be backed before allocator.resize"
            ):
                allocator.resize(_final_config(bundle))
        with self.subTest("budget without a byte count"):
            with self.assertRaises(ValueError):
                kvcache.finalize_backing(
                    SimpleNamespace(unified_memory_pool_bytes=None)
                )
        with self.subTest("budget above the reservation"):
            with self.assertRaises(ValueError):
                kvcache.finalize_backing(
                    SimpleNamespace(unified_memory_pool_bytes=2 * _RESERVED_BYTES)
                )
        with self.subTest("budget below the bs=1 floor"):
            with self.assertRaisesRegex(RuntimeError, "bs=1 floor"):
                kvcache.finalize_backing(
                    SimpleNamespace(unified_memory_pool_bytes=4096)
                )
        with self.subTest("resize with live allocations"):
            kvcache.finalize_backing(_final_config(bundle))
            # Move the grow-down allocator's frontier into the backed range
            # before allocating; it still reflects the larger reservation.
            allocator.resize(_final_config(bundle))
            handed = allocator.alloc(_PAGE_SIZE)
            self.assertIsNotNone(handed)
            with self.assertRaisesRegex(RuntimeError, "non-empty"):
                allocator.resize(_final_config(bundle))
            allocator.free(handed)

        eager = _bundle(total_bytes=_FINAL_BYTES, post_capture_active=False)
        with self.subTest("eager pool cannot be resized"):
            self.assertFalse(eager.token_to_kv_pool.post_capture_active)
            with self.assertRaises(RuntimeError):
                eager.token_to_kv_pool.finalize_backing(
                    SimpleNamespace(unified_memory_pool_bytes=_FINAL_BYTES)
                )
            with self.assertRaises(RuntimeError):
                eager.token_to_kv_pool_allocator.resize(
                    SimpleNamespace(unified_memory_pool_bytes=_FINAL_BYTES)
                )


if __name__ == "__main__":
    unittest.main()
