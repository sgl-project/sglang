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
byte budget in place. ``UnifiedSWATokenToKVPoolAllocator.resize`` then
re-derives every capacity the sub-allocators and sub-pools expose.

The oracle is the eagerly-allocated pool of the same final size: a resized pool
must hand out exactly the capacity it would have had if it had been built at
that size, without moving the tensor the graphs captured.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_RESERVED_BYTES = 64 << 20
_FINAL_BYTES = 16 << 20
_PAGE_SIZE = 4


def _bundle(*, total_bytes: int, post_capture_active: bool):
    return init_unified_swa_pools(
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
        swa_attention_layer_ids=[1, 3],
        full_attention_layer_ids=[0, 2],
        total_bytes=total_bytes,
        enable_memory_saver=False,
        post_capture_active=post_capture_active,
        need_sort=False,
        model_context_len=256,
        sliding_window_size=64,
    )


def _capacities(bundle) -> dict:
    """Every capacity-derived value a resize must keep consistent."""
    shared = bundle.unified_memory_pool
    kvcache = bundle.token_to_kv_pool
    allocator = bundle.token_to_kv_pool_allocator
    out = {
        "total_bytes": shared.total_bytes,
        "active_allocation_bytes": shared.active_allocation_bytes,
        "buf_lens": kvcache.get_contiguous_buf_infos()[1],
        "kvcache.size": kvcache.size,
        "kvcache.size_swa": kvcache.size_swa,
        "allocator.size": allocator.size,
        "allocator.available_size": allocator.available_size(),
        "allocator.full_available_size": allocator.full_available_size(),
        "allocator.swa_available_size": allocator.swa_available_size(),
    }
    for name, sub, pool in (
        ("full", allocator.full_attn_allocator, kvcache.full_kv_pool),
        ("swa", allocator.swa_attn_allocator, kvcache.swa_kv_pool),
    ):
        out[f"{name}.max_slots"] = shared.max_slots(name)
        out[f"{name}.sub.max_slots"] = sub.max_slots
        out[f"{name}.sub.num_pages"] = sub.num_pages
        out[f"{name}.sub.num_virtual_ids"] = sub.num_virtual_ids
        out[f"{name}.sub.available_size"] = sub.available_size()
        out[f"{name}.pool.size"] = pool.size
        out[f"{name}.pool.host_capacity_tokens"] = pool.host_capacity_tokens
        out[f"{name}.pool.host_capacity_bytes"] = pool.host_capacity_bytes
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestUnifiedPostCaptureSizing(unittest.TestCase):
    def _post_capture_bundle(self, total_bytes=_RESERVED_BYTES):
        return _bundle(total_bytes=total_bytes, post_capture_active=True)

    def test_reservation_backs_only_the_sink_before_capture(self):
        """Before sizing, the arena is address space: the sink is resident and
        only it may be handed to transfer engines, not the reserved bound."""
        bundle = self._post_capture_bundle()
        shared = bundle.unified_memory_pool
        self.assertTrue(bundle.token_to_kv_pool.post_capture_active)
        self.assertEqual(shared.total_bytes, _RESERVED_BYTES)
        self.assertGreater(shared.active_allocation_bytes, 0)
        self.assertLessEqual(
            shared.active_allocation_bytes, shared.post_capture_backed_bytes
        )
        self.assertLess(shared.post_capture_backed_bytes, _RESERVED_BYTES)
        _, buf_lens, _ = bundle.token_to_kv_pool.get_contiguous_buf_infos()
        self.assertEqual(buf_lens, [shared.active_allocation_bytes])

    def test_finalize_and_resize_match_the_eager_pool_of_the_final_size(self):
        bundle = self._post_capture_bundle()
        shared = bundle.unified_memory_pool
        allocator = bundle.token_to_kv_pool_allocator
        captured_ptr = shared._raw.data_ptr()
        tables = [
            table
            for sub in (allocator.full_attn_allocator, allocator.swa_attn_allocator)
            for table in (sub.virtual_to_physical, sub.physical_to_virtual)
        ]
        table_ptrs = [table.data_ptr() for table in tables]

        config = SimpleNamespace(unified_memory_pool_bytes=_FINAL_BYTES)
        bundle.token_to_kv_pool.finalize_backing(config)
        allocator.resize(config)

        # Graphs captured these addresses; sizing may not move them.
        self.assertEqual(shared._raw.data_ptr(), captured_ptr)
        self.assertEqual([table.data_ptr() for table in tables], table_ptrs)
        self.assertGreaterEqual(
            shared.post_capture_backed_bytes, shared.active_allocation_bytes
        )
        # The whole advertised span is resident, not just the sink.
        last = shared._raw[shared.active_allocation_bytes - 1]
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

    def test_sizing_contract_is_enforced(self):
        bundle = self._post_capture_bundle()
        kvcache = bundle.token_to_kv_pool
        allocator = bundle.token_to_kv_pool_allocator
        with self.subTest("resize before finalize_backing"):
            with self.assertRaisesRegex(RuntimeError, "finalize_backing must run"):
                allocator.resize(
                    SimpleNamespace(unified_memory_pool_bytes=_FINAL_BYTES)
                )
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
            kvcache.finalize_backing(
                SimpleNamespace(unified_memory_pool_bytes=_FINAL_BYTES)
            )
            handed = allocator.alloc(_PAGE_SIZE)
            self.assertIsNotNone(handed)
            with self.assertRaisesRegex(RuntimeError, "non-empty"):
                allocator.resize(
                    SimpleNamespace(unified_memory_pool_bytes=_FINAL_BYTES)
                )
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
