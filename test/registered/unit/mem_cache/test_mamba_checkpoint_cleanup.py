"""Track a checkpoint until its slot has actually been donated."""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from array import array

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.mem_cache_utils import finish_req

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMambaCheckpointCleanup(CustomTestCase):
    def setUp(self):
        self.enterContext(
            get_context().override_server_args(
                mamba_radix_cache_strategy="extra_buffer",
                _mamba_cache_chunk_size=64,
                disable_radix_cache=False,
                enable_int8_mamba_checkpoint=False,
            )
        )

    def make_fixture(self, *, lazy=False, spare=0, int8=False):
        self.enterContext(
            get_context().override_server_args(
                enable_int8_mamba_checkpoint=int8, _mamba_cache_chunk_size=64
            )
        )
        active_slots = 1 + (1 if lazy else 2)
        pool = HybridReqToTokenPool(
            size=1,
            mamba_size=active_slots + spare + (0 if int8 else 1),
            mamba_spec_state_size=1,
            max_context_len=128,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(
                shape=Mamba2StateShape.create(
                    tp_world_size=1,
                    intermediate_size=4,
                    n_groups=1,
                    num_heads=1,
                    head_dim=4,
                    state_size=4,
                    conv_kernel=2,
                ),
                layers=[0],
                dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
            ),
            mamba_layer_ids=[0],
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=lazy,
        )
        allocator = TokenToKVPoolAllocator(
            size=128, dtype=torch.float32, device="cpu", kvcache=None, need_sort=False
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=64,
                tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                enable_mamba_extra_buffer=True,
                tree_core_backend="python",
            )
        )
        prefix = array("q", [1] * 64)
        cache.insert(
            InsertParams(
                key=RadixKey(prefix),
                value=allocator.alloc(64),
                mamba_value=(
                    pool.mamba_ckpt_pool if int8 else pool.mamba_allocator
                ).alloc(1),
            )
        )
        req = Req(
            rid="checkpoint",
            origin_input_text="",
            origin_input_ids=prefix + array("q", [2] * 64),
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        req.refresh_fill_ids()
        req.last_node = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(req.origin_input_ids), cow_mamba=True, req=req
            )
        ).last_device_node
        req.lock = cache.lock(req.last_node)
        pool.alloc([req])
        pool.write(
            (req.kv.req_pool_idx, slice(0, 128)),
            torch.cat((cache.path_device_indices(req.last_node), allocator.alloc(64))),
        )
        req.kv.cache_protected_len = req.kv.cache_inserted_len = 64
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 128
        req.kv.mamba_last_track_idx = 0
        req.kv.mamba_last_track_seqlen = 128
        req.kv.mamba_prev_track_seqlen = None if lazy else 64
        self.assertEqual(pool.mamba_allocator.available_size(), spare)
        return cache, pool, req

    def test_no_donation_preserves_checkpoint_for_finish(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                cache, pool, req = self.make_fixture(lazy=lazy)
                slots = req.kv.mamba_ping_pong_track_buffer.clone()
                cache.checkpoint(req, up_to=128)
                self.assertEqual(req.kv.mamba_last_track_seqlen, 128)
                self.assertEqual(req.kv.mamba_prev_track_seqlen, None if lazy else 64)
                torch.testing.assert_close(req.kv.mamba_ping_pong_track_buffer, slots)
                self.assertEqual(pool.mamba_allocator.available_size(), 0)

                # Finish reuses the retained state without a forward or replacement slot.
                finish_req(cache, req, up_to=128)
                self.assertEqual(
                    len(
                        cache.path_device_indices(
                            cache.match_prefix(
                                MatchPrefixParams(key=RadixKey(req.origin_input_ids))
                            ).last_device_node
                        )
                    ),
                    128,
                )
                cache.evict(
                    EvictParams(num_tokens=128, mamba_num=pool.mamba_allocator.size)
                )
                self.assertEqual(
                    sorted(pool.mamba_allocator.free_slots.tolist()),
                    list(range(1, pool.mamba_allocator.size + 1)),
                )
                self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), 128)

    def test_donation_consumes_tracking_metadata(self):
        cache, pool, req = self.make_fixture(spare=1)
        donated_slot = int(req.kv.mamba_ping_pong_track_buffer[0])
        cache.checkpoint(req, up_to=128)
        self.assertIsNone(req.kv.mamba_last_track_seqlen)
        self.assertIsNone(req.kv.mamba_prev_track_seqlen)
        self.assertNotIn(donated_slot, req.kv.mamba_ping_pong_track_buffer.tolist())
        self.assertEqual(
            cache.tree_core.get_component_device_value(
                req.last_node, ComponentType.MAMBA
            ).tolist(),
            [donated_slot],
        )

    def test_int8_no_value_still_invalidates_tracking_metadata(self):
        cache, pool, req = self.make_fixture(int8=True)
        slots = req.kv.mamba_ping_pong_track_buffer.clone()
        cache.components[ComponentType.MAMBA].cleanup_after_caching_req(
            req, is_finished=False, insert_params=InsertParams()
        )
        self.assertIsNone(req.kv.mamba_last_track_seqlen)
        self.assertIsNone(req.kv.mamba_prev_track_seqlen)
        torch.testing.assert_close(req.kv.mamba_ping_pong_track_buffer, slots)
        self.assertEqual(pool.mamba_allocator.available_size(), 0)


if __name__ == "__main__":
    unittest.main()
