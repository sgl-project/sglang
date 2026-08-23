import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import retraction_backup
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.kv_cache_builder import maybe_register_hicache_draft
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.base_spec_worker import (
    HiCacheDraftMode,
    HiCacheDraftPlan,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestDecodeRetractionBackup(unittest.TestCase):
    pool_size = 32
    num_tokens = 8
    dtype = torch.bfloat16
    device = "cuda"

    def _make_pool(self, layer_num: int) -> MHATokenToKVPool:
        return MHATokenToKVPool(
            size=self.pool_size,
            page_size=1,
            head_num=2,
            head_dim=64,
            dtype=self.dtype,
            layer_num=layer_num,
            device=self.device,
            enable_memory_saver=False,
        )

    def _seed_pool(
        self, pool: MHATokenToKVPool, indices: torch.Tensor, base: int
    ) -> None:
        for layer_id, (key, value) in enumerate(
            zip(pool.k_buffer, pool.v_buffer, strict=True)
        ):
            pattern = torch.arange(
                key[indices].numel(), device=self.device, dtype=torch.float32
            ).reshape_as(key[indices])
            key[indices] = (pattern + base + layer_id * 100).to(self.dtype)
            value[indices] = (pattern + base + 50 + layer_id * 100).to(self.dtype)

    @staticmethod
    def _snapshot_pool(
        pool: MHATokenToKVPool, indices: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (key[indices].clone(), value[indices].clone())
            for key, value in zip(pool.k_buffer, pool.v_buffer, strict=True)
        ]

    def _assert_pool_equal(
        self,
        pool: MHATokenToKVPool,
        indices: torch.Tensor,
        expected: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        for (key, value), (expected_key, expected_value) in zip(
            zip(pool.k_buffer, pool.v_buffer, strict=True), expected, strict=True
        ):
            self.assertTrue(torch.equal(key[indices], expected_key))
            self.assertTrue(torch.equal(value[indices], expected_value))

    def _build_cache(self, hicache_ratio: float):
        """Bring up a UnifiedRadixCache with a draft sidecar over fresh pools."""
        server_args = ServerArgs(
            model_path="dummy",
            page_size=1,
            hicache_ratio=hicache_ratio,
            hicache_io_backend="kernel",
            hicache_mem_layout="page_first",
        )
        set_global_server_args_for_scheduler(server_args)

        req_to_token_pool = ReqToTokenPool(
            size=2,
            max_context_len=self.pool_size,
            device=self.device,
            enable_memory_saver=False,
        )
        target_pool = self._make_pool(layer_num=2)
        allocator = TokenToKVPoolAllocator(
            size=self.pool_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=target_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            disable=True,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            is_eagle=True,
            tree_components=(ComponentType.FULL,),
        )
        cache = UnifiedRadixCache(params)
        cache.init_hicache(server_args, params)
        self.addCleanup(cache.release_host_resources)

        draft_pool = self._make_pool(layer_num=1)
        maybe_register_hicache_draft(
            tree_cache=cache,
            draft_plan=HiCacheDraftPlan(
                mode=HiCacheDraftMode.SIDECAR,
                device_pools=(draft_pool,),
            ),
            server_args=server_args,
            page_size=1,
        )
        self.assertIn(PoolName.DRAFT, cache.host_pool_group.entry_map)
        cache.validate_retraction_host_capacity()
        return SimpleNamespace(
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            allocator=allocator,
            target_pool=target_pool,
            draft_pool=draft_pool,
            cache=cache,
        )

    def _admit_req(self, env, num_tokens: int):
        req = SimpleNamespace(rid="request", req_pool_idx=None, seqlen=num_tokens + 1)
        self.assertIsNotNone(env.req_to_token_pool.alloc([req]))
        source_indices = env.allocator.alloc(num_tokens)
        self.assertIsNotNone(source_indices)
        env.req_to_token_pool.write(
            (req.req_pool_idx, slice(0, num_tokens)), source_indices
        )
        return req, source_indices

    def test_backup_declined_when_host_pool_too_small(self):
        # A backup-only host pool is deliberately smaller than the device pool,
        # so a large enough request cannot be preserved.
        env = self._build_cache(hicache_ratio=0.1)
        self.assertLess(env.cache.host_pool_group.available_size(), self.num_tokens)

        req, source_indices = self._admit_req(env, self.num_tokens)
        host_free_before = env.cache.host_pool_group.available_size()

        self.assertIsNone(env.cache.retraction_backup(req))
        # The declined backup must not leak host slots.
        self.assertEqual(env.cache.host_pool_group.available_size(), host_free_before)

        # This is the signal release_req propagates so retract_decode aborts.
        self.assertFalse(
            retraction_backup(
                req,
                env.cache,
                env.req_to_token_pool,
                env.allocator,
                "host_pool",
            )
        )

        env.allocator.free(source_indices)
        env.req_to_token_pool.free(req)

    def test_restores_target_and_draft_kv(self):
        env = self._build_cache(hicache_ratio=1.0)
        req_to_token_pool = env.req_to_token_pool
        allocator = env.allocator
        target_pool = env.target_pool
        draft_pool = env.draft_pool
        cache = env.cache

        req, source_indices = self._admit_req(env, self.num_tokens)

        self._seed_pool(target_pool, source_indices, base=1000)
        self._seed_pool(draft_pool, source_indices, base=3000)
        target_expected = self._snapshot_pool(target_pool, source_indices)
        draft_expected = self._snapshot_pool(draft_pool, source_indices)

        host_free_before = cache.host_pool_group.available_size()
        backup = cache.retraction_backup(req)
        self.assertEqual(
            {transfer.name for transfer in backup.pool_transfers or []},
            {PoolName.DRAFT},
        )
        self.assertLess(cache.host_pool_group.available_size(), host_free_before)

        for buffer in (*target_pool.k_buffer, *target_pool.v_buffer):
            buffer.fill_(-1)
        for buffer in (*draft_pool.k_buffer, *draft_pool.v_buffer):
            buffer.fill_(-2)

        allocator.free(source_indices)
        blocker_indices = allocator.alloc(self.num_tokens)
        destination_indices = allocator.alloc(self.num_tokens)
        self.assertIsNotNone(blocker_indices)
        self.assertIsNotNone(destination_indices)
        self.assertFalse(torch.equal(source_indices, destination_indices))
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, self.num_tokens)), destination_indices
        )

        cache.retraction_restore(req, backup)

        self._assert_pool_equal(target_pool, destination_indices, target_expected)
        self._assert_pool_equal(draft_pool, destination_indices, draft_expected)
        self.assertEqual(cache.host_pool_group.available_size(), host_free_before)

        allocator.free(blocker_indices)
        allocator.free(destination_indices)
        req_to_token_pool.free(req)


if __name__ == "__main__":
    unittest.main()
