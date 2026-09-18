import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import RetractionBackup, retraction_backup
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.kv_cache_builder import maybe_register_hicache_draft
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_parallel
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

    def _make_pool(self, layer_num: int, *, page_size=1, use_mla=False):
        kwargs = dict(
            size=self.pool_size,
            page_size=page_size,
            dtype=self.dtype,
            layer_num=layer_num,
            device=self.device,
            enable_memory_saver=False,
        )
        if use_mla:
            return MLATokenToKVPool(**kwargs, kv_lora_rank=96, qk_rope_head_dim=32)
        return MHATokenToKVPool(**kwargs, head_num=2, head_dim=64)

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

    def _build_cache(
        self, hicache_ratio: float, *, shared_receive=False, use_mla=False, page_size=1
    ):
        """Bring up a UnifiedRadixCache over fresh pools, optionally with draft KV."""
        server_args = ServerArgs(
            model_path="dummy",
            page_size=page_size,
            hicache_ratio=hicache_ratio,
            hicache_io_backend="kernel",
            hicache_mem_layout="layer_first" if shared_receive else "page_first",
        )
        set_global_server_args_for_scheduler(server_args)

        req_to_token_pool = ReqToTokenPool(
            size=2,
            max_context_len=self.pool_size,
            device=self.device,
            enable_memory_saver=False,
        )
        target_pool = self._make_pool(layer_num=2, page_size=page_size, use_mla=use_mla)
        allocator_cls = (
            PagedTokenToKVPoolAllocator if page_size > 1 else TokenToKVPoolAllocator
        )
        allocator = allocator_cls(
            size=self.pool_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=target_pool,
            need_sort=False,
            **({"page_size": page_size} if page_size > 1 else {}),
        )
        params = CacheInitParams(
            disable=True,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            is_eagle=not shared_receive,
            tree_components=(ComponentType.FULL,),
        )
        cache = UnifiedRadixCache(params)
        cache.init_hicache(server_args, params)
        self.addCleanup(cache.release_host_resources)

        draft_pool = None
        if not shared_receive:
            draft_pool = self._make_pool(layer_num=1)
            maybe_register_hicache_draft(
                tree_cache=cache,
                draft_plan=HiCacheDraftPlan(
                    mode=HiCacheDraftMode.SIDECAR,
                    device_pools=(draft_pool,),
                ),
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
        req = SimpleNamespace(rid="request", kv=ReqKvInfo(), seqlen=num_tokens + 1)
        self.assertIsNotNone(env.req_to_token_pool.alloc([req]))
        source_indices = env.allocator.alloc(num_tokens)
        self.assertIsNotNone(source_indices)
        env.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, num_tokens)), source_indices
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
            (req.kv.req_pool_idx, slice(0, self.num_tokens)), destination_indices
        )

        cache.retraction_restore(req, backup)

        self._assert_pool_equal(target_pool, destination_indices, target_expected)
        self._assert_pool_equal(draft_pool, destination_indices, draft_expected)
        self.assertEqual(cache.host_pool_group.available_size(), host_free_before)

        allocator.free(blocker_indices)
        allocator.free(destination_indices)
        req_to_token_pool.free(req)

    def test_receive_pressure_preserves_shared_retraction_and_restore(self):
        for use_mla, page_size in ((False, 16), (True, 1)):
            with self.subTest(use_mla=use_mla, page_size=page_size):
                env = self._build_cache(
                    hicache_ratio=0.5,
                    shared_receive=True,
                    use_mla=use_mla,
                    page_size=page_size,
                )
                cache, pool = env.cache, env.target_pool
                host = cache.host_pool_group.get_pool(PoolName.KV)
                reserve = max(self.num_tokens, page_size)
                host_capacity = host.available_size()
                receive_slots = host_capacity - reserve
                receive_tokens = receive_slots - int(page_size > 1)
                host_indices = host.alloc(receive_slots)
                self.assertEqual(len(host_indices), receive_slots)
                self.assertEqual(host.available_size(), reserve)

                device_buffers = (
                    pool.kv_buffer if use_mla else pool.k_buffer + pool.v_buffer
                )
                host_buffers = host.data_refs if use_mla else host.host_kv_data_refs
                expected_receive = []
                for index, buffer in enumerate(host_buffers):
                    values = torch.arange(
                        buffer[host_indices].numel(), dtype=torch.float32
                    ).reshape_as(buffer[host_indices])
                    values = ((values + 13 * index) % 251).to(self.dtype)
                    buffer[host_indices] = values
                    expected_receive.append(values[:receive_tokens].clone())

                retracted, source_indices = self._admit_req(env, reserve)
                retracted.seqlen -= int(page_size > 1)
                expected_retraction = []
                for index, buffer in enumerate(device_buffers):
                    values = torch.arange(
                        buffer[source_indices].numel(), device=self.device
                    ).reshape_as(buffer[source_indices])
                    values = ((values + 29 * index) % 127).to(self.dtype)
                    buffer[source_indices] = values
                    expected_retraction.append(values.clone())

                backup = cache.retraction_backup(retracted)
                self.assertIsNotNone(backup)
                self.assertEqual(host.available_size(), 0)
                restored_indices = env.allocator.alloc(reserve)
                self.assertIsNotNone(restored_indices)
                self.assertFalse(torch.equal(source_indices, restored_indices))
                for buffer in device_buffers:
                    buffer.fill_(-1)
                env.allocator.free(source_indices)
                env.req_to_token_pool.write(
                    (retracted.kv.req_pool_idx, slice(0, reserve)), restored_indices
                )
                cache.retraction_restore(retracted, backup)
                self.assertEqual(host.available_size(), reserve)

                receiving, received_indices = self._admit_req(env, receive_slots)
                receiving.seqlen = receive_tokens + 1
                cache.retraction_restore(
                    receiving, RetractionBackup(host_indices=host_indices)
                )
                for buffer, restored, received in zip(
                    device_buffers,
                    expected_retraction,
                    expected_receive,
                    strict=True,
                ):
                    self.assertTrue(torch.equal(buffer[restored_indices], restored))
                    self.assertTrue(
                        torch.equal(
                            buffer[received_indices[:receive_tokens]].cpu(),
                            received,
                        )
                    )
                self.assertEqual(host.available_size(), host_capacity)
                env.allocator.free(restored_indices)
                env.allocator.free(received_indices)
                env.req_to_token_pool.free(retracted)
                env.req_to_token_pool.free(receiving)


DCP_SIZE = 4
DCP_RANK = 1
DCP_ROWS = 8


def _bare_mla_pool() -> MLATokenToKVPool:
    pool = object.__new__(MLATokenToKVPool)
    pool.layer_num = 2
    pool.cpu_offloading_chunk_size = 3
    pool.kv_buffer = [
        (torch.arange(DCP_ROWS, dtype=torch.float32) + 100 * layer).view(DCP_ROWS, 1, 1)
        for layer in range(pool.layer_num)
    ]
    return pool


def _dcp():
    return get_parallel().override(
        dcp_enabled=True, attn_dcp_size=DCP_SIZE, attn_dcp_rank=DCP_RANK
    )


class TestDcpRetractionBackup(unittest.TestCase):
    """`req_to_token` names KV slots in the widened DCP id space while
    `kv_buffer` holds only this rank's rows; a widened id used as a row index
    reads past the buffer or copies another token's row."""

    def test_restore_lands_on_new_owned_rows(self):
        pool = _bare_mla_pool()
        before = [buf.clone() for buf in pool.kv_buffer]
        old_widened = torch.arange(0, 12, dtype=torch.int64)
        new_widened = torch.arange(12, 24, dtype=torch.int64)

        with _dcp():
            pool.load_cpu_copy(pool.get_cpu_copy(old_widened), new_widened)

        old_rows = old_widened[DCP_RANK::DCP_SIZE] // DCP_SIZE
        new_rows = new_widened[DCP_RANK::DCP_SIZE] // DCP_SIZE
        self.assertEqual(new_rows.tolist(), [3, 4, 5])
        untouched = torch.tensor(
            [r for r in range(DCP_ROWS) if r not in new_rows.tolist()]
        )
        for layer in range(pool.layer_num):
            torch.testing.assert_close(
                pool.kv_buffer[layer][new_rows], before[layer][old_rows]
            )
            torch.testing.assert_close(
                pool.kv_buffer[layer][untouched], before[layer][untouched]
            )

    def test_resolved_pool_takes_ids_as_rows(self):
        pool = _bare_mla_pool()
        pool.write_loc_is_dcp_resolved = True
        rows = torch.arange(DCP_ROWS, dtype=torch.int64)

        with _dcp():
            kv_cpu = pool.get_cpu_copy(rows)

        for layer in range(pool.layer_num):
            torch.testing.assert_close(torch.cat(kv_cpu[layer]), pool.kv_buffer[layer])


if __name__ == "__main__":
    unittest.main()
