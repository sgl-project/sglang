# SPDX-License-Identifier: Apache-2.0
"""Real GPU/host/file KV round trips and publication-to-admission ownership."""

import atexit
import json
import tempfile
import time
import unittest
from array import array

import torch

from sglang.srt.managers.kv_control.contracts import (
    CopyTo,
    Location,
    Status,
    TokensTarget,
)
from sglang.srt.managers.kv_control.service import KVControlService, ServiceConfig
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestProgrammableKVCache(CustomTestCase):
    def test_roundtrip_and_preparation_handoff(self):
        for storage in (False, True):
            with self.subTest(storage=storage):
                self._roundtrip(storage)

    def _roundtrip(self, storage):
        with (
            tempfile.TemporaryDirectory() as storage_dir,
            get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1),
            get_context().override_server_args(
                device="cuda",
                enable_hierarchical_cache=True,
                hicache_ratio=2,
                hicache_mem_layout="layer_first",
                hicache_write_policy="write_back",
                hicache_io_backend="direct",
                disable_overlap_schedule=True,
                hicache_storage_backend="file" if storage else None,
                hicache_storage_backend_extra_config=json.dumps(
                    {"file_storage_path": storage_dir}
                )
                if storage
                else None,
            ),
        ):
            config = ServiceConfig(namespace="test-weights", engine_id="test-owner")
            pool = MHATokenToKVPool(
                size=64,
                page_size=1,
                dtype=torch.float16,
                head_num=1,
                head_dim=128,
                layer_num=1,
                device="cuda",
                enable_memory_saver=False,
            )
            allocator = TokenToKVPoolAllocator(
                size=64,
                dtype=torch.float16,
                device="cuda",
                kvcache=pool,
                need_sort=False,
            )
            params = CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4,
                    max_context_len=32,
                    device="cuda",
                    enable_memory_saver=False,
                ),
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                tree_components=(ComponentType.FULL,),
                tree_core_backend="python",
                kv_control_config=config.cache_config(),
            )
            cache = UnifiedRadixCache(params)
            cache.init_hicache(get_server_args(), params)
            atexit.unregister(cache.shutdown)
            preparation = None
            try:
                service = KVControlService(cache, config)
                target = TokensTarget(
                    namespace=config.namespace, token_ids=(1, 2, 3, 4)
                )
                slots = allocator.alloc(4)
                pool.k_buffer[0][slots] = 17
                pool.v_buffer[0][slots] = 18
                node_id = cache.insert(
                    InsertParams(
                        key=RadixKey(array("q", target.token_ids)),
                        value=slots.to(torch.int64),
                    )
                ).last_device_node
                node = cache.tree_core.node_by_id(node_id)
                cold_tier = "storage" if storage else "host"
                result = service.control.submit(
                    KvHintsEnvelope(
                        protocol_version="0.1",
                        message_id="save",
                        actions=[
                            KvHintAction(
                                action_id="save",
                                action_type="sglang.move_to",
                                action_version="1.0",
                                payload={
                                    "target": {
                                        "kind": "tokens",
                                        "namespace": config.namespace,
                                        "token_ids": list(target.token_ids),
                                    },
                                    "src": {"tier": "gpu"},
                                    "dst": {"tier": cold_tier},
                                },
                            )
                        ],
                    ),
                    service.control.context("test"),
                ).results[0]
                moved = self._wait(service, result.ref)
                self.assertEqual(
                    moved.update.completed_steps,
                    ("destination_ready", "source_reclaimed"),
                )
                self.assertIsNone(node.component_data[ComponentType.FULL].value)
                self.assertEqual(allocator.available_size(), 64)
                host = cache.host_pool_group.anchor_entry.host_pool
                host_slots = node.component_data[ComponentType.FULL].host_value
                self.assertTrue(torch.all(host.k_buffer[0][host_slots] == 17).item())
                self.assertTrue(torch.all(host.v_buffer[0][host_slots] == 18).item())
                if storage:
                    self.assertEqual(cache.evict_host(4), 4)
                # Reusing the same allocator slots must still require an actual copy.
                pool.k_buffer[0].fill_(-17)
                pool.v_buffer[0].fill_(-18)
                (ref,) = service.control._reserve_policy_operations(
                    (
                        (
                            "restore",
                            CopyTo(
                                target=target,
                                src=Location(tier=cold_tier),
                                dst=Location(tier="gpu"),
                            ),
                        ),
                    )
                )
                preparation = service.control.executor.prepare_prefix(ref)
                service.control._start_policy_operation(ref)
                loaded = self._wait(service, ref)
                self.assertEqual(loaded.update.completed_steps, ("destination_ready",))
                node = service.control.executor._walk(
                    target.token_ids, service.control.executor._compatibility(target)
                )[-1][0]
                restored = node.component_data[ComponentType.FULL].value
                self.assertTrue(torch.all(pool.k_buffer[0][restored] == 17).item())
                self.assertTrue(torch.all(pool.v_buffer[0][restored] == 18).item())
                self.assertTrue(preparation.ready)
                self.assertEqual(preparation.locked_tokens, 4)
                cache.evict(EvictParams(num_tokens=4))
                self.assertEqual(allocator.available_size(), 60)
                request_lock = cache.lock(node.id)
                try:
                    preparation.release()
                    cache.evict(EvictParams(num_tokens=4))
                    self.assertEqual(allocator.available_size(), 60)
                finally:
                    cache.unlock(request_lock)
                cache.evict(EvictParams(num_tokens=4))
                self.assertEqual(allocator.available_size(), 64)
                self.assertFalse(service.has_pending)
                self.assertFalse(service.blocks_prefill)
                cache.sanity_check()
            finally:
                torch.cuda.synchronize()
                if preparation is not None:
                    preparation.release()
                if storage:
                    cache._storage_attachment.shutdown()
                cache.release_host_resources()

    def _wait(self, service, ref):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            service.progress()
            result = service.control.registry.get(ref)
            if result.status.terminal:
                self.assertEqual(result.status, Status.COMPLETED, result)
                return result
            time.sleep(0.002)
        self.fail("KV transfer did not retire")


if __name__ == "__main__":
    unittest.main()
