# SPDX-License-Identifier: Apache-2.0
"""Small real-CUDA cache proof. Run with CUDA_VISIBLE_DEVICES selecting ONE GPU.

PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 python \
    test/manual/entrypoints/programmable_kv_gpu_smoke.py

This checks actual device pools/bytes and control, without loading a model.
"""

import json
import time
from array import array

import msgspec
import torch

from sglang.srt.managers.kv_control.service import (
    KVControlService,
    ServiceConfig,
    dispatch,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context


def main():
    assert torch.cuda.device_count() == 1, (
        "Select exactly one GPU with CUDA_VISIBLE_DEVICES"
    )
    started = time.monotonic()
    with get_context().override_server_args():
        config = ServiceConfig(namespace="gpu-smoke", engine_id="smoke-owner")
        pool = MHATokenToKVPool(
            size=64,
            page_size=1,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=64, dtype=torch.float16, device="cuda", kvcache=pool, need_sort=False
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4, max_context_len=32, device="cuda", enable_memory_saver=False
                ),
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                tree_components=(ComponentType.FULL,),
                tree_core_backend="python",
                kv_control_config=config.cache_config(),
            )
        )
        service = KVControlService(cache, config)

        def insert(tokens, value):
            slots = allocator.alloc(len(tokens))
            assert slots is not None and slots.is_cuda
            pool.k_buffer[0][slots] = value
            pool.v_buffer[0][slots] = value + 1
            result = cache.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens)), value=slots.to(torch.int64)
                )
            )
            return cache.tree_core.node_by_id(result.last_device_node), slots

        def length(tokens):
            return cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens)))
            ).device_prefix_len

        keep, slots = insert([1, 2, 3, 4], 7)
        insert([9, 8, 7, 6], 20)
        context = msgspec.to_builtins(service.control.context("smoke"))
        submit = {
            "kind": "submit",
            "context": context,
            "envelope": {
                "protocol_version": "0.1",
                "message_id": "keep-a",
                "actions": [
                    {
                        "action_id": "keep",
                        "action_type": "sglang.set_priority",
                        "action_version": "1.0",
                        "payload": {
                            "target": {
                                "kind": "tokens",
                                "namespace": "gpu-smoke",
                                "token_ids": [1, 2, 3, 4],
                            },
                            "location": {"tier": "gpu"},
                            "claim_id": "a",
                            "revision": 1,
                            "value": 90,
                            "ttl_seconds": 60,
                        },
                    }
                ],
            },
        }
        result = dispatch(service, submit)
        assert (
            result["success"]
            and result["data"]["results"][0]["update"]["status"] == "completed"
        ), result
        cache.evict(EvictParams(num_tokens=4))
        assert length([1, 2, 3, 4]) == 4 and length([9, 8, 7, 6]) == 0
        assert torch.all(pool.k_buffer[0][slots] == 7).item()
        assert torch.all(pool.v_buffer[0][slots] == 8).item()
        assert allocator.available_size() == 60
        lock = cache.lock(keep.id)
        assert cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted == 0
        cache.unlock(lock)
        # Soft priority can still be reclaimed under pressure.
        assert cache.evict(EvictParams(num_tokens=4)).num_tokens_evicted == 4
        assert cache.tree_core.priority_claims.capacity()["copies"] == 0
        insert([1, 2, 3, 4], 30)
        replay = dispatch(service, submit)
        assert replay["data"]["replayed"]
        observed = dispatch(
            service,
            {
                "kind": "residency",
                "context": context,
                "target": submit["envelope"]["actions"][0]["payload"]["target"],
            },
        )
        assert observed["data"]["items"][0]["effective_priority"] == 0
        cache.evict(EvictParams(num_tokens=64))
        assert allocator.available_size() == 64
        cache.sanity_check()
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "passed": True,
                    "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "steps": [
                        "real_cuda_pool",
                        "explicit_priority_victim",
                        "retained_kv_bytes",
                        "request_lock",
                        "soft_priority_reclaim",
                        "copy_lifetime_and_replay",
                        "pool_balance",
                    ],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "seconds": time.monotonic() - started,
                }
            )
        )


if __name__ == "__main__":
    main()
