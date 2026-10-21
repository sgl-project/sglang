# SPDX-License-Identifier: Apache-2.0
"""Tiny, real HiCache GPU->host MOVE and host->GPU COPY, with byte checks.

PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 python \
    test/manual/entrypoints/programmable_kv_hicache_smoke.py
"""

import argparse
import atexit
import json
import tempfile
import time
from array import array

import torch

from sglang.srt.managers.kv_control.contracts import Location, Status, TokensTarget
from sglang.srt.managers.kv_control.service import KVControlService, ServiceConfig
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage",
        action="store_true",
        help="Verify a real file-backed write and restore after evicting both local tiers",
    )
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1, "Select exactly one GPU"
    with (
        tempfile.TemporaryDirectory(prefix="sglang-kv-storage-smoke-") as storage_dir,
        get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1),
        get_context().override_server_args(
            device="cuda",
            enable_hierarchical_cache=True,
            hicache_ratio=2,
            hicache_mem_layout="layer_first",
            hicache_write_policy="write_back",
            hicache_io_backend="direct",
            disable_overlap_schedule=True,
            hicache_storage_backend="file" if args.storage else None,
            hicache_storage_backend_extra_config=json.dumps(
                {"file_storage_path": storage_dir}
            )
            if args.storage
            else None,
        ),
    ):
        config = ServiceConfig(namespace="gpu-host-smoke", engine_id="smoke")
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
        params = CacheInitParams(
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
        cache = UnifiedRadixCache(params)
        cache.init_hicache(get_server_args(), params)
        atexit.unregister(cache.shutdown)
        try:
            service = KVControlService(cache, config)
            tokens = [1, 2, 3, 4]
            slots = allocator.alloc(4)
            pool.k_buffer[0][slots] = 17
            pool.v_buffer[0][slots] = 18
            node_id = cache.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens)), value=slots.to(torch.int64)
                )
            ).last_device_node
            node = cache.tree_core.node_by_id(node_id)

            def submit(kind, src, dst, message):
                result = service.control.submit(
                    KvHintsEnvelope(
                        protocol_version="0.1",
                        message_id=message,
                        actions=[
                            KvHintAction(
                                action_id="a",
                                action_type="sglang." + kind,
                                action_version="1.0",
                                payload={
                                    "target": {
                                        "kind": "tokens",
                                        "namespace": config.namespace,
                                        "token_ids": tokens,
                                    },
                                    "src": {"tier": src},
                                    "dst": {"tier": dst},
                                },
                            )
                        ],
                    ),
                    service.control.context("smoke"),
                )
                return result.results[0]

            observed_barriers = []

            def wait(result):
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    service.progress()
                    observed_barriers.append(service.blocks_prefill)
                    current = service.control.registry.get(result.ref)
                    if current.status.terminal:
                        assert current.status == Status.COMPLETED, current
                        return current
                    time.sleep(0.002)
                raise TimeoutError("Native HiCache operation did not retire")

            moved = wait(
                submit("move_to", "gpu", "storage" if args.storage else "host", "save")
            )
            assert node.component_data[ComponentType.FULL].value is None
            assert allocator.available_size() == 64
            host_snapshot = service.control.executor.residency(
                service.control.index.resolve(
                    TokensTarget(namespace=config.namespace, token_ids=tuple(tokens))
                ),
                location=Location(tier="host"),
            )
            assert (
                host_snapshot.items and host_snapshot.items[0].location.tier == "host"
            )
            host = cache.host_pool_group.anchor_entry.host_pool
            host_slots = node.component_data[ComponentType.FULL].host_value
            assert torch.all(host.k_buffer[0][host_slots] == 17)
            assert torch.all(host.v_buffer[0][host_slots] == 18)
            if args.storage:
                backend = cache.cache_controller.storage_backend
                for index, key in zip(
                    host_slots, cache.tree_core.get_hash_values(node.id)
                ):
                    expected = host.get_data_page(index).clone()
                    actual = torch.zeros_like(expected)
                    assert backend.get(key, actual) is not None
                    assert torch.equal(actual, expected)
                assert not cache._native_storage_write_results
                assert not service.control.executor.storage.jobs
            # Poison every free device row so a skipped H2D copy cannot pass
            # merely by reusing slots that still contain their old values.
            pool.k_buffer[0].fill_(-17)
            pool.v_buffer[0].fill_(-18)
            if args.storage:
                assert cache.evict_host(4) == 4
            restore = submit(
                "copy_to", "storage" if args.storage else "host", "gpu", "restore"
            )
            if not args.storage:
                assert service.blocks_prefill
            loaded = wait(restore)
            if args.storage:
                target = TokensTarget(
                    namespace=config.namespace, token_ids=tuple(tokens)
                )
                node = service.control.executor._walk(
                    tuple(tokens), service.control.executor._compatibility(target)
                )[-1][0]
            assert any(observed_barriers)
            assert not service.blocks_prefill
            restored = node.component_data[ComponentType.FULL].value
            assert torch.all(pool.k_buffer[0][restored] == 17).item()
            assert torch.all(pool.v_buffer[0][restored] == 18).item()
            assert allocator.available_size() == 60
            assert (
                node.component_data[ComponentType.FULL].lock_ref
                == node.component_data[ComponentType.FULL].host_lock_ref
                == 0
            )
            assert not cache.ongoing_write_through and not cache.ongoing_load_back
            assert (
                not service.control.executor.transfers.jobs
                and not service.control.executor.transfers.interests
            )
            cache.sanity_check()
            torch.cuda.synchronize()
            print(
                json.dumps(
                    {
                        "passed": True,
                        "device": torch.cuda.get_device_name(0),
                        "torch": torch.__version__,
                        "move_steps": moved.update.completed_steps,
                        "copy_steps": loaded.update.completed_steps,
                        "kv_bytes_verified": True,
                        "pool_available": allocator.available_size(),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "io_backend": "direct",
                        "file_storage_bytes_verified": args.storage,
                    }
                )
            )
        finally:
            if args.storage:
                cache._storage_attachment.shutdown()
            cache.release_host_resources()


if __name__ == "__main__":
    main()
