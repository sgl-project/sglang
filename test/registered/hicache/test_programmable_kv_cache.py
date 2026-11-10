# SPDX-License-Identifier: Apache-2.0
"""Real GPU/host/storage KV round trips and resource reclamation."""

import atexit
import json
import os
import tempfile
import time
import unittest
import uuid
from array import array
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.managers.kv_control.contracts import (
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
from sglang.test.tree_slots import tree_slots_of

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@contextmanager
def mooncake_owner(extra, namespace, engine_id):
    with (
        get_parallel().override(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            attn_dcp_rank=0,
            attn_dcp_size=1,
        ),
        get_context().override_server_args(
            device="cuda",
            enable_hierarchical_cache=True,
            hicache_ratio=2,
            hicache_mem_layout="page_first_direct",
            hicache_write_policy="write_back",
            hicache_io_backend="direct",
            disable_overlap_schedule=True,
            hicache_storage_backend="mooncake",
            hicache_storage_backend_extra_config=json.dumps(extra),
        ),
    ):
        config = ServiceConfig(namespace=namespace, engine_id=engine_id)
        pool = MHATokenToKVPool(
            size=64,
            page_size=1,
            dtype=torch.float16,
            head_num=2,
            head_dim=128,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=64, dtype=torch.float16, device="cuda", kvcache=pool, need_sort=False
        )
        req = ReqToTokenPool(
            size=4, max_context_len=32, device="cuda", enable_memory_saver=False
        )
        allocator.state = req.state
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=req,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            tree_slots=tree_slots_of((ComponentType.FULL,)),
            tree_core_backend="python",
            kv_control_config=config.cache_config(),
        )
        cache = UnifiedRadixCache(params)
        cache.init_hicache(get_server_args(), params)
        atexit.unregister(cache.shutdown)
        try:
            service = KVControlService(cache, config, namespace_keys=True)
            yield cache, service, pool
        finally:
            torch.cuda.synchronize()
            cache._storage_attachment.shutdown()
            cache.release_host_resources()


def public_action(service, kind, target, source, destination, message):
    return service.control.submit(
        KvHintsEnvelope(
            protocol_version="0.1",
            message_id=message,
            actions=[
                KvHintAction(
                    action_id="a",
                    action_type="sglang." + kind,
                    action_version="1.0",
                    payload=dict(
                        target=msgspec.json.decode(msgspec.json.encode(target)),
                        source=source,
                        destination=destination,
                        timeout_ms=10000,
                    ),
                )
            ],
        ),
        service.control.context("test"),
    ).results[0]


def describe(service, target, location):
    from sglang.srt.managers.kv_control.service import Command

    command = msgspec.convert(
        dict(
            kind="storage_targets",
            context=msgspec.to_builtins(service.control.context("test")),
            target=target,
            location=location,
        ),
        type=Command,
    )
    return msgspec.to_builtins(service.execute(command))


class TestProgrammableKVCache(CustomTestCase):
    def test_mooncake_host_demote_reclaims_and_restores_exact_slots(self):
        extra = os.environ.get("KV_CONTROL_MOONCAKE_CONFIG")
        if not extra:
            self.skipTest("requires a private Mooncake master")
        namespace = "host-demote-" + uuid.uuid4().hex
        with mooncake_owner(json.loads(extra), namespace, "host-owner") as owner:
            cache, service, pool = owner
            tokens = tuple(range(1, 9))
            target = TokensTarget(namespace=namespace, token_ids=tokens)
            slots = cache.token_to_kv_pool_allocator.alloc(len(tokens))
            expected = []
            for layer in range(pool.layer_num):
                shape = pool.k_buffer[layer][slots].shape
                values = torch.arange(
                    pool.k_buffer[layer][slots].numel(),
                    device="cuda",
                    dtype=pool.k_buffer[layer].dtype,
                ).reshape(shape)
                pool.k_buffer[layer][slots] = values + layer * 3
                pool.v_buffer[layer][slots] = -values - layer * 5
                expected.append(
                    (
                        pool.k_buffer[layer][slots].clone(),
                        pool.v_buffer[layer][slots].clone(),
                    )
                )
            cache.insert(
                InsertParams(
                    key=RadixKey(
                        array("q", tokens), cache_salt=service.cache_salt(None)
                    ),
                    value=slots.to(torch.int64),
                )
            )
            self._evict_to_host(cache, len(tokens))
            host = cache.host_pool_group.anchor_entry.host_pool
            available = host.available_size()
            location = {"tier": "storage", "backend": "mooncake"}
            saved = self._wait(
                service,
                public_action(
                    service, "demote", target, {"tier": "host"}, location, "store"
                ).ref,
            )
            self.assertTrue(saved.public_result.reclaimed_scope)
            self.assertEqual(host.available_size(), available + len(tokens))
            self.assertFalse(cache.tree_core.root_node.children)
            self._wait(
                service,
                public_action(
                    service, "prefetch", target, location, {"tier": "host"}, "read"
                ).ref,
            )
            path = service.control.executor._walk(
                tokens, service.control.executor._compatibility(target)
            )
            self._load_for_inference(cache, path[-1][0])
            actual = torch.cat(
                [node.component_data[ComponentType.FULL].value for node, *_ in path]
            )
            for layer, (key, value) in enumerate(expected):
                self.assertTrue(torch.equal(pool.k_buffer[layer][actual], key))
                self.assertTrue(torch.equal(pool.v_buffer[layer][actual], value))
            cache.sanity_check()

    def test_public_transfer_roundtrip(self):
        for storage in (False, True):
            with self.subTest(storage=storage):
                self._roundtrip(storage)

    def _roundtrip(self, storage):
        with (
            tempfile.TemporaryDirectory() as storage_dir,
            get_parallel().override(
                tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_dcp_rank=0
            ),
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
                tree_slots=tree_slots_of((ComponentType.FULL,)),
                tree_core_backend="python",
                kv_control_config=config.cache_config(),
            )
            allocator.state = params.req_to_token_pool.state
            cache = UnifiedRadixCache(params)
            cache.init_hicache(get_server_args(), params)
            atexit.unregister(cache.shutdown)
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
                self._evict_to_host(cache, 4)
                self.assertIsNone(node.component_data[ComponentType.FULL].value)
                self.assertEqual(allocator.available_size(), 64)
                host = cache.host_pool_group.anchor_entry.host_pool
                host_slots = node.component_data[ComponentType.FULL].host_value
                self.assertTrue(torch.all(host.k_buffer[0][host_slots] == 17).item())
                self.assertTrue(torch.all(host.v_buffer[0][host_slots] == 18).item())
                if storage:
                    moved = self._wait(
                        service,
                        public_action(
                            service,
                            "demote",
                            target,
                            {"tier": "host"},
                            {"tier": "storage"},
                            "save",
                        ).ref,
                    )
                    self.assertEqual(
                        moved.update.completed_steps,
                        ("destination_ready", "source_reclaimed"),
                    )
                # Reusing the same allocator slots must still require an actual copy.
                pool.k_buffer[0].fill_(-17)
                pool.v_buffer[0].fill_(-18)
                if storage:
                    result = public_action(
                        service,
                        "prefetch",
                        target,
                        {"tier": "storage"},
                        {"tier": "host"},
                        "restore",
                    )
                    loaded = self._wait(service, result.ref)
                    self.assertEqual(
                        loaded.update.completed_steps, ("destination_ready",)
                    )
                node = service.control.executor._walk(
                    target.token_ids, service.control.executor._compatibility(target)
                )[-1][0]
                self._load_for_inference(cache, node)
                restored = node.component_data[ComponentType.FULL].value
                self.assertTrue(torch.all(pool.k_buffer[0][restored] == 17).item())
                self.assertTrue(torch.all(pool.v_buffer[0][restored] == 18).item())
                request_lock = cache.lock(node.id)
                try:
                    cache.evict(EvictParams(num_tokens=4))
                    self.assertEqual(allocator.available_size(), 60)
                finally:
                    cache.unlock(request_lock)
                cache.evict(EvictParams(num_tokens=4))
                self.assertEqual(allocator.available_size(), 64)
                self.assertFalse(service.has_pending)
                cache.sanity_check()
            finally:
                torch.cuda.synchronize()
                if storage:
                    cache._storage_attachment.shutdown()
                cache.release_host_resources()

    @unittest.skipUnless(
        os.environ.get("KV_CONTROL_MOONCAKE_CONFIG"),
        "Set KV_CONTROL_MOONCAKE_CONFIG to live backend JSON",
    )
    def test_live_mooncake_independent_prefetch_and_key_resolution(self):
        extra = json.loads(os.environ["KV_CONTROL_MOONCAKE_CONFIG"])
        namespace = "p2-live-" + uuid.uuid4().hex
        with ExitStack() as stack:
            a, sa, pool_a = stack.enter_context(
                mooncake_owner(extra, namespace, "writer-a")
            )
            b, sb, pool_b = stack.enter_context(
                mooncake_owner(extra, namespace, "reader-b")
            )
            caps_a = sa.capabilities()["public_protocol"]
            caps_b = sb.capabilities()["public_protocol"]
            self.assertEqual(caps_a["storage_backend"], "mooncake")
            self.assertEqual(caps_a["backend_namespace"], caps_b["backend_namespace"])
            compatibility = dict(caps_a["prefix"])
            compatibility.pop("token_material")
            compatibility["cache_salt"] = "tenant-租户"
            target = json.loads(
                json.dumps(
                    dict(
                        kind="prefix",
                        compatibility=compatibility,
                        start_token=0,
                        end_token=4,
                        token_ids=[1, 2, 3, 4],
                    )
                )
            )
            location = dict(
                tier="storage",
                backend="mooncake",
                backend_namespace=caps_a["backend_namespace"],
            )
            desc_a = describe(sa, target, location)
            desc_b = describe(sb, target, location)
            self.assertEqual(desc_a, desc_b)
            self.assertEqual(desc_b["existence"], "unknown")
            keys = [k for page in desc_b["pages"] for k in page["keys"]]
            self.assertEqual(len(keys), 8)  # four pages, K and V for each
            backend = a.cache_controller.storage_backend
            self.assertFalse(any(backend._batch_exist(keys)))
            slots = a.token_to_kv_pool_allocator.alloc(4)
            for layer in range(2):
                pool_a.k_buffer[layer][slots] = 17 + layer
                pool_a.v_buffer[layer][slots] = 27 + layer
            a.insert(
                InsertParams(
                    key=RadixKey(
                        array("q", [1, 2, 3, 4]),
                        cache_salt=sa.cache_salt("tenant-租户"),
                    ),
                    value=slots.to(torch.int64),
                )
            )
            # Ordinary insertion salt and public resolution use the same key chain.
            from sglang.srt.managers.kv_control.contracts import TokensTarget

            ordinary = TokensTarget(
                namespace=namespace, token_ids=(1, 2, 3, 4), cache_salt="tenant-租户"
            )
            path = sa.control.executor._walk(
                ordinary.token_ids, sa.control.executor._compatibility(ordinary)
            )
            self.assertEqual(
                [h for node, *_ in path for h in a.tree_core.get_hash_values(node.id)],
                [p["storage_digest"] for p in desc_a["pages"]],
            )
            writer_available = a.token_to_kv_pool_allocator.available_size()
            self._evict_to_host(a, 4)
            saved = public_action(
                sa, "demote", target, {"tier": "host"}, location, "save"
            )
            saved = self._wait(sa, saved.ref)
            self.assertTrue(saved.public_result.preserved_scope)
            self.assertTrue(saved.public_result.reclaimed_scope)
            self.assertEqual(
                a.token_to_kv_pool_allocator.available_size(), writer_available + 4
            )
            self.assertTrue(
                all(
                    node.component_data[ComponentType.FULL].value is None
                    for node, *_ in path
                )
            )
            self.assertEqual(
                saved.public_result.ready_location.backend_namespace,
                location["backend_namespace"],
            )
            self.assertIsNone(saved.public_result.ready_location.engine_id)
            self.assertTrue(all(v == 1 for v in backend._batch_exist(keys)))
            self.assertFalse(b.tree_core.root_node.children)
            self.assertFalse(sb.control.index.capacity()["continuations"])
            restored = public_action(
                sb,
                "prefetch",
                target,
                location,
                {"tier": "host", "engine_id": "reader-b"},
                "restore",
            )
            self.assertFalse(restored.public_result.ready_scope)
            from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

            demand = CacheRequestHandle("live-coalesced-demand", 0)
            stack.callback(b.release_aborted_request, demand)
            with (
                patch.object(b, "prefetch_threshold", 1),
                patch.object(b, "prefetch_stop_policy", "wait_complete"),
            ):
                b.prefetch_from_storage(
                    demand,
                    b.tree_core.root_node.id,
                    array("q", ordinary.token_ids),
                    matched_prefix_tokens=[],
                    cache_salt=sb.cache_salt("tenant-租户"),
                )
            self.assertFalse(b.check_prefetch_progress(demand))
            self.assertEqual(
                sb.control.executor.storage_reads.stats["demand_coalesced"], 1
            )
            # Joining an exact-prefix operation shares I/O; cancellation drains
            # only that interest while the other still owns the physical job.
            other = public_action(
                sb, "prefetch", target, location, {"tier": "host"}, "shared"
            )
            self.assertEqual(len(sb.control.executor.storage_reads.jobs), 1)
            sb.control.cancel(other.ref)
            done = self._wait(sb, restored.ref)
            self.assertTrue(done.public_result.ready_scope)
            self.assertTrue(b.check_prefetch_progress(demand))
            self.assertEqual(b.pop_prefetch_loaded_span(demand), (4, 0))
            self.assertEqual(done.public_result.ready_location.tier, "host")
            self.assertEqual(
                sb.control.registry.get(other.ref).status, Status.CANCELLED
            )
            nodes = sb.control.executor._walk(
                ordinary.token_ids, sb.control.executor._compatibility(ordinary)
            )
            host = b.host_pool_group.anchor_entry.host_pool
            for node, *_ in nodes:
                cd = node.component_data[ComponentType.FULL]
                self.assertIsNone(cd.value)
                for layer in range(2):
                    self.assertTrue(
                        torch.all(
                            host.k_data_refs[layer][cd.host_value] == 17 + layer
                        ).item()
                    )
                    self.assertTrue(
                        torch.all(
                            host.v_data_refs[layer][cd.host_value] == 27 + layer
                        ).item()
                    )
            # A normal host-to-GPU load after publication verifies usable bytes.
            self._load_for_inference(b, nodes[-1][0])
            for node, *_ in nodes:
                cd = node.component_data[ComponentType.FULL]
                for layer in range(2):
                    self.assertTrue(
                        torch.all(pool_b.k_buffer[layer][cd.value] == 17 + layer).item()
                    )
                    self.assertTrue(
                        torch.all(pool_b.v_buffer[layer][cd.value] == 27 + layer).item()
                    )
            self.assertFalse(sa.has_pending)
            self.assertFalse(sb.has_pending)
            a.sanity_check()
            b.sanity_check()

    def _evict_to_host(self, cache, count):
        # Let the test's DEMOTE perform the first storage write.
        with patch.object(cache, "write_backup_storage"):
            cache.evict(EvictParams(num_tokens=count))

    def _load_for_inference(self, cache, node):
        with patch.object(cache, "load_back_threshold", 1):
            self.assertTrue(cache.load_back(node.id))
        cache.ready_to_load_host_cache()
        torch.cuda.synchronize()
        cache.loading_check()

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
