# SPDX-License-Identifier: Apache-2.0
"""KV copy/move with real CPU pools and file storage; controlled DMA fences."""

import atexit
import json
import threading
import time
import unittest
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from unittest.mock import patch

import msgspec
import test_kv_control_cache as cache_fixtures
import torch

from sglang.srt.managers.kv_control.contracts import (
    Associate,
    Location,
    SessionTarget,
    Status,
)
from sglang.srt.managers.kv_control.service import (
    CacheLimits,
    KVControlService,
    ServiceConfig,
    parse_config,
)
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.l2_transfer import TransferCompletion
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.tree_slots import tree_slots_of

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
FULL = ComponentType.FULL


class HostRegistration:
    """CPU stand-in for the CUDA driver, with paired registration ownership."""

    def __init__(self):
        self.live = {}

    def cudaHostRegister(self, ptr, size, flags):
        assert ptr not in self.live, "Host range registered twice"
        self.live[ptr] = size
        return 0

    def cudaHostUnregister(self, ptr):
        assert ptr in self.live, "Unregistering an unowned host range"
        del self.live[ptr]
        return 0


class Fence:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready

    def synchronize(self):
        assert self.ready, "Prematurely waited for an unfinished transfer"


class ControlledDMA:
    def __init__(self):
        self.pending = []
        self.submitted = []

    def submit(self, direction, transfers, on_layer_done=None):
        event = Fence()
        self.pending.append((direction, transfers, event, on_layer_done))
        self.submitted.append(
            (direction, sum((len(t.device_indices) for t in transfers)))
        )
        return TransferCompletion(Fence(True), event, False)

    def submit_device_to_host(self, transfers):
        return self.submit("save", transfers)

    def submit_host_to_device(
        self, transfers, *, transfer_layer_id_max, start_event=None, on_layer_done=None
    ):
        return self.submit("load", transfers, on_layer_done)

    def finish(self):
        direction, transfers, event, on_layer_done = self.pending.pop(0)
        for transfer in transfers:
            host, device = (transfer.host_pool, transfer.device_pool)
            for layer in range(device.layer_num):
                for h, d in (
                    (host.k_buffer[layer], device.get_key_buffer(layer)),
                    (host.v_buffer[layer], device.get_value_buffer(layer)),
                ):
                    if direction == "save":
                        h[transfer.host_indices] = d[transfer.device_indices]
                    else:
                        d[transfer.device_indices] = h[transfer.host_indices]
                if on_layer_done is not None:
                    on_layer_done(layer)
        event.ready = True


def transfer(
    service,
    *,
    kind="copy_to",
    src="gpu",
    dst="host",
    caller="A",
    message="copy",
    tokens=(1, 2, 3, 4),
    selection=None,
    submission=False,
):
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
                        "target": msgspec.json.decode(
                            msgspec.json.encode(
                                selection or cache_fixtures.target(tokens)
                            )
                        ),
                        "src": {"tier": src},
                        "dst": {"tier": dst},
                    },
                )
            ],
        ),
        service.control.context(caller),
    )
    return result if submission else result.results[0]


def current(service, result):
    return service.control.registry.get(result.ref)


def seed(cache, tokens=(1, 2, 3, 4), value=7, *, salt=None):
    slots = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    assert slots is not None
    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    pool.k_buffer[0][slots] = value
    pool.v_buffer[0][slots] = value + 1
    executor = cache.kv_control_service.control.executor
    compatibility = executor._compatibility(
        cache_fixtures.target(tokens, cache_salt=salt)
    )
    for node, start, _, matched in executor._walk(tokens, compatibility):
        cd = node.component_data[ComponentType.FULL]
        old = cd.value if cd.value is not None else cd.host_value
        if old is not None:
            source = (
                pool
                if cd.value is not None
                else cache.host_pool_group.anchor_entry.host_pool
            )
            length = matched - start
            pool.k_buffer[0][slots[start:matched]] = source.k_buffer[0][old[:length]]
            pool.v_buffer[0][slots[start:matched]] = source.v_buffer[0][old[:length]]
    result = cache.insert(
        cache_fixtures.InsertParams(
            key=cache_fixtures.RadixKey(
                cache_fixtures.array("q", tokens), cache_salt=compatibility.cache_salt
            ),
            value=slots.to(torch.int64),
        )
    )
    return cache.tree_core.node_by_id(result.last_device_node)


def branched_scope(cache, service):
    seed(cache, (1, 2, 3, 4))
    seed(cache, (1, 2, 5, 6), value=11)
    for name, tokens in (("a", (1, 2, 3, 4)), ("b", (1, 2, 5, 6))):
        service.control.index.associate(
            Associate(
                session_id="task",
                continuation_id=name,
                expected_revision=0,
                target=cache_fixtures.target(tokens),
            )
        )
    return SessionTarget(namespace="ns", session_id="task")


@contextmanager
def hicache(
    case, config=None, *, storage=False, namespace_keys=False, write_policy="write_back"
):
    config = config or ServiceConfig(namespace="ns", engine_id="engine")
    storage = storage or namespace_keys
    storage_dir = TemporaryDirectory(prefix="sglang-kv-test-")
    case.addCleanup(storage_dir.cleanup)
    storage_args = (
        {
            "hicache_storage_backend": "file",
            "hicache_storage_backend_extra_config": json.dumps(
                {"file_storage_path": storage_dir.name}
            ),
        }
        if storage
        else {}
    )
    scope = get_context().override_server_args(
        device="cpu",
        enable_hierarchical_cache=True,
        hicache_ratio=2,
        hicache_write_policy=write_policy,
        hicache_mem_layout="layer_first",
        disable_overlap_schedule=True,
        **storage_args,
    )
    scope.install()
    parallel = get_parallel().override(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_dcp_rank=0
    )
    parallel.__enter__()
    registrations = HostRegistration()
    driver = patch.object(torch.cuda, "cudart", return_value=registrations)
    driver.start()
    try:
        cache, _ = case.make(
            session=False, config=config.cache_config(), limits=config.limits
        )
        cache.reset()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=cache.req_to_token_pool,
            token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
            page_size=1,
            tree_slots=tree_slots_of((ComponentType.FULL,)),
            kv_control_config=config.cache_config(),
        )
        cache.init_hicache(get_server_args(), params)
        host_pool = cache.host_pool_group.anchor_entry.host_pool
        registered = host_pool.pin_memory
        if storage:
            # File I/O uses CPU staging tensors; only the backing pool is
            # registered with the simulated driver. Restore its ownership flag
            # before destroy so real cleanup releases those registrations.
            host_pool.pin_memory = False
        atexit.unregister(cache.shutdown)
        dma = ControlledDMA()
        cache.cache_controller.l2_transfer_engine = dma
        service = KVControlService(
            cache,
            config,
            clock=case.clock,
            namespace_keys=namespace_keys,
            idle_check=(lambda: True) if namespace_keys else None,
        )
        try:
            yield (case, cache, service, dma)
            assert not service.has_pending, "Test must retire every operation"
            assert not cache.ongoing_write_through and (not cache.ongoing_load_back)
        finally:
            for pending in tuple(service.control._pending.values()):
                service.control.cancel(pending.ref)
            service.progress()
            while dma.pending:
                dma.finish()
            service.progress()
            if storage:
                cache._storage_attachment.shutdown()
            host_pool.pin_memory = registered
            # The registered driver above is simulated even with all CUDA
            # devices hidden; exercise its matching unregister path too.
            with patch("sglang.srt.mem_cache.pool_host.base._is_cuda", True):
                cache.release_host_resources()
            assert not registrations.live, "Host registrations leaked during teardown"
    finally:
        driver.stop()
        parallel.__exit__(None, None, None)
        scope.restore()


def public_transfer(
    service,
    kind,
    source,
    destination,
    message,
    *,
    tokens=(1, 2, 3, 4),
    caller="router",
    salt=None,
):
    from sglang.srt.managers.kv_control.contracts import HASH_SCHEME

    caps = service.capabilities()["public_protocol"]
    compatibility = msgspec.json.decode(msgspec.json.encode(caps["prefix"]))
    compatibility.pop("token_material")
    compatibility["cache_salt"] = salt
    assert compatibility["hash_scheme"] == HASH_SCHEME
    return service.control.submit(
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
                            "kind": "prefix",
                            "compatibility": compatibility,
                            "start_token": 0,
                            "end_token": len(tokens),
                            "token_ids": list(tokens),
                        },
                        "source": {"tier": source},
                        "destination": {"tier": destination},
                        "timeout_ms": 5000,
                    },
                )
            ],
        ),
        service.control.context(caller),
    ).results[0]


class TestKVControlTransfers(cache_fixtures.KVCacheTestCase):
    def test_public_demote_then_prefetch_observe_actual_fences(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            initial = cache.token_to_kv_pool_allocator.available_size()
            result = public_transfer(service, "demote", "gpu", "host", "demote")
            assert result.status == Status.RUNNING, result
            assert not result.public_result.ready_scope
            assert result.public_result.ready_location is None
            assert not result.public_result.reclaimed_scope
            assert node.component_data[FULL].value is not None
            service.progress()
            dma.finish()
            service.progress()
            done = current(service, result)
            assert done.status == Status.COMPLETED, done
            assert done.public_result.preserved_scope
            assert done.public_result.reclaimed_scope
            assert node.component_data[FULL].value is None
            assert cache.token_to_kv_pool_allocator.available_size() == initial + 4
            result = public_transfer(service, "prefetch", "host", "gpu", "prefetch")
            assert result.status == Status.RUNNING, result
            assert not result.public_result.ready_scope
            service.progress()
            dma.finish()
            service.progress()
            done = current(service, result)
            assert done.status == Status.COMPLETED, done
            assert done.public_result.ready_scope
            assert done.public_result.ready_location.tier == "gpu"
            assert not done.public_result.preserved_scope
            assert not done.public_result.reclaimed_scope
            cd = node.component_data[FULL]
            pool = cache.token_to_kv_pool_allocator.get_kvcache()
            assert torch.all(pool.k_buffer[0][cd.value] == 7)
            assert cd.lock_ref == cd.host_lock_ref == 0
            cache.sanity_check()

    def test_demote_retains_live_ancestors_and_reclaims_only_eligible_ranges(self):
        for destination in ("host", "storage"):
            for blocker in ("active_reference", "structural_dependency"):
                with (
                    self.subTest(destination=destination, blocker=blocker),
                    hicache(self, storage=destination == "storage") as owner,
                ):
                    _, cache, service, dma = owner
                    ancestor = seed(cache, (1, 2))
                    leaf = seed(cache)
                    sibling = (
                        seed(cache, (1, 2, 5, 6))
                        if blocker == "structural_dependency"
                        else None
                    )
                    lock = (
                        cache.lock(ancestor.id)
                        if blocker == "active_reference"
                        else None
                    )
                    initial = cache.token_to_kv_pool_allocator.available_size()
                    try:
                        result = public_transfer(
                            service, "demote", "gpu", destination, "partial"
                        )
                        stage(service, dma)
                        done = wait(service, result)
                        self.assertEqual(done.status, Status.PARTIAL)
                        self.assertEqual(
                            cache.token_to_kv_pool_allocator.available_size(),
                            initial + 2,
                        )
                        self.assertIsNone(leaf.component_data[FULL].value)
                        self.assertIsNotNone(ancestor.component_data[FULL].value)
                        self.assertEqual(
                            [
                                (o.coverage.start_token, o.coverage.end_token, o.code)
                                for o in done.public_result.blocked_scope
                            ],
                            [(0, 2, blocker)],
                        )
                        self.assertEqual(
                            [
                                (o.coverage.start_token, o.coverage.end_token)
                                for o in done.public_result.reclaimed_scope
                            ],
                            [(2, 4)],
                        )
                        self.assertEqual(
                            sum(
                                o.coverage.end_token - o.coverage.start_token
                                for o in done.public_result.preserved_scope
                            ),
                            4,
                        )
                        if sibling is not None:
                            self.assertIsNotNone(sibling.component_data[FULL].value)
                        else:
                            cache.unlock(lock)
                            lock = None
                            submissions = list(dma.submitted)
                            retry = public_transfer(
                                service,
                                "demote",
                                "gpu",
                                destination,
                                "retry",
                                tokens=(1, 2),
                            )
                            self.assertEqual(
                                wait(service, retry).status, Status.COMPLETED
                            )
                            self.assertEqual(
                                dma.submitted, submissions
                            )  # existing host copy
                            self.assertEqual(
                                cache.token_to_kv_pool_allocator.available_size(),
                                initial + 4,
                            )
                        self.assertEqual(
                            current(service, result), done
                        )  # immutable historical outcome
                        cache.sanity_check()
                    finally:
                        if lock is not None:
                            cache.unlock(lock)

    def test_demote_does_not_wait_for_foreign_load_or_cancel_it(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            saved = transfer(service, kind="move_to")
            stage(service, dma)
            self.assertEqual(wait(service, saved).status, Status.COMPLETED)
            loaded = transfer(service, src="host", dst="gpu", message="load")
            service.progress()
            initial = cache.token_to_kv_pool_allocator.available_size()
            result = public_transfer(service, "demote", "gpu", "host", "blocked")
            self.assertEqual(result.status, Status.PARTIAL)
            self.assertEqual(
                {o.code for o in result.public_result.blocked_scope}, {"inflight_load"}
            )
            self.assertTrue(result.public_result.preserved_scope)
            self.assertFalse(result.public_result.reclaimed_scope)
            self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), initial)
            self.assertEqual(current(service, loaded).status, Status.RUNNING)
            dma.finish()
            service.progress()
            self.assertEqual(current(service, loaded).status, Status.COMPLETED)
            retry = public_transfer(service, "demote", "gpu", "host", "retry")
            self.assertEqual(wait(service, retry).status, Status.COMPLETED)
            self.assertIsNone(node.component_data[FULL].value)

    def test_cancelled_shared_demote_keeps_published_scope_and_other_demand(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            seed(cache, (1, 2))
            seed(cache)
            copied = transfer(service, tokens=(1, 2))
            stage(service, dma)
            self.assertEqual(wait(service, copied).status, Status.COMPLETED)
            a = public_transfer(service, "demote", "gpu", "host", "a", caller="A")
            b = public_transfer(service, "demote", "gpu", "host", "b", caller="B")
            self.assertEqual(len(service.control.executor.transfers.jobs), 1)
            service.progress()
            self.assertEqual(len(dma.pending), 1)
            cancelled = service.control.cancel(a.ref)
            self.assertEqual(cancelled.status, Status.CANCELLED)
            self.assertEqual(
                [
                    (o.coverage.start_token, o.coverage.end_token)
                    for o in cancelled.public_result.preserved_scope
                ],
                [(0, 2)],
            )
            self.assertFalse(cancelled.public_result.reclaimed_scope)
            self.assertFalse(cancelled.public_result.missing_scope)
            dma.finish()
            done = wait(service, b)
            self.assertEqual(done.status, Status.COMPLETED)
            self.assertTrue(done.public_result.reclaimed_scope)
            self.assertEqual(
                cache.token_to_kv_pool_allocator.available_size(),
                cache.token_to_kv_pool_allocator.size,
            )
            self.assertEqual(current(service, a), cancelled)
            cache.sanity_check()

    def test_failed_public_host_backup_preserves_ready_prefix_and_all_source(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            ancestor = seed(cache, (1, 2))
            leaf = seed(cache)
            initial = cache.token_to_kv_pool_allocator.available_size()
            original = cache.cache_controller.write
            calls = 0

            def fail_tail(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original(*args, **kwargs) if calls == 1 else None

            with patch.object(cache.cache_controller, "write", fail_tail):
                result = public_transfer(service, "demote", "gpu", "host", "failed")
            service.progress()
            self.assertFalse(current(service, result).update.quiescent)
            dma.finish()
            done = wait(service, result)
            self.assertEqual(done.status, Status.FAILED)
            self.assertEqual(
                [
                    (o.coverage.start_token, o.coverage.end_token)
                    for o in done.public_result.preserved_scope
                ],
                [(0, 2)],
            )
            self.assertEqual(
                [
                    (o.coverage.start_token, o.coverage.end_token)
                    for o in done.public_result.missing_scope
                ],
                [(2, 4)],
            )
            self.assertEqual(
                {o.code for o in done.public_result.blocked_scope}, {"capacity"}
            )
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), initial)
            self.assertIsNotNone(ancestor.component_data[FULL].value)
            self.assertIsNotNone(leaf.component_data[FULL].value)
            self.assertEqual(cache.tree_core.full_protected_size(), 0)
            cache.sanity_check()

    def test_cancelled_demote_drains_backup_and_preserves_source(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            initial = cache.token_to_kv_pool_allocator.available_size()
            result = public_transfer(service, "demote", "gpu", "host", "cancel")
            service.progress()
            cancelled = service.control.cancel(result.ref)
            self.assertFalse(cancelled.update.quiescent)
            self.assertFalse(cancelled.public_result.ready_scope)
            self.assertGreater(node.component_data[FULL].lock_ref, 0)
            dma.finish()
            done = wait(service, result)
            self.assertEqual(done.status, Status.CANCELLED)
            self.assertEqual(done.update.completed_steps, ("destination_ready",))
            self.assertTrue(done.public_result.preserved_scope)
            self.assertEqual(
                {o.code for o in done.public_result.blocked_scope}, {"cancelled"}
            )
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), initial)
            self.assertIsNotNone(node.component_data[FULL].value)
            self.assertEqual(node.component_data[FULL].lock_ref, 0)

    def test_copy_waits_for_fence_preserves_bytes_and_pool_ownership(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            initial = cache.token_to_kv_pool_allocator.available_size()
            result = transfer(service)
            assert result.status == Status.RUNNING
            service.progress()
            assert current(service, result).status == Status.RUNNING
            assert node.component_data[ComponentType.FULL].lock_ref > 0
            assert node not in cache.tree_core.evictable_device_leaves
            assert cache.tree_core.full_protected_size() == 4
            assert len(dma.pending) == 1
            dma.finish()
            service.progress()
            assert current(service, result).status == Status.COMPLETED, current(
                service, result
            )
            cd = node.component_data[ComponentType.FULL]
            host = cache.host_pool_group.anchor_entry.host_pool
            assert torch.all(host.k_buffer[0][cd.host_value] == 7)
            assert cd.value is not None and cd.lock_ref == cd.host_lock_ref == 0
            assert cache.token_to_kv_pool_allocator.available_size() == initial
            assert not service.control.executor.transfers.interests
            snapshot = service.control.executor.residency(
                service.control.index.resolve(cache_fixtures.target()),
                location=Location(tier="host"),
            )
            assert snapshot.items[0].location.tier == "host"
            assert snapshot.items[0].lock_ref == 0
            cache.sanity_check()

    def test_shared_waiter_cancel_does_not_cancel_other_interest_or_release_dma(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            first = transfer(service, caller="A")
            second = transfer(service, caller="B")
            service.progress()
            service.control.cancel(first.ref)
            assert current(service, first).status == Status.CANCELLED
            assert node.component_data[ComponentType.FULL].lock_ref > 0
            assert len(dma.submitted) == 1
            service.control.cancel(second.ref)
            assert current(service, second).status == Status.CANCELLING
            assert node.component_data[ComponentType.FULL].lock_ref > 0
            dma.finish()
            service.progress()
            assert current(service, second).status == Status.CANCELLED
            assert node.component_data[ComponentType.FULL].lock_ref == 0
            assert not service.control.executor.transfers.jobs
            cache.sanity_check()

    def test_host_load_holds_whole_prefix_and_blocks_prefill_until_fence(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            saved = transfer(service, kind="move_to")
            service.progress()
            dma.finish()
            service.progress()
            assert current(service, saved).status == Status.COMPLETED
            pool = cache.token_to_kv_pool_allocator.get_kvcache()
            pool.k_buffer[0].fill_(-7)
            pool.v_buffer[0].fill_(-8)
            loaded = transfer(service, src="host", dst="gpu", message="load")
            assert service.blocks_prefill
            service.progress()
            assert current(service, loaded).status == Status.RUNNING
            assert node.component_data[ComponentType.FULL].value is not None
            assert node.component_data[ComponentType.FULL].lock_ref > 0
            assert service.blocks_prefill
            dma.finish()
            service.progress()
            assert not service.blocks_prefill
            assert current(service, loaded).status == Status.COMPLETED, current(
                service, loaded
            )
            slots = node.component_data[ComponentType.FULL].value
            pool = cache.token_to_kv_pool_allocator.get_kvcache()
            assert torch.all(pool.k_buffer[0][slots] == 7)
            assert torch.all(pool.v_buffer[0][slots] == 8)
            assert cache.load_back_threshold == 10
            cache.sanity_check()

    def test_shared_prefix_whole_scope_capacity_is_charged_once(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            selection = branched_scope(cache, service)
            assert cache.token_to_kv_pool_allocator.available_size() == 58
            saved = transfer(service, kind="move_to", selection=selection)
            service.progress()
            dma.finish()
            service.progress()
            assert current(service, saved).status == Status.COMPLETED, current(
                service, saved
            )
            assert cache.token_to_kv_pool_allocator.available_size() == 64
            other = cache.token_to_kv_pool_allocator.alloc(58)
            loaded = transfer(
                service, src="host", dst="gpu", selection=selection, message="load"
            )
            assert loaded.status == Status.RUNNING, loaded
            assert cache.token_to_kv_pool_allocator.available_size() == 0
            service.progress()
            assert dma.submitted == [("save", 6), ("load", 6)]
            assert cache.tree_core.full_protected_size() == 6
            dma.finish()
            service.progress()
            assert current(service, loaded).status == Status.COMPLETED, current(
                service, loaded
            )
            assert cache.tree_core.full_protected_size() == 0
            cache.token_to_kv_pool_allocator.free(other)
            cache.sanity_check()

    def test_insufficient_whole_scope_defers_without_reserving_or_queuing(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            selection = branched_scope(cache, service)
            saved = transfer(service, kind="move_to", selection=selection)
            service.progress()
            dma.finish()
            service.progress()
            assert current(service, saved).status == Status.COMPLETED
            other = cache.token_to_kv_pool_allocator.alloc(60)
            loaded = transfer(
                service, src="host", dst="gpu", selection=selection, message="load"
            )
            assert loaded.status == Status.QUEUED, loaded
            service.progress()
            assert not cache.cache_controller.load_queue and (not dma.pending)
            assert cache.token_to_kv_pool_allocator.available_size() == 4
            assert cache.tree_core.full_protected_size() == 0
            for node in cache.tree_core._node_arena.values():
                if node is not cache.tree_core.root_node:
                    cd = node.component_data[ComponentType.FULL]
                    assert cd.lock_ref == cd.host_lock_ref == 0
            assert service.control.cancel(loaded.ref).status == Status.CANCELLED
            cache.token_to_kv_pool_allocator.free(other)

    def test_partial_submission_failure_keeps_cause_after_cancel_ack_and_retry(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            seed(cache, (1, 2), value=7)
            seed(cache, (1, 2, 3, 4), value=11)
            controller = cache.cache_controller
            original = controller.write
            calls = 0

            def write(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original(*args, **kwargs) if calls == 1 else None

            with patch.object(controller, "write", side_effect=write):
                failed = transfer(service)
            assert failed.status == Status.CANCELLING, failed
            assert failed.update.code == "capacity"
            service.control.cancel(failed.ref)
            service.progress()
            assert current(service, failed).status == Status.CANCELLING
            assert cache.tree_core.full_protected_size() == 4
            dma.finish()
            service.progress()
            result = current(service, failed)
            assert (
                result.status == Status.FAILED and result.update.code == "capacity"
            ), result
            assert any(
                (
                    o.completed_steps == ("destination_ready",)
                    for o in result.update.target_outcomes
                )
            )
            acknowledgement = service.control.registry.acknowledge(
                failed.ref.context, failed.ref.message_id
            )
            assert acknowledgement.summaries[0].status == Status.FAILED
            replay = transfer(service, submission=True)
            assert replay.replayed and replay.acknowledged
            assert (
                replay.summaries[0].status == Status.FAILED
                and replay.summaries[0].code == "capacity"
            )
            retried = transfer(service, message="retry")
            service.progress()
            dma.finish()
            service.progress()
            assert current(service, retried).status == Status.COMPLETED, current(
                service, retried
            )
            assert cache.tree_core.full_protected_size() == 0
            cache.sanity_check()

    def test_native_load_preserves_existing_running_growth_budget_without_double_charge(
        self,
    ):
        with hicache(self) as owner:
            from array import array
            from types import SimpleNamespace

            from sglang.srt.managers.schedule_batch import Req
            from sglang.srt.managers.scheduler import Scheduler
            from sglang.srt.sampling.sampling_params import SamplingParams

            _, cache, service, dma = owner
            selection = branched_scope(cache, service)
            saved = transfer(service, kind="move_to", selection=selection)
            service.progress()
            dma.finish()
            service.progress()
            assert current(service, saved).status == Status.COMPLETED
            other = cache.token_to_kv_pool_allocator.alloc(56)
            running = Req(
                rid="running",
                origin_input_text=None,
                origin_input_ids=array("q", [99]),
                sampling_params=SamplingParams(max_new_tokens=8, ignore_eos=True),
                vocab_size=128,
            )
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.tree_cache = cache
            scheduler.token_to_kv_pool_allocator = cache.token_to_kv_pool_allocator
            scheduler.running_batch = SimpleNamespace(reqs=[running])
            scheduler.chunked_req = running
            scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.01)
            service.control.executor.gpu_copy_capacity = (
                scheduler._kv_control_gpu_copy_capacity
            )
            assert scheduler._kv_control_gpu_copy_capacity() == 0
            loaded = transfer(
                service, src="host", dst="gpu", selection=selection, message="load"
            )
            assert (
                loaded.status == Status.QUEUED
                and loaded.update.code == "whole_scope_capacity"
            )
            cache.token_to_kv_pool_allocator.free(other[:6])
            service.progress()
            assert current(service, loaded).status == Status.RUNNING, current(
                service, loaded
            )
            assert cache.token_to_kv_pool_allocator.available_size() == 8
            dma.finish()
            service.progress()
            assert current(service, loaded).status == Status.COMPLETED, current(
                service, loaded
            )
            cache.token_to_kv_pool_allocator.free(other[6:])
            cache.sanity_check()


def wait(service, result):
    deadline = time.monotonic() + 5
    while not current(service, result).status.terminal:
        service.progress()
        assert time.monotonic() < deadline, current(service, result)
        time.sleep(0.005)
    return current(service, result)


def stage(service, dma):
    service.progress()
    if dma.pending:
        dma.finish()
    service.progress()


def assert_files(cache, node):
    cc = cache.cache_controller
    pool = cache.host_pool_group.anchor_entry.host_pool
    indices = node.component_data[FULL].host_value
    hashes = cache.tree_core.get_hash_values(node.id)
    for i, key in enumerate(hashes):
        expected = pool.get_data_page(indices[i]).clone()
        actual = torch.zeros_like(expected)
        assert cc.storage_backend.get(key, actual) is not None
        assert torch.equal(actual, expected)


class TestKVControlStorage(cache_fixtures.KVCacheTestCase):
    def test_demote_classifies_foreign_storage_backup_without_waiting(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            original = cc.page_set_func

            def blocked(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return original(*args, **kwargs)

            try:
                with patch.object(cc, "page_set_func", blocked):
                    copied = transfer(service, dst="storage")
                    stage(service, dma)
                    self.assertTrue(entered.wait(5))
                    result = public_transfer(service, "demote", "gpu", "host", "demote")
                    service.progress()
                    done = current(service, result)
                    self.assertEqual(done.status, Status.PARTIAL)
                    self.assertEqual(
                        {o.code for o in done.public_result.blocked_scope},
                        {"inflight_backup"},
                    )
                    self.assertTrue(done.public_result.preserved_scope)
                    self.assertFalse(done.public_result.reclaimed_scope)
                    self.assertIsNotNone(node.component_data[FULL].value)
                    self.assertEqual(current(service, copied).status, Status.RUNNING)
                    release.set()
                    self.assertEqual(wait(service, copied).status, Status.COMPLETED)
            finally:
                release.set()
            retried = public_transfer(service, "demote", "gpu", "storage", "retry")
            # Native file backend must confirm existing copies without rewriting.
            with patch(
                "sglang.srt.mem_cache.hicache_storage.os.replace",
                side_effect=AssertionError("redundant write"),
            ):
                self.assertEqual(wait(service, retried).status, Status.COMPLETED)
            self.assertIsNone(node.component_data[FULL].value)
            assert_files(cache, node)

    def test_cancelled_storage_demote_keeps_confirmed_backup_without_reclaiming(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            initial = cache.token_to_kv_pool_allocator.available_size()
            entered, release = threading.Event(), threading.Event()
            cc = cache.cache_controller
            original = cc.page_set_func

            def blocked(*args, **kwargs):
                success = original(*args, **kwargs)
                entered.set()
                assert release.wait(5)
                return success

            try:
                with patch.object(cc, "page_set_func", blocked):
                    result = public_transfer(
                        service, "demote", "gpu", "storage", "cancel"
                    )
                    stage(service, dma)
                    self.assertTrue(entered.wait(5))
                    cancelling = service.control.cancel(result.ref)
                    self.assertFalse(cancelling.update.quiescent)
                    self.assertFalse(cancelling.public_result.preserved_scope)
                    self.assertGreater(node.component_data[FULL].host_lock_ref, 0)
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            self.assertEqual(done.status, Status.CANCELLED)
            self.assertTrue(done.public_result.preserved_scope)
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertEqual(
                {o.code for o in done.public_result.blocked_scope}, {"cancelled"}
            )
            self.assertEqual(done.update.completed_steps, ("destination_ready",))
            self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), initial)
            self.assertEqual(node.component_data[FULL].lock_ref, 0)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            assert_files(cache, node)

    def test_public_demote_partial_storage_ack_preserves_source_and_scopes(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            cc = cache.cache_controller
            original = cc.page_set_func
            calls = 0

            def partial(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original(*args, **kwargs) if calls == 1 else False

            initial = cache.token_to_kv_pool_allocator.available_size()
            with (
                patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 2),
                patch.object(cc, "page_set_func", partial),
            ):
                result = public_transfer(service, "demote", "gpu", "storage", "partial")
                stage(service, dma)
                done = wait(service, result)
            self.assertEqual(done.status, Status.FAILED)
            self.assertEqual(
                [
                    (o.coverage.start_token, o.coverage.end_token)
                    for o in done.public_result.preserved_scope
                ],
                [(0, 2)],
            )
            self.assertEqual(
                [
                    (o.coverage.start_token, o.coverage.end_token)
                    for o in done.public_result.missing_scope
                ],
                [(2, 4)],
            )
            self.assertEqual(
                {o.code for o in done.public_result.blocked_scope},
                {"incomplete_storage_write"},
            )
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), initial)
            self.assertIsNotNone(node.component_data[FULL].value)
            self.assertEqual(node.component_data[FULL].lock_ref, 0)

    def test_storage_namespace_description_and_backend_mismatch(self):
        from pathlib import Path

        from sglang.srt.managers.kv_control.contracts import ControlError
        from sglang.srt.managers.kv_control.service import Command
        from sglang.srt.mem_cache.unified_cache.storage_execution import (
            storage_execution_adapter,
        )

        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, dma = owner
            caps = service.capabilities()["public_protocol"]
            location = dict(
                tier="storage",
                backend="file",
                backend_namespace=caps["backend_namespace"],
            )
            target = dict(kind="tokens", namespace="ns", token_ids=[1, 2, 3, 4])
            command = dict(
                kind="storage_targets",
                context=msgspec.to_builtins(service.control.context("A")),
                target=target,
                location=location,
            )
            before = service.control.registry.capacity()
            description = service.execute(msgspec.convert(command, type=Command))
            self.assertEqual(description["existence"], "unknown")
            self.assertFalse(cache.tree_core.root_node.children)
            self.assertEqual(before, service.control.registry.capacity())
            paths = [Path(key) for page in description["pages"] for key in page["keys"]]
            self.assertFalse(any(path.exists() for path in paths))
            for changes in (
                {"backend_namespace": "different"},
                {"backend": "mooncake"},
            ):
                with self.assertRaises(ControlError):
                    service.execute(
                        msgspec.convert(
                            dict(command, location=dict(location, **changes)),
                            type=Command,
                        )
                    )
            seed(cache)
            result = transfer(service, dst="storage")
            service.progress()
            dma.finish()
            done = wait(service, result)
            self.assertEqual(done.status, Status.COMPLETED)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertEqual(
                description, service.execute(msgspec.convert(command, type=Command))
            )
            self.assertIsNone(done.update.target_outcomes[0].location.engine_id)
            self.assertEqual(
                done.update.target_outcomes[0].location.backend_namespace,
                location["backend_namespace"],
            )
            with patch.object(
                cache.cache_controller, "storage_backend_type", "unregistered"
            ):
                adapter, reason = storage_execution_adapter(cache)
                self.assertIsNone(adapter)
                self.assertTrue(reason)

    def test_mooncake_short_object_read_and_write_receipts_are_not_complete_pages(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        with hicache(self, storage=True) as owner:
            _, cache, _, _ = owner
            # Real writer key/buffer metadata; only the external SDK response is controlled.
            store = MooncakeStore.__new__(MooncakeStore)
            store.mem_pool_host = cache.host_pool_group.anchor_entry.host_pool
            store.storage_config = cache.cache_controller.storage_config
            store.config_prefix = "test"
            store.is_mla_backend = False
            store.should_split_heads = False
            store.mha_suffix = "0"
            store.dcp_size = 1
            store.enable_storage_metrics = False
            store._use_group_semantics = False
            indices = torch.arange(store.mem_pool_host.page_size, dtype=torch.int64)
            keys, _, sizes = store._batch_preprocess(
                store._tag_keys(["digest"]), indices
            )
            self.assertEqual(keys, ["test_digest_0_k", "test_digest_0_v"])
            for receipt, expected in (
                (sizes, [True]),
                ([sizes[0], sizes[1] - 1], [False]),
                ([sizes[0]], [False]),
                ([], [False]),
            ):
                with patch.object(
                    store, "_get_batch_zero_copy_impl", return_value=receipt
                ):
                    self.assertEqual(store.batch_get_v1(["digest"], indices), expected)
            with (
                patch.object(store, "_batch_exist", return_value=[0, 0]),
                patch.object(store, "_put_batch_zero_copy_impl", return_value=[0]),
            ):
                self.assertEqual(store.batch_set_v1(["digest"], indices), [False])
            with (
                patch.object(store, "_batch_exist", return_value=[1, 1]),
                patch.object(
                    store,
                    "_put_batch_zero_copy_impl",
                    side_effect=AssertionError(
                        "existing objects must not be rewritten"
                    ),
                ),
            ):
                self.assertEqual(store.batch_set_v1(["digest"], indices), [True])

    def test_copy_and_move_wait_for_storage_worker_not_enqueue(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            cc = cache.cache_controller
            entered, release = (threading.Event(), threading.Event())
            original = cc.page_set_func

            def blocked(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return original(*args, **kwargs)

            try:
                with patch.object(cc, "page_set_func", blocked):
                    result = transfer(service, dst="storage", kind="move_to")
                    stage(service, dma)
                    assert entered.wait(5)
                    for _ in range(3):
                        service.progress()
                    assert current(service, result).status == Status.RUNNING
                    cd = node.component_data[FULL]
                    assert (
                        cd.value is not None
                        and cd.lock_ref > 0
                        and (cd.host_lock_ref > 0)
                    )
                    assert not current(service, result).update.completed_steps
                    assert cache._storage_attachment.detach()[0] is False
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            assert done.status == Status.COMPLETED, done
            assert done.update.completed_steps == (
                "destination_ready",
                "source_reclaimed",
            )
            assert node.component_data[FULL].value is None
            assert_files(cache, node)
            assert not service.control.executor.storage.jobs
            assert not cache._native_storage_write_results
            assert cd.lock_ref == cd.host_lock_ref == 0

    def test_short_ack_preserves_source_and_actual_prefix_and_failure_on_ack(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            cc = cache.cache_controller
            original = cc.page_set_func
            count = 0

            def fail_second(*args, **kwargs):
                nonlocal count
                count += 1
                return original(*args, **kwargs) if count == 1 else False

            with (
                patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 2),
                patch.object(cc, "page_set_func", fail_second),
            ):
                result = transfer(service, dst="storage", kind="move_to")
                stage(service, dma)
                done = wait(service, result)
            assert (
                done.status == Status.FAILED
                and done.update.code == "incomplete_storage_write"
            ), done
            assert node.component_data[FULL].value is not None
            assert [
                (o.coverage.start_token, o.coverage.end_token, o.completed_steps)
                for o in done.update.target_outcomes
            ] == [(0, 2, ("destination_ready",)), (2, 4, ())]
            acknowledged = service.control.registry.acknowledge(
                result.ref.context, result.ref.message_id
            )
            assert acknowledged.summaries[0].status == Status.FAILED
            assert acknowledged.summaries[0].code == done.update.code
            retry = transfer(service, dst="storage", kind="move_to", message="retry")
            assert wait(service, retry).status == Status.COMPLETED
            assert_files(cache, node)


def cold_prefix(owner):
    _, cache, service, dma = owner
    seed(cache)
    result = transfer(service, kind="move_to", dst="storage")
    stage(service, dma)
    assert wait(service, result).status == Status.COMPLETED
    assert cache.evict_host(4) == 4
    assert not service.control.executor._walk(
        (1, 2, 3, 4), service.control.executor._compatibility(cache_fixtures.target())
    )
    return (cache, service, dma)


def wait_for(service, condition):
    deadline = time.monotonic() + 5
    while not condition():
        service.progress()
        assert time.monotonic() < deadline
        time.sleep(0.005)


def prefix_path(cache, service):
    return service.control.executor._walk(
        (1, 2, 3, 4), service.control.executor._compatibility(cache_fixtures.target())
    )


def assert_retired(cache, service):
    assert not service.control.executor.storage_reads.jobs
    assert not service.control.executor.storage_reads.interests
    assert not service.control.executor.storage_reads.demands
    assert not service.control.executor.storage_reads._joinable
    assert not cache.ongoing_prefetch
    assert not cache.prefetch_loaded_tokens_by_reqid
    assert not cache.prefetch_loaded_storage_start_by_reqid
    assert not cache._storage_prefetch_hit_remaining_by_reqid
    assert cache.cache_controller.prefetch_tokens_occupied == 0
    assert cache.cache_controller.host_mem_release_queue.empty()


class TestKVControlStorageReads(cache_fixtures.KVCacheTestCase):
    def _cold_scopes(self, owner, scopes):
        _, cache, service, dma = owner
        for index, target in enumerate(scopes):
            seed(cache, target.token_ids, value=7 + index * 2, salt=target.cache_salt)
            result = transfer(
                service,
                kind="move_to",
                dst="storage",
                selection=target,
                message=f"publish-{index}",
            )
            stage(service, dma)
            self.assertEqual(wait(service, result).status, Status.COMPLETED)
            self.assertEqual(
                cache.evict_host(len(target.token_ids)), len(target.token_ids)
            )

    def _restore(self, service, target, message, *, dst="host"):
        return transfer(
            service,
            src="storage",
            dst=dst,
            selection=target,
            message=message,
        )

    def _ordinary_prefetch(self, cache, service, target, request):
        cache.prefetch_from_storage(
            request,
            cache.tree_core.root_node.id,
            cache_fixtures.array("q", target.token_ids),
            matched_prefix_tokens=[],
            cache_salt=service.cache_salt(target.cache_salt),
        )

    def _consume_ordinary_prefetch(self, cache, service, request):
        def ready():
            cache.drain_storage_control_queues()
            return cache.check_prefetch_progress(request)

        wait_for(service, ready)
        span = cache.pop_prefetch_loaded_span(request)
        cache.discard_storage_prefetch_accounting(request)
        service.progress()
        return span

    def test_read_observations_count_physical_jobs_and_include_ordinary_reads(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [
                cache_fixtures.target(cache_salt=f"observed-{i}") for i in range(2)
            ]
            self._cold_scopes(owner, scopes)
            cc = cache.cache_controller
            get = cc.page_get_func
            entered, release = threading.Event(), threading.Event()
            demand = CacheRequestHandle("observed-demand", 0)
            ordinary = CacheRequestHandle("observed-ordinary", 0)

            def blocked_get(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return get(*args, **kwargs)

            try:
                with (
                    patch.object(cc, "page_get_func", blocked_get),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    first = self._restore(service, scopes[0], "observed-first")
                    wait_for(service, entered.is_set)
                    shared = self._restore(service, scopes[0], "observed-shared")
                    self._ordinary_prefetch(cache, service, scopes[0], demand)
                    during = cache.prefetch_outcome_stats_snapshot()
                    self.assertEqual(during["native_jobs_created"], 1)
                    self.assertEqual(during["native_jobs_started"], 1)
                    self.assertEqual(during["native_interests_coalesced"], 1)
                    self.assertEqual(during["native_demand_waiters"], 1)
                    self.assertEqual(during["native_inflight_tokens"], 4)
                    self.assertEqual(during["native_speculative_inflight_tokens"], 0)
                    self.assertEqual(during["native_reserved_host_tokens"], 0)
                    release.set()
                    self.assertEqual(wait(service, first).status, Status.COMPLETED)
                    self.assertEqual(wait(service, shared).status, Status.COMPLETED)
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, demand), (4, 0)
                    )
                    self._ordinary_prefetch(cache, service, scopes[1], ordinary)
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, ordinary),
                        (4, 0),
                    )
                stats = cache.prefetch_outcome_stats_snapshot()
                self.assertEqual(stats["storage_read_query_jobs"], 2)
                self.assertEqual(stats["storage_read_get_batches"], 2)
                self.assertEqual(stats["storage_read_get_requested_tokens"], 8)
                self.assertEqual(stats["storage_read_get_completed_tokens"], 8)
                self.assertEqual(
                    stats["storage_read_get_requested_bytes"],
                    8 * cc.storage_host_pool.size_per_token,
                )
                self.assertEqual(
                    stats["storage_read_get_completed_bytes"],
                    stats["storage_read_get_requested_bytes"],
                )
                self.assertIsNone(stats["storage_read_wire_bytes"])
                self.assertEqual(stats["native_jobs_completed"], 1)
                self.assertEqual(stats["native_loaded_tokens"], 4)
                self.assertEqual(stats["native_requested_tokens"], 4)
                self.assertEqual(stats["native_active_jobs"], 0)
                self.assertEqual(stats["native_demand_waiters"], 0)
                for key in (
                    "native_queue_seconds",
                    "native_restore_seconds",
                    "native_demand_wait_seconds",
                    "storage_read_query_seconds",
                    "storage_read_get_seconds",
                ):
                    self.assertGreaterEqual(stats[key], 0)
            finally:
                release.set()
                cache.release_aborted_request(demand)
                cache.release_aborted_request(ordinary)
            assert_retired(cache, service)
            cache.sanity_check()

    def test_queued_native_host_job_coalesces_demand_before_any_io(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        for finish in ("complete", "cancel", "expire"):
            config = ServiceConfig(
                namespace="ns",
                engine_id="engine",
                cache_limits=CacheLimits(max_concurrent_storage_reads=1),
            )
            with (
                self.subTest(finish=finish),
                hicache(self, config, namespace_keys=True) as owner,
            ):
                _, cache, service, _ = owner
                scopes = [
                    cache_fixtures.target(cache_salt=f"queued-{i}") for i in range(2)
                ]
                self._cold_scopes(owner, scopes)
                cc = cache.cache_controller
                entered, release = threading.Event(), threading.Event()
                query, get = cc._storage_hit_query, cc.page_get_func
                queries, gets = [], []
                demand = CacheRequestHandle("queued-demand", 0)

                def counted_query(operation):
                    queries.append(operation)
                    return query(operation)

                def blocked_get(operation, *args, **kwargs):
                    gets.append(operation)
                    entered.set()
                    assert release.wait(5)
                    return get(operation, *args, **kwargs)

                try:
                    with (
                        patch.object(cc, "_storage_hit_query", counted_query),
                        patch.object(cc, "page_get_func", blocked_get),
                        patch.object(cache, "prefetch_threshold", 1),
                        patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                    ):
                        first = self._restore(service, scopes[0], "queue-first")
                        wait_for(service, entered.is_set)
                        queued = public_transfer(
                            service,
                            "prefetch",
                            "storage",
                            "host",
                            "queue-second",
                            salt=scopes[1].cache_salt,
                        )
                        reads = service.control.executor.storage_reads
                        job = reads.interests[queued.ref.operation_id].job
                        self.assertIsNone(job.operation)
                        # Queuing changes no exact-scope/namespace/stop-policy rule.
                        for values, salt, extra_key, policy in (
                            (
                                (1, 2),
                                service.cache_salt(scopes[1].cache_salt),
                                None,
                                "wait_complete",
                            ),
                            (
                                (1, 2, 3, 5),
                                service.cache_salt(scopes[1].cache_salt),
                                None,
                                "wait_complete",
                            ),
                            (
                                (1, 2, 3, 4),
                                service.cache_salt("foreign"),
                                None,
                                "wait_complete",
                            ),
                            (
                                (1, 2, 3, 4),
                                service.cache_salt(scopes[1].cache_salt),
                                "foreign",
                                "wait_complete",
                            ),
                            (
                                (1, 2, 3, 4),
                                service.cache_salt(scopes[1].cache_salt),
                                None,
                                "best_effort",
                            ),
                        ):
                            with patch.object(cache, "prefetch_stop_policy", policy):
                                self.assertFalse(
                                    reads.join_demand(
                                        demand,
                                        [],
                                        cache_fixtures.RadixKey(
                                            cache_fixtures.array("q", values)
                                        ),
                                        extra_key,
                                        salt,
                                    )
                                )
                        self._ordinary_prefetch(cache, service, scopes[1], demand)
                        self.assertIs(reads.demands[demand].job, job)
                        self.assertIsNone(job.operation)
                        self.assertNotIn(demand, cache.ongoing_prefetch)
                        self.assertFalse(cache.check_prefetch_progress(demand))
                        self.assertEqual((len(queries), len(gets)), (1, 1))
                        if finish == "cancel":
                            self.assertEqual(
                                service.control.cancel(queued.ref).status,
                                Status.CANCELLING,
                            )
                        elif finish == "expire":
                            self.clock.now += 5.1
                            service.progress()
                            self.assertEqual(
                                current(service, queued).status, Status.CANCELLING
                            )
                        self.assertIn(demand, job.demands)
                        release.set()
                        self.assertEqual(wait(service, first).status, Status.COMPLETED)
                        self.assertEqual(
                            wait(service, queued).status,
                            (
                                Status.COMPLETED
                                if finish == "complete"
                                else Status.CANCELLED
                            ),
                        )
                        self.assertEqual(
                            self._consume_ordinary_prefetch(cache, service, demand),
                            (4, 0),
                        )
                        self.assertEqual((len(queries), len(gets)), (2, 2))
                finally:
                    release.set()
                    cache.release_aborted_request(demand)
                assert_retired(cache, service)
                cache.sanity_check()

    def test_aborting_joined_queued_demand_and_native_interest_leaves_no_io(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        config = ServiceConfig(
            namespace="ns",
            engine_id="engine",
            cache_limits=CacheLimits(max_concurrent_storage_reads=1),
        )
        with hicache(self, config, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [cache_fixtures.target(cache_salt=f"abort-{i}") for i in range(2)]
            self._cold_scopes(owner, scopes)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            query, get = cc._storage_hit_query, cc.page_get_func
            queries = []
            demand = CacheRequestHandle("queued-abort", 0)

            def counted_query(operation):
                queries.append(operation)
                return query(operation)

            def blocked_get(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return get(*args, **kwargs)

            try:
                with (
                    patch.object(cc, "_storage_hit_query", counted_query),
                    patch.object(cc, "page_get_func", blocked_get),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    first = self._restore(service, scopes[0], "abort-first")
                    wait_for(service, entered.is_set)
                    queued = self._restore(service, scopes[1], "abort-queued")
                    self._ordinary_prefetch(cache, service, scopes[1], demand)
                    reads = service.control.executor.storage_reads
                    self.assertIn(demand, reads.demands)
                    self.assertEqual(
                        service.control.cancel(queued.ref).status, Status.CANCELLING
                    )
                    cache.release_aborted_request(demand)
                    self.assertEqual(wait(service, queued).status, Status.CANCELLED)
                    self.assertEqual(len(queries), 1)
                    self.assertFalse(reads.demands)
                    release.set()
                    self.assertEqual(wait(service, first).status, Status.COMPLETED)
                    self.assertEqual(len(queries), 1)
            finally:
                release.set()
                cache.release_aborted_request(demand)
            assert_retired(cache, service)
            cache.sanity_check()

    def test_joined_capacity_wait_defers_and_normal_prefetch_can_continue(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            target = cache_fixtures.target(cache_salt="pinned-fallback")
            self._cold_scopes(owner, [target])
            pressure = self._host_pressure(owner)
            self._pin_pressure(service, pressure)
            demand = CacheRequestHandle("capacity-fallback", 0)
            cc = cache.cache_controller
            queries = []
            query = cc._storage_hit_query

            def counted_query(operation):
                queries.append(operation)
                return query(operation)

            try:
                with (
                    patch.object(cc, "_storage_hit_query", counted_query),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    queued = self._restore(service, target, "pinned-native")
                    self.assertEqual(
                        current(service, queued).update.code, "whole_scope_capacity"
                    )
                    self._ordinary_prefetch(cache, service, target, demand)
                    self.assertIn(
                        demand, service.control.executor.storage_reads.demands
                    )
                    done = wait(service, queued)
                    self.assertEqual(done.status, Status.DEFERRED)
                    self.assertEqual(done.update.code, "whole_scope_capacity")
                    self.assertFalse(queries)
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, demand),
                        (0, None),
                    )
                    self.assertFalse(service.control.executor.storage_reads._joinable)
                    self.assertTrue(cache.check_prefetch_progress(demand))
                    self.assertFalse(cache.has_ongoing_prefetch(demand))
                    # A fresh ordinary attempt uses its own normal hit-allocation
                    # fallback; it cannot rejoin the retired native job.
                    self._ordinary_prefetch(cache, service, target, demand)
                    self.assertNotIn(
                        demand, service.control.executor.storage_reads.demands
                    )
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, demand),
                        (0, None),
                    )
                    self.assertEqual(len(queries), 1)
                    self.assertTrue(cache.check_prefetch_progress(demand))
                    self.assertFalse(cache.has_ongoing_prefetch(demand))
                    self.assertEqual(cc.mem_pool_host.available_size(), 0)
            finally:
                cache.release_aborted_request(demand)
                self._pin_pressure(service, pressure, mode="release")
            assert_retired(cache, service)
            cache.sanity_check()

    def test_gpu_waiter_under_pin_pressure_keeps_ordinary_fallback_independent(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, dma = owner
            scopes = [
                cache_fixtures.target(cache_salt=f"gpu-pin-{i}") for i in range(2)
            ]
            self._cold_scopes(owner, scopes)
            pressure = self._host_pressure(owner)
            self._pin_pressure(service, pressure)
            demand = CacheRequestHandle("gpu-barrier-fallback", 0)
            cc = cache.cache_controller
            query, queries = cc._storage_hit_query, []

            def counted_query(operation):
                queries.append(operation.request_id)
                return query(operation)

            try:
                with (
                    patch.object(cc, "_storage_hit_query", counted_query),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    gpu = self._restore(service, scopes[0], "gpu-pin-waiter", dst="gpu")
                    host = self._restore(service, scopes[1], "host-pin-waiter")
                    self.assertEqual(
                        current(service, host).update.code,
                        "shared_prefix_serialization",
                    )
                    self._ordinary_prefetch(cache, service, scopes[1], demand)
                    self.assertNotIn(
                        demand, service.control.executor.storage_reads.demands
                    )
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, demand),
                        (0, None),
                    )
                    self.assertEqual(queries, [demand.rid])
                    self.assertTrue(cache.check_prefetch_progress(demand))
                    self.assertFalse(service.blocks_prefill)
                    self.assertFalse(dma.pending)
                    self.assertEqual(
                        service.control.cancel(gpu.ref).status, Status.CANCELLED
                    )
                    self.assertEqual(
                        service.control.cancel(host.ref).status, Status.CANCELLED
                    )
            finally:
                cache.release_aborted_request(demand)
                self._pin_pressure(service, pressure, mode="release")
            assert_retired(cache, service)
            cache.sanity_check()

    def test_joined_waiter_defers_if_gpu_load_becomes_active_after_join(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, dma = owner
            gpu_target, host_target = [
                cache_fixtures.target(cache_salt=f"late-gpu-{i}") for i in range(2)
            ]
            self._cold_scopes(owner, [gpu_target, host_target])
            self.assertEqual(
                wait(service, self._restore(service, gpu_target, "warm-host")).status,
                Status.COMPLETED,
            )
            gpu_path = service.control.executor._walk(
                gpu_target.token_ids,
                service.control.executor._compatibility(gpu_target),
            )
            gpu_rows = [(gpu_path[-1][0], gpu_target.token_ids)]
            pressure = self._host_pressure(owner)
            self._pin_pressure(service, pressure)
            self._pin_pressure(service, gpu_rows, salt=gpu_target.cache_salt)
            demand = CacheRequestHandle("late-gpu-fallback", 0)
            try:
                with (
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                    patch.object(
                        cache.cache_controller,
                        "_storage_hit_query",
                        side_effect=AssertionError("unapproved cold I/O"),
                    ),
                ):
                    host = self._restore(service, host_target, "older-host-queued")
                    self._ordinary_prefetch(cache, service, host_target, demand)
                    self.assertIn(
                        demand, service.control.executor.storage_reads.demands
                    )
                    gpu = self._restore(service, gpu_target, "new-gpu-load", dst="gpu")
                    wait_for(service, lambda: bool(dma.pending))
                    done = wait(service, host)
                    self.assertEqual(done.status, Status.DEFERRED)
                    self.assertEqual(done.update.code, "shared_prefix_serialization")
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, demand),
                        (0, None),
                    )
                    self.assertTrue(service.blocks_prefill)
                    self.assertTrue(dma.pending)
                    dma.finish()
                    self.assertEqual(wait(service, gpu).status, Status.COMPLETED)
            finally:
                cache.release_aborted_request(demand)
                self._pin_pressure(service, pressure, mode="release")
                self._pin_pressure(
                    service, gpu_rows, mode="release", salt=gpu_target.cache_salt
                )
            assert_retired(cache, service)
            cache.sanity_check()

    def test_storage_admission_config_validation_and_optional_defaults(self):
        from dataclasses import replace

        from sglang.srt.mem_cache.unified_cache.control_state import (
            CacheControlConfig,
        )
        from sglang.srt.mem_cache.unified_cache.control_storage_reads import (
            StorageCacheReads,
        )

        names = (
            "max_storage_read_inflight_tokens",
            "max_speculative_storage_read_tokens",
            "demand_host_reserve_tokens",
        )
        defaults = ServiceConfig(namespace="ns").cache_config()
        for name in names:
            self.assertEqual(getattr(defaults, name), 0)
            for bound in (0, 1 << 40):
                parsed = parse_config(
                    {"namespace": "ns", "cache_limits": {name: bound}}
                )
                self.assertEqual(getattr(parsed.cache_config(), name), bound)
            for invalid in (-1, (1 << 40) + 1, True, 1.5):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaises((ValueError, msgspec.ValidationError)):
                        CacheControlConfig(namespace="ns", **{name: invalid})
                    with self.assertRaises((ValueError, msgspec.ValidationError)):
                        parse_config(
                            {"namespace": "ns", "cache_limits": {name: invalid}}
                        )
        self.assertFalse(defaults.storage_demand_priority)
        for invalid in (1, "true", None):
            with self.assertRaises((ValueError, msgspec.ValidationError)):
                CacheControlConfig(namespace="ns", storage_demand_priority=invalid)
        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            executor = service.control.executor
            with (
                patch.object(
                    executor,
                    "config",
                    replace(defaults, max_storage_read_inflight_tokens=3),
                ),
                patch.object(cache.tree_core, "page_size", 2),
                self.assertRaisesRegex(ValueError, "page aligned"),
            ):
                StorageCacheReads(executor)
            with (
                patch.object(
                    executor,
                    "config",
                    replace(
                        defaults,
                        demand_host_reserve_tokens=cache.cache_controller.mem_pool_host.size,
                    ),
                ),
                self.assertRaisesRegex(ValueError, "below host capacity"),
            ):
                StorageCacheReads(executor)

    def test_speculative_inflight_budget_is_shared_and_demand_promotion_bypasses_it(
        self,
    ):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        config = ServiceConfig(
            namespace="ns",
            engine_id="engine",
            cache_limits=CacheLimits(max_speculative_storage_read_tokens=4),
        )
        with hicache(self, config, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [cache_fixtures.target(cache_salt=f"budget-{i}") for i in range(2)]
            self._cold_scopes(owner, scopes)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            get, query = cc.page_get_func, cc._storage_hit_query
            queries = []
            demand = CacheRequestHandle("budget-demand", 0)

            def blocked_get(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return get(*args, **kwargs)

            def counted_query(operation):
                queries.append(operation)
                return query(operation)

            try:
                with (
                    patch.object(cc, "page_get_func", blocked_get),
                    patch.object(cc, "_storage_hit_query", counted_query),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    first = self._restore(service, scopes[0], "budget-first")
                    wait_for(service, entered.is_set)
                    second = self._restore(service, scopes[1], "budget-second")
                    shared = self._restore(service, scopes[0], "budget-shared")
                    self.assertEqual(
                        current(service, second).update.code, "speculative_read_budget"
                    )
                    self.assertEqual(len(queries), 1)
                    self.assertEqual(
                        cache.prefetch_outcome_stats_snapshot()[
                            "native_speculative_inflight_tokens"
                        ],
                        4,
                    )
                    self._ordinary_prefetch(cache, service, scopes[1], demand)
                    wait_for(service, lambda: len(queries) == 2)
                    self.assertEqual(
                        cache.prefetch_outcome_stats_snapshot()[
                            "native_speculative_inflight_tokens"
                        ],
                        4,
                    )
                    release.set()
                    for result in (first, second, shared):
                        self.assertEqual(wait(service, result).status, Status.COMPLETED)
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, demand), (4, 0)
                    )
                    self.assertEqual(len(queries), 2)
            finally:
                release.set()
                cache.release_aborted_request(demand)
            assert_retired(cache, service)
            cache.sanity_check()

    def test_total_read_budget_accounts_for_ordinary_read_and_oversize_defers(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        config = ServiceConfig(
            namespace="ns",
            engine_id="engine",
            cache_limits=CacheLimits(max_storage_read_inflight_tokens=4),
        )
        with hicache(self, config, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [cache_fixtures.target(cache_salt=f"total-{i}") for i in range(2)]
            self._cold_scopes(owner, scopes)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            get = cc.page_get_func
            ordinary = CacheRequestHandle("total-ordinary", 0)

            def blocked_get(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return get(*args, **kwargs)

            try:
                with (
                    patch.object(cc, "page_get_func", blocked_get),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    self._ordinary_prefetch(cache, service, scopes[0], ordinary)
                    wait_for(
                        service,
                        lambda: (
                            cache.drain_storage_control_queues(),
                            entered.is_set(),
                        )[1],
                    )
                    second = self._restore(service, scopes[1], "total-second")
                    self.assertEqual(
                        current(service, second).update.code, "storage_inflight_budget"
                    )
                    self.assertEqual(cc.prefetch_tokens_occupied, 4)
                    release.set()
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, ordinary),
                        (4, 0),
                    )
                    self.assertEqual(wait(service, second).status, Status.COMPLETED)
                oversized = self._restore(
                    service,
                    cache_fixtures.target(range(10), cache_salt="oversized"),
                    "oversized",
                )
                self.assertEqual(oversized.status, Status.DEFERRED)
                self.assertEqual(oversized.update.code, "scope_exceeds_read_budget")
            finally:
                release.set()
                cache.release_aborted_request(ordinary)
            assert_retired(cache, service)
            cache.sanity_check()

    def test_host_reserve_is_not_allocated_and_joined_demand_can_use_it(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        config = ServiceConfig(
            namespace="ns",
            engine_id="engine",
            cache_limits=CacheLimits(demand_host_reserve_tokens=32),
        )
        with hicache(self, config, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            target = cache_fixtures.target(range(1, 41), cache_salt="reserve")
            self._cold_scopes(owner, [target])
            pressure = self._host_pressure(owner, leave_free=60)
            self._pin_pressure(service, pressure)
            demand = CacheRequestHandle("reserve-demand", 0)
            cc = cache.cache_controller
            try:
                with (
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                    patch.object(
                        cache, "evict_host", wraps=cache.evict_host
                    ) as evictions,
                ):
                    result = self._restore(service, target, "reserve-restore")
                    self.assertEqual(result.update.code, "whole_scope_capacity")
                    self.assertEqual(cc.mem_pool_host.available_size(), 60)
                    self.assertEqual(evictions.call_args.args, (12,))
                    self._ordinary_prefetch(cache, service, target, demand)
                    self.assertEqual(wait(service, result).status, Status.COMPLETED)
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, demand), (40, 0)
                    )
                    self.assertEqual(cc.mem_pool_host.available_size(), 20)
            finally:
                cache.release_aborted_request(demand)
                self._pin_pressure(service, pressure, mode="release")
            assert_retired(cache, service)
            cache.sanity_check()

    def test_queued_demand_priority_and_bounded_background_fairness(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        for demand_count in (1, 10):
            config = ServiceConfig(
                namespace="ns",
                engine_id="engine",
                cache_limits=CacheLimits(
                    max_concurrent_storage_reads=1, storage_demand_priority=True
                ),
            )
            with (
                self.subTest(demand_count=demand_count),
                hicache(self, config, namespace_keys=True) as owner,
            ):
                _, cache, service, _ = owner
                scopes = [
                    cache_fixtures.target(cache_salt=f"priority-{i}")
                    for i in range(2 + demand_count)
                ]
                self._cold_scopes(owner, scopes)
                cc = cache.cache_controller
                entered, release = threading.Event(), threading.Event()
                get, query = cc.page_get_func, cc._storage_hit_query
                queries, results, foreground_jobs = [], [], []
                demands = [
                    CacheRequestHandle(f"priority-demand-{i}", 0)
                    for i in range(demand_count)
                ]

                def blocked_get(*args, **kwargs):
                    entered.set()
                    assert release.wait(5)
                    return get(*args, **kwargs)

                def counted_query(operation):
                    queries.append(operation)
                    return query(operation)

                try:
                    with (
                        patch.object(cc, "page_get_func", blocked_get),
                        patch.object(cc, "_storage_hit_query", counted_query),
                        patch.object(cache, "prefetch_threshold", 1),
                        patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                    ):
                        results.append(
                            self._restore(service, scopes[0], "priority-first")
                        )
                        wait_for(service, entered.is_set)
                        background = self._restore(
                            service, scopes[1], "priority-background"
                        )
                        results.append(background)
                        reads = service.control.executor.storage_reads
                        background_job = reads.interests[
                            background.ref.operation_id
                        ].job
                        for index, demand in enumerate(demands):
                            result = self._restore(
                                service,
                                scopes[index + 2],
                                f"priority-foreground-{index}",
                            )
                            results.append(result)
                            foreground_jobs.append(
                                reads.interests[result.ref.operation_id].job
                            )
                            self._ordinary_prefetch(
                                cache, service, scopes[index + 2], demand
                            )
                        release.set()
                        for result in results:
                            self.assertEqual(
                                wait(service, result).status, Status.COMPLETED
                            )
                        for demand in demands:
                            self.assertEqual(
                                self._consume_ordinary_prefetch(cache, service, demand),
                                (4, 0),
                            )
                        expected = (
                            foreground_jobs[:8] + [background_job] + foreground_jobs[8:]
                        )
                        self.assertEqual(
                            [operation.request_id for operation in queries[1:]],
                            [job.handle.rid for job in expected],
                        )
                finally:
                    release.set()
                    for demand in demands:
                        cache.release_aborted_request(demand)
                assert_retired(cache, service)
                cache.sanity_check()

    def test_background_burst_turn_cannot_hide_demand_beyond_poll_limit(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        config = ServiceConfig(
            namespace="ns",
            engine_id="engine",
            cache_limits=CacheLimits(
                max_concurrent_storage_reads=2,
                max_speculative_storage_read_tokens=4,
                storage_demand_priority=True,
            ),
        )
        with hicache(self, config, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [
                cache_fixtures.target(cache_salt=f"poll-budget-{i}") for i in range(68)
            ]
            self._cold_scopes(owner, scopes)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            get = cc.page_get_func
            demand = CacheRequestHandle("poll-budget-demand", 0)
            background = []

            def blocked_get(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return get(*args, **kwargs)

            try:
                with (
                    patch.object(cc, "page_get_func", blocked_get),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    first = self._restore(service, scopes[0], "poll-budget-first")
                    wait_for(service, entered.is_set)
                    background = [
                        self._restore(service, target, f"poll-budget-background-{i}")
                        for i, target in enumerate(scopes[1:-1])
                    ]
                    foreground = self._restore(
                        service, scopes[-1], "poll-budget-foreground"
                    )
                    reads = service.control.executor.storage_reads
                    job = reads.interests[foreground.ref.operation_id].job
                    self.assertIsNone(job.operation)
                    self._ordinary_prefetch(cache, service, scopes[-1], demand)
                    # The separate burst test reaches this value through eight
                    # real admissions. Here isolate bounded poll liveness.
                    reads._demand_admissions = 8
                    wait_for(service, lambda: job.operation is not None)
                    for result in background:
                        self.assertEqual(
                            service.control.cancel(result.ref).status, Status.CANCELLED
                        )
                    release.set()
                    self.assertEqual(wait(service, first).status, Status.COMPLETED)
                    self.assertEqual(wait(service, foreground).status, Status.COMPLETED)
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, demand), (4, 0)
                    )
            finally:
                release.set()
                cache.release_aborted_request(demand)
                for result in background:
                    service.control.cancel(result.ref)
            assert_retired(cache, service)
            cache.sanity_check()

    def test_storage_pipeline_config_defaults_bounds_and_public_wiring(self):
        from sglang.srt.mem_cache.unified_cache.control_state import (
            CacheControlConfig,
        )

        self.assertEqual(
            ServiceConfig(namespace="ns").cache_config().max_concurrent_storage_reads, 4
        )
        for bound in (1, 64):
            config = parse_config(
                {
                    "namespace": "ns",
                    "cache_limits": {"max_concurrent_storage_reads": bound},
                }
            )
            self.assertEqual(config.cache_config().max_concurrent_storage_reads, bound)
        for invalid in (0, 65, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, msgspec.ValidationError)):
                    CacheControlConfig(
                        namespace="ns", max_concurrent_storage_reads=invalid
                    )
                with self.assertRaises((ValueError, msgspec.ValidationError)):
                    parse_config(
                        {
                            "namespace": "ns",
                            "cache_limits": {"max_concurrent_storage_reads": invalid},
                        }
                    )

    def test_different_salts_pipeline_queries_while_get_blocks_and_bound_is_enforced(
        self,
    ):
        for bound in (1, 2):
            config = ServiceConfig(
                namespace="ns",
                engine_id="engine",
                cache_limits=CacheLimits(max_concurrent_storage_reads=bound),
            )
            with (
                self.subTest(bound=bound),
                hicache(self, config, namespace_keys=True) as owner,
            ):
                _, cache, service, _ = owner
                scopes = [
                    cache_fixtures.target(cache_salt=f"salt-{i}") for i in range(3)
                ]
                self._cold_scopes(owner, scopes)
                cc = cache.cache_controller
                entered, release = threading.Event(), threading.Event()
                get, query = cc.page_get_func, cc._storage_hit_query
                queries = []

                def blocked_get(*args, **kwargs):
                    entered.set()
                    assert release.wait(5)
                    return get(*args, **kwargs)

                def counted_query(operation):
                    queries.append(operation)
                    return query(operation)

                results = []
                try:
                    with (
                        patch.object(cc, "page_get_func", blocked_get),
                        patch.object(cc, "_storage_hit_query", counted_query),
                    ):
                        results.append(self._restore(service, scopes[0], "restore-0"))
                        wait_for(service, entered.is_set)
                        results.extend(
                            self._restore(service, target, f"restore-{i}")
                            for i, target in enumerate(scopes[1:], 1)
                        )
                        wait_for(service, lambda: len(queries) == bound)
                        for _ in range(5):
                            service.progress()
                        reads = service.control.executor.storage_reads
                        self.assertEqual(
                            sum(j.operation is not None for j in reads.jobs.values()),
                            bound,
                        )
                        self.assertEqual(len(queries), bound)
                        self.assertEqual(
                            current(service, results[-1]).update.code,
                            "storage_concurrency_limit",
                        )
                        release.set()
                        for result in results:
                            self.assertEqual(
                                wait(service, result).status, Status.COMPLETED
                            )
                        self.assertEqual(len(queries), 3)
                finally:
                    release.set()
                for index, target in enumerate(scopes):
                    path = service.control.executor._walk(
                        target.token_ids,
                        service.control.executor._compatibility(target),
                    )
                    host = cache.host_pool_group.anchor_entry.host_pool
                    for node, _, _, _ in path:
                        cd = node.component_data[FULL]
                        self.assertTrue(
                            torch.all(host.k_buffer[0][cd.host_value] == 7 + index * 2)
                        )
                        self.assertEqual((cd.lock_ref, cd.host_lock_ref), (0, 0))
                assert_retired(cache, service)
                cache.sanity_check()

    def test_same_salt_serializes_while_resident_other_host_scope_finishes(self):
        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [
                cache_fixtures.target(cache_salt="shared"),
                cache_fixtures.target((10, 11, 12, 13), cache_salt="shared"),
                cache_fixtures.target(cache_salt="resident"),
            ]
            self._cold_scopes(owner, scopes)
            self.assertEqual(
                wait(service, self._restore(service, scopes[2], "resident")).status,
                Status.COMPLETED,
            )
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            get = cc.page_get_func

            def blocked_get(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return get(*args, **kwargs)

            try:
                with patch.object(cc, "page_get_func", blocked_get):
                    first = self._restore(service, scopes[0], "cold-first")
                    wait_for(service, entered.is_set)
                    second = self._restore(service, scopes[1], "same-domain")
                    self.assertEqual(
                        current(service, second).update.code,
                        "shared_prefix_serialization",
                    )
                    resident = self._restore(service, scopes[2], "resident-again")
                    self.assertEqual(resident.status, Status.COMPLETED)
                    self.assertEqual(len(cache.ongoing_prefetch), 1)
                    release.set()
                    self.assertEqual(wait(service, first).status, Status.COMPLETED)
                    self.assertEqual(wait(service, second).status, Status.COMPLETED)
            finally:
                release.set()
            assert_retired(cache, service)
            cache.sanity_check()

    def test_query_stage_host_reservation_prevents_double_approval_and_cancellation_waits(
        self,
    ):
        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [
                cache_fixtures.target(range(1, 41), cache_salt=f"capacity-{i}")
                for i in range(2)
            ]
            self._cold_scopes(owner, scopes)
            pressure = self._host_pressure(owner, leave_free=60)
            self._pin_pressure(service, pressure)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            query = cc._storage_hit_query
            queries = []

            def blocked_query(operation):
                queries.append(operation)
                entered.set()
                assert release.wait(5)
                return query(operation)

            try:
                with (
                    patch.object(cc, "_storage_hit_query", blocked_query),
                    patch.object(
                        cache, "evict_host", wraps=cache.evict_host
                    ) as evictions,
                ):
                    first = self._restore(service, scopes[0], "capacity-first")
                    wait_for(service, entered.is_set)
                    second = self._restore(service, scopes[1], "capacity-second")
                    reads = service.control.executor.storage_reads
                    second_job = reads.interests[second.ref.operation_id].job
                    self.assertEqual(
                        current(service, second).update.code, "whole_scope_capacity"
                    )
                    self.assertEqual(reads._unallocated_reservations(second_job), 40)
                    self.assertEqual(evictions.call_args.args, (20,))
                    self.assertEqual(
                        service.control.cancel(first.ref).status, Status.CANCELLING
                    )
                    for _ in range(10):
                        service.progress()
                    self.assertEqual(len(queries), 1)
                    self.assertEqual(evictions.call_count, 1)
                    self.assertEqual(reads._unallocated_reservations(second_job), 40)
                    release.set()
                    self.assertEqual(wait(service, first).status, Status.CANCELLED)
                    self.assertEqual(wait(service, second).status, Status.COMPLETED)
                    self.assertEqual(len(queries), 2)
            finally:
                release.set()
                self._pin_pressure(service, pressure, mode="release")
            assert_retired(cache, service)
            cache.sanity_check()

    def test_allocated_job_does_not_double_count_reservation(self):
        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [
                cache_fixtures.target(range(1, 41), cache_salt=f"allocated-{i}")
                for i in range(2)
            ]
            self._cold_scopes(owner, scopes)
            pressure = self._host_pressure(owner, leave_free=80)
            self._pin_pressure(service, pressure)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            get, query = cc.page_get_func, cc._storage_hit_query
            queries = []

            def blocked_get(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return get(*args, **kwargs)

            def counted_query(operation):
                queries.append(operation)
                return query(operation)

            try:
                with (
                    patch.object(cc, "page_get_func", blocked_get),
                    patch.object(cc, "_storage_hit_query", counted_query),
                    patch.object(
                        cache,
                        "evict_host",
                        side_effect=AssertionError("unexpected eviction"),
                    ),
                ):
                    first = self._restore(service, scopes[0], "allocated-first")
                    wait_for(service, entered.is_set)
                    self.assertEqual(cc.mem_pool_host.available_size(), 40)
                    second = self._restore(service, scopes[1], "allocated-second")
                    wait_for(service, lambda: len(queries) == 2)
                    second_job = service.control.executor.storage_reads.interests[
                        second.ref.operation_id
                    ].job
                    self.assertIsNotNone(second_job.operation)
                    release.set()
                    self.assertEqual(wait(service, first).status, Status.COMPLETED)
                    self.assertEqual(wait(service, second).status, Status.COMPLETED)
            finally:
                release.set()
                self._pin_pressure(service, pressure, mode="release")
            assert_retired(cache, service)
            cache.sanity_check()

    def test_gpu_destination_is_exclusive_through_storage_and_dma_fences(self):
        for first_dst in ("host", "gpu"):
            with (
                self.subTest(first_dst=first_dst),
                hicache(self, namespace_keys=True) as owner,
            ):
                _, cache, service, dma = owner
                scopes = [
                    cache_fixtures.target(cache_salt=f"gpu-exclusive-{i}")
                    for i in range(3)
                ]
                self._cold_scopes(owner, scopes)
                cc = cache.cache_controller
                entered, release = threading.Event(), threading.Event()
                get = cc.page_get_func

                def blocked_get(*args, **kwargs):
                    entered.set()
                    assert release.wait(5)
                    return get(*args, **kwargs)

                try:
                    with patch.object(cc, "page_get_func", blocked_get):
                        first = self._restore(
                            service, scopes[0], "exclusive-first", dst=first_dst
                        )
                        wait_for(service, entered.is_set)
                        second = self._restore(
                            service,
                            scopes[1],
                            "exclusive-second",
                            dst="gpu" if first_dst == "host" else "host",
                        )
                        self.assertEqual(
                            current(service, second).update.code,
                            "shared_prefix_serialization",
                        )
                        reads = service.control.executor.storage_reads
                        self.assertIsNone(
                            reads.interests[second.ref.operation_id].job.operation
                        )
                        release.set()
                        wait_for(service, lambda: service.blocks_prefill)
                        wait_for(service, lambda: bool(dma.pending))
                        third = self._restore(service, scopes[2], "exclusive-third")
                        self.assertEqual(
                            current(service, third).update.code,
                            "shared_prefix_serialization",
                        )
                        for _ in range(3):
                            service.progress()
                        self.assertIsNone(
                            reads.interests[third.ref.operation_id].job.operation
                        )
                        dma.finish()
                        for result in (first, second, third):
                            self.assertEqual(
                                wait(service, result).status, Status.COMPLETED
                            )
                finally:
                    release.set()
                self.assertFalse(service.blocks_prefill)
                assert_retired(cache, service)
                cache.sanity_check()

    def test_older_gpu_waiter_fences_new_host_jobs_until_drained_or_cancelled(self):
        for finish in ("complete", "cancel", "expire"):
            with (
                self.subTest(finish=finish),
                hicache(self, namespace_keys=True) as owner,
            ):
                _, cache, service, dma = owner
                scopes = [
                    cache_fixtures.target(cache_salt=f"fifo-{i}") for i in range(4)
                ]
                self._cold_scopes(owner, scopes)
                self.assertEqual(
                    wait(service, self._restore(service, scopes[3], "resident")).status,
                    Status.COMPLETED,
                )
                cc = cache.cache_controller
                entered, release = threading.Event(), threading.Event()
                query, get = cc._storage_hit_query, cc.page_get_func
                queries = []

                def counted_query(operation):
                    queries.append(operation.token_ids.cache_salt)
                    return query(operation)

                def blocked_get(*args, **kwargs):
                    entered.set()
                    assert release.wait(5)
                    return get(*args, **kwargs)

                try:
                    with (
                        patch.object(cc, "_storage_hit_query", counted_query),
                        patch.object(cc, "page_get_func", blocked_get),
                    ):
                        first = self._restore(service, scopes[0], "host-first")
                        wait_for(service, entered.is_set)
                        exclusive = public_transfer(
                            service,
                            "prefetch",
                            "storage",
                            "gpu",
                            "gpu-second",
                            salt=scopes[1].cache_salt,
                        )
                        later = self._restore(service, scopes[2], "host-third")
                        reads = service.control.executor.storage_reads
                        later_job = reads.interests[later.ref.operation_id].job
                        self.assertIsNone(later_job.operation)
                        self.assertEqual(
                            current(service, later).update.code,
                            "shared_prefix_serialization",
                        )
                        for _ in range(3):
                            service.progress()
                        self.assertIsNone(later_job.operation)
                        # A read requiring no new work still bypasses the barrier.
                        resident = self._restore(service, scopes[3], "resident-later")
                        self.assertEqual(resident.status, Status.COMPLETED)
                        expected = [service.cache_salt(scopes[0].cache_salt)]
                        if finish == "complete":
                            release.set()
                            wait_for(service, lambda: bool(dma.pending))
                            expected.append(service.cache_salt(scopes[1].cache_salt))
                            self.assertEqual(queries, expected)
                            self.assertTrue(service.blocks_prefill)
                            self.assertIsNone(later_job.operation)
                            dma.finish()
                            self.assertEqual(
                                wait(service, exclusive).status, Status.COMPLETED
                            )
                        else:
                            if finish == "cancel":
                                self.assertEqual(
                                    service.control.cancel(exclusive.ref).status,
                                    Status.CANCELLED,
                                )
                            else:
                                self.clock.now += 5.1
                            wait_for(service, lambda: len(queries) == 2)
                            self.assertEqual(
                                current(service, exclusive).status, Status.CANCELLED
                            )
                            self.assertIsNotNone(later_job.operation)
                            self.assertFalse(service.blocks_prefill)
                            self.assertFalse(dma.pending)
                            release.set()
                        self.assertEqual(wait(service, first).status, Status.COMPLETED)
                        self.assertEqual(wait(service, later).status, Status.COMPLETED)
                        expected.append(service.cache_salt(scopes[2].cache_salt))
                        self.assertEqual(queries, expected)
                finally:
                    release.set()
                assert_retired(cache, service)
                cache.sanity_check()

    def test_pipelined_job_keeps_exact_coalescing_and_demand_cancellation_ownership(
        self,
    ):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [
                cache_fixtures.target(),
                cache_fixtures.target(cache_salt="independent"),
            ]
            self._cold_scopes(owner, scopes)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            get, query = cc.page_get_func, cc._storage_hit_query
            queries, gets = [], []

            def blocked_get(operation, *args, **kwargs):
                gets.append(operation)
                entered.set()
                assert release.wait(5)
                return get(operation, *args, **kwargs)

            def counted_query(operation):
                queries.append(operation)
                return query(operation)

            demand = CacheRequestHandle("pipelined-demand", 0)
            try:
                with (
                    patch.object(cc, "page_get_func", blocked_get),
                    patch.object(cc, "_storage_hit_query", counted_query),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    first = self._restore(service, scopes[0], "coalesced-first")
                    wait_for(service, entered.is_set)
                    sibling = self._restore(service, scopes[0], "coalesced-sibling")
                    other = self._restore(service, scopes[1], "independent")
                    wait_for(service, lambda: len(queries) == 2)
                    reads = service.control.executor.storage_reads
                    job = reads.interests[first.ref.operation_id].job
                    self.assertIs(reads.interests[sibling.ref.operation_id].job, job)
                    self.assertEqual(
                        service.control.cancel(first.ref).status, Status.CANCELLED
                    )
                    self.assertFalse(job.operation.is_terminated())
                    self.assertTrue(
                        reads.join_demand(
                            demand,
                            [],
                            cache_fixtures.RadixKey(
                                cache_fixtures.array("q", [1, 2, 3, 4])
                            ),
                            None,
                            service.cache_salt(None),
                        )
                    )
                    self.assertEqual(
                        service.control.cancel(sibling.ref).status, Status.CANCELLING
                    )
                    self.assertFalse(job.operation.is_terminated())
                    release.set()
                    self.assertEqual(wait(service, sibling).status, Status.CANCELLED)
                    self.assertEqual(wait(service, other).status, Status.COMPLETED)
                    self.assertEqual(reads.release_demand(demand), (4, 0))
                    self.assertEqual(len(queries), 2)
                    self.assertEqual(len(gets), 2)
            finally:
                release.set()
                cache.release_aborted_request(demand)
            assert_retired(cache, service)
            cache.sanity_check()

    def test_query_miss_or_partial_get_releases_pipeline_turn_for_waiting_scope(self):
        for failure in ("miss", "partial"):
            config = ServiceConfig(
                namespace="ns",
                engine_id="engine",
                cache_limits=CacheLimits(max_concurrent_storage_reads=1),
            )
            with (
                self.subTest(failure=failure),
                hicache(self, config, namespace_keys=True) as owner,
            ):
                _, cache, service, _ = owner
                scopes = [
                    cache_fixtures.target(cache_salt=f"failure-{i}") for i in range(2)
                ]
                self._cold_scopes(owner, scopes)
                cc = cache.cache_controller
                entered, release = threading.Event(), threading.Event()
                query, get = cc._storage_hit_query, cc.page_get_func
                first_operation = []

                def failed_query(operation):
                    if not first_operation:
                        first_operation.append(operation)
                        if failure == "miss":
                            entered.set()
                            assert release.wait(5)
                            return [], 0
                    return query(operation)

                def partial_get(operation, hashes, indices, extra_info=None):
                    if operation is first_operation[0] and failure == "partial":
                        entered.set()
                        assert release.wait(5)
                        return get(operation, hashes[:2], indices[:2], extra_info)
                    return get(operation, hashes, indices, extra_info)

                try:
                    with (
                        patch.object(cc, "_storage_hit_query", failed_query),
                        patch.object(cc, "page_get_func", partial_get),
                    ):
                        first = self._restore(service, scopes[0], "failing-first")
                        wait_for(service, entered.is_set)
                        second = self._restore(service, scopes[1], "after-failure")
                        self.assertEqual(
                            current(service, second).update.code,
                            "storage_concurrency_limit",
                        )
                        self.assertIsNone(
                            service.control.executor.storage_reads.interests[
                                second.ref.operation_id
                            ].job.operation
                        )
                        release.set()
                        self.assertEqual(
                            wait(service, first).status,
                            Status.MISSING if failure == "miss" else Status.FAILED,
                        )
                        self.assertEqual(wait(service, second).status, Status.COMPLETED)
                finally:
                    release.set()
                assert_retired(cache, service)
                cache.sanity_check()

    def test_deadline_keeps_pipeline_turn_until_query_or_get_actually_drains(self):
        for phase in ("query", "get"):
            config = ServiceConfig(
                namespace="ns",
                engine_id="engine",
                cache_limits=CacheLimits(max_concurrent_storage_reads=1),
            )
            with (
                self.subTest(phase=phase),
                hicache(self, config, namespace_keys=True) as owner,
            ):
                _, cache, service, _ = owner
                scopes = [
                    cache_fixtures.target(),
                    cache_fixtures.target(cache_salt="after-deadline"),
                ]
                self._cold_scopes(owner, scopes)
                cc = cache.cache_controller
                entered, release = threading.Event(), threading.Event()
                query, get = cc._storage_hit_query, cc.page_get_func
                first_operation, queries = [], []

                def blocked_query(operation):
                    queries.append(operation)
                    if not first_operation:
                        first_operation.append(operation)
                        if phase == "query":
                            entered.set()
                            assert release.wait(5)
                    return query(operation)

                def blocked_get(operation, *args, **kwargs):
                    if operation is first_operation[0] and phase == "get":
                        entered.set()
                        assert release.wait(5)
                    return get(operation, *args, **kwargs)

                try:
                    with (
                        patch.object(cc, "_storage_hit_query", blocked_query),
                        patch.object(cc, "page_get_func", blocked_get),
                    ):
                        first = public_transfer(
                            service, "prefetch", "storage", "host", "deadline-first"
                        )
                        wait_for(service, entered.is_set)
                        second = self._restore(service, scopes[1], "after-deadline")
                        self.clock.now += 5.1
                        service.progress()
                        self.assertEqual(
                            current(service, first).status, Status.CANCELLING
                        )
                        for _ in range(5):
                            service.progress()
                        reads = service.control.executor.storage_reads
                        self.assertIsNone(
                            reads.interests[second.ref.operation_id].job.operation
                        )
                        self.assertFalse(first_operation[0].native_quiescent)
                        self.assertEqual(len(queries), 1)
                        release.set()
                        self.assertEqual(wait(service, first).status, Status.CANCELLED)
                        self.assertEqual(wait(service, second).status, Status.COMPLETED)
                        self.assertEqual(len(queries), 2)
                finally:
                    release.set()
                assert_retired(cache, service)
                cache.sanity_check()

    def _host_pressure(self, owner, *, leave_free=0):
        """Fill real host rows via CPU DMA and native MOVE; keep GPU empty."""
        _, cache, service, dma = owner
        pool = cache.cache_controller.mem_pool_host
        nodes = []
        while pool.available_size() > leave_free:
            count = min(
                2 if not nodes else cache.token_to_kv_pool_allocator.size,
                pool.available_size() - leave_free,
            )
            tokens = tuple(
                range(1000 + len(nodes) * 100, 1000 + len(nodes) * 100 + count)
            )
            node = seed(cache, tokens, value=11 + len(nodes))
            result = transfer(
                service, kind="move_to", tokens=tokens, message=f"pressure-{len(nodes)}"
            )
            stage(service, dma)
            self.assertEqual(wait(service, result).status, Status.COMPLETED)
            self.assertIsNone(node.component_data[FULL].value)
            self.assertIsNotNone(node.component_data[FULL].host_value)
            nodes.append((node, tokens))
        self.assertEqual(pool.available_size(), leave_free)
        return nodes

    def _pin_pressure(self, service, rows, *, mode="acquire", salt=None):
        for node, tokens in rows:
            payload = {
                "target": msgspec.json.decode(
                    msgspec.json.encode(cache_fixtures.target(tokens, cache_salt=salt))
                ),
                "location": {"tier": "host"},
                "mode": mode,
                "claim_id": f"pressure-{node.id}",
                "revision": 1 if mode == "acquire" else 2,
            }
            if mode == "acquire":
                payload["ttl_seconds"] = 10
            result = self.submit(service.control, "pin", payload, caller="keeper")
            self.assertEqual(result.status, Status.COMPLETED)

    def test_storage_prefetch_reclaims_only_shortfall_and_holds_existing_prefix(self):
        """A full host cache must not strand a valid cold prefix behind admission."""
        for existing in (0, 2):
            with (
                self.subTest(existing=existing),
                hicache(self, namespace_keys=True) as owner,
            ):
                cache, service, _ = cold_prefix(owner)
                prefix = None
                if existing:
                    restored = transfer(
                        service,
                        src="storage",
                        dst="host",
                        tokens=(1, 2),
                        message="prefix",
                    )
                    self.assertEqual(wait(service, restored).status, Status.COMPLETED)
                    prefix = prefix_path(cache, service)[0][0]
                pool = cache.cache_controller.mem_pool_host
                rows = self._host_pressure(owner, leave_free=2 - existing)
                first = rows[0][0]
                source_indices = (
                    prefix.component_data[FULL].host_value.clone()
                    if prefix is not None
                    else None
                )
                gpu_free = cache.token_to_kv_pool_allocator.available_size()
                with patch.object(
                    cache, "evict_host", wraps=cache.evict_host
                ) as evictions:
                    result = transfer(
                        service, src="storage", dst="host", message="restore"
                    )
                    self.assertIsNotNone(
                        next(
                            iter(service.control.executor.storage_reads.jobs.values())
                        ).operation
                    )
                    self.assertEqual(evictions.call_args_list[0].args, (2,))
                    done = wait(service, result)
                self.assertEqual(done.status, Status.COMPLETED)
                self.assertEqual(evictions.call_count, 1)
                self.assertIsNone(first.component_data[FULL].host_value)
                self.assertTrue(
                    all(
                        node.component_data[FULL].host_value is not None
                        for node, _ in rows[1:]
                    )
                )
                self.assertEqual(pool.available_size(), 0)
                self.assertEqual(
                    cache.token_to_kv_pool_allocator.available_size(), gpu_free
                )
                if prefix is not None:
                    self.assertTrue(
                        torch.equal(
                            prefix.component_data[FULL].host_value, source_indices
                        )
                    )
                host = cache.host_pool_group.anchor_entry.host_pool
                for node, _, _, _ in prefix_path(cache, service):
                    cd = node.component_data[FULL]
                    self.assertTrue(torch.all(host.k_buffer[0][cd.host_value] == 7))
                    self.assertEqual(cd.host_lock_ref, 0)
                    self.assertEqual(cd.lock_ref, 0)
                assert_retired(cache, service)
                cache.sanity_check()

    def test_pinned_capacity_waits_without_idle_rescans_and_live_sibling_outlives_first_timeout(
        self,
    ):
        with hicache(self, namespace_keys=True) as owner:
            cache, service, _ = cold_prefix(owner)
            rows = self._host_pressure(owner)
            self._pin_pressure(service, rows)
            held = {
                node.id: node.component_data[FULL].host_value.clone()
                for node, _ in rows
            }
            cc = cache.cache_controller
            with (
                patch.object(cache, "evict_host", wraps=cache.evict_host) as evictions,
                patch.object(
                    cc, "_storage_hit_query", wraps=cc._storage_hit_query
                ) as queries,
            ):
                first = public_transfer(
                    service, "prefetch", "storage", "host", "first", caller="A"
                )
                second = transfer(
                    service, src="storage", dst="host", caller="B", message="second"
                )
                other = transfer(
                    service,
                    src="storage",
                    dst="host",
                    caller="C",
                    message="other",
                    tokens=(1, 2),
                )
                for _ in range(20):
                    service.progress()
                self.assertEqual(evictions.call_count, 1)
                self.assertEqual(queries.call_count, 0)
                self.assertEqual(
                    current(service, first).update.code, "whole_scope_capacity"
                )
                self.assertEqual(
                    cache.cache_controller.mem_pool_host.available_size(), 0
                )
                self.assertTrue(
                    all(
                        torch.equal(node.component_data[FULL].host_value, held[node.id])
                        for node, _ in rows
                    )
                )
                self.assertEqual(
                    service.control.cancel(other.ref).status, Status.CANCELLED
                )
                # Expire only A's operation. B still owns the shared physical
                # preparation; native PIN deadlines remain at t=110.
                self._pin_pressure(service, rows[1:2], mode="release")
                self.clock.now = 105.1
                service.progress()
                self.assertEqual(current(service, first).status, Status.CANCELLED)
                self.assertEqual(wait(service, second).status, Status.COMPLETED)
                self.assertEqual(evictions.call_count, 2)
                self.assertEqual(queries.call_count, 1)
                self.assertTrue(
                    all(
                        torch.equal(node.component_data[FULL].host_value, held[node.id])
                        for node, _ in rows[:1] + rows[2:]
                    )
                )
            self._pin_pressure(service, rows[:1] + rows[2:], mode="release")
            assert_retired(cache, service)
            cache.sanity_check()

    def test_storage_prefetch_checks_destination_and_transfer_budgets_before_host_eviction(
        self,
    ):
        for blocker in ("gpu", "transfer"):
            with (
                self.subTest(blocker=blocker),
                hicache(self, namespace_keys=True) as owner,
            ):
                cache, service, _ = cold_prefix(owner)
                rows = self._host_pressure(owner)
                gpu_free = cache.token_to_kv_pool_allocator.available_size()
                held = {
                    node.id: node.component_data[FULL].host_value.clone()
                    for node, _ in rows
                }
                if blocker == "gpu":
                    service.control.executor.gpu_copy_capacity = lambda: 0
                cc = cache.cache_controller
                with (
                    patch.object(
                        cache,
                        "evict_host",
                        side_effect=AssertionError("unapproved host eviction"),
                    ),
                    patch.object(
                        cc, "prefetch_rate_limited", return_value=blocker == "transfer"
                    ),
                ):
                    result = transfer(
                        service, src="storage", dst="gpu", message="blocked"
                    )
                    for _ in range(3):
                        service.progress()
                    self.assertEqual(current(service, result).status, Status.QUEUED)
                    self.assertEqual(
                        service.control.cancel(result.ref).status, Status.CANCELLED
                    )
                self.assertEqual(
                    cache.cache_controller.mem_pool_host.available_size(), 0
                )
                self.assertEqual(
                    cache.token_to_kv_pool_allocator.available_size(), gpu_free
                )
                self.assertTrue(
                    all(
                        torch.equal(node.component_data[FULL].host_value, held[node.id])
                        for node, _ in rows
                    )
                )
                assert_retired(cache, service)
                cache.sanity_check()

    def test_waiting_prefetch_uses_freed_capacity_before_scan_retry_deadline(self):
        with hicache(self, namespace_keys=True) as owner:
            cache, service, _ = cold_prefix(owner)
            rows = self._host_pressure(owner)
            self._pin_pressure(service, rows)
            with patch.object(cache, "evict_host", wraps=cache.evict_host) as evictions:
                result = transfer(service, src="storage", dst="host", message="waiting")
                self.assertEqual(current(service, result).status, Status.QUEUED)
                self.assertEqual(evictions.call_count, 1)
                self._pin_pressure(service, rows[1:2], mode="release")
                count = len(rows[1][1])
                self.assertEqual(cache.evict_host(count), count)
                # Native memory may become available without clock advance;
                # backoff limits scans, not use of already released capacity.
                self.assertEqual(wait(service, result).status, Status.COMPLETED)
                self.assertEqual(self.clock.now, 100)
                self.assertEqual(evictions.call_count, 2)
            self._pin_pressure(service, rows[:1] + rows[2:], mode="release")
            assert_retired(cache, service)
            cache.sanity_check()

    def test_expired_waiting_prefetch_does_not_evict_newly_unpinned_host_rows(self):
        with hicache(self, namespace_keys=True) as owner:
            cache, service, _ = cold_prefix(owner)
            rows = self._host_pressure(owner)
            self._pin_pressure(service, rows)
            with patch.object(cache, "evict_host", wraps=cache.evict_host) as evictions:
                result = public_transfer(
                    service, "prefetch", "storage", "host", "expired"
                )
                self.assertEqual(evictions.call_count, 1)
                self._pin_pressure(service, rows, mode="release")
                self.clock.now = 106
                service.progress()
                self.assertEqual(current(service, result).status, Status.CANCELLED)
                self.assertEqual(evictions.call_count, 1)
            self.assertEqual(cache.cache_controller.mem_pool_host.available_size(), 0)
            self.assertTrue(
                all(
                    node.component_data[FULL].host_value is not None for node, _ in rows
                )
            )
            assert_retired(cache, service)
            cache.sanity_check()

    def test_native_demand_coalesces_backend_io(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        with hicache(self, namespace_keys=True) as owner:
            cache, service, dma = cold_prefix(owner)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            counts = {"query": 0, "get": 0}
            query, get = cc._storage_hit_query, cc.page_get_func

            def counted_query(*args, **kwargs):
                counts["query"] += 1
                return query(*args, **kwargs)

            def counted_get(*args, **kwargs):
                counts["get"] += 1
                entered.set()
                assert release.wait(5)
                return get(*args, **kwargs)

            demand = CacheRequestHandle("ordinary-demand", 0)
            try:
                with (
                    patch.object(cc, "_storage_hit_query", counted_query),
                    patch.object(cc, "page_get_func", counted_get),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    native = transfer(
                        service, src="storage", dst="host", message="native"
                    )
                    wait_for(service, entered.is_set)
                    cache.prefetch_from_storage(
                        demand,
                        cache.tree_core.root_node.id,
                        cache_fixtures.array("q", [1, 2, 3, 4]),
                        matched_prefix_tokens=[],
                        cache_salt=service.cache_salt(None),
                    )
                    release.set()
                    wait(service, native)

                    def demand_ready():
                        cache.drain_storage_control_queues()
                        return cache.check_prefetch_progress(demand)

                    wait_for(service, demand_ready)
                    span = cache.pop_prefetch_loaded_span(demand)
                    cache.discard_storage_prefetch_accounting(demand)
                    service.progress()
                    print(
                        "native/demand physical operations", counts, "demand span", span
                    )
                    self.assertEqual(counts, {"query": 1, "get": 1})
                    self.assertEqual(span, (4, 0))
            finally:
                release.set()
                cache.release_aborted_request(demand)
            assert_retired(cache, service)

    def test_native_demand_cancel_lifetimes_and_mismatches(self):
        from dataclasses import replace

        from sglang.srt.managers.schedule_batch import split_cached_prefix_by_tier
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
        from sglang.srt.mem_cache.radix_cache import RadixKey

        for abort_demand in (False, True):
            with (
                self.subTest(abort_demand=abort_demand),
                hicache(self, namespace_keys=True) as owner,
            ):
                cache, service, dma = cold_prefix(owner)
                cc = cache.cache_controller
                reads = service.control.executor.storage_reads
                entered, release = threading.Event(), threading.Event()
                original = cc.page_get_func
                calls = []

                def blocked(*args, **kwargs):
                    calls.append(1)
                    entered.set()
                    assert release.wait(5)
                    return original(*args, **kwargs)

                demand = CacheRequestHandle("demand", 3)
                try:
                    with (
                        patch.object(cc, "page_get_func", blocked),
                        patch.object(cache, "prefetch_threshold", 1),
                        patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                    ):
                        native = transfer(
                            service, src="storage", dst="host", message="native"
                        )
                        wait_for(service, entered.is_set)
                        key = lambda values: RadixKey(cache_fixtures.array("q", values))
                        for values, salt in (
                            ([1, 2, 3], service.cache_salt(None)),
                            ([1, 2, 3, 5], service.cache_salt(None)),
                            ([1, 2, 3, 4], service.cache_salt("other")),
                            ([1, 2, 3, 4], "foreign-weight-namespace"),
                        ):
                            self.assertFalse(
                                reads.join_demand(demand, [], key(values), None, salt)
                            )
                        cache.prefetch_from_storage(
                            demand,
                            cache.tree_core.root_node.id,
                            cache_fixtures.array("q", [1, 2, 3, 4]),
                            matched_prefix_tokens=[],
                            cache_salt=service.cache_salt(None),
                        )
                        self.assertTrue(cache.has_ongoing_prefetch(demand))
                        self.assertFalse(cache.check_prefetch_progress(demand))
                        self.assertTrue(
                            cache.check_prefetch_progress(
                                CacheRequestHandle("unrelated", 0)
                            )
                        )
                        # Same request re-entry cannot add a duplicate interest.
                        cache.prefetch_from_storage(
                            demand,
                            cache.tree_core.root_node.id,
                            cache_fixtures.array("q", [1, 2, 3, 4]),
                            matched_prefix_tokens=[],
                            cache_salt=service.cache_salt(None),
                        )
                        self.assertEqual(len(reads.demands), 1)
                        sibling = CacheRequestHandle("aborted-sibling", 0)
                        with patch.object(
                            service.control.executor,
                            "config",
                            replace(
                                service.control.executor.config, max_prepared_prefixes=1
                            ),
                        ):
                            self.assertFalse(
                                reads.join_demand(
                                    sibling,
                                    [],
                                    key([1, 2, 3, 4]),
                                    None,
                                    service.cache_salt(None),
                                )
                            )
                            self.assertEqual(reads.stats["demand_limit_declined"], 1)
                        self.assertTrue(
                            reads.join_demand(
                                sibling,
                                [],
                                key([1, 2, 3, 4]),
                                None,
                                service.cache_salt(None),
                            )
                        )
                        cache.release_aborted_request(sibling)
                        self.assertEqual(len(reads.demands), 1)
                        self.assertFalse(
                            next(iter(reads.jobs.values())).operation.is_terminated()
                        )
                        cancelled = service.control.cancel(native.ref)
                        self.assertEqual(cancelled.status, Status.CANCELLING)
                        job = next(iter(reads.jobs.values()))
                        self.assertFalse(job.operation.is_terminated())
                        if abort_demand:
                            cache.release_aborted_request(demand)
                            service.progress()
                            self.assertTrue(job.operation.is_terminated())
                        release.set()
                        done = wait(service, native)
                        self.assertEqual(done.status, Status.CANCELLED)
                        self.assertEqual(len(calls), 1)
                        if not abort_demand:
                            self.assertTrue(cache.check_prefetch_progress(demand))
                            self.assertTrue(service.has_pending)
                            self.assertFalse(cache._storage_attachment.detach()[0])
                            self.assertEqual(cache.evict_host(4), 0)
                            span = cache.pop_prefetch_loaded_span(demand)
                            self.assertEqual(span, (4, 0))
                            self.assertEqual(
                                split_cached_prefix_by_tier(4, 4, *span), (0, 0, 4)
                            )
                            self.assertEqual(cache.evict_host(4), 4)
                        self.assertFalse(service.has_pending)
                finally:
                    release.set()
                    cache.release_aborted_request(demand)
                assert_retired(cache, service)

    def test_native_demand_partial_failure_and_abort_leave_no_holds(self):
        from sglang.srt.managers.schedule_batch import split_cached_prefix_by_tier
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        with hicache(self, namespace_keys=True) as owner:
            cache, service, dma = cold_prefix(owner)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            original = cc.page_get_func
            calls = []

            def partial(operation, hashes, indices, extra_info=None):
                calls.append(1)
                entered.set()
                assert release.wait(5)
                # A real complete leading page, followed by a failed page read.
                return original(operation, hashes[:2], indices[:2], extra_info)

            demand = CacheRequestHandle("partial", 0)
            try:
                with (
                    patch.object(cc, "page_get_func", partial),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    native = transfer(
                        service, src="storage", dst="host", message="partial"
                    )
                    wait_for(service, entered.is_set)
                    cache.prefetch_from_storage(
                        demand,
                        cache.tree_core.root_node.id,
                        cache_fixtures.array("q", [1, 2, 3, 4]),
                        matched_prefix_tokens=[],
                        cache_salt=service.cache_salt(None),
                    )
                    release.set()
                    done = wait(service, native)
                    self.assertEqual(done.status, Status.FAILED)
                    self.assertTrue(cache.check_prefetch_progress(demand))
                    span = cache.pop_prefetch_loaded_span(demand)
                    self.assertEqual(span, (2, 0))
                    self.assertEqual(
                        split_cached_prefix_by_tier(2, 2, *span), (0, 0, 2)
                    )
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(cache.evict_host(4), 2)
            finally:
                release.set()
                cache.release_aborted_request(demand)
            assert_retired(cache, service)

    def test_native_demand_published_prefix_and_disabled_control(self):
        from sglang.srt.managers.schedule_batch import split_cached_prefix_by_tier
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

        with hicache(self, namespace_keys=True) as owner:
            cache, service, dma = cold_prefix(owner)
            native = transfer(service, src="storage", dst="host", message="ready")
            self.assertEqual(wait(service, native).status, Status.COMPLETED)
            node = prefix_path(cache, service)[-1][0]
            demand = CacheRequestHandle("arrives-after-publication", 0)
            with patch.object(
                cache.cache_controller,
                "submit_prefetch",
                side_effect=AssertionError(
                    "published host prefix needs no storage I/O"
                ),
            ):
                # Ordinary match found the complete host prefix, so only the
                # empty suffix reaches the prefetch entrypoint.
                cache.prefetch_from_storage(
                    demand,
                    node.id,
                    cache_fixtures.array("q", []),
                    matched_prefix_tokens=[1, 2, 3, 4],
                    cache_salt=service.cache_salt(None),
                )
            self.assertTrue(cache.check_prefetch_progress(demand))
            span = cache.pop_prefetch_loaded_span(demand)
            self.assertEqual(span, (0, None))
            self.assertEqual(split_cached_prefix_by_tier(4, 4, *span), (0, 4, 0))
            self.assertEqual(
                service.control.executor.storage_reads.stats["demand_coalesced"], 0
            )
            with patch.object(cache, "kv_control_service", None):
                self.assertIsNone(cache._native_storage_reads())
                self.assertTrue(cache.check_prefetch_progress(demand))
            assert_retired(cache, service)

    def test_public_file_prefetch_registers_host_without_local_prefix(self):
        with hicache(self, namespace_keys=True) as owner:
            cache, service, dma = cold_prefix(owner)
            caps = service.capabilities()["public_protocol"]
            assert caps["storage_backend"] == "file"
            assert "sglang.pin" in caps["actions"]
            assert {p["tier"] for p in caps["pin_protections"]} == {"host"}
            result = public_transfer(
                service, "prefetch", "storage", "host", "public-restore"
            )
            done = wait(service, result)
            assert done.status == Status.COMPLETED, done
            assert done.public_result.ready_scope
            assert done.public_result.ready_location.tier == "host"
            assert not dma.pending
            for node, _, _, _ in prefix_path(cache, service):
                cd = node.component_data[FULL]
                host = cache.host_pool_group.anchor_entry.host_pool
                assert cd.value is None
                assert torch.all(host.k_buffer[0][cd.host_value] == 7)
                assert torch.all(host.v_buffer[0][cd.host_value] == 8)
                assert cd.lock_ref == cd.host_lock_ref == 0
            assert_retired(cache, service)

    def test_restore_from_actual_file_without_resident_radix_nodes(self):
        with hicache(self, namespace_keys=True) as owner:
            destination = "gpu"
            cache, service, dma = cold_prefix(owner)
            pool = cache.token_to_kv_pool_allocator.get_kvcache()
            pool.k_buffer[0].fill_(-100)
            pool.v_buffer[0].fill_(-101)
            threshold = cache.prefetch_threshold
            assert threshold > 4
            result = transfer(
                service, src="storage", dst=destination, message="restore"
            )
            if destination == "gpu":
                wait_for(service, lambda: service.blocks_prefill)
                assert current(service, result).status == Status.RUNNING
                assert not current(service, result).update.completed_steps
                service.progress()
                assert len(dma.pending) == 1
                dma.finish()
            done = wait(service, result)
            assert done.status == Status.COMPLETED, done
            assert done.update.completed_steps == ("destination_ready",)
            assert cache.prefetch_threshold == threshold
            for node, _, _, _ in prefix_path(cache, service):
                cd = node.component_data[FULL]
                source = (
                    cache.host_pool_group.anchor_entry.host_pool
                    if destination == "host"
                    else pool
                )
                indices = cd.host_value if destination == "host" else cd.value
                assert torch.all(source.k_buffer[0][indices] == 7)
                assert torch.all(source.v_buffer[0][indices] == 8)
                assert cd.lock_ref == cd.host_lock_ref == 0
            assert not service.blocks_prefill
            assert_retired(cache, service)

    def test_storage_read_cancel_waits_for_actual_worker_exit(self):
        with hicache(self, namespace_keys=True) as owner:
            cache, service, dma = cold_prefix(owner)
            cc = cache.cache_controller
            entered, release = (threading.Event(), threading.Event())
            original = cc.page_get_func

            def blocked(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return original(*args, **kwargs)

            try:
                with patch.object(cc, "page_get_func", blocked):
                    result = transfer(
                        service, src="storage", dst="gpu", message="restore"
                    )
                    wait_for(service, entered.is_set)
                    assert (
                        service.control.cancel(result.ref).status == Status.CANCELLING
                    )
                    for _ in range(3):
                        service.progress()
                    assert service.has_pending
                    assert cache.ongoing_prefetch
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            assert done.status == Status.CANCELLED, done
            assert not dma.pending
            assert_retired(cache, service)


if __name__ == "__main__":
    unittest.main()
