import importlib
import os
import sys
import types
import unittest
from array import array
from enum import Enum
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _StubKVResponseStatus(Enum):
    SUCCESS = "success"
    NOTFOUND = "not_found"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    FAILED = "failed"


class _StubFlexKVConfig:
    @classmethod
    def from_env(cls) -> Any:
        raise AssertionError("FlexKV configuration must not be read")


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _module(name: str, **attributes: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


_FLEXKV_STUBS = {
    "flexkv": _package("flexkv"),
    "flexkv.common": _package("flexkv.common"),
    "flexkv.integration": _package("flexkv.integration"),
    "flexkv.server": _package("flexkv.server"),
    "flexkv.transfer": _package("flexkv.transfer"),
    "flexkv.common.request": _module(
        "flexkv.common.request", KVResponseStatus=_StubKVResponseStatus
    ),
    "flexkv.common.storage": _module(
        "flexkv.common.storage",
        KVCacheLayout=object,
        KVCacheLayoutType=SimpleNamespace(LAYERFIRST="layerfirst"),
    ),
    "flexkv.integration.config": _module(
        "flexkv.integration.config", FlexKVConfig=_StubFlexKVConfig
    ),
    "flexkv.kvmanager": _module("flexkv.kvmanager", KVManager=object),
    "flexkv.server.client": _module("flexkv.server.client", KVTPClient=object),
    "flexkv.transfer.layerwise": _module(
        "flexkv.transfer.layerwise",
        build_layerwise_eventfd_socket_path=lambda **_: "",
    ),
    "flexkv.transfer_manager": _module(
        "flexkv.transfer_manager", TransferManagerOnRemote=object
    ),
}

with mock.patch.dict(sys.modules, _FLEXKV_STUBS):
    connector_module = importlib.import_module(
        "sglang.srt.mem_cache.storage.flexkv.flexkv_connector"
    )
    radix_module = importlib.import_module(
        "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache"
    )


class _Response:
    def __init__(self, task_id: int, status: _StubKVResponseStatus) -> None:
        self.task_id = task_id
        self.status = status


class _FakeKVManager:
    def __init__(self, matches: list[Any]) -> None:
        self.matches = list(matches)
        self.get_match_masks: list[np.ndarray] = []
        self.cancel_calls: list[list[int]] = []
        self.launch_calls: list[dict[str, Any]] = []
        self.wait_calls: list[dict[str, Any]] = []
        self.terminal_task_ids = [900]
        self.wait_status = _StubKVResponseStatus.SUCCESS
        self.cancel_error: Optional[Exception] = None
        self.omit_wait_response = False

    def get_match(
        self,
        *,
        token_ids: np.ndarray,
        token_mask: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        self.get_match_masks.append(token_mask.copy())
        result = self.matches.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def cancel(self, task_ids: list[int]) -> None:
        self.cancel_calls.append(list(task_ids))
        if self.cancel_error is not None:
            raise self.cancel_error

    def launch(self, **kwargs: Any) -> list[int]:
        self.launch_calls.append(kwargs)
        return list(self.terminal_task_ids)

    def wait(self, **kwargs: Any) -> dict[int, _Response]:
        self.wait_calls.append(kwargs)
        if self.omit_wait_response:
            return {}
        return {
            task_id: _Response(task_id, self.wait_status)
            for task_id in kwargs["task_ids"]
        }


class _FakeSyncContext:
    def __init__(self, *, send_to_remote: bool = False) -> None:
        self.is_sync_leader = True
        self.needs_sync = False
        self.should_send_slot_mapping_to_remote = send_to_remote
        self.is_pp_receiver = False
        self.is_pp_sender = False

    def scatter(self, payload: Any) -> Any:
        return payload

    def all_reduce_min(self, value: int) -> int:
        return value


class _ScriptedSyncContext:
    def __init__(
        self,
        *,
        scatter_results: list[Any],
        reduce_results: list[int],
        send_to_remote: bool = False,
        is_sync_leader: bool = False,
    ) -> None:
        self.is_sync_leader = is_sync_leader
        self.needs_sync = True
        self.should_send_slot_mapping_to_remote = send_to_remote
        self.is_pp_receiver = False
        self.is_pp_sender = False
        self.scatter_results = list(scatter_results)
        self.reduce_results = list(reduce_results)
        self.events: list[tuple[str, Any]] = []

    def scatter(self, payload: Any) -> Any:
        self.events.append(("scatter", payload))
        return self.scatter_results.pop(0)

    def all_reduce_min(self, value: int) -> int:
        self.events.append(("reduce", value))
        return self.reduce_results.pop(0)


class _FakeTPClient:
    def __init__(self) -> None:
        self.mapping_calls: list[tuple[int, np.ndarray]] = []

    def set_slot_mapping(self, task_id: int, mapping: np.ndarray) -> None:
        self.mapping_calls.append((task_id, mapping.copy()))


def _make_connector(
    manager: _FakeKVManager,
    *,
    allocator_page_size: int = 4,
    send_to_remote: bool = False,
) -> Any:
    connector = object.__new__(connector_module.FlexKVConnector)
    connector.allocator_page_size = allocator_page_size
    connector.storage_page_size = 1
    connector.enable_layerwise = False
    connector.kv_manager = manager
    connector._sync_ctx = _FakeSyncContext(send_to_remote=send_to_remote)
    connector._pending_lookups = {}
    connector._launched_load_tids = []
    connector.tp_client = _FakeTPClient()
    return connector


class _FakeLayerEvent:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_for_new_transfer(self) -> None:
        self.reset_calls += 1


class _FakeLayerDoneCounter:
    def __init__(self) -> None:
        self.events = [_FakeLayerEvent()]
        self.register_calls: list[tuple[int, int]] = []
        self.consumer_calls: list[int] = []

    def update_producer(self) -> int:
        return 0

    def register_task(self, task_id: int, producer_id: int) -> None:
        self.register_calls.append((task_id, producer_id))

    def set_consumer(self, task_id: int) -> None:
        self.consumer_calls.append(task_id)


class _FakeAllocator:
    def __init__(
        self,
        slots: Optional[torch.Tensor],
        page_size: int = 4,
        available: int = 1024,
    ) -> None:
        self.page_size = page_size
        self.slots = slots
        self.available = available
        self.available_error: Optional[Exception] = None
        self.alloc_error: Optional[Exception] = None
        self.alloc_sizes: list[int] = []
        self.free_calls: list[torch.Tensor] = []
        self.events: list[str] = []

    def available_size(self) -> int:
        self.events.append("available")
        if self.available_error is not None:
            raise self.available_error
        return self.available

    def alloc(self, size: int) -> Optional[torch.Tensor]:
        self.events.append("alloc")
        self.alloc_sizes.append(size)
        if self.alloc_error is not None:
            raise self.alloc_error
        if self.slots is None:
            return None
        return self.slots[:size].clone()

    def free(self, slots: torch.Tensor) -> None:
        self.events.append("free")
        self.free_calls.append(slots.clone())


class _FakeRadixConnector:
    def __init__(self, result: Any, force_eviction: Optional[bool] = None) -> None:
        self.result = result
        self.force_eviction = force_eviction
        self.retrieve_calls: list[dict[str, Any]] = []
        self.eviction_decisions: list[bool] = []

    def requires_mp_eviction(self, *, local_has_capacity: bool) -> bool:
        self.eviction_decisions.append(local_has_capacity)
        if self.force_eviction is not None:
            return self.force_eviction
        return not local_has_capacity

    def retrieve_kv(self, **kwargs: Any) -> Any:
        self.retrieve_calls.append(kwargs)
        return self.result

    def release_pending(self, rid: str) -> None:
        raise AssertionError(f"unexpected release_pending for {rid}")


def _make_radix_cache(allocator: _FakeAllocator, connector: Any) -> Any:
    cache = object.__new__(radix_module.FlexKVRadixCache)
    cache.token_to_kv_pool_allocator = allocator
    cache.flexkv_connector = connector
    cache._allocator_page_size = allocator.page_size
    cache._pending_mp_leases = {}
    cache._next_mp_lease_id = 0
    cache.page_size = 1
    cache.evictable_size_ = 0
    cache._update_leaf_status = mock.Mock()
    cache._record_store_event = mock.Mock()
    return cache


def _make_production_mp_cache(
    *,
    allocator: _FakeAllocator,
    scatter_results: list[Any],
    reduce_results: list[int],
    expected_slots: int = 4,
    send_to_remote: bool = False,
) -> tuple[Any, _FakeKVManager, Any, _ScriptedSyncContext]:
    manager = _FakeKVManager(matches=[])
    connector = _make_connector(manager, send_to_remote=send_to_remote)
    connector._pending_lookups["request"] = (
        connector_module._PendingFlexKVLookup(
            lookup_task_id=202,
            expected_slots=expected_slots,
        )
    )
    sync_context = _ScriptedSyncContext(
        scatter_results=scatter_results,
        reduce_results=reduce_results,
        send_to_remote=send_to_remote,
        is_sync_leader=True,
    )
    connector._sync_ctx = sync_context
    cache = _make_radix_cache(allocator, connector)
    return cache, manager, connector, sync_context


class TestFlexKVConnectorPageAlignedLoad(unittest.TestCase):
    def test_lookup_cancels_raw_task_and_relooks_up_aligned_mask(self) -> None:
        """A partial raw match is cancelled before an aligned task is retained."""
        manager = _FakeKVManager(
            matches=[
                (101, np.ones(6, dtype=np.bool_)),
                (202, np.array([True, True, True, True, False, False])),
            ]
        )
        connector = _make_connector(manager)

        task_id, hit = connector.lookup_kv(
            token_ids=list(range(6)),
            token_mask=torch.ones(6, dtype=torch.bool),
            rid="request",
        )

        self.assertEqual(task_id, 202)
        self.assertEqual(hit, 4)
        self.assertEqual(manager.cancel_calls, [[101]])
        self.assertEqual([int(mask.sum()) for mask in manager.get_match_masks], [6, 4])
        self.assertEqual(
            connector._pending_lookups["request"].lookup_task_id,
            202,
        )

    def test_lookup_rebinds_nonprefix_mask_to_exact_leading_page(self) -> None:
        """A hole after an aligned prefix forces cancellation and exact relookup."""
        manager = _FakeKVManager(
            matches=[
                (101, np.array([True, True, True, True, False, True])),
                (202, np.array([True, True, True, True, False, False])),
            ]
        )
        connector = _make_connector(manager)

        task_id, hit = connector.lookup_kv(
            token_ids=list(range(6)),
            token_mask=torch.ones(6, dtype=torch.bool),
            rid="request",
        )

        self.assertEqual((task_id, hit), (202, 4))
        self.assertEqual(manager.cancel_calls, [[101]])
        self.assertEqual(
            manager.get_match_masks[1].tolist(),
            [True, True, True, True, False, False],
        )

    def test_lookup_hole_before_first_page_cancels_and_misses(self) -> None:
        """A hole before a full page prevents later hits from being published."""
        manager = _FakeKVManager(
            matches=[(101, np.array([True, True, False, True, True, True]))]
        )
        connector = _make_connector(manager)

        task_id, hit = connector.lookup_kv(
            token_ids=list(range(6)),
            token_mask=torch.ones(6, dtype=torch.bool),
            rid="request",
        )

        self.assertEqual((task_id, hit), (-1, 0))
        self.assertEqual(manager.cancel_calls, [[101]])
        self.assertNotIn("request", connector._pending_lookups)

    def test_relookup_exception_has_no_held_task(self) -> None:
        """A failed aligned relookup leaves the already-cancelled raw task absent."""
        manager = _FakeKVManager(
            matches=[
                (101, np.ones(6, dtype=np.bool_)),
                RuntimeError("relookup failed"),
            ]
        )
        connector = _make_connector(manager)

        task_id, hit = connector.lookup_kv(
            token_ids=list(range(6)),
            token_mask=torch.ones(6, dtype=torch.bool),
            rid="request",
        )

        self.assertEqual((task_id, hit), (-1, 0))
        self.assertEqual(manager.cancel_calls, [[101]])
        self.assertNotIn("request", connector._pending_lookups)

    def test_cancel_failure_retains_ambiguous_raw_lookup(self) -> None:
        """A failed raw-task cancellation preserves identity and fails loudly."""
        manager = _FakeKVManager(matches=[(101, np.ones(6, dtype=np.bool_))])
        manager.cancel_error = RuntimeError("cancel failed")
        connector = _make_connector(manager)

        with self.assertRaises(connector_module._FlexKVFatalTransferError):
            connector.lookup_kv(
                token_ids=list(range(6)),
                token_mask=torch.ones(6, dtype=torch.bool),
                rid="request",
            )

        self.assertEqual(
            connector._pending_lookups["request"].lookup_task_id,
            101,
        )

    def test_retrieve_waits_only_on_launch_terminal_ids(self) -> None:
        """MP completion uses launch terminal ids with a complete wait proof."""
        manager = _FakeKVManager(
            matches=[(202, np.array([True, True, True, True]))]
        )
        manager.terminal_task_ids = [900]
        connector = _make_connector(manager, send_to_remote=True)
        connector.lookup_kv(
            token_ids=list(range(4)),
            token_mask=torch.ones(4, dtype=torch.bool),
            rid="request",
        )

        result = connector.retrieve_kv(
            rid="request",
            slot_mapping=torch.tensor([10, 11, 12, 13]),
            expected_lookup_task_id=202,
        )

        self.assertEqual(result.lookup_task_id, 202)
        self.assertEqual(result.terminal_task_ids, (900,))
        self.assertTrue(result.terminal_proof)
        self.assertTrue(result.terminal_success)
        self.assertEqual(manager.launch_calls[0]["task_ids"], [202])
        self.assertEqual(manager.wait_calls[0]["task_ids"], [900])
        self.assertTrue(manager.wait_calls[0]["completely"])
        self.assertEqual(connector.tp_client.mapping_calls[0][0], 202)

    def test_allocation_failure_cancels_without_launching(self) -> None:
        """A local allocation failure becomes a clean collective prelaunch miss."""
        manager = _FakeKVManager(
            matches=[(202, np.array([True, True, True, True]))]
        )
        connector = _make_connector(manager)
        connector.lookup_kv(
            token_ids=list(range(4)),
            token_mask=torch.ones(4, dtype=torch.bool),
            rid="request",
        )

        result = connector.retrieve_kv(
            rid="request",
            slot_mapping=None,
            expected_lookup_task_id=202,
        )

        self.assertTrue(result.prelaunch_miss)
        self.assertFalse(result.terminal_proof)
        self.assertEqual(manager.cancel_calls, [[202]])
        self.assertEqual(manager.launch_calls, [])
        self.assertEqual(manager.wait_calls, [])

    def test_cross_rank_slot_id_mismatch_cancels_before_launch(self) -> None:
        """Different fragmented slot ids fail exact manifest consensus prelaunch."""
        manager = _FakeKVManager(matches=[])
        connector = _make_connector(manager)
        connector._pending_lookups["request"] = connector_module._PendingFlexKVLookup(
            lookup_task_id=202,
            expected_slots=4,
        )
        connector._sync_ctx = _ScriptedSyncContext(
            scatter_results=[
                {
                    "rid": "request",
                    "lookup_task_id": 202,
                    "expected_slots": 4,
                    "slot_mapping": [10, 11, 12, 13],
                },
                {
                    "lookup_task_id": 202,
                    "cancelled": True,
                    "error": None,
                },
            ],
            reduce_results=[-1, 0],
        )

        result = connector.retrieve_kv(
            rid="request",
            slot_mapping=torch.tensor([20, 21, 22, 23]),
            expected_lookup_task_id=202,
        )

        self.assertTrue(result.prelaunch_contract_error)
        self.assertEqual(manager.launch_calls, [])
        self.assertNotIn("request", connector._pending_lookups)

    def test_remote_mapping_send_exception_cancels_without_launch(self) -> None:
        """A fire-and-forget send exception is a clean prelaunch miss, not an ACK."""
        manager = _FakeKVManager(
            matches=[(202, np.array([True, True, True, True]))]
        )
        connector = _make_connector(manager, send_to_remote=True)
        connector.tp_client.set_slot_mapping = mock.Mock(
            side_effect=RuntimeError("send failed")
        )
        connector.lookup_kv(
            token_ids=list(range(4)),
            token_mask=torch.ones(4, dtype=torch.bool),
            rid="request",
        )

        result = connector.retrieve_kv(
            rid="request",
            slot_mapping=torch.tensor([10, 11, 12, 13]),
            expected_lookup_task_id=202,
        )

        self.assertTrue(result.prelaunch_miss)
        self.assertEqual(manager.cancel_calls, [[202]])
        self.assertEqual(manager.launch_calls, [])

    def test_follower_uses_broadcast_terminal_proof_without_launching(self) -> None:
        """A follower consumes leader terminal proof and never launches locally."""
        manager = _FakeKVManager(matches=[])
        connector = _make_connector(manager)
        connector._pending_lookups["request"] = connector_module._PendingFlexKVLookup(
            lookup_task_id=202,
            expected_slots=4,
        )
        connector._sync_ctx = _ScriptedSyncContext(
            scatter_results=[
                {
                    "rid": "request",
                    "lookup_task_id": 202,
                    "expected_slots": 4,
                    "slot_mapping": [10, 11, 12, 13],
                },
                {
                    "terminal_task_ids": [900],
                    "requested_slots": 4,
                    "error": None,
                },
                {"complete": True, "successful": True, "error": None},
            ],
            reduce_results=[1],
        )

        result = connector.retrieve_kv(
            rid="request",
            slot_mapping=torch.tensor([10, 11, 12, 13]),
            expected_lookup_task_id=202,
        )

        self.assertTrue(result.terminal_proof)
        self.assertEqual(result.terminal_task_ids, (900,))
        self.assertEqual(manager.launch_calls, [])
        self.assertEqual(manager.wait_calls, [])

    def test_proven_terminal_failure_releases_connector_identity(self) -> None:
        """A complete failed terminal response is safe to clean up as a miss."""
        manager = _FakeKVManager(
            matches=[(202, np.array([True, True, True, True]))]
        )
        manager.wait_status = _StubKVResponseStatus.FAILED
        connector = _make_connector(manager)
        connector.lookup_kv(
            token_ids=list(range(4)),
            token_mask=torch.ones(4, dtype=torch.bool),
            rid="request",
        )

        result = connector.retrieve_kv(
            rid="request",
            slot_mapping=torch.tensor([10, 11, 12, 13]),
            expected_lookup_task_id=202,
        )

        self.assertTrue(result.terminal_proof)
        self.assertFalse(result.terminal_success)
        self.assertNotIn("request", connector._pending_lookups)

    def test_terminal_failure_retains_unproven_lookup_state(self) -> None:
        """A missing terminal proof remains fail-stop and cannot be cancelled."""
        manager = _FakeKVManager(
            matches=[(202, np.array([True, True, True, True]))]
        )
        manager.wait_status = _StubKVResponseStatus.NOTFOUND
        connector = _make_connector(manager)
        connector.lookup_kv(
            token_ids=list(range(4)),
            token_mask=torch.ones(4, dtype=torch.bool),
            rid="request",
        )

        with self.assertRaises(connector_module._FlexKVFatalTransferError):
            connector.retrieve_kv(
                rid="request",
                slot_mapping=torch.tensor([10, 11, 12, 13]),
                expected_lookup_task_id=202,
            )
        with self.assertRaises(connector_module._FlexKVFatalTransferError):
            connector.release_pending("request")
        with self.assertRaises(connector_module._FlexKVFatalTransferError):
            connector.reset()

        self.assertIn("request", connector._pending_lookups)
        self.assertEqual(manager.cancel_calls, [])

    def test_timeout_and_missing_terminal_response_remain_ambiguous(self) -> None:
        """Timeout and missing terminal identities never release destination ownership."""
        for omit_response in (False, True):
            with self.subTest(omit_response=omit_response):
                manager = _FakeKVManager(
                    matches=[(202, np.array([True, True, True, True]))]
                )
                manager.wait_status = _StubKVResponseStatus.TIMEOUT
                manager.omit_wait_response = omit_response
                connector = _make_connector(manager)
                connector.lookup_kv(
                    token_ids=list(range(4)),
                    token_mask=torch.ones(4, dtype=torch.bool),
                    rid="request",
                )

                with self.assertRaises(connector_module._FlexKVFatalTransferError):
                    connector.retrieve_kv(
                        rid="request",
                        slot_mapping=torch.tensor([10, 11, 12, 13]),
                        expected_lookup_task_id=202,
                    )

                self.assertIn("request", connector._pending_lookups)

    def test_page_size_one_layerwise_lookup_launches_original_holey_task(self) -> None:
        """Legacy layerwise lookup keeps a holey match and launches its original id."""
        manager = _FakeKVManager(
            matches=[(101, np.array([True, False, True], dtype=np.bool_))]
        )
        connector = _make_connector(manager, allocator_page_size=1)
        connector.enable_layerwise = True
        connector.layer_done_counter = _FakeLayerDoneCounter()

        task_id, hit = connector.lookup_kv(
            token_ids=[10, 11, 12],
            token_mask=torch.ones(3, dtype=torch.bool),
            rid="request",
        )
        loaded, producer_id = connector.start_load_kv_layerwise(
            rid="request",
            slot_mapping=torch.tensor([40, 42]),
        )

        self.assertEqual((task_id, hit), (101, 2))
        self.assertEqual(len(manager.get_match_masks), 1)
        self.assertEqual(manager.cancel_calls, [])
        self.assertEqual((loaded, producer_id), (2, 0))
        self.assertEqual(manager.launch_calls[0]["task_ids"], [101])
        self.assertTrue(manager.launch_calls[0]["layerwise_transfer"])
        self.assertEqual(connector.layer_done_counter.register_calls, [(101, 0)])
        self.assertEqual(connector.layer_done_counter.consumer_calls, [101])
        self.assertEqual(
            connector.layer_done_counter.events[0].reset_calls,
            1,
        )

    def test_connector_layerwise_gate_precedes_configuration(self) -> None:
        """The defensive page-size gate runs before FlexKV configuration."""
        with mock.patch.dict(
            os.environ,
            {"FLEXKV_ENABLE_LAYERWISE_TRANSFER": "1"},
        ), mock.patch.object(
            connector_module.FlexKVConfig,
            "from_env",
        ) as from_env:
            with self.assertRaisesRegex(ValueError, "allocator page size 1"):
                connector_module.FlexKVConnector(
                    sgl_model_config=None,
                    server_args=None,
                    page_size=1,
                    allocator_page_size=4,
                    kvcache=None,
                    tp_rank=0,
                    dp_rank=0,
                    pp_rank=0,
                    attn_cp_rank=0,
                )

        from_env.assert_not_called()


class TestFlexKVRadixPageOwnership(unittest.TestCase):
    def test_page_size_one_layerwise_helper_preserves_legacy_route(self) -> None:
        """Page-size-one layerwise load still publishes every retrieved token."""
        allocator = _FakeAllocator(
            slots=torch.tensor([5, 6]),
            page_size=1,
        )
        cache = _make_radix_cache(allocator, connector=mock.Mock())
        parent = radix_module.TreeNode(priority=0)

        loaded_slots, node = cache._allocate_and_load_layerwise(
            key=radix_module.RadixKey(array("q", range(2))),
            value_numel=0,
            uncached_len=2,
            last_node=parent,
            load_fn=lambda slot_mapping: int(slot_mapping.numel()),
        )

        self.assertEqual(loaded_slots.tolist(), [5, 6])
        self.assertEqual(node.value.tolist(), [5, 6])
        self.assertEqual(allocator.free_calls, [])

    def test_remote_rank_shortage_forces_symmetric_evict_before_alloc(self) -> None:
        """A scripted remote shortage makes every successful rank evict in order."""
        allocator = _FakeAllocator(slots=torch.tensor([1, 2, 3, 4]))
        cache, manager, connector, sync_context = _make_production_mp_cache(
            allocator=allocator,
            scatter_results=[
                {
                    "rid": "request",
                    "lookup_task_id": 202,
                    "expected_slots": 4,
                    "slot_mapping": [1, 2, 3, 4],
                },
                {
                    "lookup_task_id": 202,
                    "cancelled": True,
                    "error": None,
                },
            ],
            reduce_results=[0, 0, 0],
        )
        cache.evict = mock.Mock(
            side_effect=lambda params: allocator.events.append("evict")
        )

        loaded = cache._allocate_and_load_mp(
            rid="request",
            lookup_task_id=202,
            key=radix_module.RadixKey(array("q", range(4))),
            value_numel=0,
            uncached_len=4,
            last_node=radix_module.TreeNode(priority=0),
        )

        self.assertIsNone(loaded)
        self.assertEqual(
            allocator.events,
            ["available", "evict", "alloc", "free"],
        )
        self.assertEqual(
            [event for event, _ in sync_context.events],
            ["reduce", "scatter", "reduce", "reduce", "scatter"],
        )
        self.assertEqual(allocator.free_calls[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(manager.cancel_calls, [[202]])
        self.assertEqual(manager.launch_calls, [])
        self.assertNotIn("request", connector._pending_lookups)

    def test_capacity_exception_still_enters_symmetric_evict_and_consensus(self) -> None:
        """A capacity exception preserves collective order and prevents launch."""
        allocator = _FakeAllocator(slots=torch.tensor([1, 2, 3, 4]))
        allocator.available_error = RuntimeError("capacity failed")
        cache, manager, connector, sync_context = _make_production_mp_cache(
            allocator=allocator,
            scatter_results=[
                {
                    "rid": "request",
                    "lookup_task_id": 202,
                    "expected_slots": 4,
                    "slot_mapping": None,
                },
                {
                    "lookup_task_id": 202,
                    "cancelled": True,
                    "error": None,
                },
            ],
            reduce_results=[0, 0, 0],
        )
        cache.evict = mock.Mock(
            side_effect=lambda params: allocator.events.append("evict")
        )

        loaded = cache._allocate_and_load_mp(
            rid="request",
            lookup_task_id=202,
            key=radix_module.RadixKey(array("q", range(4))),
            value_numel=0,
            uncached_len=4,
            last_node=radix_module.TreeNode(priority=0),
        )

        self.assertIsNone(loaded)
        self.assertEqual(allocator.events, ["available", "evict"])
        self.assertEqual(
            [event for event, _ in sync_context.events],
            ["reduce", "scatter", "reduce", "reduce", "scatter"],
        )
        self.assertEqual(manager.cancel_calls, [[202]])
        self.assertEqual(manager.launch_calls, [])
        self.assertNotIn("request", connector._pending_lookups)

    def test_evict_and_alloc_exceptions_reach_prelaunch_consensus(self) -> None:
        """Eviction and allocation exceptions become local prelaunch failures."""
        for failure in ("evict", "alloc"):
            with self.subTest(failure=failure):
                allocator = _FakeAllocator(
                    slots=torch.tensor([1, 2, 3, 4]),
                    available=0 if failure == "evict" else 1024,
                )
                if failure == "alloc":
                    allocator.alloc_error = RuntimeError("alloc failed")
                cache, manager, connector, sync_context = (
                    _make_production_mp_cache(
                        allocator=allocator,
                        scatter_results=[
                            {
                                "rid": "request",
                                "lookup_task_id": 202,
                                "expected_slots": 4,
                                "slot_mapping": None,
                            },
                            {
                                "lookup_task_id": 202,
                                "cancelled": True,
                                "error": None,
                            },
                        ],
                        reduce_results=[0 if failure == "evict" else 1, 0, 0],
                    )
                )
                if failure == "evict":

                    def fail_eviction(params: Any) -> None:
                        allocator.events.append("evict")
                        raise RuntimeError("evict failed")

                    cache.evict = mock.Mock(side_effect=fail_eviction)

                loaded = cache._allocate_and_load_mp(
                    rid="request",
                    lookup_task_id=202,
                    key=radix_module.RadixKey(array("q", range(4))),
                    value_numel=0,
                    uncached_len=4,
                    last_node=radix_module.TreeNode(priority=0),
                )

                self.assertIsNone(loaded)
                self.assertEqual(
                    [event for event, _ in sync_context.events],
                    ["reduce", "scatter", "reduce", "reduce", "scatter"],
                )
                self.assertEqual(manager.cancel_calls, [[202]])
                self.assertEqual(manager.launch_calls, [])
                self.assertNotIn("request", connector._pending_lookups)

    def test_slot_count_mismatch_uses_production_prelaunch_consensus(self) -> None:
        """A real connector count mismatch cancels before launch and frees once."""
        allocator = _FakeAllocator(slots=torch.tensor([1, 2, 3, 4]))
        cache, manager, connector, sync_context = _make_production_mp_cache(
            allocator=allocator,
            scatter_results=[
                {
                    "rid": "request",
                    "lookup_task_id": 202,
                    "expected_slots": 4,
                    "slot_mapping": None,
                },
                {
                    "lookup_task_id": 202,
                    "cancelled": True,
                    "error": None,
                },
            ],
            reduce_results=[1, -1, 0],
        )

        with self.assertRaisesRegex(RuntimeError, "manifest validation failed"):
            cache._allocate_and_load_mp(
                rid="request",
                lookup_task_id=202,
                key=radix_module.RadixKey(array("q", range(3))),
                value_numel=0,
                uncached_len=3,
                last_node=radix_module.TreeNode(priority=0),
            )

        self.assertEqual(len(allocator.free_calls), 1)
        self.assertEqual(allocator.free_calls[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(
            [event for event, _ in sync_context.events],
            ["reduce", "scatter", "reduce", "reduce", "scatter"],
        )
        self.assertEqual(manager.cancel_calls, [[202]])
        self.assertEqual(manager.launch_calls, [])
        self.assertNotIn("request", connector._pending_lookups)

    def test_remote_send_exception_uses_production_prelaunch_consensus(self) -> None:
        """A real remote mapping send failure cancels and frees before launch."""
        allocator = _FakeAllocator(slots=torch.tensor([5, 6, 7, 8]))
        cache, manager, connector, sync_context = _make_production_mp_cache(
            allocator=allocator,
            scatter_results=[
                {
                    "rid": "request",
                    "lookup_task_id": 202,
                    "expected_slots": 4,
                    "slot_mapping": [5, 6, 7, 8],
                },
                {
                    "lookup_task_id": 202,
                    "cancelled": True,
                    "error": None,
                },
            ],
            reduce_results=[1, 0, 0],
            send_to_remote=True,
        )
        connector.tp_client.set_slot_mapping = mock.Mock(
            side_effect=RuntimeError("send failed")
        )

        loaded = cache._allocate_and_load_mp(
            rid="request",
            lookup_task_id=202,
            key=radix_module.RadixKey(array("q", range(4))),
            value_numel=0,
            uncached_len=4,
            last_node=radix_module.TreeNode(priority=0),
        )

        self.assertIsNone(loaded)
        self.assertEqual(len(allocator.free_calls), 1)
        self.assertEqual(allocator.free_calls[0].tolist(), [5, 6, 7, 8])
        self.assertEqual(
            [event for event, _ in sync_context.events],
            ["reduce", "scatter", "reduce", "reduce", "scatter"],
        )
        self.assertEqual(manager.cancel_calls, [[202]])
        self.assertEqual(manager.launch_calls, [])
        self.assertNotIn("request", connector._pending_lookups)

    def test_partial_hit_publishes_one_page_and_frees_disjoint_tail(self) -> None:
        """A partial trailing hit never enters the published radix node."""
        slots = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
        allocator = _FakeAllocator(slots=slots)
        result = connector_module._FlexKVRetrieveResult(
            lookup_task_id=202,
            terminal_task_ids=(900,),
            requested_slots=6,
            terminal_proof=True,
            terminal_success=True,
            prelaunch_miss=False,
            prelaunch_contract_error=False,
        )
        connector = _FakeRadixConnector(result)
        cache = _make_radix_cache(allocator, connector)
        parent = radix_module.TreeNode(priority=0)

        loaded_slots, node = cache._allocate_and_load_mp(
            rid="request",
            lookup_task_id=202,
            key=radix_module.RadixKey(array("q", range(6))),
            value_numel=0,
            uncached_len=6,
            last_node=parent,
        )

        self.assertEqual(allocator.alloc_sizes, [8])
        self.assertEqual(connector.retrieve_calls[0]["slot_mapping"].numel(), 6)
        self.assertEqual(loaded_slots.tolist(), [10, 11, 12, 13])
        self.assertEqual(len(node.key), 4)
        self.assertEqual(node.value.tolist(), [10, 11, 12, 13])
        self.assertEqual(cache.evictable_size_, 4)
        self.assertEqual(allocator.free_calls[0].tolist(), [14, 15, 16, 17])

        allocator.free(node.value)
        self.assertTrue(
            set(allocator.free_calls[0].tolist()).isdisjoint(
                allocator.free_calls[1].tolist()
            )
        )

    def test_subpage_hit_frees_lease_without_creating_child(self) -> None:
        """A hit shorter than one allocator page creates no radix child."""
        allocator = _FakeAllocator(slots=torch.tensor([20, 21, 22, 23]))
        result = connector_module._FlexKVRetrieveResult(
            lookup_task_id=303,
            terminal_task_ids=(901,),
            requested_slots=2,
            terminal_proof=True,
            terminal_success=True,
            prelaunch_miss=False,
            prelaunch_contract_error=False,
        )
        cache = _make_radix_cache(allocator, _FakeRadixConnector(result))
        parent = radix_module.TreeNode(priority=0)

        loaded = cache._allocate_and_load_mp(
            rid="request",
            lookup_task_id=303,
            key=radix_module.RadixKey(array("q", range(2))),
            value_numel=0,
            uncached_len=2,
            last_node=parent,
        )

        self.assertIsNone(loaded)
        self.assertEqual(len(parent.children), 0)
        self.assertEqual(cache.evictable_size_, 0)
        self.assertEqual(allocator.free_calls[0].tolist(), [20, 21, 22, 23])
        self.assertEqual(cache._pending_mp_leases, {})

    def test_proven_terminal_failure_frees_full_lease(self) -> None:
        """A proven failed transfer returns its complete allocation once."""
        allocator = _FakeAllocator(slots=torch.tensor([24, 25, 26, 27]))
        result = connector_module._FlexKVRetrieveResult(
            lookup_task_id=304,
            terminal_task_ids=(902,),
            requested_slots=4,
            terminal_proof=True,
            terminal_success=False,
            prelaunch_miss=False,
            prelaunch_contract_error=False,
        )
        cache = _make_radix_cache(allocator, _FakeRadixConnector(result))
        parent = radix_module.TreeNode(priority=0)

        loaded = cache._allocate_and_load_mp(
            rid="request",
            lookup_task_id=304,
            key=radix_module.RadixKey(array("q", range(4))),
            value_numel=0,
            uncached_len=4,
            last_node=parent,
        )

        self.assertIsNone(loaded)
        self.assertEqual(len(parent.children), 0)
        self.assertEqual(allocator.free_calls[0].tolist(), [24, 25, 26, 27])

    def test_radix_layerwise_gate_precedes_kvcache_access(self) -> None:
        """The radix gate rejects page-sized layerwise mode before KV access."""
        allocator = mock.Mock()
        allocator.page_size = 4
        params = SimpleNamespace(
            page_size=1,
            token_to_kv_pool_allocator=allocator,
        )

        initialize_base = mock.Mock()

        with mock.patch.object(
            radix_module.RadixCache,
            "__init__",
            new=initialize_base,
        ), mock.patch.dict(
            os.environ,
            {"FLEXKV_ENABLE_LAYERWISE_TRANSFER": "1"},
        ):
            with self.assertRaisesRegex(ValueError, "allocator page size 1"):
                radix_module.FlexKVRadixCache(
                    params=params,
                    model_config=None,
                    server_args=None,
                    tp_rank=0,
                    tp_size=1,
                    dp_rank=0,
                    pp_rank=0,
                    attn_cp_rank=0,
                )

        allocator.get_kvcache.assert_not_called()
        initialize_base.assert_not_called()

    def test_ambiguous_retrieve_retains_full_allocator_lease(self) -> None:
        """An ambiguous launched transfer retains every allocated destination."""
        allocator = _FakeAllocator(slots=torch.tensor([30, 31, 32, 33]))
        connector = mock.Mock()
        connector.requires_mp_eviction.return_value = False
        connector.retrieve_kv.side_effect = connector_module._FlexKVFatalTransferError(
            "terminal timeout"
        )
        cache = _make_radix_cache(allocator, connector)
        parent = radix_module.TreeNode(priority=0)

        with self.assertRaises(connector_module._FlexKVFatalTransferError):
            cache._allocate_and_load_mp(
                rid="request",
                lookup_task_id=404,
                key=radix_module.RadixKey(array("q", range(4))),
                value_numel=0,
                uncached_len=4,
                last_node=parent,
            )

        self.assertEqual(len(cache._pending_mp_leases), 1)
        retained = next(iter(cache._pending_mp_leases.values()))
        self.assertEqual(retained.slots.tolist(), [30, 31, 32, 33])
        self.assertEqual(allocator.free_calls, [])
        with self.assertRaisesRegex(RuntimeError, "lack terminal proof"):
            cache.reset()
        with self.assertRaisesRegex(RuntimeError, "lack terminal proof"):
            cache.shutdown()
        with self.assertRaisesRegex(RuntimeError, "lacks terminal proof"):
            cache.release_aborted_request("request")


if __name__ == "__main__":
    unittest.main()
