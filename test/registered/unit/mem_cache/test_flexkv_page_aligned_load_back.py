import importlib
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from enum import Enum
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_CONNECTOR_MODULE_NAME = "sglang.srt.mem_cache.storage.flexkv.flexkv_connector"
_RADIX_MODULE_NAME = "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache"
_MISSING = object()


class _ResponseStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


def _make_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


def _fake_flexkv_modules() -> dict[str, types.ModuleType]:
    return {
        "flexkv": _make_module("flexkv"),
        "flexkv.common": _make_module("flexkv.common"),
        "flexkv.common.request": _make_module(
            "flexkv.common.request", KVResponseStatus=_ResponseStatus
        ),
        "flexkv.common.storage": _make_module(
            "flexkv.common.storage",
            KVCacheLayout=object,
            KVCacheLayoutType=object,
        ),
        "flexkv.integration": _make_module("flexkv.integration"),
        "flexkv.integration.config": _make_module(
            "flexkv.integration.config", FlexKVConfig=object
        ),
        "flexkv.kvmanager": _make_module("flexkv.kvmanager", KVManager=object),
        "flexkv.server": _make_module("flexkv.server"),
        "flexkv.server.client": _make_module("flexkv.server.client", KVTPClient=object),
        "flexkv.transfer": _make_module("flexkv.transfer"),
        "flexkv.transfer.layerwise": _make_module(
            "flexkv.transfer.layerwise",
            build_layerwise_eventfd_socket_path=object,
        ),
        "flexkv.transfer_manager": _make_module(
            "flexkv.transfer_manager", TransferManagerOnRemote=object
        ),
    }


@contextmanager
def _import_flexkv_modules() -> Iterator[tuple[types.ModuleType, types.ModuleType]]:
    module_names = (_RADIX_MODULE_NAME, _CONNECTOR_MODULE_NAME)
    existing_modules = {name: sys.modules.pop(name, None) for name in module_names}
    parent_attributes: dict[str, tuple[types.ModuleType, str, Any]] = {}
    for module_name in module_names:
        parent_name, _, attribute = module_name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            parent_attributes[module_name] = (
                parent,
                attribute,
                getattr(parent, attribute, _MISSING),
            )
    try:
        with patch.dict(sys.modules, _fake_flexkv_modules()):
            connector_module = importlib.import_module(_CONNECTOR_MODULE_NAME)
            radix_module = importlib.import_module(_RADIX_MODULE_NAME)
            yield connector_module, radix_module
    finally:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
            existing_module = existing_modules[module_name]
            if existing_module is not None:
                sys.modules[module_name] = existing_module
            parent_state = parent_attributes.get(module_name)
            if parent_state is None:
                continue
            parent, attribute, value = parent_state
            if value is _MISSING:
                parent.__dict__.pop(attribute, None)
            else:
                setattr(parent, attribute, value)


class _FakeSyncContext:
    def __init__(self) -> None:
        self.is_sync_leader = True
        self.needs_sync = False
        self.should_send_slot_mapping_to_remote = False
        self.is_pp_receiver = False
        self.minimum_inputs: list[int] = []
        self.scatter_manifest: Any = None
        self.stage_manifest: Any = None

    def all_reduce_min(self, value: int) -> int:
        self.minimum_inputs.append(value)
        return value

    def scatter(self, value: Any) -> Any:
        return value if self.scatter_manifest is None else self.scatter_manifest

    def scatter_stage(self, value: Any) -> Any:
        return value if self.stage_manifest is None else self.stage_manifest


def _make_connector(
    module: types.ModuleType,
) -> tuple[Any, _FakeSyncContext, MagicMock]:
    connector = module.FlexKVConnector.__new__(module.FlexKVConnector)
    sync_context = _FakeSyncContext()
    manager = MagicMock()
    connector.storage_page_size = 1
    connector.allocator_page_size = 2
    connector.enable_layerwise = False
    connector.layer_done_counter = None
    connector._sync_ctx = sync_context
    connector.kv_manager = manager
    connector._pending_lookups = {}
    connector._ambiguous_loads = {}
    connector._poison_reason = None
    connector._launched_load_tids = []
    connector._inflight_stores = {}
    connector._ambiguous_stores = {}
    connector._store_owner_required = set()
    connector._ongoing_prefetches = {}
    connector._inflight_loads = {}
    connector._completed_layerwise = set()
    return connector, sync_context, manager


def _response(task_id: int, status: _ResponseStatus) -> SimpleNamespace:
    return SimpleNamespace(task_id=task_id, status=status)


def _make_store_cache(radix_module: types.ModuleType) -> tuple[Any, MagicMock]:
    cache = radix_module.FlexKVRadixCache.__new__(radix_module.FlexKVRadixCache)
    cache.device = torch.device("cpu")
    cache._allocator_page_size = 2
    cache.is_eagle = False
    cache._load_markers = {}
    cache._node_lock = threading.Lock()
    cache._inflight_store_nodes = {}
    cache.store_stream = object()
    cache.inc_lock_ref = MagicMock()
    cache.dec_lock_ref = MagicMock()
    cache.flexkv_connector = MagicMock()
    cache.flexkv_connector.store_kv.return_value = -1
    cache.flexkv_connector.store_requires_owner_lock.return_value = False
    return cache, cache.flexkv_connector


class TestFlexKVPageAlignedLoadBack(unittest.TestCase):
    def test_exact_lookup_is_consumed_without_requery(self) -> None:
        """A held aligned lookup drives the exact launch and wait once."""
        with _import_flexkv_modules() as (module, _):
            connector, _, manager = _make_connector(module)
            manager.get_match.return_value = (
                17,
                np.asarray([False, False, True, True, True, True]),
            )
            manager.launch.return_value = [91]
            manager.wait.return_value = {
                91: _response(task_id=91, status=_ResponseStatus.SUCCESS)
            }

            _, hit_length = connector.lookup_kv(
                token_ids=list(range(6)),
                token_mask=torch.tensor([False, False, True, True, True, True]),
                rid="request",
            )
            result = connector.retrieve_kv(
                rid="request",
                slot_mapping=torch.tensor([100, 101, 102, 103]),
            )

            self.assertEqual(hit_length, 4)
            self.assertIs(result.status, module.FlexKVRetrieveStatus.SUCCESS)
            manager.get_match.assert_called_once()
            launch_kwargs = manager.launch.call_args.kwargs
            self.assertEqual(launch_kwargs["task_ids"], [17])
            self.assertTrue(
                torch.equal(
                    launch_kwargs["slot_mappings"][0],
                    torch.tensor([100, 101, 102, 103]),
                )
            )
            self.assertFalse(launch_kwargs["as_batch"])
            self.assertFalse(launch_kwargs["layerwise_transfer"])
            manager.wait.assert_called_once_with(
                [91],
                timeout=30.0,
                completely=True,
            )

    def test_layerwise_large_page_rejects_before_external_setup(self) -> None:
        """Layerwise transfer rejects multi-token allocator pages immediately."""
        with _import_flexkv_modules() as (module, _):
            with (
                patch.dict("os.environ", {"FLEXKV_ENABLE_LAYERWISE_TRANSFER": "1"}),
                self.assertRaisesRegex(ValueError, "requires allocator page size 1"),
            ):
                module.FlexKVConnector(
                    sgl_model_config=object(),
                    server_args=object(),
                    page_size=1,
                    allocator_page_size=2,
                    kvcache=object(),
                    tp_rank=0,
                    dp_rank=0,
                    pp_rank=0,
                    attn_cp_rank=0,
                )

    def test_prelaunch_failure_still_coordinates_and_never_launches(self) -> None:
        """An invalid slot manifest reaches the fixed decision collective."""
        with _import_flexkv_modules() as (module, _):
            connector, sync_context, manager = _make_connector(module)
            connector._pending_lookups["request"] = module._PendingFlexKVLookup(
                task_id=17,
                expected_slots=4,
            )

            result = connector.retrieve_kv(
                rid="request",
                slot_mapping=torch.tensor([100, 101, 102]),
            )

            self.assertIs(
                result.status,
                module.FlexKVRetrieveStatus.DEFINITE_TERMINAL_FAILURE,
            )
            self.assertEqual(sync_context.minimum_inputs, [0])
            manager.cancel.assert_called_once_with([17])
            manager.launch.assert_not_called()

    def test_stage_mapping_mismatch_cancels_before_store_launch(self) -> None:
        """A stage-local store mapping mismatch cancels the held put task."""
        with _import_flexkv_modules() as (module, _):
            connector, sync_context, manager = _make_connector(module)
            sync_context.stage_manifest = [200, 201]
            manager.put_match.return_value = (
                33,
                np.asarray([True, True, False, False]),
            )

            result = connector.store_kv(
                rid="request",
                token_ids=list(range(4)),
                kv_indices=torch.tensor([100, 101, 102, 103]),
            )

            self.assertEqual(result, -1)
            manager.cancel.assert_called_once_with([33])
            manager.launch.assert_not_called()

    def test_asymmetric_store_preparation_failure_reaches_fixed_consensus(
        self,
    ) -> None:
        """A rank-local source failure joins the shared preflight rejection."""
        with _import_flexkv_modules() as (module, _):
            connector, sync_context, manager = _make_connector(module)
            sync_context.needs_sync = True
            sync_context.scatter_manifest = {
                "rid": "request",
                "token_ids": [0, 1],
                "reason": None,
            }

            result = connector.store_kv(
                rid="request",
                token_ids=[],
                kv_indices=torch.empty((0,), dtype=torch.int64),
                local_preparation_error="canonical rematch failed",
            )

            self.assertEqual(result, -1)
            self.assertEqual(sync_context.minimum_inputs, [0])
            manager.put_match.assert_not_called()
            manager.launch.assert_not_called()

    def test_cache_finished_store_uses_canonical_mapping_after_duplicate_handoff(
        self,
    ) -> None:
        """STORE reads the canonical node after base releases duplicate slots."""
        with _import_flexkv_modules() as (_, radix_module):
            cache, connector = _make_store_cache(radix_module)
            cache.req_to_token_pool = SimpleNamespace()
            node = object()
            canonical_indices = torch.tensor([200, 201, 202, 203])
            req = SimpleNamespace(
                rid="request",
                origin_input_ids=[0, 1, 2],
                output_ids=[3, 4, 5],
                extra_key=None,
                kv_committed_len=6,
            )

            with (
                patch.object(
                    radix_module.RadixCache,
                    "cache_finished_req",
                    return_value=radix_module.CacheFinishedReqResult(
                        unhandled_kv_start=4
                    ),
                ),
                patch.object(
                    radix_module.RadixCache,
                    "match_prefix",
                    return_value=SimpleNamespace(
                        device_indices=canonical_indices,
                        last_device_node=node,
                    ),
                ),
                patch.object(torch.cuda, "stream", return_value=MagicMock()),
            ):
                result = cache.cache_finished_req(
                    req,
                    is_insert=True,
                    kv_len_to_handle=6,
                )

            self.assertEqual(result.unhandled_kv_start, 4)
            connector.store_kv.assert_called_once()
            store_kwargs = connector.store_kv.call_args.kwargs
            self.assertEqual(store_kwargs["token_ids"], [0, 1, 2, 3])
            self.assertTrue(torch.equal(store_kwargs["kv_indices"], canonical_indices))
            self.assertIsNone(store_kwargs["local_preparation_error"])
            cache.inc_lock_ref.assert_called_once_with(node)
            cache.dec_lock_ref.assert_called_once_with(node)

    def test_cache_finished_store_length_comes_only_from_base_handoff(self) -> None:
        """A shorter base handoff overrides the request committed-length field."""
        with _import_flexkv_modules() as (_, radix_module):
            cache, connector = _make_store_cache(radix_module)
            node = object()
            canonical_indices = torch.tensor([300, 301])
            req = SimpleNamespace(
                rid="request",
                origin_input_ids=[0, 1, 2],
                output_ids=[3, 4, 5],
                extra_key=None,
                kv_committed_len=6,
            )

            with (
                patch.object(
                    radix_module.RadixCache,
                    "cache_finished_req",
                    return_value=radix_module.CacheFinishedReqResult(
                        unhandled_kv_start=2
                    ),
                ),
                patch.object(
                    radix_module.RadixCache,
                    "match_prefix",
                    return_value=SimpleNamespace(
                        device_indices=canonical_indices,
                        last_device_node=node,
                    ),
                ),
                patch.object(torch.cuda, "stream", return_value=MagicMock()),
            ):
                cache.cache_finished_req(
                    req,
                    is_insert=True,
                    kv_len_to_handle=6,
                )

            store_kwargs = connector.store_kv.call_args.kwargs
            self.assertEqual(store_kwargs["token_ids"], [0, 1])
            self.assertTrue(torch.equal(store_kwargs["kv_indices"], canonical_indices))

    def test_eagle_store_rematches_with_raw_boundary_token(self) -> None:
        """Eagle rematch adds one raw token without extending STORE payload."""
        with _import_flexkv_modules() as (_, radix_module):
            cache, connector = _make_store_cache(radix_module)
            cache.is_eagle = True
            node = object()
            canonical_indices = torch.tensor([400, 401, 402, 403])
            req = SimpleNamespace(
                rid="request",
                origin_input_ids=[0, 1, 2],
                output_ids=[3, 4, 5],
                extra_key=None,
            )

            with (
                patch.object(
                    radix_module.RadixCache,
                    "cache_finished_req",
                    return_value=radix_module.CacheFinishedReqResult(
                        unhandled_kv_start=4
                    ),
                ),
                patch.object(
                    radix_module.RadixCache,
                    "match_prefix",
                    return_value=SimpleNamespace(
                        device_indices=canonical_indices,
                        last_device_node=node,
                    ),
                ) as match_prefix,
                patch.object(torch.cuda, "stream", return_value=MagicMock()),
            ):
                cache.cache_finished_req(
                    req,
                    is_insert=True,
                    kv_len_to_handle=5,
                )

            match_key = match_prefix.call_args.args[0].key
            self.assertTrue(match_key.is_bigram)
            self.assertEqual(list(match_key.raw_token_ids()), [0, 1, 2, 3, 4])
            store_kwargs = connector.store_kv.call_args.kwargs
            self.assertEqual(store_kwargs["token_ids"], [0, 1, 2, 3])
            self.assertTrue(torch.equal(store_kwargs["kv_indices"], canonical_indices))

    def test_cache_finished_rematch_failure_enters_connector_preflight(self) -> None:
        """A local canonical rematch error is passed to connector consensus."""
        with _import_flexkv_modules() as (_, radix_module):
            cache, connector = _make_store_cache(radix_module)
            req = SimpleNamespace(
                rid="request",
                origin_input_ids=[0, 1],
                output_ids=[],
                extra_key=None,
            )

            with (
                patch.object(
                    radix_module.RadixCache,
                    "cache_finished_req",
                    return_value=radix_module.CacheFinishedReqResult(
                        unhandled_kv_start=2
                    ),
                ),
                patch.object(
                    radix_module.RadixCache,
                    "match_prefix",
                    side_effect=RuntimeError("rematch failed"),
                ),
                patch.object(torch.cuda, "stream", return_value=MagicMock()),
            ):
                result = cache.cache_finished_req(
                    req,
                    is_insert=True,
                    kv_len_to_handle=2,
                )

            self.assertEqual(result.unhandled_kv_start, 2)
            store_kwargs = connector.store_kv.call_args.kwargs
            self.assertEqual(store_kwargs["token_ids"], [0, 1])
            self.assertEqual(store_kwargs["kv_indices"].numel(), 0)
            self.assertIn("rematch failed", store_kwargs["local_preparation_error"])
            cache.inc_lock_ref.assert_not_called()
            cache.dec_lock_ref.assert_not_called()

    def test_cache_finished_preowner_exception_releases_canonical_node(self) -> None:
        """A pre-owner connector exception releases the canonical source lock."""
        with _import_flexkv_modules() as (_, radix_module):
            cache, connector = _make_store_cache(radix_module)
            connector.store_kv.side_effect = RuntimeError("pre-owner failure")
            node = object()
            canonical_indices = torch.tensor([500, 501])
            req = SimpleNamespace(
                rid="request",
                origin_input_ids=[0, 1],
                output_ids=[],
                extra_key=None,
            )

            with (
                patch.object(
                    radix_module.RadixCache,
                    "cache_finished_req",
                    return_value=radix_module.CacheFinishedReqResult(
                        unhandled_kv_start=2
                    ),
                ),
                patch.object(
                    radix_module.RadixCache,
                    "match_prefix",
                    return_value=SimpleNamespace(
                        device_indices=canonical_indices,
                        last_device_node=node,
                    ),
                ),
                patch.object(torch.cuda, "stream", return_value=MagicMock()),
                self.assertRaisesRegex(RuntimeError, "pre-owner failure"),
            ):
                cache.cache_finished_req(
                    req,
                    is_insert=True,
                    kv_len_to_handle=2,
                )

            cache.inc_lock_ref.assert_called_once_with(node)
            cache.dec_lock_ref.assert_called_once_with(node)
            self.assertNotIn("request", cache._inflight_store_nodes)

    def test_cache_finished_installed_owner_exception_retains_canonical_node(
        self,
    ) -> None:
        """An installed owner keeps the canonical source lock for finalization."""
        with _import_flexkv_modules() as (_, radix_module):
            cache, connector = _make_store_cache(radix_module)
            connector.store_kv.side_effect = RuntimeError("post-owner failure")
            connector.store_requires_owner_lock.return_value = True
            node = object()
            canonical_indices = torch.tensor([600, 601])
            req = SimpleNamespace(
                rid="request",
                origin_input_ids=[0, 1],
                output_ids=[],
                extra_key=None,
            )

            with (
                patch.object(
                    radix_module.RadixCache,
                    "cache_finished_req",
                    return_value=radix_module.CacheFinishedReqResult(
                        unhandled_kv_start=2
                    ),
                ),
                patch.object(
                    radix_module.RadixCache,
                    "match_prefix",
                    return_value=SimpleNamespace(
                        device_indices=canonical_indices,
                        last_device_node=node,
                    ),
                ),
                patch.object(torch.cuda, "stream", return_value=MagicMock()),
                self.assertRaisesRegex(RuntimeError, "post-owner failure"),
            ):
                cache.cache_finished_req(
                    req,
                    is_insert=True,
                    kv_len_to_handle=2,
                )

            cache.inc_lock_ref.assert_called_once_with(node)
            cache.dec_lock_ref.assert_not_called()
            self.assertIs(cache._inflight_store_nodes["request"], node)

    def test_postattempt_store_failure_retains_owner_and_blocks_reset(self) -> None:
        """An attempted store launch poisons ownership instead of clearing it."""
        with _import_flexkv_modules() as (module, _):
            connector, _, manager = _make_connector(module)
            manager.put_match.return_value = (
                33,
                np.asarray([True, True, False, False]),
            )
            manager.launch.side_effect = RuntimeError("launch transport failed")

            result = connector.store_kv(
                rid="request",
                token_ids=list(range(4)),
                kv_indices=torch.tensor([100, 101, 102, 103]),
            )

            self.assertEqual(result, 33)
            self.assertIn("request", connector._inflight_stores)
            self.assertIn("request", connector._store_owner_required)
            self.assertIn("request", connector._ambiguous_stores)
            self.assertIn("launch transport failed", connector._poison_reason)
            with self.assertRaisesRegex(RuntimeError, "Cannot reset FlexKV"):
                connector.ensure_reset_safe()

    def test_mixed_store_status_retries_only_remaining_observation_ids(self) -> None:
        """Successful store observations retire while timeout observations remain."""
        with _import_flexkv_modules() as (module, _):
            connector, _, manager = _make_connector(module)
            connector._inflight_stores["request"] = module._InflightFlexKVStore(
                version=1,
                remaining_task_ids=(41, 42),
                successful_task_ids=(),
                terminal_ready=False,
            )
            connector._store_owner_required.add("request")
            manager.wait.side_effect = [
                {
                    41: _response(41, _ResponseStatus.SUCCESS),
                    42: _response(42, _ResponseStatus.TIMEOUT),
                },
                {42: _response(42, _ResponseStatus.SUCCESS)},
            ]

            first_completed = connector.check_completed_stores()
            second_completed = connector.check_completed_stores()

            self.assertEqual(first_completed, [])
            self.assertEqual(second_completed, ["request"])
            self.assertEqual(
                manager.wait.call_args_list,
                [
                    call(task_ids=[41, 42], timeout=0, completely=True),
                    call(task_ids=[42], timeout=0, completely=True),
                ],
            )

    def test_store_timeout_exception_poison_prevents_retry(self) -> None:
        """An indeterminate terminal wait poisons the store before any retry."""
        with _import_flexkv_modules() as (module, _):
            connector, _, manager = _make_connector(module)
            connector._inflight_stores["request"] = module._InflightFlexKVStore(
                version=1,
                remaining_task_ids=(41,),
                successful_task_ids=(),
                terminal_ready=False,
            )
            connector._store_owner_required.add("request")
            manager.wait.side_effect = TimeoutError("deadline")

            with self.assertRaisesRegex(RuntimeError, "deadline"):
                connector.check_completed_stores()
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                connector.check_completed_stores()

            manager.wait.assert_called_once()

    def test_poison_rejects_new_store_before_external_calls(self) -> None:
        """A poisoned connector rejects new stores without contacting FlexKV."""
        with _import_flexkv_modules() as (module, _):
            connector, _, manager = _make_connector(module)
            connector.poison_load_back("ambiguous prior transfer")

            with self.assertRaisesRegex(RuntimeError, "ambiguous prior transfer"):
                connector.store_kv(
                    rid="request",
                    token_ids=[1, 2],
                    kv_indices=torch.tensor([100, 101]),
                )

            manager.put_match.assert_not_called()
            manager.launch.assert_not_called()

    def test_store_finalization_commits_only_after_owner_release_consensus(
        self,
    ) -> None:
        """Terminal state remains owned until node release reaches consensus."""
        with _import_flexkv_modules() as (_, radix_module):
            cache = radix_module.FlexKVRadixCache.__new__(radix_module.FlexKVRadixCache)
            cache._node_lock = threading.Lock()
            node = object()
            cache._inflight_store_nodes = {"request": node}
            order: list[str] = []
            plan = object()
            cache.dec_lock_ref = MagicMock(
                side_effect=lambda current_node: order.append("dec")
            )
            cache.flexkv_connector = MagicMock()
            cache.flexkv_connector.check_completed_stores.side_effect = lambda: [
                "request"
            ]
            cache.flexkv_connector.prepare_store_finalization.side_effect = (
                lambda rids: order.append("prepare") or plan
            )
            cache.flexkv_connector.coordinate_store_owner_release.side_effect = (
                lambda local_success: order.append("min") or True
            )
            cache.flexkv_connector.commit_store_finalization.side_effect = (
                lambda current_plan: order.append("commit")
            )

            cache._drain_completed_stores()

            self.assertEqual(order, ["prepare", "dec", "min", "commit"])
            self.assertNotIn("request", cache._inflight_store_nodes)
            cache.flexkv_connector.commit_store_finalization.assert_called_once_with(
                plan
            )


if __name__ == "__main__":
    unittest.main()
