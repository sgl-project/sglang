import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _KVResponseStatus(Enum):
    SUCCESS = "success"


def _module(name: str, **attrs):
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def _load_connector_class():
    """Load the connector with lightweight stubs for optional FlexKV."""
    module_name = "_flexkv_connector_lifecycle_under_test"
    stubs = {
        "sglang.srt.mem_cache.storage.flexkv.flexkv_comm": _module(
            "sglang.srt.mem_cache.storage.flexkv.flexkv_comm",
            CMD_LAYERWISE=3,
            CMD_PUT_META=2,
            CMD_STORE_COMPLETE=5,
            FlexKVComm=object,
            FlexKVLayerDoneCounter=object,
            send_fds=lambda *args, **kwargs: None,
        ),
        "flexkv": _module("flexkv"),
        "flexkv.common": _module("flexkv.common"),
        "flexkv.common.config": _module(
            "flexkv.common.config",
            LayerGroupSpec=object,
            recompute_cache_block_counts=lambda *args, **kwargs: False,
        ),
        "flexkv.common.request": _module(
            "flexkv.common.request", KVResponseStatus=_KVResponseStatus
        ),
        "flexkv.common.storage": _module(
            "flexkv.common.storage", KVCacheLayout=object, KVCacheLayoutType=object
        ),
        "flexkv.integration": _module("flexkv.integration"),
        "flexkv.integration.config": _module(
            "flexkv.integration.config", FlexKVConfig=object
        ),
        "flexkv.kvmanager": _module("flexkv.kvmanager", KVManager=object),
        "flexkv.server": _module("flexkv.server"),
        "flexkv.server.client": _module("flexkv.server.client", KVTPClient=object),
        "flexkv.transfer": _module("flexkv.transfer"),
        "flexkv.transfer.layerwise": _module(
            "flexkv.transfer.layerwise",
            build_layerwise_eventfd_socket_path=lambda **kwargs: "",
        ),
        "flexkv.transfer_manager": _module(
            "flexkv.transfer_manager", TransferManagerOnRemote=object
        ),
    }

    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_connector.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module.FlexKVConnector


FlexKVConnector = _load_connector_class()


def _sync_context():
    return SimpleNamespace(
        is_pp_receiver=False,
        should_send_slot_mapping_to_remote=False,
        is_pp_sender=False,
        is_sync_leader=True,
        needs_sync=False,
    )


def test_layerwise_reset_waits_for_returned_batch_task_id():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.enable_layerwise = True
    connector.layer_done_counter = MagicMock()
    connector.layer_done_counter.update_producer.return_value = 1
    connector.layer_done_counter.events = [MagicMock(), MagicMock(), MagicMock()]
    connector._pending_lookups = {"request": 17}
    connector._sync_ctx = _sync_context()
    connector.kv_manager = MagicMock()
    connector.kv_manager.launch.return_value = [91]
    connector.kv_manager.wait.return_value = {
        91: SimpleNamespace(status=_KVResponseStatus.SUCCESS)
    }
    connector._to_cpu_int64 = lambda value: value
    connector._build_swa_slot_mapping = lambda value: None
    connector._launched_load_tids = []
    connector._ongoing_prefetches = {}
    connector._inflight_loads = {}
    connector._completed_layerwise = []
    connector._inflight_stores = {}

    loaded, producer_id = connector.start_load_kv_layerwise(
        "request", torch.tensor([3, 4], dtype=torch.int64)
    )

    assert (loaded, producer_id) == (2, 1)
    assert connector._launched_load_tids == [91]

    connector.reset()

    connector.kv_manager.wait.assert_called_once_with(
        [91], timeout=30.0, completely=True
    )
    connector.layer_done_counter.reset.assert_called_once_with()
    assert connector._launched_load_tids == []


def test_drain_launched_loads_preserves_unfinished_tasks():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._sync_ctx = _sync_context()
    connector.kv_manager = MagicMock()
    connector.kv_manager.wait.return_value = {
        91: SimpleNamespace(status=_KVResponseStatus.SUCCESS)
    }
    connector._launched_load_tids = [91, 92]

    connector.drain_launched_loads(threshold=2)

    connector.kv_manager.wait.assert_called_once_with(
        [91, 92], timeout=0.0, completely=True
    )
    assert connector._launched_load_tids == [92]
