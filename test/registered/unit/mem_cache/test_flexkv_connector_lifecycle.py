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


def _load_layer_done_counter_class():
    module_name = "_flexkv_comm_lifecycle_under_test"
    parallel_state_name = "sglang.srt.distributed.parallel_state"
    parallel_state_stub = _module(
        parallel_state_name,
        get_world_group=lambda: None,
    )
    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_comm.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with patch.dict(
            sys.modules, {parallel_state_name: parallel_state_stub}
        ), patch("ctypes.CDLL", return_value=MagicMock()):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module.FlexKVLayerDoneCounter


FlexKVLayerDoneCounter = _load_layer_done_counter_class()


class _FakeLayerLoadingEvent:
    def __init__(self, num_layers):
        self._num_layers = num_layers
        self._finished = True
        self._wait_started = False
        self.wait_remaining = [0] * num_layers
        self.wait_calls = []

    def reset_for_new_transfer(self):
        self._finished = False
        self._wait_started = False
        self.wait_remaining = [0] * self._num_layers

    def add_transfer(self):
        assert not self._finished and not self._wait_started
        self.wait_remaining = [count + 1 for count in self.wait_remaining]

    def wait(self, layer_index, count=1):
        self._wait_started = True
        self.wait_calls.append((layer_index, count))
        if layer_index == self._num_layers - 1:
            self._finished = True

    def mark_reusable(self):
        self._finished = True
        self._wait_started = False
        self.wait_remaining = [0] * self._num_layers

    def close(self):
        pass


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


def test_lookup_uses_swa_aware_match_when_swa_pool_is_registered():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._sync_ctx = _sync_context()
    connector.kv_manager = MagicMock()
    connector.kv_manager.get_match.return_value = (
        17,
        torch.tensor([True, True, False]).numpy(),
    )
    connector._swa_kv_pool = object()
    connector._pending_lookups = {}
    connector.page_size = 1

    task_id, hit_length = connector.lookup_kv(
        [11, 12, 13],
        torch.tensor([True, True, True]),
        rid="request",
    )

    assert (task_id, hit_length) == (17, 2)
    connector.kv_manager.get_match.assert_called_once()
    assert connector.kv_manager.get_match.call_args.kwargs["swa_aware"] is True
    assert connector._pending_lookups == {"request": 17}


def test_layerwise_counter_groups_restore_requests_in_one_prefill_batch():
    counter = FlexKVLayerDoneCounter.__new__(FlexKVLayerDoneCounter)
    counter.num_layers = 2
    counter.num_counters = 3
    counter.events = [_FakeLayerLoadingEvent(2) for _ in range(3)]
    counter.producer_index = -1
    counter.consumer_index = -1
    counter._task_to_producer = {}

    producer_ids = []
    for task_id in (11, 12, 13):
        producer_id = counter.update_producer()
        producer_ids.append(producer_id)
        counter.register_task(task_id, producer_id)
        counter.set_consumer(task_id)

    assert producer_ids == [0, 0, 0]
    counter.wait_until(0)
    counter.wait_until(1)
    assert counter.events[0].wait_calls == [(0, 3), (1, 3)]
    assert counter.events[0]._finished
    assert counter.consumer_index == -1

    assert counter.update_producer() == 1
