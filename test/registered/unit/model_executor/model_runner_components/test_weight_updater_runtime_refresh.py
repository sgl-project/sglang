from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
LOAD_MODEL_UTILS_PATH = (
    REPO_ROOT
    / "python/sglang/srt/model_executor/model_runner_components/load_model_utils.py"
)
WEIGHT_UPDATER_PATH = (
    REPO_ROOT
    / "python/sglang/srt/model_executor/model_runner_components/weight_updater.py"
)


class _FakeDType:
    pass


class _FakeTensor:
    def to(self, device):
        del device
        return self


class _FakeHandle:
    def wait(self):
        return None


class _MsgspecStruct:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()


class _DefaultModelLoader:
    class Source:
        @staticmethod
        def init_new(config, model):
            return config, model


class _LoadConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FlattenedTensorMetadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FlattenedTensorBucket:
    def __init__(self, named_tensors=None, flattened_tensor=None, metadata=None):
        self.named_tensors = named_tensors
        self.flattened_tensor = flattened_tensor
        self.metadata = metadata

    def get_flattened_tensor(self):
        return self.flattened_tensor or _FakeTensor()

    def reconstruct_tensors(self):
        if self.named_tensors is not None:
            return self.named_tensors
        return [(metadata.name, _FakeTensor()) for metadata in self.metadata]


class _Model:
    def __init__(self, *, fail_load: bool = False):
        self.fail_load = fail_load
        self.refresh_count = 0
        self.child = SimpleNamespace(refresh_count=0)
        self.child.refresh_runtime_weight_state = self._refresh_child
        self.non_refreshable = SimpleNamespace(refresh_runtime_weight_state=None)
        self.parameter = SimpleNamespace()

    def _refresh_child(self):
        self.child.refresh_count += 1

    def refresh_runtime_weight_state(self):
        self.refresh_count += 1

    def modules(self):
        return (self, self.child, self.non_refreshable)

    def named_parameters(self):
        return (("weight", self.parameter),)

    def load_weights(self, named_tensors):
        del named_tensors
        if self.fail_load:
            raise RuntimeError("load failed")


def _ensure_package(monkeypatch, name: str) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = _ensure_package(monkeypatch, parent_name)
    else:
        parent = None
        child_name = name

    module = ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if parent is not None:
        monkeypatch.setattr(parent, child_name, module, raising=False)
    return module


def _install_module(monkeypatch, name: str, *, package=False, **attributes):
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = _ensure_package(monkeypatch, parent_name)
    else:
        parent = None
        child_name = name

    module = ModuleType(name)
    if package:
        module.__path__ = []
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    if parent is not None:
        monkeypatch.setattr(parent, child_name, module, raising=False)
    return module


def _load_source_module(monkeypatch, name: str, path: Path) -> ModuleType:
    parent_name, child_name = name.rsplit(".", 1)
    parent = _ensure_package(monkeypatch, parent_name)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(parent, child_name, module, raising=False)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime_modules(monkeypatch):
    _install_module(monkeypatch, "msgspec", Struct=_MsgspecStruct)
    distributed = _install_module(
        monkeypatch,
        "torch.distributed",
        is_initialized=lambda: True,
        broadcast=lambda *args, **kwargs: _FakeHandle(),
        destroy_process_group=lambda group: None,
    )
    torch = _install_module(
        monkeypatch,
        "torch",
        package=True,
        Tensor=_FakeTensor,
        dtype=_FakeDType,
        device=lambda device: device,
        distributed=distributed,
        empty=lambda shape, dtype, device: _FakeTensor(),
        get_device_module=lambda device: SimpleNamespace(current_device=lambda: 0),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.configs.device_config",
        DeviceConfig=lambda device, gpu_id: (device, gpu_id),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.configs.load_config",
        LoadConfig=_LoadConfig,
        LoadFormat=SimpleNamespace(REMOTE_INSTANCE="remote_instance"),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.constants",
        GPU_MEMORY_TYPE_WEIGHTS="weights",
    )
    _install_module(
        monkeypatch,
        "sglang.srt.debug_utils.tensor_dump_forward_hook",
        register_forward_hook_for_model=lambda *args, **kwargs: None,
    )
    _install_module(
        monkeypatch,
        "sglang.srt.distributed",
        package=True,
        get_tp_group=lambda: SimpleNamespace(cpu_group=None),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.distributed.parallel_state",
        monkey_patch_vllm_parallel_state=lambda reverse=False: None,
    )
    loader_module = _install_module(
        monkeypatch,
        "sglang.srt.model_loader.loader",
        DefaultModelLoader=_DefaultModelLoader,
        get_model_loader=lambda *args, **kwargs: None,
    )
    _install_module(
        monkeypatch,
        "sglang.srt.model_loader.remote_instance_weight_loader_utils",
        RemoteInstanceWeightLoaderBackend=SimpleNamespace(NCCL="nccl"),
        trigger_init_weights_send_group_for_remote_instance_request=(
            lambda *args, **kwargs: None
        ),
    )

    @contextmanager
    def set_default_torch_dtype(dtype):
        del dtype
        yield

    _install_module(
        monkeypatch,
        "sglang.srt.model_loader.utils",
        set_default_torch_dtype=set_default_torch_dtype,
    )
    _install_module(
        monkeypatch,
        "sglang.srt.model_loader.weight_utils",
        default_weight_loader=lambda parameter, tensor: None,
    )
    _install_module(
        monkeypatch,
        "sglang.srt.model_executor.model_runner_components.weight_update_coordination",
        begin_uncoordinated_update=lambda: None,
        coordinated_weight_update=lambda method: method,
        finish_uncoordinated_update=lambda token, success: None,
    )
    _install_module(
        monkeypatch,
        "sglang.srt.platforms",
        current_platform=SimpleNamespace(
            is_out_of_tree=lambda: False,
            support_cuda_graph=lambda: False,
        ),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.utils",
        package=True,
        MultiprocessingSerializer=SimpleNamespace(
            deserialize=lambda serialized: serialized
        ),
        dynamic_import=lambda path: None,
        get_available_gpu_memory=lambda *args, **kwargs: 1.0,
        init_custom_process_group=lambda **kwargs: object(),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.utils.common",
        is_npu=lambda: False,
    )
    _install_module(
        monkeypatch,
        "sglang.srt.utils.network",
        NetworkAddress=lambda host, port: SimpleNamespace(
            to_tcp=lambda: f"tcp://{host}:{port}"
        ),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.utils.patch_torch",
        monkey_patch_torch_reductions=lambda: None,
    )
    _install_module(
        monkeypatch,
        "sglang.srt.weight_sync.tensor_bucket",
        FlattenedTensorBucket=_FlattenedTensorBucket,
        FlattenedTensorMetadata=_FlattenedTensorMetadata,
    )

    load_model_utils = _load_source_module(
        monkeypatch,
        "sglang.srt.model_executor.model_runner_components.load_model_utils",
        LOAD_MODEL_UTILS_PATH,
    )
    weight_updater = _load_source_module(
        monkeypatch,
        "sglang.srt.model_executor.model_runner_components.weight_updater",
        WEIGHT_UPDATER_PATH,
    )
    return SimpleNamespace(
        load_model_utils=load_model_utils,
        loader_module=loader_module,
        torch=torch,
        weight_updater=weight_updater,
    )


def _make_updater(module, model, **overrides):
    values = {
        "tp_rank": 0,
        "device": "cpu",
        "gpu_id": 0,
        "model_config": SimpleNamespace(dtype=None, model_path="original"),
        "custom_weight_loaders": {"custom_loader": object()},
        "get_model": lambda: model,
        "update_model_fields": lambda *args, **kwargs: None,
        "recapture_cuda_graph": lambda: None,
        "get_model_runner": lambda: object(),
    }
    values.update(overrides)
    return module.WeightUpdater(**values)


def _flattened_payload():
    return {
        "flattened_tensor": _FakeTensor(),
        "metadata": [
            SimpleNamespace(
                name="weight",
                shape=(1,),
                dtype=_FakeDType(),
                start_idx=0,
                end_idx=1,
                numel=1,
            )
        ],
    }


def _assert_refreshed(model):
    assert model.refresh_count == 1
    assert model.child.refresh_count == 1


def _assert_not_refreshed(model):
    assert model.refresh_count == 0
    assert model.child.refresh_count == 0


def test_refresh_helper_calls_every_callable_module(runtime_modules):
    model = _Model()

    runtime_modules.load_model_utils.refresh_runtime_weight_state(model)

    _assert_refreshed(model)


@pytest.mark.parametrize(
    "load_format",
    ["direct", None, "custom_loader", "flattened_bucket"],
)
def test_tensor_update_refreshes_runtime_state_after_success(
    runtime_modules, monkeypatch, load_format
):
    module = runtime_modules.weight_updater
    model = _Model()
    updater = _make_updater(module, model)
    monkeypatch.setattr(
        module,
        "dynamic_import",
        lambda path: lambda loaded_model, tensors: None,
    )
    payload = (
        _flattened_payload()
        if load_format == "flattened_bucket"
        else [("weight", _FakeTensor())]
    )

    success, _ = updater.update_weights_from_tensor(payload, load_format=load_format)

    assert success is True
    _assert_refreshed(model)


@pytest.mark.parametrize(
    "load_format",
    ["direct", None, "custom_loader", "flattened_bucket"],
)
def test_tensor_update_does_not_refresh_after_failure(
    runtime_modules, monkeypatch, load_format
):
    module = runtime_modules.weight_updater
    model = _Model(fail_load=load_format in (None, "flattened_bucket"))
    updater = _make_updater(module, model)

    def fail(*args, **kwargs):
        raise RuntimeError("load failed")

    if load_format == "direct":
        monkeypatch.setattr(module, "default_weight_loader", fail)
    elif load_format == "custom_loader":
        monkeypatch.setattr(module, "dynamic_import", lambda path: fail)

    payload = (
        _flattened_payload()
        if load_format == "flattened_bucket"
        else [("weight", _FakeTensor())]
    )

    with pytest.raises(RuntimeError, match="load failed"):
        updater.update_weights_from_tensor(payload, load_format=load_format)

    _assert_not_refreshed(model)


@pytest.mark.parametrize("load_format", [None, "flattened_bucket"])
def test_distributed_update_refreshes_runtime_state_after_success(
    runtime_modules, load_format
):
    module = runtime_modules.weight_updater
    model = _Model()
    updater = _make_updater(
        module,
        model,
        _model_update_group={"group": object()},
    )

    success, _ = updater.update_weights_from_distributed(
        names=["weight"],
        dtypes=[_FakeDType()],
        shapes=[(1,)],
        group_name="group",
        load_format=load_format,
    )

    assert success is True
    _assert_refreshed(model)


@pytest.mark.parametrize("load_format", [None, "flattened_bucket"])
def test_distributed_update_does_not_refresh_after_failure(
    runtime_modules, load_format
):
    module = runtime_modules.weight_updater
    model = _Model(fail_load=True)
    updater = _make_updater(
        module,
        model,
        _model_update_group={"group": object()},
    )

    success, _ = updater.update_weights_from_distributed(
        names=["weight"],
        dtypes=[_FakeDType()],
        shapes=[(1,)],
        group_name="group",
        load_format=load_format,
    )

    assert success is False
    _assert_not_refreshed(model)


def test_disk_update_refreshes_runtime_state_after_success(
    runtime_modules, monkeypatch
):
    module = runtime_modules.weight_updater
    model = _Model()

    class Loader(_DefaultModelLoader):
        def _get_weights_iterator(self, source):
            del source
            return iter(())

        def load_weights_and_postprocess(self, loaded_model, weights, device):
            del loaded_model, weights, device

    monkeypatch.setattr(module, "get_model_loader", lambda *args, **kwargs: Loader())
    updater = _make_updater(module, model)

    success, _ = updater.update_weights_from_disk("new-model", "auto")

    assert success is True
    _assert_refreshed(model)


def test_disk_update_does_not_refresh_after_failure(runtime_modules, monkeypatch):
    module = runtime_modules.weight_updater
    model = _Model()

    class Loader(_DefaultModelLoader):
        load_calls = 0

        def _get_weights_iterator(self, source):
            del source
            return iter(())

        def load_weights_and_postprocess(self, loaded_model, weights, device):
            del loaded_model, weights, device
            self.load_calls += 1
            if self.load_calls == 1:
                raise RuntimeError("load failed")

    monkeypatch.setattr(module, "get_model_loader", lambda *args, **kwargs: Loader())
    updater = _make_updater(module, model)

    success, _ = updater.update_weights_from_disk("new-model", "auto")

    assert success is False
    _assert_not_refreshed(model)


def test_ipc_update_refreshes_runtime_state_after_success(runtime_modules, monkeypatch):
    module = runtime_modules.weight_updater
    model = _Model()

    class Worker:
        def __init__(self, model_runner):
            del model_runner

        def update_weights_from_ipc(self, handles):
            del handles

    _install_module(
        monkeypatch,
        "sglang.srt.checkpoint_engine.checkpoint_engine_worker",
        SGLangCheckpointEngineWorkerExtensionImpl=Worker,
    )
    updater = _make_updater(module, model)

    success, _ = updater.update_weights_from_ipc(
        SimpleNamespace(zmq_handles=["handle"])
    )

    assert success is True
    _assert_refreshed(model)


def test_ipc_update_does_not_refresh_after_failure(runtime_modules, monkeypatch):
    module = runtime_modules.weight_updater
    model = _Model()

    class Worker:
        def __init__(self, model_runner):
            del model_runner

        def update_weights_from_ipc(self, handles):
            del handles
            raise RuntimeError("IPC failed")

    _install_module(
        monkeypatch,
        "sglang.srt.checkpoint_engine.checkpoint_engine_worker",
        SGLangCheckpointEngineWorkerExtensionImpl=Worker,
    )
    updater = _make_updater(module, model)

    success, _ = updater.update_weights_from_ipc(
        SimpleNamespace(zmq_handles=["handle"])
    )

    assert success is False
    _assert_not_refreshed(model)
