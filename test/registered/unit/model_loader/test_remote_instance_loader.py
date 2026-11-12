"""Unit tests for RemoteInstanceModelLoader construction - no server, no weights."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

import sglang.srt.connector.remote_instance as connector_mod
import sglang.srt.model_executor.model_runner_components.weight_exporter as exporter_mod
import sglang.srt.model_loader.loader as loader_mod
import sglang.srt.model_loader.remote_instance_weight_loader_utils as rfork_utils
import sglang.srt.utils.device as device_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EXTRA_CONFIG = {"enable_multithread_load": True, "num_threads": 64}

# ServerArgs forwards the backend as a plain str, so every case covers both that
# and the enum member; the loader's comparison holds only via the str mixin.
MODELEXPRESS_BACKENDS = (
    RemoteInstanceWeightLoaderBackend.MODELEXPRESS,
    "modelexpress",
)
OTHER_BACKENDS = (
    RemoteInstanceWeightLoaderBackend.NCCL,
    "nccl",
    RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE,
    "transfer_engine",
)
ALL_BACKENDS = MODELEXPRESS_BACKENDS + OTHER_BACKENDS


def _load_config(backend, extra_config=None):
    return LoadConfig(
        load_format=LoadFormat.REMOTE_INSTANCE,
        model_loader_extra_config=extra_config or {},
        remote_instance_weight_loader_backend=backend,
    )


class TestRemoteInstanceModelLoaderExtraConfig(CustomTestCase):
    def test_modelexpress_backend_accepts_extra_config(self):
        for backend in MODELEXPRESS_BACKENDS:
            with self.subTest(backend=backend):
                loader = loader_mod.RemoteInstanceModelLoader(
                    _load_config(backend, EXTRA_CONFIG)
                )
                # ModelExpress hands the extra config to the DefaultModelLoader
                # it falls back to, so it must survive construction unmodified.
                self.assertEqual(
                    loader.load_config.model_loader_extra_config, EXTRA_CONFIG
                )

    def test_modelexpress_backend_rejects_unknown_extra_config_key(self):
        for backend in MODELEXPRESS_BACKENDS:
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    loader_mod.RemoteInstanceModelLoader(
                        _load_config(backend, {"num_thread": 64})
                    )

    def test_other_backends_reject_extra_config(self):
        for backend in OTHER_BACKENDS:
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    loader_mod.RemoteInstanceModelLoader(
                        _load_config(backend, EXTRA_CONFIG)
                    )

    def test_all_backends_construct_without_extra_config(self):
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend):
                loader = loader_mod.RemoteInstanceModelLoader(_load_config(backend))
                self.assertFalse(loader.load_config.model_loader_extra_config)


def _client_group_kwargs(device_type, *, xpu):
    """The process-group kwargs the client-side connector passes on ``device_type``."""
    connector = connector_mod.RemoteInstanceConnector(
        "tcp://127.0.0.1:29500", torch.device(device_type)
    )
    with (
        patch.object(connector_mod, "is_xpu", return_value=xpu),
        patch.object(connector_mod, "is_npu", return_value=False),
        patch.object(connector_mod, "init_custom_process_group") as init_group,
        patch.object(connector_mod.dist, "barrier"),
    ):
        ok, _ = connector.build_group(gpu_id=1, tp_rank=0, instance_ip="10.0.0.2")
    assert ok
    return init_group.call_args.kwargs


def _seed_group_kwargs(*, xpu):
    """The process-group kwargs the seed-side exporter passes."""
    exporter = exporter_mod.WeightExporter(
        tp_rank=0,
        tp_size=1,
        gpu_id=1,
        get_model_path=lambda: "",
        get_model=lambda: nn.Linear(2, 2),
    )
    with (
        patch.object(exporter_mod, "is_xpu", return_value=xpu),
        patch.object(exporter_mod, "is_npu", return_value=False),
        patch.object(exporter_mod, "init_custom_process_group") as init_group,
        patch.object(exporter_mod.dist, "barrier"),
        patch.object(
            exporter_mod.torch.distributed, "is_initialized", return_value=True
        ),
        patch.object(exporter_mod.current_platform, "empty_cache"),
    ):
        ok, _ = exporter.init_weights_send_group_for_remote_instance(
            master_address="10.0.0.1",
            ports="29500",
            group_rank=0,
            world_size=2,
            group_name="send_weights_10.0.0.2",
        )
    assert ok
    return init_group.call_args.kwargs


class TestRemoteInstanceNcclGroupDevice(CustomTestCase):
    def test_xpu_seed_and_client_build_an_xccl_group(self):
        """Seed and client must agree on the backend, or the group init hangs."""
        client = _client_group_kwargs("xpu", xpu=True)
        seed = _seed_group_kwargs(xpu=True)
        self.assertEqual((client["backend"], seed["backend"]), ("xccl", "xccl"))
        self.assertEqual(client["device_id"], torch.device("xpu", 1))
        self.assertEqual(seed["device_id"], torch.device("xpu", 1))

    def test_cuda_keeps_nccl(self):
        client = _client_group_kwargs("cuda", xpu=False)
        seed = _seed_group_kwargs(xpu=False)
        self.assertEqual((client["backend"], seed["backend"]), ("nccl", "nccl"))
        self.assertEqual(seed["device_id"], torch.device("cuda", 1))


class TestRemoteInstanceTransferEngineRegistration(CustomTestCase):
    def test_xpu_registers_each_weight_without_the_cuda_allocator(self):
        """The merged path reads torch.cuda's allocator snapshot, absent on XPU."""
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2, bias=False))
        engine = MagicMock()
        engine.register_memory.return_value = 0
        with (
            patch.object(device_mod, "is_xpu", return_value=True),
            patch.object(
                torch.cuda.memory,
                "memory_snapshot",
                side_effect=AssertionError("read the CUDA allocator"),
            ),
        ):
            info = rfork_utils.register_memory_region(model, engine)
        registered = {c.args for c in engine.register_memory.call_args_list}
        expected = {
            (p.data_ptr(), p.numel() * p.element_size()) for p in model.parameters()
        }
        self.assertEqual(registered, expected)
        self.assertEqual(set(info), {name for name, _ in model.named_parameters()})


if __name__ == "__main__":
    unittest.main()
