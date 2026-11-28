"""Unit tests for RemoteInstanceModelLoader construction - no server, no weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
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


class TestRemoteInstanceModelLoaderNccl(CustomTestCase):
    def test_nccl_group_initialization_failure_stops_loading(self):
        load_config = _load_config("nccl")
        load_config.tp_rank = 0
        load_config.remote_instance_weight_loader_seed_instance_ip = "127.0.0.1"
        load_config.remote_instance_weight_loader_seed_instance_service_port = 30000
        load_config.remote_instance_weight_loader_send_weights_group_ports = [30001]
        loader = loader_mod.RemoteInstanceModelLoader(load_config)
        model = torch.nn.Linear(2, 2)
        client = SimpleNamespace(
            build_group=MagicMock(return_value=(False, "injected init failure"))
        )
        model_config = SimpleNamespace(dtype=torch.float32)
        device_config = SimpleNamespace(gpu_id=0)

        with (
            patch.object(loader_mod.socket, "gethostbyname", return_value="127.0.0.1"),
            patch.object(loader_mod.threading, "Thread") as thread_cls,
            patch.object(
                loader_mod, "trigger_transferring_weights_request"
            ) as trigger_transfer,
            patch.object(loader_mod.torch.distributed, "broadcast") as broadcast,
            patch.object(loader_mod.current_platform, "synchronize") as synchronize,
            patch.object(loader_mod.current_platform, "empty_cache") as empty_cache,
            patch.object(
                loader_mod.torch.distributed.distributed_c10d, "destroy_process_group"
            ) as destroy_process_group,
            patch.object(loader_mod, "post_load_weights") as post_load_weights,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected init failure"):
                loader.load_model_from_remote_instance_by_nccl(
                    model, client, model_config, device_config
                )

        client.build_group.assert_called_once_with(
            gpu_id=0, tp_rank=0, instance_ip="127.0.0.1"
        )
        thread_cls.assert_not_called()
        thread_cls.return_value.start.assert_not_called()
        trigger_transfer.assert_not_called()
        broadcast.assert_not_called()
        synchronize.assert_not_called()
        post_load_weights.assert_not_called()
        destroy_process_group.assert_not_called()
        empty_cache.assert_not_called()

    def test_nccl_group_initialization_success_loads_and_cleans_up(self):
        load_config = _load_config("nccl")
        load_config.tp_rank = 0
        load_config.remote_instance_weight_loader_seed_instance_ip = "127.0.0.1"
        load_config.remote_instance_weight_loader_seed_instance_service_port = 30000
        load_config.remote_instance_weight_loader_send_weights_group_ports = [30001]
        loader = loader_mod.RemoteInstanceModelLoader(load_config)
        model = torch.nn.Linear(2, 2)
        group = object()
        client = SimpleNamespace(
            build_group=MagicMock(return_value=(True, "initialized")),
            _model_update_group=group,
        )
        model_config = SimpleNamespace(dtype=torch.float32)
        device_config = SimpleNamespace(gpu_id=0)

        with (
            patch.object(loader_mod.socket, "gethostbyname", return_value="127.0.0.1"),
            patch.object(loader_mod.threading, "Thread") as thread_cls,
            patch.object(
                loader_mod, "trigger_transferring_weights_request"
            ) as trigger_transfer,
            patch.object(loader_mod.torch.distributed, "broadcast") as broadcast,
            patch.object(loader_mod.current_platform, "synchronize") as synchronize,
            patch.object(loader_mod.current_platform, "empty_cache") as empty_cache,
            patch.object(
                loader_mod.torch.distributed.distributed_c10d, "destroy_process_group"
            ) as destroy_process_group,
            patch.object(loader_mod, "post_load_weights") as post_load_weights,
        ):
            result = loader.load_model_from_remote_instance_by_nccl(
                model, client, model_config, device_config
            )

        client.build_group.assert_called_once_with(
            gpu_id=0, tp_rank=0, instance_ip="127.0.0.1"
        )
        thread_cls.assert_called_once_with(
            target=trigger_transfer,
            args=(
                "127.0.0.1",
                30000,
                [30001],
                "127.0.0.1",
            ),
        )
        thread_cls.return_value.start.assert_called_once_with()
        self.assertEqual(broadcast.call_count, 2)
        for broadcast_call in broadcast.call_args_list:
            self.assertIs(broadcast_call.kwargs["group"], group)
        self.assertEqual(synchronize.call_count, 2)
        post_load_weights.assert_called_once_with(model)
        destroy_process_group.assert_called_once_with(group)
        empty_cache.assert_called_once_with()
        self.assertIs(result, model)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
