"""Unit tests for RemoteInstanceModelLoader construction - no server, no weights."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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


class TestRemoteInstanceTransferEngineRank(CustomTestCase):
    def test_transfer_engine_queries_metadata_by_global_rank(self):
        transfer_engine = MagicMock()
        transfer_engine.batch_transfer_sync_read.return_value = 0
        load_config = LoadConfig(
            load_format=LoadFormat.REMOTE_INSTANCE,
            tp_rank=1,
            remote_instance_weight_loader_backend=(
                RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE
            ),
            remote_instance_weight_loader_rank=3,
            remote_instance_weight_loader_transfer_engine=transfer_engine,
            remote_instance_weight_loader_seed_instance_ip="127.0.0.1",
            remote_instance_weight_loader_seed_instance_service_port=12345,
        )
        loader = loader_mod.RemoteInstanceModelLoader(load_config)
        model = MagicMock()
        model.named_parameters.return_value = []
        model.eval.return_value = model

        with (
            patch.object(loader_mod, "_get_quantization_config", return_value=None),
            patch.object(
                loader_mod, "set_default_torch_dtype", return_value=nullcontext()
            ),
            patch.object(loader_mod.torch, "device", return_value=nullcontext()),
            patch.object(loader_mod, "_initialize_model", return_value=model),
            patch.object(loader_mod, "register_memory_region"),
            patch.object(
                loader_mod,
                "get_remote_instance_transfer_engine_info_per_rank",
                return_value=("seed-session", {}),
            ) as get_remote_info,
            patch.object(loader_mod, "_post_load_weights"),
        ):
            loaded_model = loader.load_model(
                model_config=SimpleNamespace(dtype=None),
                device_config=SimpleNamespace(device="cpu"),
            )

        self.assertIs(loaded_model, model)
        get_remote_info.assert_called_once_with("http://127.0.0.1:12345", 3)


if __name__ == "__main__":
    unittest.main()
