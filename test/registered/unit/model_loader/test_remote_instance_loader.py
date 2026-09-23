"""Remote loader construction and synthetic host-storage registration tests."""

import unittest
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region_v2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

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


class TestHostCheckpointRegistration(CustomTestCase):
    def test_cpu_storage_aliases_offsets_and_empty(self):
        backing = torch.arange(32, dtype=torch.float32)
        model = torch.nn.Module()
        model.register_parameter("left", torch.nn.Parameter(backing[4:12]))
        model.register_parameter("right", torch.nn.Parameter(backing[12:24]))
        model.register_parameter("empty", torch.nn.Parameter(torch.empty(0)))
        engine = Mock()
        engine.register_memory.return_value = 0
        with patch("torch.cuda.memory.memory_snapshot", return_value=[]):
            published = register_memory_region_v2(model, engine)
        engine.register_memory.assert_called_once_with(
            backing.data_ptr(), backing.nbytes
        )
        self.assertEqual(published["left"], (model.left.data_ptr(), 8, 4))
        self.assertEqual(published["right"], (model.right.data_ptr(), 12, 4))
        self.assertEqual(published["empty"][1], 0)

    def test_registration_failure_is_not_published(self):
        model = torch.nn.Linear(8, 4, bias=False, device="cpu")
        engine = Mock()
        engine.register_memory.return_value = -7
        with patch("torch.cuda.memory.memory_snapshot", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "register memory failed.*-7"):
                register_memory_region_v2(model, engine)


if __name__ == "__main__":
    unittest.main()
