"""Unit tests for remote loader configuration and memory registration."""

import unittest
from unittest.mock import Mock, patch

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


class TestRemoteInstanceWeightMemoryRegistration(CustomTestCase):
    def _register_memory(self, hip):
        allocation_base = 0x100000
        allocation_size = 0x200000
        first_pointer = allocation_base + 2048
        second_pointer = allocation_base + 4096
        parameters = [
            (
                name,
                Mock(
                    data_ptr=Mock(return_value=pointer),
                    numel=Mock(return_value=128),
                    element_size=Mock(return_value=2),
                ),
            )
            for name, pointer in (("first", first_pointer), ("second", second_pointer))
        ]
        model = Mock()
        model.named_parameters.return_value = parameters
        transfer_engine = Mock()
        transfer_engine.register_memory.return_value = 0
        snapshot = [
            {
                "address": allocation_base,
                "total_size": allocation_size,
                "blocks": [
                    {
                        "address": first_pointer,
                        "size": 2048,
                        "state": "active_allocated",
                    },
                    {
                        "address": second_pointer,
                        "size": 4096,
                        "state": "active_allocated",
                    },
                ],
            },
            {
                "address": 0x500000,
                "total_size": allocation_size,
                "blocks": [
                    {
                        "address": 0x500000,
                        "size": 4096,
                        "state": "active_allocated",
                    }
                ],
            },
        ]
        with (
            patch("torch.version.hip", hip),
            patch("torch.cuda.memory.memory_snapshot", return_value=snapshot),
        ):
            metadata = register_memory_region_v2(model, transfer_engine)
        self.assertEqual(
            metadata,
            {
                "first": (first_pointer, 128, 2),
                "second": (second_pointer, 128, 2),
            },
        )
        return transfer_engine

    def test_hip_registers_allocation_containing_interior_weights(self):
        transfer_engine = self._register_memory("7.2")
        # Both weights share this allocation; the unrelated segment is omitted.
        transfer_engine.register_memory.assert_called_once_with(0x100000, 0x200000)

    def test_non_hip_merges_adjacent_weight_blocks(self):
        transfer_engine = self._register_memory(None)
        transfer_engine.register_memory.assert_called_once_with(0x100000 + 2048, 6144)


if __name__ == "__main__":
    unittest.main()
