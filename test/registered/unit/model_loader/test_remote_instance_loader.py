"""Unit tests for RemoteInstanceModelLoader construction - no server, no weights."""

import importlib.util
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_loader.loader as loader_mod
import sglang.srt.model_loader.remote_instance_weight_loader_utils as utils_mod
from sglang.srt.arg_groups.model_path_hook import _assert_mori_seed_usable
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region_mori,
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


class TestRemoteInstanceTransferEngineRole(CustomTestCase):
    def test_each_runner_loads_the_seed_weights_of_its_own_role(self):
        loader = loader_mod.RemoteInstanceModelLoader(
            LoadConfig(
                load_format=LoadFormat.REMOTE_INSTANCE,
                remote_instance_weight_loader_backend=RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE,
                remote_instance_weight_loader_transfer_engine=object(),
                remote_instance_weight_loader_seed_instance_ip="127.0.0.1",
                remote_instance_weight_loader_seed_instance_service_port=30000,
                tp_rank=1,
            )
        )
        for is_draft_model, role in ((False, "target"), (True, "draft")):
            with self.subTest(role=role):
                queried = []
                with (
                    patch.object(loader_mod, "_get_quantization_config"),
                    patch.object(
                        loader_mod,
                        "_initialize_model",
                        return_value=torch.nn.Linear(1, 1),
                    ),
                    patch.object(loader_mod, "register_memory_region"),
                    patch.object(
                        loader_mod,
                        "get_remote_instance_transfer_engine_info_per_rank",
                        side_effect=lambda *args: queried.append(args),
                    ),
                ):
                    with self.assertRaises(RuntimeError):
                        loader.load_model(
                            model_config=SimpleNamespace(
                                is_draft_model=is_draft_model, dtype=torch.float32
                            ),
                            device_config=SimpleNamespace(device="cpu"),
                        )
                self.assertEqual(queried, [("http://127.0.0.1:30000", 1, role)])


class TestMoriSeed(CustomTestCase):
    def _cfg(self, **overrides):
        return SimpleNamespace(
            **{
                "remote_instance_weight_loader_start_seed_via_transfer_engine": False,
                "remote_instance_weight_loader_start_seed_via_nixl": False,
                "load_format": "auto",
                "remote_instance_weight_loader_backend": "nccl",
                "enable_memory_saver": False,
                **overrides,
            }
        )

    def test_unusable_setups_are_refused_at_startup(self):
        """Each would start without publishing MoRI weights, or abort inside MoRI."""
        cases = {
            "set one seed transport": dict(
                remote_instance_weight_loader_start_seed_via_nixl=True
            ),
            "cannot seed weights loaded over transfer_engine": dict(
                load_format="remote_instance",
                remote_instance_weight_loader_backend="transfer_engine",
            ),
            "cannot seed weights loaded over modelexpress": dict(
                load_format="remote_instance",
                remote_instance_weight_loader_backend="modelexpress",
            ),
            "--enable-memory-saver": dict(enable_memory_saver=True),
        }
        with patch.object(importlib.util, "find_spec", lambda name: object()):
            _assert_mori_seed_usable(self._cfg())
            for message, overrides in cases.items():
                with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                    _assert_mori_seed_usable(self._cfg(**overrides))
        with patch.object(importlib.util, "find_spec", lambda name: None):
            with self.assertRaisesRegex(ValueError, "needs the mori package"):
                _assert_mori_seed_usable(self._cfg())

    def _register(self, env, registered):
        engine = SimpleNamespace(
            register_memory=lambda *args: (
                registered.append(args) or f"desc{len(registered)}"
            )
        )
        mori_io = ModuleType("mori.io")
        mori_io.MemoryLocationType = SimpleNamespace(GPU="GPU")
        blocks = [(0x1000, 64), (0x9000, 32)]
        with (
            patch.dict(sys.modules, {"mori": ModuleType("mori"), "mori.io": mori_io}),
            patch.dict(os.environ, env, clear=True),
            patch.object(
                utils_mod,
                "_collect_weight_blocks",
                lambda model: ({"w": (1, 2, 3)}, blocks),
            ),
        ):
            return register_memory_region_mori(None, engine, 5)

    def test_each_weight_block_is_registered_as_gpu_memory(self):
        registered = []
        weights, published = self._register({}, registered)

        self.assertEqual(weights, {"w": (1, 2, 3)})
        self.assertEqual(registered, [(0x1000, 64, 5, "GPU"), (0x9000, 32, 5, "GPU")])
        self.assertEqual(published, [(0x1000, 64, "desc1"), (0x9000, 32, "desc2")])

    def test_expandable_segments_are_refused_before_registering_anything(self):
        """MoRI cannot register VMM memory, and aborts the process trying."""
        for name in (
            "PYTORCH_ALLOC_CONF",
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_HIP_ALLOC_CONF",
        ):
            with self.subTest(name):
                registered = []
                with self.assertRaisesRegex(RuntimeError, "expandable_segments"):
                    self._register({name: "expandable_segments:True"}, registered)
                self.assertEqual(registered, [])


if __name__ == "__main__":
    unittest.main()
