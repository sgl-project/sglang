"""Unit tests for RemoteInstanceModelLoader construction — no server, no weights."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
)
from sglang.test.test_utils import CustomTestCase

EXTRA_CONFIG = {"enable_multithread_load": True, "num_threads": 64}


def _load_config(backend, extra_config=None):
    return LoadConfig(
        load_format=LoadFormat.REMOTE_INSTANCE,
        model_loader_extra_config=extra_config,
        remote_instance_weight_loader_backend=backend,
    )


class TestRemoteInstanceModelLoaderExtraConfig(CustomTestCase):
    def test_modelexpress_backend_accepts_extra_config(self):
        load_config = _load_config(
            RemoteInstanceWeightLoaderBackend.MODELEXPRESS, EXTRA_CONFIG
        )
        loader = loader_mod.RemoteInstanceModelLoader(load_config)
        # ModelExpress hands the extra config to the native loader it falls back
        # to, so it must survive construction unmodified.
        self.assertEqual(loader.load_config.model_loader_extra_config, EXTRA_CONFIG)

    def test_modelexpress_backend_accepts_extra_config_as_string(self):
        load_config = _load_config(
            RemoteInstanceWeightLoaderBackend.MODELEXPRESS,
            '{"enable_multithread_load": true, "num_threads": 64}',
        )
        loader = loader_mod.RemoteInstanceModelLoader(load_config)
        self.assertEqual(loader.load_config.model_loader_extra_config, EXTRA_CONFIG)

    def test_modelexpress_backend_without_extra_config(self):
        load_config = _load_config(RemoteInstanceWeightLoaderBackend.MODELEXPRESS)
        loader = loader_mod.RemoteInstanceModelLoader(load_config)
        self.assertFalse(loader.load_config.model_loader_extra_config)

    def test_nccl_backend_rejects_extra_config(self):
        load_config = _load_config(RemoteInstanceWeightLoaderBackend.NCCL, EXTRA_CONFIG)
        with self.assertRaises(ValueError):
            loader_mod.RemoteInstanceModelLoader(load_config)

    def test_transfer_engine_backend_rejects_extra_config(self):
        load_config = _load_config(
            RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE, EXTRA_CONFIG
        )
        with self.assertRaises(ValueError):
            loader_mod.RemoteInstanceModelLoader(load_config)

    def test_backends_without_extra_config_still_construct(self):
        for backend in (
            RemoteInstanceWeightLoaderBackend.NCCL,
            RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE,
        ):
            with self.subTest(backend=backend):
                loader = loader_mod.RemoteInstanceModelLoader(_load_config(backend))
                self.assertFalse(loader.load_config.model_loader_extra_config)


if __name__ == "__main__":
    unittest.main()
