"""Every in-place reload path puts postprocessed weights back into checkpoint form
before the loader writes into them."""

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _checkpoint_engine_stand_in():
    """The worker module imports the optional checkpoint-engine extra at import time;
    get_model_loader never calls into it."""
    worker = ModuleType("checkpoint_engine.worker")
    worker.update_weights_from_ipc = Mock()
    package = ModuleType("checkpoint_engine")
    package.worker = worker
    return {"checkpoint_engine": package, "checkpoint_engine.worker": worker}


class TestInplaceWeightReloadRestoresFirst(CustomTestCase):
    def _disk_updater(self, model):
        return SimpleNamespace(
            get_model=lambda: model,
            _assert_weight_cache_inactive=Mock(),
            device="cpu",
            gpu_id=0,
            model_config=SimpleNamespace(model_path="old", dtype=torch.float32),
            update_model_fields=Mock(),
        )

    def _disk_update(self, loader, restore):
        model = torch.nn.Linear(1, 1)
        with (
            patch.object(weight_updater, "get_model_loader", return_value=loader),
            patch.object(weight_updater, "get_available_gpu_memory", return_value=0.0),
            patch.object(DefaultModelLoader, "restore_weights_before_loading", restore),
            patch.object(DefaultModelLoader.Source, "init_new", return_value=None),
        ):
            return WeightUpdater.update_weights_from_disk(
                self._disk_updater(model), "new", "auto"
            )

    def test_update_weights_from_disk_restores_before_every_load(self):
        calls = Mock()
        loader = Mock(spec=DefaultModelLoader)
        loader._get_weights_iterator.return_value = iter(())
        calls.attach_mock(loader.load_weights_and_postprocess, "load")
        calls.attach_mock(Mock(), "restore")

        ok, _ = self._disk_update(loader, calls.restore)

        self.assertTrue(ok)
        self.assertEqual([c[0] for c in calls.mock_calls], ["restore", "load"])

    def test_the_disk_rollback_restores_before_reloading_the_original(self):
        calls = Mock()
        loader = Mock(spec=DefaultModelLoader)
        loader._get_weights_iterator.side_effect = lambda _: iter(())
        loader.load_weights_and_postprocess.side_effect = [RuntimeError("bad"), None]
        calls.attach_mock(loader.load_weights_and_postprocess, "load")
        calls.attach_mock(Mock(), "restore")

        ok, message = self._disk_update(loader, calls.restore)

        self.assertFalse(ok)
        self.assertIn("Rolling back", message)
        self.assertEqual(
            [c[0] for c in calls.mock_calls], ["restore", "load", "restore", "load"]
        )

    def test_the_checkpoint_engine_loader_restores_before_every_bucket(self):
        with patch.dict(sys.modules, _checkpoint_engine_stand_in()):
            sys.modules.pop(
                "sglang.srt.checkpoint_engine.checkpoint_engine_worker", None
            )
            worker_module = importlib.import_module(
                "sglang.srt.checkpoint_engine.checkpoint_engine_worker"
            )
            try:
                calls = Mock()
                model = SimpleNamespace(load_weights=calls.load)
                worker = worker_module.SGLangCheckpointEngineWorkerExtensionImpl(
                    SimpleNamespace(model=model)
                )
                with (
                    patch.object(worker_module, "get_device", return_value="cuda"),
                    patch.object(
                        worker_module,
                        "get_device_module",
                        return_value=SimpleNamespace(current_device=lambda: 0),
                    ),
                    patch.object(
                        DefaultModelLoader,
                        "restore_weights_before_loading",
                        calls.restore,
                    ),
                ):
                    load = worker.get_model_loader()
                    load(["bucket 1"])
                    load(["bucket 2"])
            finally:
                sys.modules.pop(
                    "sglang.srt.checkpoint_engine.checkpoint_engine_worker", None
                )

        self.assertEqual(
            [c[0] for c in calls.mock_calls], ["restore", "load", "restore", "load"]
        )
        self.assertIs(calls.restore.call_args.args[0], model)
        self.assertEqual(calls.restore.call_args.args[1], torch.device("cuda", 0))


if __name__ == "__main__":
    unittest.main()
