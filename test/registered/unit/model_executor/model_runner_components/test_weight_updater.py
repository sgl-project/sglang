"""Failed disk refits must preserve the checkpoint that is currently served."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("first", torch.tensor([2.0]))
        self.register_buffer("second", torch.tensor([3.0]))

    def load_weights(self, weights):
        for name, tensor in weights:
            getattr(self, name).copy_(tensor)

    def forward(self):
        return self.first + self.second


class TestDiskWeightUpdate(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())
        self.model = _Model()
        self.config = SimpleNamespace(
            model_path="original", revision=None, dtype=torch.float32, quantization=None
        )
        self.runner = SimpleNamespace(load_config=LoadConfig(load_format="safetensors"))
        self.sources = []
        self.load_groups = []
        self.checkpoints = {
            "original": [
                ("first", torch.tensor([2.0])),
                ("second", torch.tensor([3.0])),
            ],
            "replacement": [
                ("first", torch.tensor([7.0])),
                ("second", torch.tensor([9.0])),
            ],
            "broken": [("first", torch.tensor([7.0])), ("second", torch.ones(2))],
        }
        self.updater = weight_updater.WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=self.config,
            custom_weight_loaders={},
            get_model=lambda: self.model,
            update_model_fields=self._commit,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: self.runner,
        )

        def iterator(loader, source):
            self.sources.append((source.model_or_path, loader.load_config.load_format))
            self.load_groups.append(loader.load_config.load_group)
            if source.model_or_path == "missing":
                raise OSError("checkpoint unavailable")
            return iter(self.checkpoints[source.model_or_path])

        for patcher in (
            patch.object(DefaultModelLoader, "_get_weights_iterator", iterator),
            patch.object(weight_updater, "get_available_gpu_memory", return_value=0.0),
            patch.object(
                weight_updater,
                "get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _commit(self, model, *, model_path, load_format, load_config):
        self.model = model
        self.runner.load_config = load_config

    def test_partial_failure_restores_original_checkpoint_and_loader(self):
        """A failed second tensor must not leave the first tensor from a new checkpoint."""
        original_load_config = self.runner.load_config

        success, message = self.updater.update_weights_from_disk("broken", "pt")

        self.assertFalse(success)
        self.assertIn("Rolling back", message)
        self.assertEqual(self.model().item(), 5.0)
        self.assertEqual(self.config.model_path, "original")
        self.assertIs(self.runner.load_config, original_load_config)
        self.assertEqual(
            self.sources,
            [
                ("broken", LoadConfig(load_format="pt").load_format),
                ("original", original_load_config.load_format),
            ],
        )

    def test_preload_failures_preserve_original_model_path(self):
        for path, load_format in (("missing", "pt"), ("replacement", "dummy")):
            with self.subTest(path=path):
                success, _ = self.updater.update_weights_from_disk(path, load_format)

                self.assertFalse(success)
                self.assertEqual(self.config.model_path, "original")
                self.assertEqual(self.model().item(), 5.0)

    def test_successful_refit_commits_checkpoint_for_later_rollback(self):
        """A second failed refit must restore the latest successfully loaded checkpoint."""
        success, message = self.updater.update_weights_from_disk("replacement", "pt")
        self.assertTrue(success, message)
        self.assertEqual(self.model().item(), 16.0)
        self.assertEqual(self.config.model_path, "replacement")

        success, _ = self.updater.update_weights_from_disk("broken", "safetensors")

        self.assertFalse(success)
        self.assertEqual(self.model().item(), 16.0)
        self.assertEqual(self.config.model_path, "replacement")

    def test_rollback_uses_stage_load_group_without_changing_startup_config(self):
        """Sequential PP refits must not restore through a WORLD load collective."""
        original_load_group = self.runner.load_config.load_group
        stage_group = SimpleNamespace(cpu_group=None)
        with get_parallel().override(pp_size=2, tp_group=stage_group):
            success, _ = self.updater.update_weights_from_disk("broken", "pt")

        self.assertFalse(success)
        self.assertEqual(self.model().item(), 5.0)
        self.assertEqual(self.load_groups, [stage_group, stage_group])
        self.assertIs(self.runner.load_config.load_group, original_load_group)


if __name__ == "__main__":
    unittest.main()
