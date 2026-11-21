"""A failed disk update must roll back to the checkpoint served before it."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ORIGINAL, NEW = "/ckpt/original", "/ckpt/new"


class _FakeLoader(DefaultModelLoader):
    """Serves each checkpoint path from a dict; a ``None`` weight raises."""

    def __init__(self, checkpoints):
        self.checkpoints = checkpoints
        self.loaded_paths = []

    def _get_weights_iterator(self, source):
        self.loaded_paths.append(source.model_or_path)
        return iter(self.checkpoints[source.model_or_path])

    def load_weights_and_postprocess(self, model, weights, target_device):
        params = dict(model.named_parameters())
        for name, value in weights:
            if value is None:
                raise OSError(f"corrupt shard at {name}")
            params[name].data.copy_(value)


class TestUpdateWeightsFromDiskRollback(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)
        )
        self.original = {
            n: p.detach().clone() for n, p in self.model.named_parameters()
        }
        self.config = SimpleNamespace(
            model_path=ORIGINAL, revision=None, dtype=torch.float32
        )
        self.updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=self.config,
            custom_weight_loaders={},
            get_model=lambda: self.model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )

    def _update(self, loader):
        with ExitStack() as stack:
            for name, value in [
                ("get_model_loader", lambda *a, **k: loader),
                ("get_available_gpu_memory", lambda *a, **k: 0.0),
                ("get_pp_stage_load_group", lambda: None),
                ("_discard_pending_weight_halves", lambda model: []),
            ]:
                stack.enter_context(patch.object(weight_updater, name, value))
            stack.enter_context(
                patch.object(
                    DefaultModelLoader,
                    "restore_weights_before_loading",
                    lambda *a, **k: None,
                )
            )
            stack.enter_context(
                patch.object(
                    WeightUpdater, "_assert_weight_cache_inactive", lambda *a: None
                )
            )
            return self.updater.update_weights_from_disk(NEW, load_format="auto")

    def _assert_original_weights(self):
        for name, param in self.model.named_parameters():
            self.assertTrue(torch.equal(param, self.original[name]), name)

    def _original_checkpoint(self):
        return [(n, t.clone()) for n, t in self.original.items()]

    def test_partial_load_failure_restores_original_weights(self):
        # The new checkpoint writes layer 0, then fails on layer 1.
        loader = _FakeLoader(
            {
                ORIGINAL: self._original_checkpoint(),
                NEW: [("0.weight", torch.zeros(2, 2)), ("1.weight", None)],
            }
        )
        success, message = self._update(loader)
        self.assertFalse(success)
        self.assertIn("Rolled back to original weights", message)
        self.assertEqual(loader.loaded_paths, [NEW, ORIGINAL])
        self.assertEqual(self.config.model_path, ORIGINAL)
        self._assert_original_weights()

    def test_failed_rollback_is_reported_not_raised(self):
        loader = _FakeLoader(
            {
                ORIGINAL: [("0.weight", None)],
                NEW: [("0.weight", None)],
            }
        )
        success, message = self._update(loader)
        self.assertFalse(success)
        self.assertIn("Failed to roll back to original weights", message)
        self.assertEqual(self.config.model_path, ORIGINAL)

    def test_iterator_failure_keeps_original_path(self):
        class _NoIterator(_FakeLoader):
            def _get_weights_iterator(self, source):
                raise FileNotFoundError(source.model_or_path)

        success, message = self._update(_NoIterator({}))
        self.assertFalse(success)
        self.assertIn("Failed to get weights iterator", message)
        self.assertEqual(self.config.model_path, ORIGINAL)
        self._assert_original_weights()


if __name__ == "__main__":
    unittest.main()
