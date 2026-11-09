"""A failed disk update rolls back from the checkpoint the engine was serving."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MODULE = "sglang.srt.model_executor.model_runner_components.weight_updater"


class TestUpdateWeightsFromDiskRollback(CustomTestCase):
    def test_failed_load_reloads_the_original_checkpoint(self):
        model = torch.nn.Linear(1, 1)
        updater = SimpleNamespace(
            device="cpu",
            gpu_id=0,
            model_config=SimpleNamespace(
                model_path="/ckpt/original", dtype=torch.float32
            ),
            get_model=lambda: model,
            _assert_weight_cache_inactive=Mock(),
        )
        loader = Mock(spec=DefaultModelLoader)
        # Stand in for the checkpoint path each load would read.
        loader._get_weights_iterator.side_effect = lambda source: source
        loader.load_weights_and_postprocess.side_effect = [
            RuntimeError("bad shard"),
            None,
        ]
        with (
            patch(f"{_MODULE}.get_model_loader", return_value=loader),
            patch(f"{_MODULE}.LoadConfig"),
            patch(f"{_MODULE}.get_pp_stage_load_group"),
            patch(f"{_MODULE}.get_available_gpu_memory", return_value=0.0),
            patch.object(
                DefaultModelLoader.Source,
                "init_new",
                side_effect=lambda config, model: config.model_path,
            ),
        ):
            ok, message = WeightUpdater.update_weights_from_disk(
                updater, "/ckpt/new", "auto"
            )

        self.assertFalse(ok)
        self.assertIn("Rolling back", message)
        self.assertEqual(
            [c.args[1] for c in loader.load_weights_and_postprocess.call_args_list],
            ["/ckpt/new", "/ckpt/original"],
        )
        self.assertEqual(updater.model_config.model_path, "/ckpt/original")


if __name__ == "__main__":
    unittest.main()
