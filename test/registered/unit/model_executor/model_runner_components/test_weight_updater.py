"""CPU tests for model-runner weight reload configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.model_runner_components import (  # noqa: E402
    weight_updater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestWeightUpdaterDiskReload(CustomTestCase):
    def _reload_and_capture_configs(self, draft_model_idx):
        load_configs = []
        updated_load_configs = []
        model = SimpleNamespace(fall_back_to_pt_during_load=True)
        model_config = SimpleNamespace(
            model_path="/models/original",
            revision=None,
            dtype=torch.float32,
        )
        runner = SimpleNamespace(draft_model_idx=draft_model_idx)

        def get_loader(load_config, received_model_config):
            self.assertIs(received_model_config, model_config)
            load_configs.append(load_config)
            loader = DefaultModelLoader(load_config)
            loader._get_weights_iterator = MagicMock(return_value=iter(()))
            loader.load_weights_and_postprocess = MagicMock(
                side_effect=lambda received_model, weights, target_device: (
                    received_model
                )
            )
            return loader

        def update_model_fields(received_model, **kwargs):
            self.assertIs(received_model, model)
            updated_load_configs.append(kwargs["load_config"])

        updater = weight_updater.WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=model_config,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=update_model_fields,
            recapture_cuda_graph=lambda: self.fail("unexpected graph recapture"),
            get_model_runner=lambda: runner,
        )

        with (
            patch.object(weight_updater, "get_model_loader", side_effect=get_loader),
            patch.object(
                weight_updater,
                "get_available_gpu_memory",
                return_value=0.0,
            ),
        ):
            result = updater.update_weights_from_disk(
                "/models/reloaded",
                "safetensors",
            )

        self.assertEqual(result, (True, "Succeeded to update model weights."))
        self.assertEqual(len(load_configs), 1)
        self.assertEqual(updated_load_configs, load_configs)
        return load_configs[0]

    def test_reload_load_config_preserves_target_and_each_draft_model_index(self):
        for draft_model_idx in (None, 0, 1, 2):
            with self.subTest(draft_model_idx=draft_model_idx):
                load_config = self._reload_and_capture_configs(draft_model_idx)
                self.assertEqual(load_config.draft_model_idx, draft_model_idx)

    def test_default_loader_keeps_shared_weights_and_only_selected_mtp_layer(self):
        shared_embedding = object()
        shared_head = object()
        stage_payloads = {index: (object(), object()) for index in range(3)}
        weights = [
            ("model.embed_tokens.weight", shared_embedding),
            *[
                (name, payload)
                for index in range(3)
                for name, payload in (
                    (
                        f"model.mtp.layers.{index}.block.weight",
                        stage_payloads[index][0],
                    ),
                    (
                        f"model.mtp.layers.{index}.norm.weight",
                        stage_payloads[index][1],
                    ),
                )
            ],
            ("lm_head.weight", shared_head),
        ]

        for draft_model_idx in range(3):
            with self.subTest(draft_model_idx=draft_model_idx):
                filtered = list(
                    DefaultModelLoader._filter_mtp_weights(
                        iter(weights),
                        prefix="draft.",
                        draft_model_idx=draft_model_idx,
                    )
                )

                self.assertEqual(
                    [name for name, _ in filtered],
                    [
                        "draft.model.embed_tokens.weight",
                        "draft.model.mtp.layers.0.block.weight",
                        "draft.model.mtp.layers.0.norm.weight",
                        "draft.lm_head.weight",
                    ],
                )
                self.assertIs(filtered[0][1], shared_embedding)
                self.assertIs(filtered[1][1], stage_payloads[draft_model_idx][0])
                self.assertIs(filtered[2][1], stage_payloads[draft_model_idx][1])
                self.assertIs(filtered[3][1], shared_head)


if __name__ == "__main__":
    unittest.main()
