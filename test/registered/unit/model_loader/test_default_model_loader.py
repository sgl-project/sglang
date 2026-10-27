import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDefaultModelLoader(CustomTestCase):
    def test_load_weights_only_precedes_postprocessing(self):
        events = []
        model = Mock()
        model.quant_config = None
        module = Mock()
        module.quant_method.process_weights_after_loading.side_effect = lambda _: (
            events.append("postprocess")
        )
        model.load_weights.side_effect = lambda _: events.append("load")
        model.named_modules.return_value = [("layer", module)]

        with (
            patch.object(loader_mod, "is_cuda_alike", return_value=False),
            patch.object(
                loader_mod,
                "device_loading_context",
                side_effect=lambda *_: nullcontext(),
            ),
        ):
            DefaultModelLoader.load_weights_and_postprocess(
                model,
                iter(()),
                torch.device("cpu"),
            )

        self.assertEqual(events, ["load", "postprocess"])

    def test_reload_requires_explicit_initial_load_marker(self):
        self.assertFalse(loader_mod.QuantizedRLModelLoader.is_reload_scenario(Mock()))
        model = torch.nn.Module()
        model.original_weights_rebuild_keys = set()
        model.recorded_loader = {}
        for marker in (None, False, 1, Mock()):
            with self.subTest(marker=marker):
                model.flash_rl_initial_load_complete = marker
                self.assertFalse(
                    loader_mod.QuantizedRLModelLoader.is_reload_scenario(model)
                )
        model.flash_rl_initial_load_complete = True
        self.assertTrue(loader_mod.QuantizedRLModelLoader.is_reload_scenario(model))

    def test_boot_paths_preserve_load_order_and_custom_override(self):
        class CustomModelLoader(DefaultModelLoader):
            def load_weights_and_postprocess(self, model, weights, target_device):
                events.append("override")
                DefaultModelLoader.load_weights_and_postprocess(
                    model, weights, target_device
                )

        for loader_class in (DefaultModelLoader, CustomModelLoader):
            with self.subTest(loader=loader_class.__name__):
                events = []
                loader = object.__new__(loader_class)
                loader.load_config = object()
                model = Mock(quant_config=None)
                model.eval.return_value = model
                model.load_weights.side_effect = lambda weights: events.append(
                    ("load", list(weights))
                )
                module = Mock()
                module.quant_method.process_weights_after_loading.side_effect = (
                    lambda _: events.append("postprocess")
                )
                model.named_modules.return_value = [("layer", module)]
                model_config = SimpleNamespace(modelopt_quant=None, dtype=torch.float32)
                resolved_source = SimpleNamespace(source=object())

                with (
                    patch.object(loader_mod, "is_cuda_alike", return_value=False),
                    patch.object(
                        loader_mod, "_get_quantization_config", return_value=None
                    ),
                    patch.object(loader_mod, "_initialize_model", return_value=model),
                    patch.object(
                        loader_mod,
                        "device_loading_context",
                        side_effect=lambda *_: nullcontext(),
                    ),
                    patch.object(
                        loader, "_get_all_weights", return_value=iter([("direct", 1)])
                    ),
                    patch.object(
                        loader,
                        "_get_weights_iterator",
                        return_value=iter([("deferred", 2)]),
                    ) as get_weights,
                ):
                    self.assertIs(
                        loader.load_model(
                            model_config=model_config,
                            device_config=SimpleNamespace(device="cpu"),
                        ),
                        model,
                    )
                    loader.commit_model_weights(
                        model=model,
                        model_config=model_config,
                        resolved_sources=(resolved_source,),
                        target_device=torch.device("cpu"),
                        startup_prefetch_active=True,
                    )

                expected = []
                for weights in ([("direct", 1)], [("deferred", 2)]):
                    if loader_class is CustomModelLoader:
                        expected.append("override")
                    expected.extend([("load", weights), "postprocess"])
                self.assertEqual(events, expected)
                get_weights.assert_called_once_with(
                    resolved_source.source,
                    resolved_source=resolved_source,
                    startup_prefetch_started=True,
                    startup_prefetch_active=True,
                )

    def test_prepare_weights_finds_shards_the_index_lists(self):
        # Top-level shards keep the glob path; shards in tensors/ come from the
        # index, and a stray subfolder file the index does not list stays out.
        for shard_folder in ("", "tensors/"):
            with (
                self.subTest(shard_folder=shard_folder),
                tempfile.TemporaryDirectory() as model_folder,
            ):
                shard_names = [
                    f"{shard_folder}model-0000{i}-of-00002.safetensors" for i in (1, 2)
                ]
                for name in shard_names + ["extras/unlisted.safetensors"]:
                    path = os.path.join(model_folder, name)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    open(path, "w").close()
                with open(
                    os.path.join(model_folder, "model.safetensors.index.json"), "w"
                ) as f:
                    json.dump({"weight_map": dict(zip("ab", shard_names))}, f)
                loader = object.__new__(DefaultModelLoader)
                loader.load_config = LoadConfig()

                with patch.object(loader_mod, "get_server_args", return_value=None):
                    _, files, use_safetensors = loader._prepare_weights(
                        model_folder, None, fall_back_to_pt=False
                    )

                self.assertTrue(use_safetensors)
                expected = [os.path.join(model_folder, n) for n in shard_names]
                self.assertEqual(files, expected)

    def test_weight_name_filter_reads_only_the_shards_holding_kept_tensors(self):
        # A draft sharing its target's checkpoint opens only the shards the index
        # places its tensors in, and sees none of the target's tensors.
        with tempfile.TemporaryDirectory() as model_folder:
            weight_map = {
                "mtp.fc.weight": "tensors/model-00001-of-00002.safetensors",
                "model.layers.0.mlp.experts.0.weight": (
                    "tensors/model-00001-of-00002.safetensors"
                ),
                "model.layers.1.mlp.experts.0.weight": (
                    "tensors/model-00002-of-00002.safetensors"
                ),
            }
            with open(
                os.path.join(model_folder, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump({"weight_map": weight_map}, f)
            shard_files = [
                os.path.join(model_folder, shard)
                for shard in sorted(set(weight_map.values()))
            ]
            source = DefaultModelLoader.Source(
                model_folder,
                None,
                weight_name_filter=lambda name: name.startswith("mtp."),
            )
            resolved = DefaultModelLoader.ResolvedSource(
                source=source,
                hf_folder=model_folder,
                weight_files=tuple(shard_files),
                use_safetensors=True,
            )
            opened_files = []

            def first_shard_tensors(files, **_):
                opened_files.extend(files)
                return iter(
                    [
                        ("mtp.fc.weight", torch.zeros(1)),
                        ("model.layers.0.mlp.experts.0.weight", torch.zeros(1)),
                    ]
                )

            loader = DefaultModelLoader(LoadConfig(load_format="instanttensor"))
            with patch.object(
                loader_mod,
                "instanttensor_weights_iterator",
                side_effect=first_shard_tensors,
            ):
                names = [
                    name
                    for name, _ in loader._get_weights_iterator(
                        source, resolved_source=resolved
                    )
                ]

        self.assertEqual(opened_files, shard_files[:1])
        self.assertEqual(names, ["mtp.fc.weight"])


if __name__ == "__main__":
    unittest.main()
