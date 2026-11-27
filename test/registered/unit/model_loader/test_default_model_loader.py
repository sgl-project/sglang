import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.connector import ConnectorType
from sglang.srt.connector.utils import pull_files_from_db
from sglang.srt.model_loader.loader import (
    DefaultModelLoader,
    ModelOptModelLoader,
    RemoteModelLoader,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _InMemoryRemoteConnector:
    def __init__(self, local_dir):
        self.local_dir = local_dir
        self.tensors = {}
        self.files = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def set(self, key, tensor):
        self.tensors[key] = tensor

    def setstr(self, key, value):
        self.files[key] = value

    def get_local_dir(self):
        return self.local_dir

    def list(self, prefix):
        return [key for key in self.files if key.startswith(prefix)]

    def getstr(self, key):
        return self.files[key]


class TestRemoteModelLoader(CustomTestCase):
    def test_save_model_preserves_relative_metadata_paths(self):
        source_files = {
            "config.json": '{"name": "example"}',
            "generation_config.json": '{"temperature": 0.7}',
            "custom/nested.json": '{"nested": true}',
            "custom/a.py": "VALUE = 'custom'\n",
            "other/a.py": "VALUE = 'other'\n",
            ".hidden.json": "hidden",
            "custom/.hidden.py": "hidden",
            "README.md": "unsupported",
            "custom/notes.txt": "unsupported",
        }
        expected_files = {
            "model/files/config.json": source_files["config.json"],
            "model/files/generation_config.json": source_files[
                "generation_config.json"
            ],
            "model/files/custom/nested.json": source_files["custom/nested.json"],
            "model/files/custom/a.py": source_files["custom/a.py"],
            "model/files/other/a.py": source_files["other/a.py"],
        }

        with (
            tempfile.TemporaryDirectory() as model_path,
            tempfile.TemporaryDirectory() as local_dir,
        ):
            for relative_path, content in source_files.items():
                path = os.path.join(model_path, *relative_path.split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as file:
                    file.write(content)

            client = _InMemoryRemoteConnector(local_dir)
            with (
                patch.object(
                    loader_mod, "create_remote_connector", return_value=client
                ),
                patch.object(
                    loader_mod, "get_connector_type", return_value=ConnectorType.KV
                ),
                patch.object(loader_mod, "_checkpoint_shard_rank", return_value=0),
            ):
                RemoteModelLoader.save_model(
                    torch.nn.Module(), model_path, "redis://localhost:6379/model"
                )

            self.assertEqual(client.files, expected_files)

            pull_files_from_db(client, "model")
            pulled_files = {}
            for root, _, file_names in os.walk(local_dir):
                for file_name in file_names:
                    path = os.path.join(root, file_name)
                    with open(path, encoding="utf-8") as file:
                        relative_path = os.path.relpath(path, local_dir).replace(
                            os.sep, "/"
                        )
                        pulled_files[relative_path] = file.read()

            self.assertEqual(
                pulled_files,
                {
                    key.removeprefix("model/files/"): value
                    for key, value in expected_files.items()
                },
            )


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

    def test_mistral_honors_index_opt_out(self):
        """safetensors_index_file=None must skip the consolidated index too."""
        with tempfile.TemporaryDirectory() as model_path:
            files = [
                os.path.join(model_path, f"consolidated-0000{i}.safetensors")
                for i in (1, 2)
            ]
            for path in files:
                open(path, "w").close()
            with open(
                os.path.join(model_path, "consolidated.safetensors.index.json"), "w"
            ) as f:
                json.dump({"weight_map": {"w": os.path.basename(files[0])}}, f)
            loader = DefaultModelLoader(LoadConfig(load_format=LoadFormat.MISTRAL))
            with patch.object(loader_mod, "get_server_args", return_value=None):
                for index_file, expected in (
                    ("model.safetensors.index.json", files[:1]),
                    (None, files),
                ):
                    with self.subTest(index_file=index_file):
                        _, weight_files, _ = loader._prepare_weights(
                            model_path,
                            None,
                            fall_back_to_pt=False,
                            safetensors_index_file=index_file,
                        )
                        self.assertEqual(sorted(weight_files), expected)

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


class TestModelOptLoaderSelection(CustomTestCase):
    def test_every_modelopt_quantization_selects_the_modelopt_loader(self):
        """A pre-quantized checkpoint under any ModelOpt quantization value must
        load through ModelOptModelLoader, not fall through to the default loader."""
        for quantization in (
            "modelopt",
            "modelopt_fp8",
            "modelopt_fp4",
            "modelopt_mixed",
        ):
            with self.subTest(quantization=quantization):
                model_config = SimpleNamespace(
                    quantization=quantization, _is_already_quantized=lambda: True
                )
                self.assertIsInstance(
                    loader_mod.get_model_loader(LoadConfig(), model_config),
                    ModelOptModelLoader,
                )


if __name__ == "__main__":
    unittest.main()
