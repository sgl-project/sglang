import json
import sys
import tempfile
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
import sglang.srt.model_loader.weight_utils as weight_utils
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeSafeOpen:
    def __init__(self, tensors):
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def keys(self):
        return [name for name, _ in self._tensors]

    def tensors(self):
        yield from self._tensors


class TestInstantTensorLoader(CustomTestCase):
    def test_extra_config_reaches_safe_open(self):
        class Backend(Enum):
            URING = 0
            AIO = 1

        class BackendPolicy(Enum):
            BUFFERED = 0

        options = {
            "buffer_size": 1024 * 1024 * 1024,
            "chunk_size": 8 * 1024 * 1024,
            "io_depth": 64,
            "backend": ["URING", "AIO"],
        }
        config = options.copy()
        load_config = LoadConfig(
            load_format="instanttensor", model_loader_extra_config=json.dumps(config)
        )
        model_loader = loader_mod.DefaultModelLoader(load_config)
        safe_open = MagicMock(return_value=_FakeSafeOpen([]))
        module = SimpleNamespace(
            safe_open=safe_open, Backend=Backend, BackendPolicy=BackendPolicy
        )
        with (
            patch.dict(sys.modules, {"instanttensor": module}),
            patch.object(weight_utils.torch.cuda, "current_device", return_value=0),
            patch.object(
                weight_utils.current_platform,
                "get_device",
                return_value=torch.device("cuda:0"),
            ),
            patch.object(
                weight_utils.torch.distributed, "is_initialized", return_value=False
            ),
            patch.object(
                model_loader,
                "_prepare_weights",
                return_value=("model", ["model.safetensors"], True),
            ),
        ):
            self.assertEqual(
                list(
                    model_loader._get_weights_iterator(
                        loader_mod.DefaultModelLoader.Source("model", None)
                    )
                ),
                [],
            )

        safe_open.assert_called_once_with(
            ["model.safetensors"],
            framework="pt",
            device=torch.device("cuda:0"),
            process_group=None,
            copy=True,
            **{**options, "backend": [Backend.URING, Backend.AIO]},
        )
        self.assertEqual(load_config.model_loader_extra_config, config)

    def test_backend_names(self):
        class Backend(Enum):
            MMAP = 0

        class BackendPolicy(Enum):
            BUFFERED = 0

        safe_open = MagicMock(return_value=_FakeSafeOpen([]))
        module = SimpleNamespace(
            safe_open=safe_open, Backend=Backend, BackendPolicy=BackendPolicy
        )
        with (
            patch.dict(sys.modules, {"instanttensor": module}),
            patch.object(weight_utils.torch.cuda, "current_device", return_value=0),
            patch.object(
                weight_utils.current_platform,
                "get_device",
                return_value=torch.device("cuda:0"),
            ),
            patch.object(
                weight_utils.torch.distributed, "is_initialized", return_value=False
            ),
        ):
            for backend, expected in [
                ("MMAP", [Backend.MMAP]),
                ("BUFFERED", [BackendPolicy.BUFFERED]),
                (None, None),
            ]:
                with self.subTest(backend=backend):
                    list(
                        weight_utils.instanttensor_weights_iterator(
                            [], {"backend": backend}
                        )
                    )
                    self.assertEqual(safe_open.call_args.kwargs["backend"], expected)
            null_options = dict.fromkeys(
                [
                    "backend",
                    "buffer_size",
                    "chunk_size",
                    "io_depth",
                ]
            )
            list(weight_utils.instanttensor_weights_iterator([], null_options))
            for key in null_options:
                self.assertIsNone(safe_open.call_args.kwargs[key])
            for backend in ["unknown", [], ["MMAP", "unknown"], 1, [1], {"MMAP": True}]:
                with (
                    self.subTest(backend=backend),
                    self.assertRaisesRegex(ValueError, "backend"),
                ):
                    list(
                        weight_utils.instanttensor_weights_iterator(
                            [], {"backend": backend}
                        )
                    )

    def test_duplicate_options_are_rejected_by_python(self):
        def safe_open(filename, *, framework, device, process_group, copy):
            self.fail("Duplicate arguments must fail before entering safe_open")

        with (
            patch.dict(
                sys.modules, {"instanttensor": SimpleNamespace(safe_open=safe_open)}
            ),
            patch.object(weight_utils.torch.cuda, "current_device", return_value=0),
            patch.object(
                weight_utils.current_platform,
                "get_device",
                return_value=torch.device("cuda:0"),
            ),
            patch.object(
                weight_utils.torch.distributed, "is_initialized", return_value=False
            ),
        ):
            for key in ["filename", "device", "framework", "process_group", "copy"]:
                with self.subTest(key=key):
                    model_loader = loader_mod.DefaultModelLoader(
                        LoadConfig(
                            load_format="instanttensor",
                            model_loader_extra_config={
                                key: False if key == "copy" else None
                            },
                        )
                    )
                    source = loader_mod.DefaultModelLoader.Source("model", None)
                    resolved = loader_mod.DefaultModelLoader.ResolvedSource(
                        source=source,
                        hf_folder="model",
                        weight_files=("model.safetensors",),
                        use_safetensors=True,
                    )
                    with self.assertRaisesRegex(TypeError, f"multiple values.*'{key}'"):
                        list(
                            model_loader._get_weights_iterator(
                                source, resolved_source=resolved
                            )
                        )

    def test_parameter_errors_propagate_from_instanttensor(self):
        error = ValueError("chunk_size must be greater than zero")

        def safe_open(files, *, framework, device, process_group, copy, chunk_size):
            self.assertEqual(chunk_size, 0)
            raise error

        with (
            patch.dict(
                sys.modules, {"instanttensor": SimpleNamespace(safe_open=safe_open)}
            ),
            patch.object(weight_utils.torch.cuda, "current_device", return_value=0),
            patch.object(
                weight_utils.current_platform,
                "get_device",
                return_value=torch.device("cuda:0"),
            ),
            patch.object(
                weight_utils.torch.distributed, "is_initialized", return_value=False
            ),
        ):
            model_loader = loader_mod.DefaultModelLoader(
                LoadConfig(
                    load_format="instanttensor",
                    model_loader_extra_config={"chunk_size": 0},
                )
            )
            source = loader_mod.DefaultModelLoader.Source("model", None)
            resolved = loader_mod.DefaultModelLoader.ResolvedSource(
                source=source,
                hf_folder="model",
                weight_files=("model.safetensors",),
                use_safetensors=True,
            )
            with self.assertRaises(ValueError) as raised:
                list(
                    model_loader._get_weights_iterator(source, resolved_source=resolved)
                )
            self.assertIs(raised.exception, error)

    def test_instanttensor_options_do_not_change_other_formats(self):
        for load_format in ["auto", "safetensors", "fastsafetensors"]:
            with (
                self.subTest(load_format=load_format),
                self.assertRaisesRegex(ValueError, "Unexpected extra config"),
            ):
                loader_mod.DefaultModelLoader(
                    LoadConfig(
                        load_format=load_format,
                        model_loader_extra_config={"io_depth": 2},
                    )
                )

    def test_iterator_uses_current_device_and_world_group(self):
        tensors = [("weight", torch.tensor([1]))]
        device_group = object()
        calls = []

        def safe_open(files, **kwargs):
            calls.append((files, kwargs))
            return _FakeSafeOpen(tensors)

        files = ["b", "a"]
        module = SimpleNamespace(safe_open=safe_open)
        with (
            patch.dict(sys.modules, {"instanttensor": module}),
            patch.object(
                weight_utils.current_platform,
                "get_device",
                return_value=torch.device("cuda:0"),
            ),
            patch.object(weight_utils.torch.cuda, "current_device", return_value=0),
            patch.object(
                weight_utils,
                "get_parallel",
                return_value=SimpleNamespace(
                    world_group=SimpleNamespace(
                        world_size=2,
                        device_group=device_group,
                    )
                ),
            ),
            patch.object(
                weight_utils.torch.distributed,
                "is_initialized",
                return_value=True,
            ),
            patch.object(weight_utils.torch.distributed, "get_rank", return_value=0),
        ):
            result = list(weight_utils.instanttensor_weights_iterator(files))

        self.assertEqual(result, tensors)
        self.assertIs(calls[0][0], files)
        self.assertEqual(calls[0][1]["framework"], "pt")
        self.assertEqual(calls[0][1]["device"], torch.device("cuda:0"))
        self.assertIs(calls[0][1]["process_group"], device_group)
        self.assertIs(calls[0][1]["copy"], True)
        self.assertEqual(
            set(calls[0][1]), {"framework", "device", "process_group", "copy"}
        )

    def test_iterator_without_initialized_world_group(self):
        module = SimpleNamespace(
            safe_open=lambda files, **kwargs: _FakeSafeOpen([]),
        )
        with (
            patch.dict(sys.modules, {"instanttensor": module}),
            patch.object(
                weight_utils.current_platform,
                "get_device",
                return_value=torch.device("cuda:0"),
            ),
            patch.object(weight_utils.torch.cuda, "current_device", return_value=0),
            patch.object(weight_utils, "get_parallel") as get_parallel,
            patch.object(
                weight_utils.torch.distributed,
                "is_initialized",
                return_value=False,
            ),
        ):
            self.assertEqual(
                list(
                    weight_utils.instanttensor_weights_iterator(["model.safetensors"])
                ),
                [],
            )

        get_parallel.assert_not_called()

    def test_iterator_reports_missing_dependency(self):
        with (
            patch.dict(sys.modules, {"instanttensor": None}),
            self.assertRaisesRegex(ImportError, 'pip install "instanttensor>=0.1.9"'),
        ):
            list(weight_utils.instanttensor_weights_iterator(["model.safetensors"]))

    def test_format_uses_safetensors_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = f"{folder}/model.safetensors"
            open(checkpoint, "wb").close()
            model_loader = loader_mod.DefaultModelLoader(
                LoadConfig(load_format=LoadFormat.INSTANTTENSOR)
            )
            with patch.object(loader_mod, "get_server_args", return_value=None):
                _, weight_files, use_safetensors = model_loader._prepare_weights(
                    folder,
                    revision=None,
                    fall_back_to_pt=True,
                )

        self.assertEqual(weight_files, [checkpoint])
        self.assertTrue(use_safetensors)


if __name__ == "__main__":
    unittest.main()
