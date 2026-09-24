import json
import sys
import unittest
from contextlib import ExitStack
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
import sglang.srt.model_loader.weight_utils as weight_utils
from sglang.srt.configs.load_config import LoadConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Backend(Enum):
    MMAP = 0
    URING = 1
    AIO = 2


class _BackendPolicy(Enum):
    BUFFERED = 0


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
    def setUp(self):
        super().setUp()
        patches = ExitStack()
        self.addCleanup(patches.close)
        self.tensors = [("weight", torch.tensor([1]))]
        self.safe_open = MagicMock(return_value=_FakeSafeOpen(self.tensors))
        patches.enter_context(
            patch.dict(
                sys.modules,
                {
                    "instanttensor": SimpleNamespace(
                        safe_open=self.safe_open,
                        Backend=_Backend,
                        BackendPolicy=_BackendPolicy,
                    )
                },
            )
        )
        patches.enter_context(
            patch.object(weight_utils.torch.cuda, "current_device", return_value=1)
        )
        self.get_device = patches.enter_context(
            patch.object(
                weight_utils.current_platform,
                "get_device",
                return_value=torch.device("cuda:1"),
            )
        )
        self.is_initialized = patches.enter_context(
            patch.object(
                weight_utils.torch.distributed, "is_initialized", return_value=False
            )
        )
        self.get_parallel = patches.enter_context(
            patch.object(weight_utils, "get_parallel")
        )

    def test_extra_config_reaches_safe_open(self):
        options = {
            "buffer_size": 1024 * 1024 * 1024,
            "chunk_size": 8 * 1024 * 1024,
            "io_depth": 64,
            "backend": ["URING", "AIO"],
        }
        config = LoadConfig(
            load_format="instanttensor", model_loader_extra_config=json.dumps(options)
        )
        model_loader = loader_mod.DefaultModelLoader(config)
        source = loader_mod.DefaultModelLoader.Source("model", None)
        resolved = loader_mod.DefaultModelLoader.ResolvedSource(
            source=source,
            hf_folder="model",
            weight_files=("model.safetensors",),
            use_safetensors=True,
        )
        result = list(
            model_loader._get_weights_iterator(source, resolved_source=resolved)
        )

        self.assertEqual(result, self.tensors)
        self.safe_open.assert_called_once_with(
            ["model.safetensors"],
            framework="pt",
            device=torch.device("cuda:1"),
            process_group=None,
            copy=True,
            **{**options, "backend": [_Backend.URING, _Backend.AIO]},
        )
        self.assertEqual(config.model_loader_extra_config, options)
        self.get_device.assert_called_once_with(1)
        self.get_parallel.assert_not_called()

    def test_backend_conversion(self):
        for backend, expected in [
            ("MMAP", [_Backend.MMAP]),
            ("BUFFERED", [_BackendPolicy.BUFFERED]),
            (["BUFFERED", "MMAP"], [_BackendPolicy.BUFFERED, _Backend.MMAP]),
            (None, None),
        ]:
            with self.subTest(backend=backend):
                self.safe_open.reset_mock()
                list(
                    weight_utils.instanttensor_weights_iterator(
                        [], {"backend": backend}
                    )
                )
                self.assertEqual(self.safe_open.call_args.kwargs["backend"], expected)
                self.safe_open.assert_called_once()
        for backend in ["unknown", [], ["MMAP", "unknown"], 1]:
            with self.subTest(backend=backend):
                self.safe_open.reset_mock()
                with self.assertRaisesRegex(ValueError, "backend"):
                    list(
                        weight_utils.instanttensor_weights_iterator(
                            [], {"backend": backend}
                        )
                    )
                self.safe_open.assert_not_called()

    def test_iterator_uses_initialized_world_group(self):
        self.is_initialized.return_value = True
        device_group = object()
        with patch.object(weight_utils.torch.distributed, "get_rank", return_value=0):
            for world_size in (1, 2):
                with self.subTest(world_size=world_size):
                    self.safe_open.reset_mock()
                    self.get_parallel.return_value = SimpleNamespace(
                        world_group=SimpleNamespace(
                            world_size=world_size, device_group=device_group
                        )
                    )
                    result = list(
                        weight_utils.instanttensor_weights_iterator(
                            ["model.safetensors"]
                        )
                    )
                    self.assertEqual(result, self.tensors)
                    self.safe_open.assert_called_once_with(
                        ["model.safetensors"],
                        framework="pt",
                        device=torch.device("cuda:1"),
                        process_group=device_group if world_size > 1 else None,
                        copy=True,
                    )

    def test_iterator_rejects_unsupported_files(self):
        for files in (["model.pt"], ["model.bin"], ["model.safetensors", "model.pt"]):
            with self.subTest(files=files):
                with self.assertRaisesRegex(
                    ValueError, r"only supports \.safetensors"
                ) as raised:
                    list(weight_utils.instanttensor_weights_iterator(files))
                self.assertIn(files[-1], str(raised.exception))
                self.safe_open.assert_not_called()
                self.get_device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
