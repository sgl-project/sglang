import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
            self.assertRaisesRegex(ImportError, "pip install instanttensor"),
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
