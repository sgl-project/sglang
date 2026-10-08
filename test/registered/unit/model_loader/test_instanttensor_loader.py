import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import safe_open, save_file

import sglang.srt.model_loader.loader as loader_mod
import sglang.srt.model_loader.weight_utils as weight_utils
from sglang.srt.configs.load_config import _DEFAULT_LOAD_GROUP, LoadConfig
from sglang.srt.environ import envs
from sglang.srt.platforms.cuda import CudaSRTPlatform
from sglang.srt.platforms.rocm import RocmSRTPlatform
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
            patch.object(weight_utils.current_platform, "device_type", "cuda")
        )
        self.current_device = patches.enter_context(
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

    def test_loader_passes_extra_config_and_load_group(self):
        self.is_initialized.return_value = True
        group = SimpleNamespace(world_size=2, device_group=object())
        self.get_parallel.return_value = SimpleNamespace(
            world_group=SimpleNamespace(world_size=4, device_group=object())
        )
        options = {
            "buffer_size": 1024 * 1024 * 1024,
            "chunk_size": 8 * 1024 * 1024,
            "io_depth": 64,
            "backend": ["URING", "AIO"],
        }
        config = LoadConfig(
            load_format="instanttensor",
            model_loader_extra_config=json.dumps(options),
            load_group=group,
        )
        model_loader = loader_mod.DefaultModelLoader(config)
        source = loader_mod.DefaultModelLoader.Source("model", None)
        resolved = loader_mod.DefaultModelLoader.ResolvedSource(
            source=source,
            hf_folder="model",
            weight_files=("model.safetensors",),
            use_safetensors=True,
        )
        with patch.object(weight_utils.torch.distributed, "get_rank", return_value=0):
            result = list(
                model_loader._get_weights_iterator(source, resolved_source=resolved)
            )

        self.assertEqual(result, self.tensors)
        self.safe_open.assert_called_once_with(
            ["model.safetensors"],
            framework="pt",
            device=torch.device("cuda:1"),
            process_group=group.device_group,
            copy=True,
            **{**options, "backend": [_Backend.URING, _Backend.AIO]},
        )
        self.assertEqual(config.model_loader_extra_config, options)

    def test_iterator_accepts_cuda_and_rocm(self):
        for platform in (CudaSRTPlatform(), RocmSRTPlatform()):
            with (
                self.subTest(platform=platform.device_name),
                patch.object(weight_utils, "current_platform", platform),
            ):
                self.safe_open.reset_mock()
                self.assertEqual(
                    list(
                        weight_utils.instanttensor_weights_iterator(
                            ["model.safetensors"]
                        )
                    ),
                    self.tensors,
                )
                self.safe_open.assert_called_once_with(
                    ["model.safetensors"],
                    framework="pt",
                    device=torch.device("cuda:1"),
                    process_group=None,
                    copy=True,
                )

    def test_iterator_rejects_non_cuda_devices_before_import(self):
        for device_type in ("cpu", "xpu", "npu", "musa", "hpu", "mps"):
            with (
                self.subTest(device_type=device_type),
                patch.object(weight_utils.current_platform, "device_type", device_type),
                patch.dict(sys.modules, {"instanttensor": None}),
                self.assertRaisesRegex(
                    ValueError, "InstantTensor requires a CUDA-compatible device"
                ) as raised,
            ):
                list(weight_utils.instanttensor_weights_iterator(["model.safetensors"]))
            self.assertIn(repr(device_type), str(raised.exception))
        self.current_device.assert_not_called()
        self.get_device.assert_not_called()
        self.safe_open.assert_not_called()

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
                        ),
                        tp_group=SimpleNamespace(world_size=1, device_group=object()),
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

    def test_iterator_uses_explicit_load_group(self):
        self.is_initialized.return_value = True
        world_group = SimpleNamespace(world_size=4, device_group=object())
        tp_group = SimpleNamespace(world_size=2, device_group=object())
        self.get_parallel.return_value = SimpleNamespace(
            world_group=world_group, tp_group=tp_group
        )

        with patch.object(weight_utils.torch.distributed, "get_rank", return_value=2):
            for group in (
                tp_group,
                None,
                SimpleNamespace(world_size=1, device_group=object()),
            ):
                with self.subTest(group=group):
                    self.safe_open.reset_mock()
                    list(
                        weight_utils.instanttensor_weights_iterator(
                            ["model.safetensors"], load_group=group
                        )
                    )
                    self.safe_open.assert_called_once()
                    self.assertIs(
                        self.safe_open.call_args.kwargs["process_group"],
                        tp_group.device_group if group is tp_group else None,
                    )

    def test_pp_stage_load_group(self):
        tp_group = object()
        for pp_size in (1, 2):
            with self.subTest(pp_size=pp_size):
                self.get_parallel.return_value = SimpleNamespace(
                    pp_size=pp_size, tp_group=tp_group
                )
                self.assertIs(
                    weight_utils.get_pp_stage_load_group(),
                    tp_group if pp_size > 1 else _DEFAULT_LOAD_GROUP,
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

    def test_pp_embedding_can_reopen_instanttensor_checkpoint(self):
        from sglang.srt.speculative.pp_draft_embedding import (
            load_embedding_tensor,
            prepare_checkpoint_files,
        )

        expected = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        name = "model.embed_tokens.weight"
        with tempfile.TemporaryDirectory() as folder:
            save_file({name: expected}, f"{folder}/model.safetensors")
            with get_context().override_server_args():
                hf_folder, weight_files, use_safetensors = prepare_checkpoint_files(
                    folder,
                    revision=None,
                    load_config=LoadConfig(load_format="instanttensor"),
                )
                key, actual = load_embedding_tensor(
                    hf_folder, weight_files, use_safetensors=use_safetensors
                )
        self.assertEqual(key, name)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        self.safe_open.assert_not_called()


class TestInstantTensorHostResidentWeights(CustomTestCase):
    """Checkpoint files holding host-resident tensors bypass InstantTensor."""

    # Stands in for the GPU: tensors placed on the device show up as "meta".
    DEVICE = torch.device("meta")

    def setUp(self):
        super().setUp()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # DeepSeek-V4.1 checkpoint names; the Engram file also holds a tensor
        # that is not host-resident.
        layout = {
            "model-00001.safetensors": ["layers.0.attn.wq_a.weight"],
            "model-00002.safetensors": [
                "layers.1.engram.embed.weight",
                "layers.1.engram.embed.scale",
                "layers.1.engram.wkv.weight",
            ],
            "model-00003.safetensors": ["layers.2.ffn.w1.weight"],
        }
        torch.manual_seed(0)
        self.files, self.expected = [], {}
        for file_name, names in layout.items():
            tensors = {name: torch.randn(4, 8) for name in names}
            path = os.path.join(folder.name, file_name)
            save_file(tensors, path)
            self.files.append(path)
            self.expected.update(tensors)
        self.engram_file = self.files[1]

        self.instanttensor_calls = []
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            patch.object(
                loader_mod,
                "instanttensor_weights_iterator",
                side_effect=self._fake_instanttensor,
            )
        )
        patches.enter_context(
            patch.object(
                loader_mod.current_platform, "get_device", return_value=self.DEVICE
            )
        )
        patches.enter_context(
            patch.object(loader_mod.torch.cuda, "current_device", return_value=0)
        )

    def _fake_instanttensor(self, files, **_kwargs):
        self.instanttensor_calls.append(list(files))
        for path in files:
            with safe_open(path, framework="pt") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name).to(self.DEVICE)

    def _load(self, source):
        model_loader = loader_mod.DefaultModelLoader(
            LoadConfig(load_format="instanttensor")
        )
        resolved = loader_mod.DefaultModelLoader.ResolvedSource(
            source=source,
            hf_folder=os.path.dirname(self.files[0]),
            weight_files=tuple(self.files),
            use_safetensors=True,
        )
        with get_context().override_server_args():
            return list(
                model_loader._get_weights_iterator(source, resolved_source=resolved)
            )

    def _dsv4_source(self, host_table: bool, model_cls=None):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        model_cls = model_cls or DeepseekV4ForCausalLM
        model = model_cls.__new__(model_cls)
        model_config = SimpleNamespace(model_path="model", revision=None)
        with envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(host_table):
            return loader_mod.DefaultModelLoader.Source.init_new(model_config, model)

    def _host_resident_names(self, result):
        """Check each tensor arrives once and return the names left on the host."""
        self.assertCountEqual([name for name, _ in result], self.expected)
        on_host = set()
        for name, tensor in result:
            expected = self.expected[name]
            self.assertEqual(
                (tensor.shape, tensor.dtype), (expected.shape, expected.dtype)
            )
            if tensor.device != self.DEVICE:
                self.assertEqual(tensor.device.type, "cpu")
                torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
                on_host.add(name)
        return on_host

    def test_engram_tables_stay_on_host(self):
        result = self._load(self._dsv4_source(host_table=True))

        self.assertEqual(self.instanttensor_calls, [[self.files[0], self.files[2]]])
        # The Engram file's other tensor is read on the host but placed on the
        # device like every InstantTensor tensor.
        self.assertEqual(
            self._host_resident_names(result),
            {"layers.1.engram.embed.weight", "layers.1.engram.embed.scale"},
        )

    def test_device_tables_keep_every_file_on_instanttensor(self):
        result = self._load(self._dsv4_source(host_table=False))

        self.assertEqual(self.instanttensor_calls, [self.files])
        self.assertEqual(self._host_resident_names(result), set())

    def test_only_host_resident_files_skip_instanttensor(self):
        self.files = [self.engram_file]
        self.expected = {k: v for k, v in self.expected.items() if ".engram." in k}
        result = self._load(self._dsv4_source(host_table=True))

        self.assertEqual(self.instanttensor_calls, [])
        self.assertEqual(len(self._host_resident_names(result)), 2)

    def test_non_safetensors_files_are_left_to_instanttensor(self):
        # InstantTensor reports unsupported files; the split must not open them.
        self.assertEqual(
            loader_mod._split_host_resident_files(
                ["model.bin", self.engram_file], (".engram.embed.",)
            ),
            (["model.bin"], [self.engram_file]),
        )

    def test_dspark_draft_declares_the_same_tables(self):
        from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark

        for enabled, expected in ((False, None), (True, (".engram.embed.",))):
            source = self._dsv4_source(enabled, DeepseekV4ForCausalLMDSpark)
            self.assertEqual(source.host_resident_weight_patterns, expected)


if __name__ == "__main__":
    unittest.main()
