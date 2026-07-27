import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

import sglang.srt.model_loader.loader as loader_mod
import sglang.srt.model_loader.weight_utils as weight_utils
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.models.deepseek_common import deepseek_weight_loader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _FakeModel:
    def eval(self):
        return self


class TestRunaiModelStreamerLoader(CustomTestCase):
    def _write_index(self, folder, weight_map):
        with open(
            os.path.join(folder, "model.safetensors.index.json"),
            "w",
            encoding="utf-8",
        ) as index_file:
            json.dump({"weight_map": weight_map}, index_file)

    def test_selects_only_requested_native_mtp_layer_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write_index(
                folder,
                {
                    "model.layers.0.weight": "model-00001.safetensors",
                    "mtp.0.decoder.weight": "mtp-00000.safetensors",
                    "mtp.1.decoder.weight": "mtp-00001.safetensors",
                    "mtp.shared_norm.weight": "mtp-shared.safetensors",
                },
            )
            files = [
                os.path.join(folder, filename)
                for filename in (
                    "model-00001.safetensors",
                    "mtp-00000.safetensors",
                    "mtp-00001.safetensors",
                    "mtp-shared.safetensors",
                )
            ]

            selected = loader_mod._select_runai_draft_weight_files(
                folder, folder, files, draft_model_idx=1
            )

        self.assertEqual(selected, [files[2], files[3]])

    def test_selects_hf_mtp_layer_and_common_mtp_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write_index(
                folder,
                {
                    "model.mtp.layers.0.decoder.weight": "mtp-0.safetensors",
                    "model.mtp.layers.1.decoder.weight": "mtp-1.safetensors",
                    "model.mtp.norm.weight": "mtp-common.safetensors",
                },
            )
            files = [
                os.path.join(folder, filename)
                for filename in (
                    "mtp-0.safetensors",
                    "mtp-1.safetensors",
                    "mtp-common.safetensors",
                )
            ]

            selected = loader_mod._select_runai_draft_weight_files(
                folder, folder, files, draft_model_idx=0
            )

        self.assertEqual(selected, [files[0], files[2]])

    def test_reads_object_storage_index_from_metadata_cache(self):
        with tempfile.TemporaryDirectory() as metadata_folder:
            self._write_index(
                metadata_folder,
                {"mtp.0.decoder.weight": "mtp.safetensors"},
            )
            files = [
                "s3://bucket/model/model.safetensors",
                "s3://bucket/model/mtp.safetensors",
            ]

            with patch(
                "sglang.srt.utils.runai_utils.ObjectStorageModel.get_path",
                return_value=metadata_folder,
            ):
                selected = loader_mod._select_runai_draft_weight_files(
                    "s3://bucket/model",
                    "s3://bucket/model",
                    files,
                    draft_model_idx=0,
                )

        self.assertEqual(selected, [files[1]])

    def test_falls_back_to_all_shards_for_unrecognized_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write_index(
                folder,
                {"model.layers.0.weight": "model.safetensors"},
            )

            selected = loader_mod._select_runai_draft_weight_files(
                folder,
                folder,
                [os.path.join(folder, "model.safetensors")],
                draft_model_idx=0,
            )

        self.assertIsNone(selected)

    def test_get_weights_iterator_passes_only_draft_shards_to_runai(self):
        runai_loader = loader_mod.RunaiModelStreamerLoader(
            LoadConfig(
                load_format=LoadFormat.RUNAI_STREAMER,
                model_loader_extra_config={},
                draft_model_idx=0,
            )
        )
        runai_loader.target_device_str = "cpu"
        runai_loader._is_distributed = False

        with tempfile.TemporaryDirectory() as folder:
            self._write_index(
                folder,
                {
                    "model.layers.0.weight": "model.safetensors",
                    "mtp.0.decoder.weight": "mtp.safetensors",
                },
            )
            files = [
                os.path.join(folder, "model.safetensors"),
                os.path.join(folder, "mtp.safetensors"),
            ]
            source = loader_mod.RunaiModelStreamerLoader.Source(
                model_or_path=folder,
                revision=None,
                model_config=cast(
                    ModelConfig,
                    SimpleNamespace(
                        is_draft_model=True,
                        hf_config=SimpleNamespace(architectures=[None]),
                    ),
                ),
            )

            with (
                patch.object(
                    runai_loader, "_prepare_weights", return_value=(folder, files)
                ),
                patch.object(
                    weight_utils,
                    "runai_safetensors_weights_iterator",
                    return_value=iter([("mtp.0.decoder.weight", torch.tensor([1]))]),
                ) as mock_iterator,
            ):
                list(runai_loader._get_weights_iterator(source))

        mock_iterator.assert_called_once_with([files[1]], False, "cpu")

    def test_passes_quant_config_to_model_init(self):
        quant_config = object()
        fake_model = _FakeModel()

        with (
            patch.object(
                loader_mod,
                "_get_quantization_config",
                return_value=quant_config,
            ),
            patch.object(loader_mod, "_initialize_model") as mock_initialize_model,
            patch.object(
                loader_mod.DefaultModelLoader,
                "load_weights_and_postprocess",
            ) as mock_load_weights,
        ):
            mock_initialize_model.return_value = fake_model
            runai_loader = loader_mod.RunaiModelStreamerLoader(
                LoadConfig(
                    load_format=LoadFormat.RUNAI_STREAMER,
                    model_loader_extra_config={},
                )
            )
            model_config = cast(
                ModelConfig,
                SimpleNamespace(dtype=torch.float16, modelopt_quant=False),
            )

            model = runai_loader.load_model(
                model_config=model_config,
                device_config=DeviceConfig("cpu"),
            )

        self.assertIs(model, fake_model)
        self.assertIs(mock_load_weights.call_args.args[0], fake_model)
        self.assertIs(mock_initialize_model.call_args.args[2], quant_config)

    def test_marks_streamer_tensors(self):
        source_tensor = torch.tensor([1], dtype=torch.int32)

        class FakeStreamer:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def stream_files(self, *_args, **_kwargs):
                self.files_to_tensors_metadata = {0: [object()]}

            def get_tensors(self):
                yield "weight", source_tensor

        with patch.dict(
            sys.modules,
            {"runai_model_streamer": SimpleNamespace(SafetensorsStreamer=FakeStreamer)},
        ):
            weights = list(
                weight_utils.runai_safetensors_weights_iterator(["model.safetensors"])
            )

        self.assertEqual(weights[0][0], "weight")
        self.assertTrue(getattr(weights[0][1], weight_utils.RUNAI_STREAMER_TENSOR_ATTR))

    def test_deepseek_clone_only_clones_marked_tensors(self):
        unmarked = torch.tensor([1], dtype=torch.int32)

        self.assertIs(
            deepseek_weight_loader._clone_if_runai_streamed_tensor(unmarked),
            unmarked,
        )

        marked = torch.tensor([1], dtype=torch.int32)
        setattr(marked, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)

        cloned = deepseek_weight_loader._clone_if_runai_streamed_tensor(marked)

        self.assertIsNot(cloned, marked)
        marked.fill_(2)
        self.assertEqual(cloned.item(), 1)

    def test_get_model_loader_uses_runai_for_prequantized_modelopt(self):
        load_config = LoadConfig(
            load_format=LoadFormat.RUNAI_STREAMER,
            model_loader_extra_config={},
        )
        model_config = cast(
            ModelConfig,
            SimpleNamespace(
                quantization="modelopt_fp4",
                modelopt_quant=False,
                _is_already_quantized=lambda: True,
            ),
        )

        model_loader = loader_mod.get_model_loader(load_config, model_config)

        self.assertIsInstance(model_loader, loader_mod.RunaiModelStreamerLoader)

    def test_get_model_loader_uses_remote_instance_for_prequantized_modelopt(self):
        load_config = LoadConfig(
            load_format=LoadFormat.REMOTE_INSTANCE,
            model_loader_extra_config={},
        )
        model_config = cast(
            ModelConfig,
            SimpleNamespace(
                quantization="modelopt_fp4",
                modelopt_quant=False,
                _is_already_quantized=lambda: True,
            ),
        )

        model_loader = loader_mod.get_model_loader(load_config, model_config)

        self.assertIsInstance(model_loader, loader_mod.RemoteInstanceModelLoader)


if __name__ == "__main__":
    unittest.main()
