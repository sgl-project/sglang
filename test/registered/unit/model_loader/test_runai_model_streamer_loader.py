import sys
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
from sglang.srt.models.deepseek_v4 import (
    _dequant_fp8_wo_a,
    _dequant_fp8_wo_a_streaming,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _FakeModel:
    def eval(self):
        return self


class TestRunaiModelStreamerLoader(CustomTestCase):
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

    def test_deepseek_v4_streaming_dequant_fp8_wo_a_pairs_weight_and_scale(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)

        for weights in (
            [
                ("layers.0.attn.wo_a.scale", scale),
                ("layers.0.attn.wo_a.weight", weight),
                ("layers.0.attn.wq.weight", torch.tensor([3])),
            ],
            [
                ("layers.0.attn.wo_a.weight", weight),
                ("layers.0.attn.wq.weight", torch.tensor([3])),
                ("layers.0.attn.wo_a.scale", scale),
            ],
        ):
            converted = list(_dequant_fp8_wo_a_streaming(weights))

            converted_names = [name for name, _ in converted]
            self.assertIn("layers.0.attn.wo_a.weight", converted_names)
            self.assertNotIn("layers.0.attn.wo_a.scale", converted_names)
            converted_weight = dict(converted)["layers.0.attn.wo_a.weight"]
            self.assertEqual(converted_weight.dtype, torch.bfloat16)

    def test_deepseek_v4_streaming_dequant_matches_legacy_by_name(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)
        ordinary = torch.tensor([3])
        weights = [
            ("layers.0.attn.wo_a.weight", weight),
            ("layers.0.attn.wq.weight", ordinary),
            ("layers.0.attn.wo_a.scale", scale),
        ]

        legacy = list(_dequant_fp8_wo_a(weights))
        streaming = list(_dequant_fp8_wo_a_streaming(weights))

        self.assertNotEqual(
            [name for name, _ in legacy], [name for name, _ in streaming]
        )
        self.assertEqual(set(dict(legacy)), set(dict(streaming)))
        for name, legacy_tensor in dict(legacy).items():
            torch.testing.assert_close(legacy_tensor, dict(streaming)[name])

    def test_deepseek_v4_streaming_dequant_clones_pending_runai_tensors(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)
        setattr(scale, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)

        def weights():
            yield "layers.0.attn.wo_a.scale", scale
            scale.fill_(0)
            yield "layers.0.attn.wo_a.weight", weight

        converted = dict(_dequant_fp8_wo_a_streaming(weights()))

        converted_weight = converted["layers.0.attn.wo_a.weight"]
        self.assertGreater(converted_weight.abs().sum().item(), 0)

    def test_deepseek_v4_streaming_dequant_preserves_missing_scale_behavior(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        ordinary = torch.tensor([3])

        converted = dict(
            _dequant_fp8_wo_a_streaming(
                [
                    ("layers.0.attn.wo_a.weight", weight),
                    ("layers.0.attn.wq.weight", ordinary),
                ]
            )
        )

        self.assertIs(converted["layers.0.attn.wo_a.weight"], weight)
        self.assertIs(converted["layers.0.attn.wq.weight"], ordinary)

        with self.assertRaises(AssertionError):
            list(
                _dequant_fp8_wo_a_streaming(
                    [
                        ("layers.0.attn.wo_a.weight", weight),
                        ("layers.1.attn.wo_a.scale", torch.ones((1, 1))),
                    ]
                )
            )

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
