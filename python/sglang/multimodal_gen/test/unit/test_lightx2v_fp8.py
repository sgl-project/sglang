# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.layers import linear
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.lightx2v_fp8 import (
    LightX2VFp8Config,
    LightX2VFp8LinearMethod,
    build_lightx2v_fp8_config,
)
from sglang.multimodal_gen.runtime.layers.quantization.rowwise_fp8 import (
    RowwiseFp8Config,
)
from sglang.multimodal_gen.runtime.loader import transformer_load_utils
from sglang.multimodal_gen.runtime.models import parameter

PREFIX = "transformer_blocks.0.img_mod.1"


def _checkpoint_tensors():
    return {
        "img_in.weight": torch.zeros(3072, 64, dtype=torch.bfloat16),
        f"{PREFIX}.weight": torch.zeros(4, 8, dtype=torch.float8_e4m3fn),
        f"{PREFIX}.weight_scale": torch.ones(4, 1, dtype=torch.bfloat16),
    }


class TestLightX2VFp8(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_detection_and_declared_format_precedence(self):
        path = str(self.tmp_path / "transformer.safetensors")
        save_file(_checkpoint_tensors(), path)
        (config, detected) = (
            transformer_load_utils._resolve_weight_override_quantization([path], {}, {})
        )
        assert detected
        assert isinstance(config, LightX2VFp8Config)
        self.assertFalse(config.supports_fsdp)
        assert config.fp8_layers == frozenset({PREFIX})
        assert isinstance(
            config.get_quant_method(ReplicatedLinear(8, 4), PREFIX[:-2]),
            LightX2VFp8LinearMethod,
        )
        assert isinstance(
            config.get_quant_method(ReplicatedLinear(64, 3072), "img_in"),
            UnquantizedLinearMethod,
        )
        tensors = _checkpoint_tensors()
        tensors[f"{PREFIX}.weight_scale"] = torch.ones(4, dtype=torch.float32)
        metadata = {
            "format_version": "1.1",
            "layers": {
                PREFIX: {"format": "rowwise-float8_e4m3fn"},
                "img_in": {"format": "bf16"},
            },
            "quant_configs": {"rowwise-float8_e4m3fn": {"scale_recipe": "rowwise"}},
        }
        declared_path = str(self.tmp_path / "declared.safetensors")
        save_file(
            tensors,
            declared_path,
            metadata={"_quantization_metadata": json.dumps(metadata)},
        )
        with mock.patch(
            "sglang.multimodal_gen.runtime.layers.quantization.lightx2v_fp8."
            "build_lightx2v_fp8_config"
        ) as fallback:
            resolved, detected = (
                transformer_load_utils._resolve_weight_override_quantization(
                    [declared_path], {}, {}
                )
            )
        self.assertTrue(detected)
        self.assertIs(type(resolved), RowwiseFp8Config)
        self.assertEqual(resolved.fp8_layers, frozenset({PREFIX}))
        fallback.assert_not_called()

    def test_rejects_aiter_preshuffled_gemm(self):
        config = LightX2VFp8Config(format_version="1.1", fp8_layers=frozenset({PREFIX}))
        with mock.patch(
            "sglang.multimodal_gen.runtime.layers.quantization.lightx2v_fp8."
            "use_aiter_bpreshuffle_gemm",
            return_value=True,
        ) as use_preshuffle:
            with self.assertRaisesRegex(ValueError, "SGLANG_USE_AITER=0"):
                ReplicatedLinear(8, 3072, quant_config=config, prefix=PREFIX[:-2])
            use_preshuffle.assert_called_once_with(3072)

    def test_bf16_checkpoint_skips_fp8_fallback(self):
        tensors = _checkpoint_tensors()
        del tensors[f"{PREFIX}.weight_scale"]
        tensors[f"{PREFIX}.weight"] = tensors[f"{PREFIX}.weight"].to(torch.bfloat16)
        path = str(self.tmp_path / "bf16.safetensors")
        save_file(tensors, path)
        with mock.patch(
            "sglang.multimodal_gen.runtime.layers.quantization.lightx2v_fp8."
            "build_lightx2v_fp8_config"
        ) as fallback:
            config, detected = (
                transformer_load_utils._resolve_weight_override_quantization(
                    [path], {}, {}
                )
            )
        self.assertIsNone(config)
        self.assertFalse(detected)
        fallback.assert_not_called()

    def test_rejects_invalid_scales(self):
        for invalid in ["shape", "dtype", "missing", "orphan", "input"]:
            with self.subTest(invalid=invalid):
                tensors = _checkpoint_tensors()
                scale_key = f"{PREFIX}.weight_scale"
                if invalid == "shape":
                    tensors[scale_key] = tensors[scale_key].squeeze(-1).contiguous()
                elif invalid == "dtype":
                    tensors[scale_key] = tensors[scale_key].float()
                elif invalid == "missing":
                    del tensors[scale_key]
                elif invalid == "orphan":
                    tensors["orphan.weight_scale"] = torch.ones(
                        1, 1, dtype=torch.bfloat16
                    )
                else:
                    tensors[f"{PREFIX}.input_scale"] = torch.ones(
                        1, dtype=torch.bfloat16
                    )
                path = str(self.tmp_path / "invalid.safetensors")
                save_file(tensors, path)
                with self.assertRaisesRegex(ValueError, "LightX2V"):
                    build_lightx2v_fp8_config([path])

    def test_unrelated_layout_is_not_detected(self):
        tensors = _checkpoint_tensors()
        tensors["img_in.weight"] = torch.zeros(16, 64, dtype=torch.bfloat16)
        path = str(self.tmp_path / "other.safetensors")
        save_file(tensors, path)
        assert build_lightx2v_fp8_config([path]) is None

    def test_scale_conversion_preserves_existing_linear_loaders(self):
        config = LightX2VFp8Config(format_version="1.1", fp8_layers=frozenset({PREFIX}))
        with (
            mock.patch.object(linear, "get_tp_group", return_value=object()),
            mock.patch.object(linear, "get_group_size", return_value=2),
            mock.patch.object(linear, "get_group_rank", return_value=1),
            mock.patch.object(parameter, "get_tp_rank", return_value=1),
        ):
            kwargs = dict(
                bias=False,
                quant_config=config,
                prefix=PREFIX[:-2],
                params_dtype=torch.bfloat16,
            )
            replicated = ReplicatedLinear(8, 8, **kwargs)
            column = ColumnParallelLinear(8, 12, **kwargs)
            merged = MergedColumnParallelLinear(8, [8, 12], **kwargs)
            row = RowParallelLinear(12, 8, **kwargs)
            for layer, rows, expected in (
                (replicated, 8, torch.arange(8)),
                (column, 12, torch.arange(6, 12)),
                (merged, 20, torch.cat((torch.arange(4, 8), torch.arange(14, 20)))),
                (row, 8, torch.arange(8)),
            ):
                scale = torch.arange(rows, dtype=torch.bfloat16).reshape(-1, 1)
                layer.weight_scale.weight_loader(layer.weight_scale, scale)
                assert layer.weight_scale.dtype == torch.float32
                torch.testing.assert_close(layer.weight_scale, expected.float())
            weight = torch.ones(8, 8, dtype=torch.float8_e4m3fn)
            replicated.weight.weight_loader(replicated.weight, weight)
            torch.testing.assert_close(replicated.weight.float(), weight.float())


if __name__ == "__main__":
    unittest.main()
