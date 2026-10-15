# SPDX-License-Identifier: Apache-2.0
"""Unit tests for model-agnostic rowwise W8A8 FP8 quantization."""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.layers import linear
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.rowwise_fp8 import (
    RowwiseFp8Config,
    RowwiseFp8Error,
    build_rowwise_fp8_config_from_metadata,
)
from sglang.multimodal_gen.runtime.models import parameter
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    get_quant_config_from_safetensors_metadata,
)

FP8_E4M3_MAX = 448.0


def _config(fp8_layers=("proj",), input_scale_layers=()):
    return RowwiseFp8Config(
        format_version="1.1",
        fp8_layers=frozenset(fp8_layers),
        input_scale_layers=frozenset(input_scale_layers),
    )


def _metadata(*, version="1.1", fp8_format="rowwise-float8_e4m3fn", recipe="rowwise"):
    return {
        "format_version": version,
        "layers": {"a": {"format": fp8_format}, "b": {"format": "bf16"}},
        "quant_configs": {"rowwise-float8_e4m3fn": {"scale_recipe": recipe}},
    }


def _tensor_headers(**overrides):
    headers = {
        "a.weight": {"dtype": "F8_E4M3", "shape": [4, 8]},
        "a.weight_scale": {"dtype": "F32", "shape": [4]},
        "b.weight": {"dtype": "BF16", "shape": [4, 8]},
    }
    headers.update(overrides)
    return headers


@contextmanager
def _mock_tp(tp_size=1, tp_rank=0):
    group = object()
    with (
        mock.patch.object(linear, "get_tp_group", return_value=group),
        mock.patch.object(linear, "get_group_size", return_value=tp_size),
        mock.patch.object(linear, "get_group_rank", return_value=tp_rank),
        mock.patch.object(parameter, "get_tp_rank", return_value=tp_rank),
    ):
        yield


class TestRowwiseFp8Contract(unittest.TestCase):
    def test_safetensors_metadata_routes_to_rowwise_config(self):
        metadata = _metadata()
        metadata["layers"]["c"] = {"format": "rowwise-float8_e4m3fn"}
        tensors = {
            "a.input_scale": torch.ones(1),
            "b.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
        }
        for prefix in ("a", "c"):
            tensors[f"{prefix}.weight"] = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
            tensors[f"{prefix}.weight_scale"] = torch.ones(4)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "dit.safetensors")
            save_file(
                tensors, path, metadata={"_quantization_metadata": json.dumps(metadata)}
            )
            config = get_quant_config_from_safetensors_metadata(path)

        self.assertIsInstance(config, RowwiseFp8Config)
        self.assertEqual(config.fp8_layers, frozenset({"a", "c"}))
        self.assertTrue(config.has_input_scale("a"))
        self.assertFalse(config.has_input_scale("c"))

    def test_rejects_contract_violations(self):
        per_tensor_scale = {"a.weight_scale": {"dtype": "F32", "shape": [1]}}
        orphan_scale = {"b.weight_scale": {"dtype": "F32", "shape": [4]}}
        cases = (
            (_metadata(version="2.0"), _tensor_headers(), "format_version"),
            (_metadata(fp8_format="int4"), _tensor_headers(), "unknown quant format"),
            (_metadata(recipe="tensorwise"), _tensor_headers(), "scale_recipe"),
            (_metadata(), _tensor_headers(**per_tensor_scale), "must have shape"),
            (_metadata(), _tensor_headers(**orphan_scale), "non-FP8 layer"),
        )
        for metadata, headers, reason in cases:
            with (
                self.subTest(reason),
                self.assertRaisesRegex(RowwiseFp8Error, reason),
            ):
                build_rowwise_fp8_config_from_metadata(metadata, headers)


class TestRowwiseFp8Config(unittest.TestCase):
    def test_v2_loaders_shard_channel_scales(self):
        config = _config(
            fp8_layers=("column", "merged", "row"),
            input_scale_layers=("column",),
        )
        with _mock_tp(tp_size=2, tp_rank=1):
            column = ColumnParallelLinear(
                8, 12, bias=False, prefix="column", quant_config=config
            )
            merged = MergedColumnParallelLinear(
                8, [8, 12], bias=False, prefix="merged", quant_config=config
            )
            row = RowParallelLinear(
                12, 8, bias=False, prefix="row", quant_config=config
            )

            column.weight_scale.weight_loader(
                column.weight_scale, torch.arange(12, dtype=torch.float32)
            )
            column.input_scale.weight_loader(column.input_scale, torch.tensor([0.125]))
            merged.weight_scale.weight_loader(
                merged.weight_scale, torch.arange(20, dtype=torch.float32)
            )
            row.weight_scale.weight_loader(
                row.weight_scale, torch.arange(8, dtype=torch.float32)
            )

        torch.testing.assert_close(
            column.weight_scale, torch.arange(6, 12, dtype=torch.float32)
        )
        torch.testing.assert_close(column.input_scale, torch.tensor([0.125]))
        self.assertIsNone(merged.input_scale)
        torch.testing.assert_close(
            merged.weight_scale,
            torch.cat(
                (
                    torch.arange(4, 8, dtype=torch.float32),
                    torch.arange(14, 20, dtype=torch.float32),
                )
            ),
        )
        torch.testing.assert_close(
            row.weight_scale, torch.arange(8, dtype=torch.float32)
        )

    def test_per_tensor_serialized_fp8_keeps_a_scalar_weight_scale(self):
        with _mock_tp():
            layer = ColumnParallelLinear(
                8,
                8,
                bias=False,
                quant_config=Fp8Config(is_checkpoint_fp8_serialized=True),
            )
        self.assertEqual(tuple(layer.weight_scale.shape), (1,))
        self.assertFalse(layer.quant_method.per_channel_serialized)


def _fp8_capable():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 9 or (major == 8 and minor >= 9)


@unittest.skipUnless(_fp8_capable(), "requires FP8-capable CUDA GPU")
class TestFp8LinearMethodRowwise(unittest.TestCase):
    def test_apply_matches_scaled_mm_reference(self):
        torch.manual_seed(0)
        m, k, n = 8, 128, 64
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        weight_scale = (weight.float().abs().amax(dim=1) / FP8_E4M3_MAX).clamp(min=1e-8)
        # [K, N] view of [N, K] storage, as Fp8LinearMethod leaves it after loading.
        quantized_weight = (
            (weight.float() / weight_scale[:, None])
            .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
            .t()
        )
        dynamic_scale = (
            x.float().abs().amax(dim=1, keepdim=True) / FP8_E4M3_MAX
        ).clamp(min=1e-8)
        static_scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)

        for input_scale in (None, static_scale):
            with self.subTest(static=input_scale is not None):
                layer = SimpleNamespace(
                    weight=quantized_weight,
                    weight_scale=weight_scale,
                    input_scale=input_scale,
                )
                output = Fp8LinearMethod(_config()).apply(layer, x)

                scale = dynamic_scale if input_scale is None else input_scale
                quantized_input = (
                    (x.float() / scale)
                    .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
                    .to(torch.float8_e4m3fn)
                )
                reference = torch._scaled_mm(
                    quantized_input,
                    quantized_weight,
                    scale_a=scale.expand(m, 1).contiguous(),
                    scale_b=weight_scale.reshape(1, -1),
                    out_dtype=torch.bfloat16,
                )
                cosine = F.cosine_similarity(
                    output.float().flatten(), reference.float().flatten(), dim=0
                )
                self.assertGreater(cosine.item(), 0.999)


if __name__ == "__main__":
    unittest.main()
