"""Unit tests for compressed-tensors KV cache scale loading — CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsKVCacheMethod,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.test.test_utils import CustomTestCase

_FP8_TENSOR_KV_SCHEME = {
    "type": "float",
    "num_bits": 8,
    "strategy": "tensor",
    "symmetric": True,
    "dynamic": False,
}

_NVFP4_TENSOR_KV_SCHEME = dict(_FP8_TENSOR_KV_SCHEME, num_bits=4)


def _config(kv_cache_scheme):
    cfg = {
        "format": "float-quantized",
        "quant_method": "compressed-tensors",
        "ignore": [],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
            }
        },
    }
    if kv_cache_scheme is not None:
        cfg["kv_cache_scheme"] = kv_cache_scheme
    return CompressedTensorsConfig.from_config(cfg)


def _attn():
    # __new__ is enough: get_quant_method only isinstance-checks the layer.
    return RadixAttention.__new__(RadixAttention)


class TestCompressedTensorsKVCacheMethod(CustomTestCase):
    def test_declared_scheme_gets_kv_cache_method(self):
        """A declared supported scheme must produce the KV cache method;
        without it the calibrated k_scale/v_scale have no parameters to
        load into and fp8 KV runs unscaled."""
        config = _config(_FP8_TENSOR_KV_SCHEME)
        method = config.get_quant_method(_attn(), "model.layers.0.attn")
        self.assertIsInstance(method, CompressedTensorsKVCacheMethod)

    def test_no_scheme_returns_none(self):
        config = _config(None)
        self.assertIsNone(config.get_quant_method(_attn(), "model.layers.0.attn"))

    def test_nvfp4_checkpoint_scales_survive_loading(self):
        config = _config(_NVFP4_TENSOR_KV_SCHEME)
        layer = torch.nn.Module()
        method = config.get_quant_method(_attn(), "model.layers.0.attn")
        self.assertIsInstance(method, CompressedTensorsKVCacheMethod)
        method.create_weights(layer)
        # The checkpoint serializer stores scalar global scales, independently
        # of the per-block E4M3 scales generated when the KV cache is written.
        params = {
            "model.layers.0.self_attn.attn.k_scale": layer.k_scale,
            "model.layers.0.self_attn.attn.v_scale": layer.v_scale,
        }
        for suffix, value in (("k_scale", 0.001), ("v_scale", 0.002)):
            name = maybe_remap_kv_scale_name(
                f"model.layers.0.self_attn.{suffix}", params
            )
            default_weight_loader(params[name], torch.tensor([value]))
        method.process_weights_after_loading(layer)
        self.assertAlmostEqual(layer.k_scale_float, 0.001)
        self.assertAlmostEqual(layer.v_scale_float, 0.002)
        self.assertAlmostEqual(layer.k_scale.item(), layer.k_scale_float)
        self.assertAlmostEqual(layer.v_scale.item(), layer.v_scale_float)

    def test_kv_cache_quant_algo_resolves_auto_dtype(self):
        """configure_kv_cache_dtype duck-types this field for --kv-cache-dtype
        auto: four-bit calibration must not select the FP8 pool."""
        self.assertEqual(_config(_FP8_TENSOR_KV_SCHEME).kv_cache_quant_algo, "FP8")
        self.assertEqual(_config(_NVFP4_TENSOR_KV_SCHEME).kv_cache_quant_algo, "NVFP4")
        self.assertIsNone(_config(None).kv_cache_quant_algo)
        self.assertIsNone(
            _config(dict(_FP8_TENSOR_KV_SCHEME, dynamic=True)).kv_cache_quant_algo
        )

    def test_unsupported_scheme_degrades_to_none(self):
        """Unsupported declared schemes must skip the method, not fail
        the boot: such checkpoints serve with an unquantized-scale cache."""
        for bad in (
            dict(_FP8_TENSOR_KV_SCHEME, type="int"),
            dict(_FP8_TENSOR_KV_SCHEME, strategy="channel"),
            dict(_FP8_TENSOR_KV_SCHEME, symmetric=False),
            dict(_FP8_TENSOR_KV_SCHEME, dynamic=True),
        ):
            self.assertIsNone(
                _config(bad).get_quant_method(_attn(), "model.layers.0.attn")
            )

    def test_unsupported_nvfp4_schemes_do_not_load_scalar_scales(self):
        for update in (
            {"type": "int"},
            {"num_bits": 3},
            {"strategy": "channel"},
            {"strategy": "head"},
            {"symmetric": False},
            {"dynamic": True},
        ):
            with self.subTest(update=update):
                config = _config(dict(_NVFP4_TENSOR_KV_SCHEME, **update))
                self.assertIsNone(config.kv_cache_quant_algo)
                self.assertIsNone(
                    config.get_quant_method(_attn(), "model.layers.0.attn")
                )

    def _resolve_dtype(self, dtype, **overrides):
        kwargs = dict(
            server_args_kv_cache_dtype=dtype,
            model=SimpleNamespace(quant_config=_config(_NVFP4_TENSOR_KV_SCHEME)),
            model_dtype=torch.bfloat16,
            is_draft_worker=False,
            is_dflash=False,
            speculative_draft_attention_backend="flashinfer",
        )
        kwargs.update(overrides)
        return configure_kv_cache_dtype(**kwargs)

    def test_nvfp4_calibration_requires_matched_cache_dtype(self):
        for dtype in ("auto", "bf16", "fp8_e4m3", "fp4_mx_block16"):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(
                ValueError, "Checkpoint NVFP4 KV calibration requires"
            ):
                self._resolve_dtype(dtype)
        self.assertEqual(
            self._resolve_dtype("nvfp4")[1], torch.float4_e2m1fn_x2
        )

    def test_nvfp4_calibration_checks_effective_draft_dtype(self):
        for draft_dtype in ("bf16", "fp4_mx_block16"):
            with self.subTest(draft_dtype=draft_dtype), self.assertRaisesRegex(
                ValueError, "NVFP4 KV calibration requires"
            ):
                self._resolve_dtype(
                    "nvfp4",
                    is_draft_worker=True,
                    speculative_draft_kv_cache_dtype=draft_dtype,
                )
        self.assertEqual(
            self._resolve_dtype(
                "bf16",
                is_draft_worker=True,
                speculative_draft_kv_cache_dtype="nvfp4",
            )[1],
            torch.float4_e2m1fn_x2,
        )

    def test_nvfp4_calibration_rejects_dflash_fa4_fallback(self):
        with self.assertRaisesRegex(ValueError, "NVFP4 KV calibration requires"):
            self._resolve_dtype(
                "nvfp4",
                is_draft_worker=True,
                is_dflash=True,
                speculative_draft_attention_backend="fa4",
            )

    def test_other_cache_calibration_keeps_existing_resolution(self):
        self.assertEqual(
            self._resolve_dtype(
                "auto", model=SimpleNamespace(quant_config=_config(_FP8_TENSOR_KV_SCHEME))
            )[1],
            torch.float8_e4m3fn,
        )
        self.assertEqual(
            self._resolve_dtype(
                "bf16", model=SimpleNamespace(quant_config=_config(None))
            )[1],
            torch.bfloat16,
        )


if __name__ == "__main__":
    unittest.main()
