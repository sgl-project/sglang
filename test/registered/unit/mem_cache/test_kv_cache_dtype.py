"""Unit tests for srt/mem_cache/kv_cache_dtype.py - no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.kv_cache_dtype import (
    TORCH_DTYPE_TO_KV_CACHE_STR,
    _is_hip,
    configure_kv_cache_dtype,
)
from sglang.test.test_utils import CustomTestCase

_FP8_E4M3 = fp8_dtype if _is_hip else torch.float8_e4m3fn
_FP8_E5M2 = fp8_dtype if _is_hip else torch.float8_e5m2
_MODEL_DTYPE = torch.bfloat16


def _make_quant_config(algo):
    return SimpleNamespace(kv_cache_quant_algo=algo)


def _make_model(quant_config):
    return SimpleNamespace(quant_config=quant_config)


def _call(**overrides):
    """Call configure_kv_cache_dtype with defaults overridable per test."""
    kwargs = dict(
        server_args_kv_cache_dtype="auto",
        model=None,
        model_dtype=_MODEL_DTYPE,
        is_draft_worker=False,
        is_dflash=False,
        speculative_draft_attention_backend="",
        speculative_draft_kv_cache_dtype=None,
    )
    kwargs.update(overrides)
    return configure_kv_cache_dtype(**kwargs)


class TestTorchDtypeToStrMapping(CustomTestCase):
    def test_mapping_covers_quantized_and_bf16_dtypes(self):
        self.assertEqual(
            TORCH_DTYPE_TO_KV_CACHE_STR,
            {
                torch.float8_e4m3fn: "fp8_e4m3",
                torch.float8_e4m3fnuz: "fp8_e4m3",
                torch.float8_e5m2: "fp8_e5m2",
                torch.bfloat16: "bf16",
            },
        )


class TestAutoDtypeResolution(CustomTestCase):
    def test_auto_with_fp8_quant_algo_uppercase(self):
        resolved, dtype = _call(model=_make_model(_make_quant_config("FP8")))
        self.assertEqual(resolved, "fp8_e4m3")
        self.assertEqual(dtype, _FP8_E4M3)

    def test_auto_with_fp8_quant_algo_lowercase(self):
        resolved, dtype = _call(model=_make_model(_make_quant_config("fp8")))
        self.assertEqual(resolved, "fp8_e4m3")
        self.assertEqual(dtype, _FP8_E4M3)

    def test_auto_with_non_fp8_quant_algo_falls_back_to_model_dtype(self):
        resolved, dtype = _call(model=_make_model(_make_quant_config("INT8")))
        self.assertIsNone(resolved)
        self.assertEqual(dtype, _MODEL_DTYPE)

    def test_auto_with_non_string_quant_algo_falls_back_to_model_dtype(self):
        model = _make_model(SimpleNamespace(kv_cache_quant_algo={"FP8": True}))
        resolved, dtype = _call(model=model)
        self.assertIsNone(resolved)
        self.assertEqual(dtype, _MODEL_DTYPE)

    def test_auto_without_quant_config_falls_back_to_model_dtype(self):
        resolved, dtype = _call(model=_make_model(None))
        self.assertIsNone(resolved)
        self.assertEqual(dtype, _MODEL_DTYPE)

    def test_auto_without_model_falls_back_to_model_dtype(self):
        resolved, dtype = _call(model=None)
        self.assertIsNone(resolved)
        self.assertEqual(dtype, _MODEL_DTYPE)


class TestExplicitDtypeResolution(CustomTestCase):
    def test_fp8_e5m2(self):
        resolved, dtype = _call(server_args_kv_cache_dtype="fp8_e5m2")
        self.assertIsNone(resolved)
        self.assertEqual(dtype, _FP8_E5M2)

    def test_fp8_e4m3(self):
        resolved, dtype = _call(server_args_kv_cache_dtype="fp8_e4m3")
        self.assertIsNone(resolved)
        self.assertEqual(dtype, _FP8_E4M3)

    def test_mxfp8(self):
        resolved, dtype = _call(server_args_kv_cache_dtype="mxfp8")
        self.assertIsNone(resolved)
        self.assertEqual(dtype, torch.float8_e4m3fn)

    def test_bf16(self):
        resolved, dtype = _call(server_args_kv_cache_dtype="bf16")
        self.assertIsNone(resolved)
        self.assertEqual(dtype, torch.bfloat16)

    def test_bfloat16_alias(self):
        resolved, dtype = _call(server_args_kv_cache_dtype="bfloat16")
        self.assertIsNone(resolved)
        self.assertEqual(dtype, torch.bfloat16)

    def test_fp4_e2m1_is_deprecated_and_raises(self):
        with self.assertRaisesRegex(ValueError, "deprecated"):
            _call(server_args_kv_cache_dtype="fp4_e2m1")

    def test_unsupported_dtype_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported kv_cache_dtype"):
            _call(server_args_kv_cache_dtype="foo")

    def test_nvfp4(self):
        if not hasattr(torch, "float4_e2m1fn_x2"):
            with self.assertRaises(ValueError):
                _call(server_args_kv_cache_dtype="nvfp4")
            return
        resolved, dtype = _call(server_args_kv_cache_dtype="nvfp4")
        self.assertIsNone(resolved)
        self.assertEqual(dtype, torch.float4_e2m1fn_x2)

    def test_fp4_mx_block16(self):
        if not hasattr(torch, "float4_e2m1fn_x2"):
            with self.assertRaises(ValueError):
                _call(server_args_kv_cache_dtype="fp4_mx_block16")
            return
        resolved, dtype = _call(server_args_kv_cache_dtype="fp4_mx_block16")
        self.assertIsNone(resolved)
        self.assertEqual(dtype, torch.float4_e2m1fn_x2)


class TestDraftWorkerDtype(CustomTestCase):
    def test_draft_dtype_overrides_server_args_dtype(self):
        resolved, dtype = _call(
            server_args_kv_cache_dtype="bf16",
            is_draft_worker=True,
            speculative_draft_kv_cache_dtype="fp8_e5m2",
        )
        self.assertEqual(resolved, "fp8_e5m2")
        self.assertEqual(dtype, _FP8_E5M2)

    def test_draft_dtype_overrides_auto_with_explicit(self):
        resolved, dtype = _call(
            server_args_kv_cache_dtype="auto",
            model=_make_model(None),
            is_draft_worker=True,
            speculative_draft_kv_cache_dtype="fp8_e4m3",
        )
        self.assertEqual(resolved, "fp8_e4m3")
        self.assertEqual(dtype, _FP8_E4M3)

    def test_draft_dtype_auto_keeps_auto_resolution(self):
        resolved, dtype = _call(
            server_args_kv_cache_dtype="bf16",
            model=_make_model(None),
            is_draft_worker=True,
            speculative_draft_kv_cache_dtype="auto",
        )
        self.assertIsNone(resolved)
        self.assertEqual(dtype, _MODEL_DTYPE)


class TestDflashFa4DtypeOverride(CustomTestCase):
    def test_fa4_overrides_quantized_dtype_with_model_dtype(self):
        resolved, dtype = _call(
            server_args_kv_cache_dtype="fp8_e4m3",
            model_dtype=_MODEL_DTYPE,
            is_draft_worker=True,
            is_dflash=True,
            speculative_draft_attention_backend="fa4",
        )
        self.assertEqual(resolved, "auto")
        self.assertEqual(dtype, _MODEL_DTYPE)

    def test_fa4_keeps_dtype_when_it_matches_model_dtype(self):
        resolved, dtype = _call(
            server_args_kv_cache_dtype="bf16",
            model_dtype=_MODEL_DTYPE,
            is_draft_worker=True,
            is_dflash=True,
            speculative_draft_attention_backend="fa4",
        )
        self.assertIsNone(resolved)
        self.assertEqual(dtype, _MODEL_DTYPE)

    def test_dflash_without_fa4_does_not_override(self):
        resolved, dtype = _call(
            server_args_kv_cache_dtype="fp8_e4m3",
            model_dtype=_MODEL_DTYPE,
            is_draft_worker=True,
            is_dflash=True,
            speculative_draft_attention_backend="fused",
        )
        self.assertIsNone(resolved)
        self.assertEqual(dtype, _FP8_E4M3)


if __name__ == "__main__":
    unittest.main()
