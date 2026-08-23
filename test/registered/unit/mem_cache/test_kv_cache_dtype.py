"""Unit tests for srt/mem_cache/kv_cache_dtype - CPU only, no server, no model."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv_cache_dtype import (
    TORCH_DTYPE_TO_KV_CACHE_STR,
    configure_kv_cache_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _call(server_args_kv_cache_dtype="auto", model=None, **overrides):
    kwargs = dict(
        server_args_kv_cache_dtype=server_args_kv_cache_dtype,
        model=model,
        model_dtype=torch.bfloat16,
        is_draft_worker=False,
        is_dflash=False,
        speculative_draft_attention_backend="default",
    )
    kwargs.update(overrides)
    return configure_kv_cache_dtype(**kwargs)


class TestConfigureKvCacheDtype(CustomTestCase):
    def test_auto_without_quant_config_uses_model_dtype(self):
        resolved, dtype = _call()
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.bfloat16)

    def test_auto_with_fp8_quant_config(self):
        model = torch.nn.Module()
        # kv_cache_quant_algo set after init via SimpleNamespace-like attr
        model.quant_config = type("QC", (), {"kv_cache_quant_algo": "FP8"})()
        resolved, dtype = _call(model=model, model_dtype=torch.bfloat16)
        self.assertEqual(resolved, "fp8_e4m3")
        self.assertIs(dtype, torch.float8_e4m3fn)

    def test_auto_with_fp8_quant_config_on_hip(self):
        # On HIP, fp8_kernel.fp8_dtype itself is the fnuz variant; emulate the
        # whole platform, not just the _is_hip flag.
        model = torch.nn.Module()
        model.quant_config = type("QC", (), {"kv_cache_quant_algo": "fp8"})()
        with (
            patch("sglang.srt.mem_cache.kv_cache_dtype._is_hip", True),
            patch("sglang.srt.mem_cache.kv_cache_dtype.fp8_dtype", torch.float8_e4m3fnuz),
        ):
            resolved, dtype = _call(model=model, model_dtype=torch.bfloat16)
        self.assertEqual(resolved, "fp8_e4m3")
        self.assertIs(dtype, torch.float8_e4m3fnuz)

    def test_auto_with_non_fp8_quant_config_falls_back_to_model_dtype(self):
        model = torch.nn.Module()
        model.quant_config = type("QC", (), {"kv_cache_quant_algo": "QQQ"})()
        resolved, dtype = _call(model=model, model_dtype=torch.bfloat16)
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.bfloat16)

    def test_auto_with_quant_config_model_without_attr(self):
        # quant_config models may lack the attribute entirely
        model = torch.nn.Module()
        resolved, dtype = _call(model=model, model_dtype=torch.bfloat16)
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.bfloat16)

    def test_fp8_e4m3_cuda(self):
        # Explicit dtypes resolve to a dtype only; `resolved` stays None
        # (it is set only by the "auto" path, as the pool's resolve tag).
        resolved, dtype = _call(server_args_kv_cache_dtype="fp8_e4m3")
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.float8_e4m3fn)

    def test_fp8_e4m3_hip(self):
        with (
            patch("sglang.srt.mem_cache.kv_cache_dtype._is_hip", True),
            patch("sglang.srt.mem_cache.kv_cache_dtype.fp8_dtype", torch.float8_e4m3fnuz),
        ):
            resolved, dtype = _call(server_args_kv_cache_dtype="fp8_e4m3")
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.float8_e4m3fnuz)

    def test_fp8_e5m2_cuda(self):
        resolved, dtype = _call(server_args_kv_cache_dtype="fp8_e5m2")
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.float8_e5m2)

    def test_fp8_e5m2_hip(self):
        with (
            patch("sglang.srt.mem_cache.kv_cache_dtype._is_hip", True),
            patch("sglang.srt.mem_cache.kv_cache_dtype.fp8_dtype", torch.float8_e4m3fnuz),
        ):
            resolved, dtype = _call(server_args_kv_cache_dtype="fp8_e5m2")
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.float8_e4m3fnuz)

    def test_mxfp8(self):
        resolved, dtype = _call(server_args_kv_cache_dtype="mxfp8")
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.float8_e4m3fn)

    def test_bf16_spellings(self):
        for spelling in ("bf16", "bfloat16"):
            with self.subTest(spelling=spelling):
                resolved, dtype = _call(server_args_kv_cache_dtype=spelling)
                self.assertIsNone(resolved)
                self.assertIs(dtype, torch.bfloat16)

    def test_fp4_e2m1_deprecated(self):
        with self.assertRaisesRegex(
            ValueError, "fp4_e2m1 is deprecated"
        ):
            _call(server_args_kv_cache_dtype="fp4_e2m1")

    def test_unsupported_dtype_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported kv_cache_dtype"):
            _call(server_args_kv_cache_dtype="not-a-dtype")

    def test_nvfp4_error_when_dtype_unavailable(self):
        if hasattr(torch, "float4_e2m1fn_x2"):
            self.skipTest("this torch build has float4_e2m1fn_x2")
        with self.assertRaisesRegex(ValueError, "requires.*float4_e2m1fn_x2 support"):
            _call(server_args_kv_cache_dtype="nvfp4")

    def test_nvfp4_with_torch_support(self):
        fake = torch.bfloat16
        with patch("torch.float4_e2m1fn_x2", create=True, new=fake):
            resolved, dtype = _call(server_args_kv_cache_dtype="nvfp4")
        self.assertIsNone(resolved)
        self.assertIs(dtype, fake)

    def test_draft_worker_overrides_server_dtype(self):
        # Draft worker with an explicit speculative dtype wins over the server's.
        resolved, dtype = _call(
            server_args_kv_cache_dtype="fp8_e4m3",
            is_draft_worker=True,
            speculative_draft_kv_cache_dtype="bf16",
            model_dtype=torch.bfloat16,
        )
        self.assertEqual(resolved, "bf16")
        self.assertIs(dtype, torch.bfloat16)

    def test_draft_worker_auto_passthrough(self):
        # "auto" from the draft config keeps the model dtype and no resolved tag.
        resolved, dtype = _call(
            server_args_kv_cache_dtype="fp8_e4m3",
            is_draft_worker=True,
            speculative_draft_kv_cache_dtype="auto",
            model_dtype=torch.bfloat16,
        )
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.bfloat16)

    def test_dflash_fa4_overrides_quantized_kv(self):
        # fa4 draft attention cannot read quantized KV (needs K.dtype == Q.dtype).
        model = torch.nn.Module()
        model.quant_config = type("QC", (), {"kv_cache_quant_algo": "FP8"})()
        resolved, dtype = _call(
            model=model,
            model_dtype=torch.bfloat16,
            is_draft_worker=True,
            is_dflash=True,
            speculative_draft_attention_backend="fa4",
        )
        self.assertEqual(resolved, "auto")
        self.assertIs(dtype, torch.bfloat16)

    def test_dflash_fa4_keeps_bf16_kv(self):
        # No override when the KV dtype already matches the compute dtype.
        resolved, dtype = _call(
            model_dtype=torch.bfloat16,
            is_draft_worker=True,
            is_dflash=True,
            speculative_draft_attention_backend="fa4",
        )
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.bfloat16)

    def test_dflash_non_fa4_backend_keeps_quantized_kv(self):
        model = torch.nn.Module()
        model.quant_config = type("QC", (), {"kv_cache_quant_algo": "FP8"})()
        resolved, dtype = _call(
            model=model,
            model_dtype=torch.bfloat16,
            is_draft_worker=True,
            is_dflash=True,
            speculative_draft_attention_backend="default",
        )
        self.assertEqual(resolved, "fp8_e4m3")
        self.assertIs(dtype, torch.float8_e4m3fn)

    def test_dtype_to_str_mapping(self):
        self.assertEqual(TORCH_DTYPE_TO_KV_CACHE_STR[torch.float8_e4m3fn], "fp8_e4m3")
        self.assertEqual(TORCH_DTYPE_TO_KV_CACHE_STR[torch.float8_e4m3fnuz], "fp8_e4m3")
        self.assertEqual(TORCH_DTYPE_TO_KV_CACHE_STR[torch.float8_e5m2], "fp8_e5m2")
        self.assertEqual(TORCH_DTYPE_TO_KV_CACHE_STR[torch.bfloat16], "bf16")


if __name__ == "__main__":
    unittest.main()
