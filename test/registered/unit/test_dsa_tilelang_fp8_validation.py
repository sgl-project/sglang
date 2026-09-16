"""Validate the CUDA TileLang group-scaled NoPE capability and its boundaries.

Regression: the combination used to boot the server and crash at decode
CUDA-graph capture with ``kernel main input KV dtype expected bfloat16,
but got float8_e4m3fn``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import (
    ResolvedView,
    _check_dsa_backend_constraints,
    _check_tilelang_dsa_fp8_kv,
    _dsa_split_backend_resolution,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDsaTilelangFp8Validation(CustomTestCase):
    def test_cuda_group_scaled_nope_allowed(self):
        _check_dsa_backend_constraints(
            "fp8_e4m3", "tilelang", "tilelang", hip=False, nope_group_scaled=True
        )

    def test_cuda_group_scaled_requires_matching_consumers(self):
        for prefill, decode in (
            ("tilelang", "trtllm"),
            ("flashmla_kv", "tilelang"),
            ("tilelang", "fa3"),
            ("tilelang", None),
        ):
            with self.subTest(prefill=prefill, decode=decode):
                with self.assertRaises(ValueError):
                    _check_dsa_backend_constraints(
                        "fp8_e4m3", prefill, decode, hip=False, nope_group_scaled=True
                    )

    def test_cuda_group_scaled_dcp_rejected(self):
        with self.assertRaisesRegex(ValueError, "dcp-size 1"):
            _check_dsa_backend_constraints(
                "fp8_e4m3",
                "tilelang",
                "tilelang",
                hip=False,
                nope_group_scaled=True,
                dcp_size=2,
            )

    def test_cuda_triton_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "only supported on ROCm/HIP"):
            _check_dsa_backend_constraints(
                "fp8_e4m3", "triton", "tilelang", hip=False, nope_group_scaled=True
            )

    def test_production_resolution_checks_model_geometry(self):
        def make_view(rank=512, rope=0, dtype=torch.bfloat16, **kwargs):
            model = SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Glm5NextForConditionalGeneration"]
                ),
                kv_lora_rank=rank,
                qk_rope_head_dim=rope,
                dtype=dtype,
            )
            options = dict(
                _model_config=model,
                kv_cache_dtype="fp8_e4m3",
                dsa_prefill_backend="tilelang",
                dsa_decode_backend="tilelang",
                enable_hisparse=False,
                dcp_size=1,
            )
            options.update(kwargs)
            return ResolvedView(SimpleNamespace(**options))

        with (
            patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            override_platform(is_npu=False, is_xpu=False, is_hip=False),
        ):
            self.assertEqual(_dsa_split_backend_resolution(make_view()), {})
            for options in (
                {"rank": 256},
                {"rope": 64},
                {"dtype": torch.float16},
                {"dcp_size": 2},
                {"enable_hisparse": True},
            ):
                with self.subTest(options=options):
                    with self.assertRaises(ValueError):
                        _dsa_split_backend_resolution(make_view(**options))

    def test_cuda_fp8_tilelang_decode_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "tilelang", hip=False)

    def test_cuda_fp8_tilelang_prefill_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "trtllm", hip=False)

    def test_hip_fp8_tilelang_allowed(self):
        # ROCm has a real fp8 tilelang kernel
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=True)

    def test_bf16_tilelang_allowed(self):
        # what the CUDA kernel expects
        _check_tilelang_dsa_fp8_kv("bfloat16", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_non_tilelang_allowed(self):
        # fp8-capable backends must pass
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "trtllm", hip=False)


if __name__ == "__main__":
    unittest.main()
