"""Quark FP8 GDN packing: actual TP widths, scales, padding and dispatch."""

import os
import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")


class _FP8Linear(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
            QuarkW8A8Fp8,
        )

        self.weight = torch.nn.Parameter(
            torch.randn(rows, 4096, device="cuda").to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            torch.linspace(0.001, 0.1, rows, device="cuda", dtype=torch.float32),
            requires_grad=False,
        )
        self.scheme = QuarkW8A8Fp8(
            {"qscheme": "per_channel"},
            {"qscheme": "per_channel", "is_dynamic": True},
        )
        self.scheme.process_weights_after_loading(self)
        self.bias = None
        self.quant_method = object()

    def forward(self, x):
        return self.scheme.apply_weights(self, x), None


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "ROCm FP8 GEMM")
class TestQwen35GDNPackedFP8InProj(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_USE_AITER", "1")
        from sglang.srt.models import qwen3_5

        if not qwen3_5._use_aiter:
            raise unittest.SkipTest("SGLANG_USE_AITER was disabled at import")
        cls.module = qwen3_5
        cls.gdn_cls = qwen3_5.Qwen3_5GatedDeltaNet

    def make_gdn(self, tp, pp_size=1, model_type="qwen3_5_moe_text"):
        gdn = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type=model_type),
            in_proj_qkvz=_FP8Linear(20480 // tp),
            in_proj_ba=_FP8Linear(128 // tp),
            _fused_in_proj_weight=None,
            _fused_in_proj_scale=None,
            _fused_in_proj_sources=None,
            alt_stream=None,
            _fused_input_proj_cpu_enabled=types.SimpleNamespace(value=False),
        )
        for name in (
            "_finalize_fused_fp8_in_proj",
            "_fused_fp8_in_proj_sources_valid",
            "_forward_input_proj_fused_quant_amd",
        ):
            setattr(gdn, name, types.MethodType(getattr(self.gdn_cls, name), gdn))
        with (
            patch.object(
                self.module,
                "get_parallel",
                return_value=types.SimpleNamespace(pp_size=pp_size),
            ),
            patch.object(
                self.module,
                "get_lora",
                return_value=types.SimpleNamespace(enable_lora=False, lora_paths=None),
            ),
            patch.dict(os.environ, {"SGLANG_QWEN35_PACKED_IN_PROJ": "1"}),
        ):
            self.gdn_cls.finalize_fused_in_proj(gdn)
        return gdn

    def test_fp8_dispatch_matches_separate_projections(self):
        torch.manual_seed(42)
        for tp in (2, 4):
            gdn = self.make_gdn(tp)
            self.assertIsNotNone(gdn._fused_in_proj_scale)
            self.assertEqual(gdn._fused_in_proj_weight.shape[0] % 64, 0)
            self.assertTrue(gdn._fused_fp8_in_proj_sources_valid())
            for m in (1, 4, 33, 64, 65, 300):
                with self.subTest(tp=tp, tokens=m):
                    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
                    expected = (gdn.in_proj_qkvz(x)[0], gdn.in_proj_ba(x)[0])
                    for hidden in (x, (x, None, None)):
                        got = self.gdn_cls._forward_input_proj(gdn, hidden)
                        for actual, reference in zip(got, expected):
                            self.assertEqual(actual.shape, reference.shape)
                            error = (actual.float() - reference.float()).norm()
                            self.assertLess(
                                (error / reference.float().norm()).item(), 0.008
                            )
                    if m > 64:
                        self.assertTrue(
                            all(torch.equal(a, b) for a, b in zip(got, expected))
                        )

    def test_replaced_parameters_disable_cached_projection(self):
        gdn = self.make_gdn(4)
        gdn.in_proj_ba.weight = torch.nn.Parameter(
            gdn.in_proj_ba.weight.clone(), requires_grad=False
        )
        self.assertFalse(gdn._fused_fp8_in_proj_sources_valid())
        x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
        got = self.gdn_cls._forward_input_proj(gdn, x)
        expected = (gdn.in_proj_qkvz(x)[0], gdn.in_proj_ba(x)[0])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, expected)))

    def test_pipeline_parallel_keeps_original_projections(self):
        gdn = self.make_gdn(4, pp_size=2)
        self.assertIsNone(gdn._fused_in_proj_weight)

    def test_fp8_packing_is_limited_to_qwen35(self):
        for model_type, supported in (
            ("qwen3_5_text", True),
            ("qwen3_5_moe_text", True),
            ("qwen4_exp_text", False),
            ("other_text_model", False),
        ):
            with self.subTest(model_type=model_type):
                gdn = self.make_gdn(4, model_type=model_type)
                self.assertEqual(gdn._fused_in_proj_weight is not None, supported)
                self.assertEqual(gdn._fused_in_proj_scale is not None, supported)
                self.assertEqual(hasattr(gdn, "_derived_weight_cache_error"), supported)


if __name__ == "__main__":
    unittest.main()
