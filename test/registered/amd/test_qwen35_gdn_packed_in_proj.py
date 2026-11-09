"""Packed BF16 GDN input projection for Qwen3.5 decode on ROCm.

``finalize_fused_in_proj`` stacks ``in_proj_qkvz`` and ``in_proj_ba`` into one
weight and re-points the module weights at row views of it; on ROCm
``_forward_input_proj`` then runs one aiter GEMM for verify-sized batches and
splits the result. Guards: the views alias the packed buffer and keep their
values, and the packed and separate paths agree on both sides of the token gate.
"""

import os
import types
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

HIDDEN = 4096
# Qwen3.5-397B at TP4: 4 K heads x 128 + 16 V heads x 128.
QKVZ = 2 * 4 * 128 + 2 * 16 * 128
BA = 2 * 16
# bf16 carries ~8 mantissa bits, so one ULP is ~4e-3 relative.
TOL = 8e-3


class _Linear(torch.nn.Module):
    """Stand-in for the unquantized column-parallel projections."""

    def __init__(self, rows, device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02,
            requires_grad=False,
        )
        self.bias = None
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class _GDN:
    """Only the attributes Qwen3_5GatedDeltaNet's input projection touches."""

    def __init__(self, device):
        self.config = types.SimpleNamespace(model_type="qwen3_5_moe_text")
        self.in_proj_qkvz = _Linear(QKVZ, device)
        self.in_proj_ba = _Linear(BA, device)
        self._fused_in_proj_weight = None
        self._fused_in_proj_qkvz_width = None
        self._fused_in_proj_ba_width = 0
        self._fused_in_proj_scale = None
        self._fused_in_proj_sources = None
        self.alt_stream = None
        self._fused_input_proj_cpu_enabled = types.SimpleNamespace(value=False)


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "ROCm aiter GEMM")
class TestQwen35GDNPackedInProj(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_USE_AITER", "1")
        from sglang.srt.models import qwen3_5

        if not qwen3_5._use_aiter:
            raise unittest.SkipTest("qwen3_5 was imported without SGLANG_USE_AITER")
        lora = patch.object(
            qwen3_5,
            "get_lora",
            return_value=types.SimpleNamespace(enable_lora=False, lora_paths=None),
        )
        lora.start()
        cls.addClassCleanup(lora.stop)
        cls.module = qwen3_5
        cls.gdn_cls = qwen3_5.Qwen3_5GatedDeltaNet
        torch.manual_seed(0)
        cls.gdn = _GDN(torch.device("cuda", 0))
        for name in (
            "_finalize_fused_fp8_in_proj",
            "_fused_fp8_in_proj_sources_valid",
            "_forward_input_proj_fused_quant_amd",
        ):
            setattr(
                cls.gdn, name, types.MethodType(getattr(cls.gdn_cls, name), cls.gdn)
            )
        cls.pre = [
            m.weight.data.clone() for m in (cls.gdn.in_proj_qkvz, cls.gdn.in_proj_ba)
        ]
        cls.gdn_cls.finalize_fused_in_proj(cls.gdn)

    @staticmethod
    def _scope_gdn(model_type):
        def linear(rows):
            return types.SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.arange(rows * 4, dtype=torch.bfloat16).reshape(rows, 4),
                    requires_grad=False,
                ),
                bias=None,
                quant_method=UnquantizedLinearMethod(),
            )

        return types.SimpleNamespace(
            config=types.SimpleNamespace(model_type=model_type),
            in_proj_qkvz=linear(8),
            in_proj_ba=linear(4),
            _fused_in_proj_weight=None,
            _finalize_fused_fp8_in_proj=lambda: False,
        )

    def test_rocm_packing_is_limited_to_qwen35(self):
        for model_type, supported in (
            ("qwen3_5_text", True),
            ("qwen3_5_moe_text", True),
            ("qwen4_exp_text", False),
            ("other_text_model", False),
        ):
            with self.subTest(model_type=model_type):
                gdn = self._scope_gdn(model_type)
                original = (gdn.in_proj_qkvz.weight, gdn.in_proj_ba.weight)
                pointers = [weight.data_ptr() for weight in original]
                self.gdn_cls.finalize_fused_in_proj(gdn)
                self.assertEqual(gdn._fused_in_proj_weight is not None, supported)
                if not supported:
                    self.assertEqual(
                        [weight.data_ptr() for weight in original], pointers
                    )

    def test_cuda_packing_keeps_qwen4_support(self):
        gdn = self._scope_gdn("qwen4_exp_text")
        with (
            patch.object(self.module, "_is_cuda", True),
            patch.object(self.module, "_use_aiter", False),
        ):
            self.gdn_cls.finalize_fused_in_proj(gdn)
        self.assertIsNotNone(gdn._fused_in_proj_weight)
        self.assertEqual(
            gdn.in_proj_qkvz.weight.data_ptr(), gdn._fused_in_proj_weight.data_ptr()
        )

    def test_rocm_prepare_hook_is_limited_to_qwen35(self):
        for model_type, supported in (
            ("qwen3_5_text", True),
            ("qwen3_5_moe_text", True),
            ("qwen4_exp_text", False),
            ("other_text_model", False),
        ):
            with self.subTest(model_type=model_type):
                gdn = Mock(spec=self.gdn_cls)
                gdn._fused_in_proj_weight = None
                model = types.SimpleNamespace(
                    config=types.SimpleNamespace(model_type=model_type),
                    modules=Mock(return_value=[gdn]),
                    flashinfer_mnnvl_cutedsl_fusion=None,
                )
                self.module.Qwen3_5ForCausalLM.prepare_before_cuda_graph_capture(
                    model, None
                )
                self.assertEqual(model.modules.call_count, int(supported))
                self.assertEqual(gdn.finalize_fused_in_proj.call_count, int(supported))

    def test_packed_views_alias_and_preserve_values(self):
        gdn = self.gdn
        fused = gdn._fused_in_proj_weight
        self.assertIsNotNone(fused)
        self.assertEqual(tuple(fused.shape), (QKVZ + BA, HIDDEN))
        self.assertTrue(fused.is_contiguous())
        self.assertEqual(gdn._fused_in_proj_qkvz_width, QKVZ)
        self.assertEqual(gdn.in_proj_qkvz.weight.data_ptr(), fused.data_ptr())
        self.assertEqual(
            gdn.in_proj_ba.weight.data_ptr(),
            fused.data_ptr() + QKVZ * HIDDEN * fused.element_size(),
        )
        for got, want in zip((gdn.in_proj_qkvz, gdn.in_proj_ba), self.pre):
            self.assertTrue(torch.equal(got.weight.data, want))
        # finalize is idempotent: a second call keeps the packed buffer
        self.gdn_cls.finalize_fused_in_proj(gdn)
        self.assertIs(gdn._fused_in_proj_weight, fused)

    def test_packed_and_separate_paths_agree(self):
        gdn = self.gdn
        for tokens in (1, 4, 33, 64, 65, 300):
            with self.subTest(tokens=tokens):
                x = torch.randn(
                    tokens,
                    HIDDEN,
                    dtype=torch.bfloat16,
                    device=gdn._fused_in_proj_weight.device,
                )
                want_qkvz = torch.nn.functional.linear(x, self.pre[0])
                want_ba = torch.nn.functional.linear(x, self.pre[1])
                # the fused AR+RMSNorm path hands the bf16 side in a tuple
                for hidden in (x, (x, None, None)):
                    qkvz, ba = self.gdn_cls._forward_input_proj(gdn, hidden)
                    self.assertEqual(tuple(qkvz.shape), (tokens, QKVZ))
                    self.assertEqual(tuple(ba.shape), (tokens, BA))
                    for name, got, want in (
                        ("qkvz", qkvz, want_qkvz),
                        ("ba", ba, want_ba),
                    ):
                        scale = want.float().abs().max().clamp_min(1e-6)
                        rel = ((got.float() - want.float()).abs().max() / scale).item()
                        self.assertLess(
                            rel, TOL, f"{name} rel err {rel:.2e} at {tokens} tokens"
                        )


if __name__ == "__main__":
    unittest.main()
