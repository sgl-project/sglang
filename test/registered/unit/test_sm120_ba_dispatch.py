"""The experimental BA path must not change default or unsupported dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.quantization import unquant
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, stage="stage-a")


class TestSm120BaDispatch(unittest.TestCase):
    def test_default_is_disabled(self):
        with unquant.envs.SGLANG_ENABLE_SM120_BA_GEMM.override(False):
            self.assertFalse(
                unquant.should_enable_sm120_ba_gemm(unquant.Bf16GemmBackend.AUTO)
            )

    def test_enable_guards(self):
        for cuda, enabled, backend, deterministic, invariant, capability, expected in (
            (True, True, "auto", False, False, (12, 0), True),
            (False, True, "auto", False, False, (12, 0), False),
            (True, False, "auto", False, False, (12, 0), False),
            (True, True, "torch", False, False, (12, 0), False),
            (True, True, "cutedsl", False, False, (12, 0), False),
            (True, True, "gemv", False, False, (12, 0), False),
            (True, True, "auto", True, False, (12, 0), False),
            (True, True, "auto", False, True, (12, 0), False),
            (True, True, "auto", False, False, (12, 1), False),
            (True, True, "auto", False, False, (10, 0), False),
            (True, True, "auto", False, False, (9, 0), False),
        ):
            with self.subTest(backend=backend, capability=capability, enabled=enabled):
                runtime = SimpleNamespace(
                    deterministic=SimpleNamespace(
                        enable_deterministic_inference=deterministic
                    )
                )
                with (
                    patch.object(unquant, "_is_cuda", cuda),
                    unquant.envs.SGLANG_ENABLE_SM120_BA_GEMM.override(enabled),
                    patch.object(unquant, "get_exec", return_value=runtime),
                    patch.object(
                        unquant,
                        "is_batch_invariant_mode_enabled",
                        return_value=invariant,
                    ),
                    patch("torch.cuda.get_device_capability", return_value=capability),
                ):
                    self.assertEqual(
                        unquant.should_enable_sm120_ba_gemm(
                            unquant.Bf16GemmBackend(backend)
                        ),
                        expected,
                    )

    def test_layer_and_mode_guards(self):
        for prefix, enabled, compiling, invariant, expected in (
            ("model.layers.0.linear_attn.in_proj_ba", True, False, False, True),
            ("in_proj_ba", True, False, False, True),
            ("model.in_proj_ba", False, False, False, False),
            ("model.in_proj_ba", True, True, False, False),
            ("model.in_proj_ba", True, False, True, False),
            ("model.other_in_proj_ba", True, False, False, False),
            ("model.in_proj_qkvz", True, False, False, False),
            ("", True, False, False, False),
        ):
            with self.subTest(prefix=prefix, enabled=enabled, compiling=compiling):
                x, w = torch.randn(4, 8), torch.randn(3, 8)
                layer = SimpleNamespace(prefix=prefix, weight=w)
                candidate = Mock(return_value=torch.zeros(4, 3))
                with (
                    patch.object(
                        unquant, "_sm120_ba_linear", candidate if enabled else None
                    ),
                    patch.object(unquant, "use_intel_amx_backend", return_value=False),
                    patch.object(unquant, "_use_aiter", False),
                    patch.object(
                        unquant,
                        "get_bf16_gemm_backend",
                        return_value=unquant.Bf16GemmBackend.AUTO,
                    ),
                    patch.object(
                        unquant,
                        "is_batch_invariant_mode_enabled",
                        return_value=invariant,
                    ),
                    patch("torch.compiler.is_compiling", return_value=compiling),
                ):
                    out = unquant.UnquantizedLinearMethod().apply(layer, x)
                self.assertEqual(candidate.call_count, int(expected))
                if expected:
                    candidate.assert_called_once_with(x, w, None)
                else:
                    torch.testing.assert_close(
                        out, torch.nn.functional.linear(x, w), rtol=0, atol=0
                    )

    def test_reinitialize_clears_previous_opt_in(self):
        runtime = SimpleNamespace(
            kernel=SimpleNamespace(bf16_gemm_backend="auto"),
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
        )
        with (
            patch.object(unquant, "get_exec", return_value=runtime),
            patch.object(
                unquant, "get_platform", return_value=SimpleNamespace(is_sm100=False)
            ),
            patch.object(unquant, "should_enable_bf16_splitk_gemm", return_value=False),
            patch.object(
                unquant, "should_enable_sm120_ba_gemm", return_value=True
            ) as enable,
            patch.object(unquant, "_sm120_ba_linear", None),
            patch.object(unquant, "_BF16_GEMM_BACKEND", None),
            patch.object(unquant, "_enable_bf16_splitk_gemm", False),
        ):
            unquant.initialize_bf16_gemm_config()
            self.assertIsNotNone(unquant._sm120_ba_linear)
            enable.return_value = False
            unquant.initialize_bf16_gemm_config()
            self.assertIsNone(unquant._sm120_ba_linear)


if __name__ == "__main__":
    unittest.main()
