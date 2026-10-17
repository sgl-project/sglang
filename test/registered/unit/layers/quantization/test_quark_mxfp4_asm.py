"""CPU-only tests for Quark MXFP4 AITER GEMM layout helpers and the ROCm
--fp4-gemm-backend choices (aiter / triton)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import argparse
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups.choices import FP4_GEMM_RUNNER_BACKEND_CHOICES
from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.fp4_utils import (
    Fp4GemmRunnerBackend,
    initialize_fp4_gemm_config,
    resolve_rocm_fp4_gemm_backend,
)
from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    _asm_fp4_scale_swizzle_supported,
    _swizzle_asm_fp4_weight_scale,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


def _resolve(backend, *, is_gfx95=True, legacy_use_aiter_asm=None):
    return resolve_rocm_fp4_gemm_backend(
        backend=backend,
        is_gfx95=is_gfx95,
        legacy_use_aiter_asm=legacy_use_aiter_asm,
    )


class TestFp4GemmBackendCli(CustomTestCase):
    def test_rocm_backends_parse(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for backend in ("aiter", "triton"):
            with self.subTest(backend=backend):
                args = parser.parse_args(
                    ["--model", "dummy", "--fp4-gemm-backend", backend]
                )
                server_args = ServerArgs.from_cli_args(args)
                self.assertEqual(server_args.fp4_gemm_runner_backend, backend)

    def test_every_cli_choice_is_a_runner_backend(self):
        for choice in FP4_GEMM_RUNNER_BACKEND_CHOICES:
            with self.subTest(choice=choice):
                self.assertEqual(Fp4GemmRunnerBackend(choice).value, choice)


class TestResolveRocmFp4GemmBackend(CustomTestCase):
    def test_auto_keeps_triton_default(self):
        """AITER is opt-in: auto must not preshuffle weights that the Quark
        tuple-input fusion paths (DeepSeek MLA) still read in Triton layout."""
        self.assertEqual(_resolve("auto"), "triton")
        self.assertEqual(_resolve("auto", is_gfx95=False), "triton")

    def test_explicit_backends(self):
        self.assertEqual(_resolve("aiter"), "aiter")
        self.assertEqual(_resolve("triton"), "triton")
        self.assertEqual(_resolve("triton", is_gfx95=False), "triton")

    def test_aiter_requires_gfx95(self):
        with self.assertRaisesRegex(ValueError, "gfx95"):
            _resolve("aiter", is_gfx95=False)

    def test_legacy_env_is_honored_under_auto(self):
        cases = [
            # (legacy value, is_gfx95, expected backend)
            (True, True, "aiter"),
            # The env var was a no-op off gfx95; it must not start raising.
            (True, False, "triton"),
            (False, True, "triton"),
        ]
        for legacy, is_gfx95, expected in cases:
            with self.subTest(legacy=legacy, is_gfx95=is_gfx95):
                with self.assertWarnsRegex(
                    DeprecationWarning, "SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM"
                ):
                    backend = _resolve(
                        "auto", is_gfx95=is_gfx95, legacy_use_aiter_asm=legacy
                    )
                self.assertEqual(backend, expected)

    def test_explicit_flag_overrides_legacy_env(self):
        with self.assertWarnsRegex(DeprecationWarning, "ignored"):
            backend = _resolve("triton", legacy_use_aiter_asm=True)
        self.assertEqual(backend, "triton")

    def test_no_warning_without_legacy_env(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            self.assertEqual(_resolve("aiter"), "aiter")


class TestInitializeFp4GemmConfig(CustomTestCase):
    def setUp(self):
        saved = fp4_utils.FP4_GEMM_RUNNER_BACKEND
        self.addCleanup(setattr, fp4_utils, "FP4_GEMM_RUNNER_BACKEND", saved)

    def _initialize(self, backend, *, is_hip, is_gfx95=True):
        exec_ctx = SimpleNamespace(
            kernel=SimpleNamespace(fp4_gemm_runner_backend=backend)
        )
        with (
            mock.patch.object(fp4_utils, "get_exec", return_value=exec_ctx),
            mock.patch.object(fp4_utils, "is_hip", return_value=is_hip),
            mock.patch.object(fp4_utils, "is_gfx95_supported", return_value=is_gfx95),
        ):
            initialize_fp4_gemm_config()
        return fp4_utils.get_fp4_gemm_runner_backend()

    def test_rocm_auto_defaults_to_triton(self):
        with envs.SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM.override(False):
            # Restored to the pre-test state when the override exits.
            envs.SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM.clear()
            backend = self._initialize("auto", is_hip=True)
        self.assertEqual(backend, Fp4GemmRunnerBackend.TRITON)

    def test_rocm_legacy_env_reaches_global_backend(self):
        with envs.SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM.override(True):
            with self.assertWarns(DeprecationWarning):
                backend = self._initialize("auto", is_hip=True)
        self.assertEqual(backend, Fp4GemmRunnerBackend.AITER)

    def test_rocm_only_backends_rejected_off_rocm(self):
        for backend in ("aiter", "triton"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "only supported on ROCm"):
                    self._initialize(backend, is_hip=False)


class TestQuarkMxfp4BackendSelection(CustomTestCase):
    def setUp(self):
        saved = fp4_utils.FP4_GEMM_RUNNER_BACKEND
        self.addCleanup(setattr, fp4_utils, "FP4_GEMM_RUNNER_BACKEND", saved)

    def _make_scheme(self, backend):
        fp4_utils.FP4_GEMM_RUNNER_BACKEND = backend
        with mock.patch.object(quark_w4a4_mxfp4, "_is_hip", True):
            return quark_w4a4_mxfp4.QuarkW4A4MXFP4(
                weight_quant_spec={}, input_quant_spec={}
            )

    def test_scheme_follows_fp4_gemm_backend(self):
        self.assertTrue(
            self._make_scheme(Fp4GemmRunnerBackend.AITER).use_aiter_fp4_gemm
        )
        for backend in (
            Fp4GemmRunnerBackend.TRITON,
            Fp4GemmRunnerBackend.AUTO,
            Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
        ):
            with self.subTest(backend=backend):
                self.assertFalse(self._make_scheme(backend).use_aiter_fp4_gemm)


class TestQuarkMxfp4AsmScaleLayout(CustomTestCase):
    def test_supported_shape(self):
        scale = torch.empty((32, 8), dtype=torch.uint8)
        self.assertTrue(_asm_fp4_scale_swizzle_supported(scale))

    def test_rejects_unsupported_shapes(self):
        self.assertFalse(
            _asm_fp4_scale_swizzle_supported(torch.empty((16, 8), dtype=torch.uint8))
        )
        self.assertFalse(
            _asm_fp4_scale_swizzle_supported(torch.empty((32, 4), dtype=torch.uint8))
        )
        self.assertFalse(
            _asm_fp4_scale_swizzle_supported(torch.empty((1, 32, 8), dtype=torch.uint8))
        )

    def test_swizzle_matches_aiter_tile_permutation(self):
        scale = torch.arange(32 * 8, dtype=torch.int32).view(32, 8)
        expected = (
            scale.view(1, 2, 16, 1, 2, 4, 1)
            .permute(0, 3, 5, 2, 4, 1, 6)
            .contiguous()
            .view(32, 8)
        )
        actual = _swizzle_asm_fp4_weight_scale(scale)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, scale.shape)
        self.assertTrue(actual.is_contiguous())


if __name__ == "__main__":
    unittest.main()
