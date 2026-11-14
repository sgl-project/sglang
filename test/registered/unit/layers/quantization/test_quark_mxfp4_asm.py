"""CPU-only tests for Quark MXFP4 AITER GEMM layout helpers and the ROCm
--fp4-gemm-backend choices (aiter / triton)."""

from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")

import argparse
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

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
    _asm_fp4_prequantized_input,
    _asm_fp4_scale_swizzle_supported,
    _shuffle_asm_fp4_act_scale,
    _swizzle_asm_fp4_scale,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase


def _resolve(backend, *, mxfp4=True, legacy_use_aiter_asm=None):
    return resolve_rocm_fp4_gemm_backend(
        backend=backend,
        mxfp4_gemm_supported=mxfp4,
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
    def test_resolution(self):
        cases = [
            # (flag, native MXFP4, expected); AITER is opt-in, so auto is Triton.
            ("auto", True, "triton"),
            ("auto", False, "triton"),
            ("aiter", True, "aiter"),
            ("triton", True, "triton"),
            ("triton", False, "triton"),
        ]
        for backend, mxfp4, expected in cases:
            with self.subTest(backend=backend, mxfp4=mxfp4):
                self.assertEqual(_resolve(backend, mxfp4=mxfp4), expected)
        with self.assertRaisesRegex(ValueError, "native MXFP4"):
            _resolve("aiter", mxfp4=False)

    def test_legacy_env_is_honored_under_auto(self):
        cases = [
            # (legacy value, mxfp4, expected backend)
            (True, True, "aiter"),
            # The env var was a no-op without native MXFP4; it must not start raising.
            (True, False, "triton"),
            (False, True, "triton"),
        ]
        for legacy, mxfp4, expected in cases:
            with self.subTest(legacy=legacy, mxfp4=mxfp4):
                with self.assertWarnsRegex(
                    DeprecationWarning, "SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM"
                ):
                    backend = _resolve("auto", mxfp4=mxfp4, legacy_use_aiter_asm=legacy)
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

    def _initialize(self, backend, *, is_hip, mxfp4=True):
        exec_ctx = SimpleNamespace(
            kernel=SimpleNamespace(fp4_gemm_runner_backend=backend)
        )
        with (
            mock.patch.object(fp4_utils, "get_exec", return_value=exec_ctx),
            mock.patch.object(fp4_utils, "is_hip", return_value=is_hip),
            mock.patch.object(fp4_utils, "is_mxfp4_gemm_supported", return_value=mxfp4),
        ):
            initialize_fp4_gemm_config()
        return fp4_utils.get_fp4_gemm_runner_backend()

    def test_rocm_auto(self):
        for mxfp4, expected in (
            (True, Fp4GemmRunnerBackend.TRITON),
            (False, Fp4GemmRunnerBackend.TRITON),
        ):
            with self.subTest(mxfp4=mxfp4):
                with envs.SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM.override(False):
                    # Restored to the pre-test state when the override exits.
                    envs.SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM.clear()
                    backend = self._initialize("auto", is_hip=True, mxfp4=mxfp4)
                self.assertEqual(backend, expected)

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
            # AUTO is what get_fp4_gemm_runner_backend() returns before the
            # scheduler resolves the flag; it must not enable the AITER path.
            Fp4GemmRunnerBackend.AUTO,
            Fp4GemmRunnerBackend.TRITON,
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
        actual = _swizzle_asm_fp4_scale(scale)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, scale.shape)
        self.assertTrue(actual.is_contiguous())


class TestQuarkMxfp4AsmTupleInput(CustomTestCase):
    def test_act_scale_shuffle_matches_aiter_padded_layout(self):
        """Row-major (M, K/32) activation scales from the Triton fused-quant
        kernels must land in the (pad256(M), pad8(K/32)) tile order that
        per_1x32_f4_quant(shuffle=True) emits, with E8M0 1.0 in the padding."""
        rows, cols = 3, 6
        scale = torch.randint(0, 0x7F, (rows, cols), dtype=torch.uint8)

        shuffled = _shuffle_asm_fp4_act_scale(scale)

        self.assertEqual(shuffled.dtype, torch.float8_e8m0fnu)
        self.assertEqual(tuple(shuffled.shape), (256, 8))
        # Invert the AITER tile permutation (0, 3, 5, 2, 4, 1, 6).
        unshuffled = (
            shuffled.view(torch.uint8)
            .view(256 // 32, 8 // 8, 4, 16, 2, 2, 1)
            .permute(0, 5, 3, 1, 4, 2, 6)
            .reshape(256, 8)
        )
        expected = torch.full((256, 8), 0x7F, dtype=torch.uint8)
        expected[:rows, :cols] = scale
        torch.testing.assert_close(unshuffled, expected, rtol=0, atol=0)

    def test_quantized_pair_is_passed_through(self):
        rows, k = 4, 64
        x_q = torch.zeros((rows, k // 2), dtype=torch.uint8)
        x_scales = torch.full((rows, k // 32), 0x7F, dtype=torch.uint8)

        a, a_scales = _asm_fp4_prequantized_input(x_q, x_scales)
        self.assertEqual(a.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(a.data_ptr(), x_q.data_ptr())
        self.assertEqual(tuple(a_scales.shape), (256, 8))


class TestQuarkMxfp4AsmFusedEpilogueInputs(CustomTestCase):
    """The ASM path runs the Triton-only fused epilogues unfused, so every tuple
    input the Triton path accepts works on a layer that uses gemm_a4w4."""

    M, K, HEADS, S1, S2, ROPE = 3, 64, 2, 4, 6, 5

    def _apply(self, x):
        n = self.HEADS * (self.S1 + self.S2)
        layer = _make_mxfp4_layer(n, self.K)
        layer.use_aiter_asm_fp4_gemm = True
        self.gemm_out = torch.randn(self.M, n, dtype=torch.bfloat16)
        scheme = quark_w4a4_mxfp4.QuarkW4A4MXFP4.__new__(
            quark_w4a4_mxfp4.QuarkW4A4MXFP4
        )
        scheme.out_dtype = torch.bfloat16
        quantized = (torch.empty(self.M, self.K // 2), torch.empty(self.M, 1))
        quant = mock.Mock(return_value=quantized)
        with (
            mock.patch.object(
                quark_w4a4_mxfp4, "per_1x32_f4_quant", quant, create=True
            ),
            mock.patch.object(
                quark_w4a4_mxfp4,
                "_asm_fp4_prequantized_input",
                return_value=quantized,
            ) as prequant,
            mock.patch.object(
                quark_w4a4_mxfp4,
                "gemm_a4w4",
                return_value=self.gemm_out,
                create=True,
            ),
        ):
            out = scheme.apply_weights(layer, x)
        return out, quant, prequant

    def test_three_tuple_drops_the_preallocated_output(self):
        x = torch.randn(self.M, self.K, dtype=torch.bfloat16)
        unused = torch.zeros(self.M, self.HEADS * (self.S1 + self.S2))
        out, quant, prequant = self._apply((x, None, unused))
        quant.assert_called_once()
        self.assertIs(quant.call_args.args[0], x)
        prequant.assert_not_called()
        self.assertIs(out, self.gemm_out)

    def test_three_tuple_with_scales_uses_them(self):
        x_q = torch.zeros(self.M, self.K // 2, dtype=torch.uint8)
        x_s = torch.zeros(self.M, self.K // 32, dtype=torch.uint8)
        out, quant, prequant = self._apply((x_q, x_s, torch.empty(0)))
        quant.assert_not_called()
        prequant.assert_called_once_with(x_q, x_s)
        self.assertIs(out, self.gemm_out)

    def test_five_tuple_matches_split_cat_reference(self):
        x = torch.randn(self.M, self.K, dtype=torch.bfloat16)
        k_pe = torch.randn(self.M, 1, self.ROPE, dtype=torch.bfloat16).expand(
            -1, self.HEADS, -1
        )
        (k, v), _, _ = self._apply((x, k_pe, self.S1, self.S2, torch.float32))

        # The docstring reference of aiter's fused_gemm_afp4wfp4_split_cat; like
        # the Triton path, the outputs take k_pe's dtype.
        c = self.gemm_out.view(-1, self.HEADS, self.S1 + self.S2)
        c1, c2 = c.split([self.S1, self.S2], dim=-1)
        torch.testing.assert_close(k, torch.cat([c1, k_pe], dim=-1), rtol=0, atol=0)
        torch.testing.assert_close(v, c2, rtol=0, atol=0)
        self.assertEqual((k.dtype, v.dtype), (k_pe.dtype, k_pe.dtype))
        self.assertTrue(v.is_contiguous())


def _make_mxfp4_layer(n, k):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((n, k // 2), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros((n, k // 32), dtype=torch.uint8), requires_grad=False
    )
    return layer


class TestQuarkMxfp4AiterLayerSelection(CustomTestCase):
    N, K = 1024, 5120

    def setUp(self):
        saved = fp4_utils.FP4_GEMM_RUNNER_BACKEND
        self.addCleanup(setattr, fp4_utils, "FP4_GEMM_RUNNER_BACKEND", saved)

    def _process(self, backend, n=None):
        fp4_utils.FP4_GEMM_RUNNER_BACKEND = backend
        layer = _make_mxfp4_layer(n or self.N, self.K)
        self.loaded_weight = layer.weight
        with (
            mock.patch.object(quark_w4a4_mxfp4, "_is_hip", True),
            mock.patch.object(
                quark_w4a4_mxfp4,
                "shuffle_weight",
                side_effect=lambda w, layout: w.clone(),
                create=True,
            ),
        ):
            scheme = quark_w4a4_mxfp4.QuarkW4A4MXFP4({}, {})
            scheme.process_weights_after_loading(layer)
        return layer

    def test_only_explicit_aiter_uses_the_aiter_gemm(self):
        self.assertTrue(
            self._process(Fp4GemmRunnerBackend.AITER).use_aiter_asm_fp4_gemm
        )
        for backend in (Fp4GemmRunnerBackend.AUTO, Fp4GemmRunnerBackend.TRITON):
            with self.subTest(backend=backend):
                layer = self._process(backend)
                self.assertFalse(layer.use_aiter_asm_fp4_gemm)
                self.assertIs(layer.weight, self.loaded_weight)

    def test_narrow_layer_stays_on_triton(self):
        """With --fp4-gemm-backend aiter, a layer with N <= 128 (Qwen3.8 GDN
        in_proj_ba, N=96) must keep the Triton GEMM: AITER's untuned ASM
        heuristic returns wrong rows 1..M-1 for it at small batch sizes."""
        layer = self._process(Fp4GemmRunnerBackend.AITER, n=96)
        self.assertFalse(layer.use_aiter_asm_fp4_gemm)
        self.assertIs(layer.weight, self.loaded_weight)  # not shuffled for AITER


@unittest.skipUnless(is_hip(), "needs a ROCm build with aiter")
class TestQuarkMxfp4DynamicQuantFake(CustomTestCase):
    def test_fake_scale_matches_aiter_column_major_layout(self):
        """The fake must report aiter's column-major (1, M) scale strides;
        any other layout fails Inductor's assert_size_stride under torch.compile."""
        m, n = 37, 2080
        with FakeTensorMode():
            x = torch.empty((m, n), dtype=torch.bfloat16)
            x_fp4, scale = torch.ops.sglang.aiter_dynamic_mxfp4_quant(x)

        self.assertEqual(tuple(x_fp4.shape), (m, n // 2))
        self.assertEqual(tuple(scale.shape), (m, n // 32))
        self.assertEqual(scale.stride(), (1, m))


if __name__ == "__main__":
    unittest.main()
