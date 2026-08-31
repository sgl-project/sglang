"""Numerics for the FP8 dense-linear GEMM backends (--fp8-gemm-backend).

Real layer path vs a dequantized-reference matmul, in three formats: FP8
blockwise, MXFP8, and per-tensor FP8 (auto dispatch). Backend sets adapt to
the device SM, so one file covers SM90 / SM100 / SM120.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import (
    assert_output_close,
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

FP8_MAX = 448.0

# (M, N, K), N and K multiples of the (128, 128) weight block.
FP8_BLOCK_SHAPES = [
    (64, 512, 512),
    (5, 384, 896),
    (128, 1024, 1024),
]

# (M, N, K); K must be a multiple of 256 (flashinfer trtllm mxfp8 requirement).
MXFP8_SHAPES = [
    (64, 512, 512),
    (5, 384, 768),
]

# (M, N, K); per-tensor has no block-alignment constraints.
PER_TENSOR_SHAPES = [
    (64, 512, 512),
    (5, 384, 896),
]


def _fp8_block_backends():
    sm = get_device_sm()
    if 100 <= sm < 110:
        return ["triton", "deep_gemm", "flashinfer_trtllm", "flashinfer_cutlass"]
    if sm >= 120:
        # cutlass is the SM120-only explicit backend; the trtllm / deepgemm
        # kernels do not support consumer Blackwell.
        return ["triton", "cutlass"]
    if sm == 90:
        # flashinfer_deepgemm (swapAB) is SM90-only.
        return ["triton", "deep_gemm", "flashinfer_deepgemm"]
    return []


def _mxfp8_backends():
    # MXFP8 linear is validated on SM100/103 only.
    if get_device_sm() in (100, 103):
        return [
            "auto",
            "flashinfer_trtllm",
            "flashinfer_cutlass",
            "flashinfer_cutedsl",
        ]
    return []


def _quantize_fp8_blockwise(w: torch.Tensor, block: int = 128):
    """Per (block, block) tile fp8 quantization; returns checkpoint-format
    (w_fp8 [N, K], scale_inv fp32 [N/block, K/block]) and the dequant reference."""
    n, k = w.shape
    tiles = w.float().reshape(n // block, block, k // block, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / FP8_MAX
    w_fp8 = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    w_dequant = (w_fp8.float() * scale[:, None, :, None]).reshape(n, k)
    return w_fp8.reshape(n, k), scale, w_dequant


def _quantize_mxfp8(w: torch.Tensor, block: int = 32):
    """Per (1, block) group e8m0 quantization; returns checkpoint-format
    (w_fp8 [N, K], scale uint8 [N, K/block]) and the dequant reference."""
    n, k = w.shape
    groups = w.float().reshape(n, k // block, block)
    amax = groups.abs().amax(dim=-1).clamp(min=1e-12)
    exp = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(min=-127, max=127)
    scale = torch.pow(2.0, exp)
    w_fp8 = (groups / scale[..., None]).to(torch.float8_e4m3fn)
    w_dequant = (w_fp8.float() * scale[..., None]).reshape(n, k)
    scale_e8m0 = (exp + 127).to(torch.uint8)
    return w_fp8.reshape(n, k), scale_e8m0, w_dequant


def _make_linear(quant_config, n: int, k: int):
    return make_tp1_column_parallel_linear(
        quant_config, n, k, skip_block_quant_check=True
    )


class _LinearBackendCheck(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _check_backend(self, backend: str, allowed, shapes, build_layer):
        if backend not in allowed:
            self.skipTest(f"{backend} not in SM{get_device_sm()} backend set")
        torch.manual_seed(7)
        for m, n, k in shapes:
            with self.subTest(backend=backend, shape=(m, n, k)):
                with mock.patch.object(
                    fp8_utils,
                    "FP8_GEMM_RUNNER_BACKEND",
                    Fp8GemmRunnerBackend(backend),
                ):
                    layer, w_dequant = build_layer(n, k)
                    layer.quant_method.process_weights_after_loading(layer)

                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
                    out, _ = layer(x)

                    ref = x.float() @ w_dequant.T
                    # atol covers single-element UE8M0 scale-rounding outliers
                    # (deep_gemm); a wrong kernel/layout fails by orders more.
                    assert_output_close(self, out, ref, rtol=5e-2, atol=1e-1)


class TestModelOptFp8PrequantizedDispatch(unittest.TestCase):
    @staticmethod
    def _make_method():
        method = object.__new__(ModelOptFp8LinearMethod)
        method.use_marlin = False
        method.use_sm120_gemv = False
        method.cutlass_fp8_supported = True
        return method

    def test_three_item_tuple_preserves_output_dtype(self):
        method = self._make_method()
        scale = torch.ones(1)
        layer = mock.Mock(
            use_flashinfer_bmm=False,
            weight=torch.empty((8, 8), dtype=torch.float8_e4m3fn),
            weight_scale=torch.ones(1),
            input_scale=scale,
            orig_dtype=torch.float16,
        )
        qx = torch.empty((2, 8), dtype=torch.float8_e4m3fn)
        expected = torch.empty((2, 8), dtype=torch.float16)

        with mock.patch(
            "sglang.srt.layers.quantization.modelopt_quant.apply_fp8_linear",
            return_value=expected,
        ) as apply_fp8:
            actual = method.apply(layer, (qx, scale, torch.float16))

        self.assertIs(actual, expected)
        self.assertEqual(
            apply_fp8.call_args.kwargs["pre_quant_output_dtype"], torch.float16
        )

    def test_prequantized_m1_keeps_sm120_gemv_fast_path(self):
        method = self._make_method()
        method.use_sm120_gemv = True
        scale = torch.ones(1)
        # ModelOpt stores a [K, N] view whose transpose recovers contiguous [N, K].
        weight_nk = torch.empty((16, 512), dtype=torch.float8_e4m3fn)
        layer = mock.Mock(
            use_flashinfer_bmm=False,
            weight=weight_nk.t(),
            weight_scale=torch.ones(1),
            input_scale=scale,
            orig_dtype=torch.bfloat16,
            sm120_gemv_alpha=torch.ones(1),
        )
        qx = torch.empty((1, 512), dtype=torch.float8_e4m3fn)
        expected = torch.empty((1, 16), dtype=torch.bfloat16)

        with (
            mock.patch(
                "sglang.kernels.ops.gemm.sm120_fp8_gemv.use_sm120_fp8_gemv",
                return_value=True,
            ),
            mock.patch(
                "sglang.kernels.ops.gemm.sm120_fp8_gemv.sm120_fp8_gemv",
                return_value=expected,
            ) as gemv,
            mock.patch(
                "sglang.srt.layers.quantization.modelopt_quant.apply_fp8_linear",
                side_effect=AssertionError("generic FP8 GEMM should not run"),
            ),
        ):
            actual = method.apply(layer, (qx, scale, torch.bfloat16))

        self.assertIs(actual, expected)
        gemv.assert_called_once()
        self.assertIs(gemv.call_args.args[0], qx)
        self.assertEqual(gemv.call_args.args[1].data_ptr(), weight_nk.data_ptr())
        self.assertEqual(gemv.call_args.args[1].shape, weight_nk.shape)
        self.assertIs(gemv.call_args.args[2], layer.sm120_gemv_alpha)

    def test_prequantized_tuple_rejects_another_layers_scale(self):
        method = self._make_method()
        layer_scale = torch.ones(1)
        layer = mock.Mock(
            use_flashinfer_bmm=False,
            input_scale=layer_scale,
            orig_dtype=torch.bfloat16,
        )
        qx = torch.empty((2, 8), dtype=torch.float8_e4m3fn)

        with self.assertRaisesRegex(ValueError, "layer's input_scale"):
            method.apply(layer, (qx, layer_scale.clone(), torch.bfloat16))

    def test_prequantized_tuple_rejects_non_fp8_payload(self):
        method = self._make_method()
        scale = torch.ones(1)
        layer = mock.Mock(
            use_flashinfer_bmm=False,
            input_scale=scale,
            orig_dtype=torch.bfloat16,
        )

        with self.assertRaisesRegex(TypeError, "FP8 tensor"):
            method.apply(layer, (torch.empty((2, 8), dtype=torch.bfloat16), scale))

    def test_prequantized_tuple_rejects_e4m3fnuz_payload(self):
        method = self._make_method()
        scale = torch.ones(1)
        layer = mock.Mock(
            use_flashinfer_bmm=False,
            input_scale=scale,
            orig_dtype=torch.bfloat16,
        )
        qx = torch.empty((2, 8), dtype=torch.float8_e4m3fnuz)

        with self.assertRaisesRegex(TypeError, "E4M3FN"):
            method.apply(layer, (qx, scale))

    def test_prequantized_tuple_rejects_wrong_output_dtype(self):
        method = self._make_method()
        scale = torch.ones(1)
        layer = mock.Mock(
            use_flashinfer_bmm=False,
            input_scale=scale,
            orig_dtype=torch.bfloat16,
        )
        qx = torch.empty((2, 8), dtype=torch.float8_e4m3fn)

        with self.assertRaisesRegex(ValueError, "original dtype"):
            method.apply(layer, (qx, scale, torch.float16))

    def test_bare_fp8_input_is_not_a_prequantized_contract(self):
        method = self._make_method()
        layer = mock.Mock(
            use_flashinfer_bmm=False,
            input_scale=torch.ones(1),
            orig_dtype=torch.bfloat16,
        )
        for dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            with self.subTest(dtype=dtype):
                qx = torch.empty((2, 8), dtype=dtype)
                with self.assertRaisesRegex(TypeError, "tuple contract"):
                    method.apply(layer, qx)


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestFp8BlockwiseLinearBackends(_LinearBackendCheck):
    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        layer = _make_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_fp8, scale_inv, w_dequant = _quantize_fp8_blockwise(w)
        load_linear_weights(layer, weight=w_fp8, weight_scale_inv=scale_inv)
        return layer, w_dequant

    def _run(self, backend: str):
        self._check_backend(
            backend, _fp8_block_backends(), FP8_BLOCK_SHAPES, self._build_layer
        )

    def test_triton(self):
        self._run("triton")

    def test_deep_gemm(self):
        self._run("deep_gemm")

    def test_flashinfer_trtllm(self):
        self._run("flashinfer_trtllm")

    def test_flashinfer_cutlass(self):
        self._run("flashinfer_cutlass")

    def test_flashinfer_deepgemm(self):
        self._run("flashinfer_deepgemm")

    def test_cutlass(self):
        self._run("cutlass")


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestMxfp8LinearBackends(_LinearBackendCheck):
    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            use_mxfp8=True,
        )
        layer = _make_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_fp8, scale_e8m0, w_dequant = _quantize_mxfp8(w)
        load_linear_weights(layer, weight=w_fp8, weight_scale_inv=scale_e8m0)
        return layer, w_dequant

    def _run(self, backend: str):
        self._check_backend(backend, _mxfp8_backends(), MXFP8_SHAPES, self._build_layer)

    def test_flashinfer_trtllm(self):
        self._run("flashinfer_trtllm")

    def test_flashinfer_cutlass(self):
        self._run("flashinfer_cutlass")

    def test_flashinfer_cutedsl(self):
        self._run("flashinfer_cutedsl")

    def test_auto(self):
        if "auto" not in _mxfp8_backends():
            self.skipTest(f"auto not in SM{get_device_sm()} MXFP8 backend set")
        with mock.patch.object(
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend.AUTO,
        ):
            self.assertEqual(
                fp8_utils.resolve_mxfp8_dense_gemm_backend(),
                fp8_utils.Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL,
            )
        self._run("auto")

    @unittest.skipUnless(get_device_sm() >= 100, "Requires Blackwell FlashInfer")
    def test_auto_falls_back_when_cutedsl_is_unsupported(self):
        with (
            mock.patch.object(
                fp8_utils,
                "FP8_GEMM_RUNNER_BACKEND",
                Fp8GemmRunnerBackend.AUTO,
            ),
            mock.patch.object(fp8_utils, "get_device_sm", return_value=107),
            mock.patch.object(
                fp8_utils._raw_flashinfer_mm_mxfp8,
                "is_backend_supported",
                return_value=False,
            ) as is_backend_supported,
        ):
            self.assertEqual(
                fp8_utils.resolve_mxfp8_dense_gemm_backend(),
                fp8_utils.Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS,
            )
            is_backend_supported.assert_called_once_with("cute-dsl", 107)


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestModeloptFp8PerTensorLinear(_LinearBackendCheck):
    """Per-tensor FP8 (ModelOptFp8LinearMethod, static scales) on the auto
    dispatch path -- the checkpoint style of nvidia/*-FP8 models."""

    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True, packed_modules_mapping={}
        )
        layer = _make_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        scale = (w.float().abs().max() / FP8_MAX).clamp(min=1e-12)
        w_fp8 = (w.float() / scale).to(torch.float8_e4m3fn)
        # 0-dim scales exercise weight_loader_v2's scalar reshape branch.
        load_linear_weights(
            layer,
            weight=w_fp8,
            weight_scale=scale,
            input_scale=torch.tensor(1.0 / FP8_MAX, device="cuda"),
        )
        w_dequant = w_fp8.float() * scale
        return layer, w_dequant

    def test_auto(self):
        self._check_backend("auto", ["auto"], PER_TENSOR_SHAPES, self._build_layer)

    def test_prequantized_input_skips_static_quant(self):
        layer, _ = self._build_layer(512, 512)
        layer.quant_method.process_weights_after_loading(layer)
        # Exercise the generic scaled-mm/CUTLASS route here. The FlashInfer BMM
        # helper has its own dispatch test; separating them makes a failure
        # identify which consumer mishandled the pre-quantized activation.
        layer.use_flashinfer_bmm = False
        x = torch.randn((8, 512), device="cuda", dtype=torch.bfloat16) / 10
        qx, scale = fp8_utils.static_quant_fp8(x, layer.input_scale, repeat_scale=False)
        expected, _ = layer(x)

        with mock.patch.object(
            fp8_utils,
            "static_quant_fp8",
            side_effect=AssertionError("pre-quantized input was quantized again"),
        ):
            actual, _ = layer((qx, scale))

        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=1e-1)

    def test_prequantized_flashinfer_bmm_dispatch(self):
        qx = torch.zeros((8, 512), device="cuda", dtype=torch.float8_e4m3fn)
        weight = torch.zeros((512, 512), device="cuda", dtype=torch.float8_e4m3fn)
        input_scale = torch.ones(1, device="cuda")
        weight_scale = torch.ones(1, device="cuda")
        expected = torch.zeros((8, 512), device="cuda", dtype=torch.bfloat16)

        with (
            mock.patch.object(
                fp8_utils,
                "static_quant_fp8",
                side_effect=AssertionError("pre-quantized input was quantized again"),
            ),
            mock.patch.object(
                fp8_utils, "flashinfer_bmm_fp8", return_value=expected
            ) as bmm,
        ):
            actual = fp8_utils.apply_fp8_linear_bmm_flashinfer(
                qx, weight, weight_scale, input_scale
            )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(bmm.call_args.args[0].data_ptr(), qx.data_ptr())
        self.assertEqual(bmm.call_args.args[-1], torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
