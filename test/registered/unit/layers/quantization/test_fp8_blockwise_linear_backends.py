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
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
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


if __name__ == "__main__":
    unittest.main()
