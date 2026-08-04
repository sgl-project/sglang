"""Numerics for the FP8 blockwise / MXFP8 dense-linear GEMM backends
(--fp8-gemm-backend).

Runs Fp8LinearMethod end to end (create_weights ->
process_weights_after_loading -> apply) for each backend choice and checks
the output against a dequantized-reference matmul. This covers the
per-backend weight preparation (e.g. UE8M0 scale requant for DeepGEMM,
per-backend MXFP8 scale packing) and the GEMM kernel dispatch.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

FP8_MAX = 448.0

# (M, N, K), N and K multiples of the (128, 128) weight block.
FP8_BLOCK_SHAPES = [
    (64, 512, 512),
    (5, 384, 896),
    (128, 1024, 1024),
]

# SM100-capable backends; flashinfer_deepgemm is SM90-only and aiter is ROCm.
FP8_BLOCK_BACKENDS = [
    "triton",
    "deep_gemm",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
]

# (M, N, K); K must be a multiple of 256 (flashinfer trtllm mxfp8 requirement).
MXFP8_SHAPES = [
    (64, 512, 512),
    (5, 384, 768),
]

MXFP8_BACKENDS = [
    "triton",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
]


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


def _make_layer(quant_config: Fp8Config, n: int, k: int, device: str = "cuda"):
    method = Fp8LinearMethod(quant_config)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=k,
        output_partition_sizes=[n],
        input_size=k,
        output_size=n,
        params_dtype=torch.bfloat16,
        # The shape check reads TP world size (needs distributed init); skip it here.
        skip_block_quant_check=True,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer = layer.to(device)
    return method, layer


class _LinearBackendCheck(CustomTestCase):
    def _check_backend(self, backend: str, shapes, build_layer):
        torch.manual_seed(7)
        for m, n, k in shapes:
            with self.subTest(backend=backend, shape=(m, n, k)):
                with mock.patch.object(
                    fp8_utils,
                    "FP8_GEMM_RUNNER_BACKEND",
                    Fp8GemmRunnerBackend(backend),
                ):
                    method, layer, w_dequant = build_layer(n, k)
                    method.process_weights_after_loading(layer)

                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
                    out = method.apply(layer, x)

                    ref = x.float() @ w_dequant.T
                    self.assertEqual(out.shape, (m, n))
                    cos = torch.nn.functional.cosine_similarity(
                        out.float().flatten(), ref.flatten(), dim=0
                    ).item()
                    self.assertGreater(cos, 0.99)
                    # atol covers single-element UE8M0 scale-rounding outliers
                    # (deep_gemm); a wrong kernel/layout fails by orders more.
                    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=1e-1)


@unittest.skipIf(get_device_sm() < 100, "targets the SM100 backend set")
class TestFp8BlockwiseLinearBackends(_LinearBackendCheck):
    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        method, layer = _make_layer(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_fp8, scale_inv, w_dequant = _quantize_fp8_blockwise(w)
        layer.weight.data.copy_(w_fp8)
        layer.weight_scale_inv.data.copy_(scale_inv)
        return method, layer, w_dequant

    def test_triton(self):
        self._check_backend("triton", FP8_BLOCK_SHAPES, self._build_layer)

    def test_deep_gemm(self):
        self._check_backend("deep_gemm", FP8_BLOCK_SHAPES, self._build_layer)

    def test_flashinfer_trtllm(self):
        self._check_backend("flashinfer_trtllm", FP8_BLOCK_SHAPES, self._build_layer)

    def test_flashinfer_cutlass(self):
        self._check_backend("flashinfer_cutlass", FP8_BLOCK_SHAPES, self._build_layer)


@unittest.skipIf(get_device_sm() < 100, "targets the SM100 backend set")
class TestMxfp8LinearBackends(_LinearBackendCheck):
    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            use_mxfp8=True,
        )
        method, layer = _make_layer(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_fp8, scale_e8m0, w_dequant = _quantize_mxfp8(w)
        layer.weight.data.copy_(w_fp8)
        layer.weight_scale_inv.data.copy_(scale_e8m0)
        return method, layer, w_dequant

    def test_triton(self):
        self._check_backend("triton", MXFP8_SHAPES, self._build_layer)

    def test_flashinfer_trtllm(self):
        self._check_backend("flashinfer_trtllm", MXFP8_SHAPES, self._build_layer)

    def test_flashinfer_cutlass(self):
        self._check_backend("flashinfer_cutlass", MXFP8_SHAPES, self._build_layer)


if __name__ == "__main__":
    unittest.main()
