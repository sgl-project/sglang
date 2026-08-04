"""Numerics for the NVFP4 dense-linear GEMM backends (--fp4-gemm-backend).

Runs ModelOptFp4LinearMethod end to end (create_weights ->
process_weights_after_loading -> apply) for each SM100 backend choice and
checks the output against a dequantized-reference matmul. This covers both
the per-backend weight preparation (padding / interleave / TRTLLM shuffle)
and the GEMM kernel dispatch.
"""

import unittest
from unittest import mock

import torch
from flashinfer import fp4_quantize

from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0

# (M, N, K). The second shape hits the padding paths: N=160 is not a multiple
# of 128 (TRTLLM shuffle pad) and K=336 is neither a multiple of 32 (CUTLASS
# K pad) nor K/16 a multiple of 4 (TRTLLM scale pad).
SHAPES = [
    (64, 256, 512),
    (5, 160, 336),
    (128, 1024, 1024),
]

BACKENDS = [
    "flashinfer_cutedsl",
    "flashinfer_cutlass",
    "flashinfer_cudnn",
    "flashinfer_trtllm",
]


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    # Crop the K-tile padding too: k // block_size scale columns, not k.
    return out[0:m, 0 : k // block_size]


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2).to(dtype=dtype)


def dequantize_nvfp4_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def _make_quantized_layer(n: int, k: int, device: str = "cuda"):
    """Build a linear layer holding NVFP4 checkpoint-format weights; returns
    (method, layer, w_dequant) with w_dequant the fp32 quant->dequant reference."""
    quant_config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        use_per_token_activation=False,
    )
    method = ModelOptFp4LinearMethod(quant_config)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=k,
        output_partition_sizes=[n],
        input_size=k,
        output_size=n,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer = layer.to(device)

    w = torch.randn((n, k), device=device, dtype=torch.bfloat16) / 10
    w_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w.abs().max().to(torch.float32)
    w_q, w_sf_swizzled = fp4_quantize(w, w_gs)
    w_sf_linear = convert_swizzled_to_linear(
        w_sf_swizzled.view(torch.float8_e4m3fn), n, k, 16
    )
    w_dequant = dequantize_nvfp4_to_dtype(
        w_q, w_sf_swizzled, w_gs, torch.float32, device
    )

    layer.weight.data.copy_(w_q)
    layer.weight_scale.data.copy_(w_sf_linear)
    layer.weight_scale_2.data.fill_(1.0 / w_gs)
    # Calibrated activation amax stand-in (inputs are randn/10).
    layer.input_scale.data.fill_(1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX))
    return method, layer, w_dequant


@unittest.skipIf(get_device_sm() < 100, "NVFP4 dense GEMM backends require SM100+")
class TestNvFp4LinearBackends(CustomTestCase):
    def _run_backend(self, backend: str):
        torch.manual_seed(7)
        for m, n, k in SHAPES:
            with self.subTest(backend=backend, shape=(m, n, k)):
                with mock.patch.object(
                    fp4_utils,
                    "FP4_GEMM_RUNNER_BACKEND",
                    Fp4GemmRunnerBackend(backend),
                ):
                    method, layer, w_dequant = _make_quantized_layer(n, k)
                    method.process_weights_after_loading(layer)

                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
                    out = method.apply(layer, x)

                    x_gs = layer.input_scale_inv.data.float()
                    x_q, x_sf = fp4_quantize(x, x_gs)
                    x_dequant = dequantize_nvfp4_to_dtype(
                        x_q, x_sf, x_gs, torch.float32, x.device
                    )
                    ref = x_dequant @ w_dequant.T

                    self.assertEqual(out.shape, (m, n))
                    cos = torch.nn.functional.cosine_similarity(
                        out.float().flatten(), ref.flatten(), dim=0
                    ).item()
                    self.assertGreater(cos, 0.99)
                    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)

    def test_flashinfer_cutedsl(self):
        self._run_backend("flashinfer_cutedsl")

    def test_flashinfer_cutlass(self):
        self._run_backend("flashinfer_cutlass")

    def test_flashinfer_cudnn(self):
        self._run_backend("flashinfer_cudnn")

    def test_flashinfer_trtllm(self):
        self._run_backend("flashinfer_trtllm")


if __name__ == "__main__":
    unittest.main()
