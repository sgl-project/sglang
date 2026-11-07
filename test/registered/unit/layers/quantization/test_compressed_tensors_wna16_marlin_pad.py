"""Regression test: compressed-tensors WNA16 with an N not divisible by 64.

Marlin repack and GEMM tile the output dimension by 64 columns.  Checkpoints
whose intermediate size is not a multiple of 64 (e.g. the 8608-wide
Gemma-4-31B AWQ checkpoint in sgl-project/sglang#43342) used to abort weight
loading in gptq_marlin_repack.  The scheme now zero-pads the last tile and
drops those columns after the GEMM.
"""

import unittest

import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    CompressedTensorsWNA16,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

PACK_FACTOR = 8
M = 4
K = 256
GROUP_SIZE = 128


def _pack_last_dim(q: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(
        (q.shape[0], q.shape[1] // PACK_FACTOR), dtype=torch.int32, device=q.device
    )
    for i in range(PACK_FACTOR):
        out |= q[:, i::PACK_FACTOR].to(torch.int32) << (4 * i)
    return out


def _quantize_awq(weight: torch.Tensor, group_size: int, symmetric: bool):
    n, k = weight.shape
    grouped = weight.reshape(n, k // group_size, group_size)
    if symmetric:
        w_absmax = grouped.abs().amax(-1, keepdim=True)
        scale = (w_absmax / 7).clamp(min=1e-8)
        zp = torch.full_like(scale, PACK_FACTOR)
    else:
        w_min = grouped.amin(-1, keepdim=True)
        w_max = grouped.amax(-1, keepdim=True)
        scale = ((w_max - w_min) / 15).clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, 15)
    q = (torch.round(grouped / scale) + zp).clamp(0, 15)
    dequant = ((q - zp) * scale).reshape(n, k)
    return q.reshape(n, k), scale.squeeze(-1), zp.squeeze(-1), dequant


class TestCompressedTensorsWNA16MarlinPad(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        major, _ = torch.cuda.get_device_capability(0)
        if major < 8:
            raise unittest.SkipTest("Marlin requires compute capability >= 8.0")

    def _run_case(self, size_n: int, symmetric: bool):
        torch.manual_seed(0)
        device = "cuda"
        weight = torch.randn((size_n, K), dtype=torch.float32, device=device)
        q, scale, zp, dequant = _quantize_awq(weight, GROUP_SIZE, symmetric)

        layer = torch.nn.Module()
        scheme = CompressedTensorsWNA16(
            strategy="group",
            num_bits=4,
            group_size=GROUP_SIZE,
            symmetric=symmetric,
            actorder=None,
        )
        scheme.create_weights(
            layer,
            output_size=size_n,
            input_size=K,
            output_partition_sizes=[size_n],
            input_size_per_partition=K,
            params_dtype=torch.float16,
            weight_loader=lambda *args, **kwargs: None,
        )
        layer.weight_packed.data = _pack_last_dim(q.to(torch.int32))
        layer.weight_scale.data = scale.to(torch.float16)
        if not symmetric:
            layer.weight_zero_point.data = _pack_last_dim(zp.to(torch.int32).t()).t()

        scheme.process_weights_after_loading(layer)

        expected_padded = -(-size_n // 64) * 64
        if expected_padded != size_n:
            self.assertEqual(layer.marlin_size_n_padded, expected_padded)
        else:
            self.assertIsNone(getattr(layer, "marlin_size_n_padded", None))

        x = torch.randn((M, K), dtype=torch.float16, device=device)
        bias = torch.randn((size_n,), dtype=torch.float16, device=device)
        out = scheme.apply_weights(layer, x, bias)
        ref = x.float() @ dequant.float().T + bias.float()

        self.assertEqual(tuple(out.shape), (M, size_n))
        torch.testing.assert_close(out.float(), ref, atol=0.2, rtol=0.02)

    def test_n_not_multiple_of_64(self):
        self._run_case(8608, symmetric=False)

    def test_n_not_multiple_of_64_symmetric(self):
        self._run_case(8608, symmetric=True)

    def test_n_multiple_of_64(self):
        self._run_case(8640, symmetric=False)


if __name__ == "__main__":
    unittest.main()
