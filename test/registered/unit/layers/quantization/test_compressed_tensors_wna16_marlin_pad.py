"""Regression test: compressed-tensors WNA16 with an N not divisible by 64.

Marlin repack and GEMM tile the output dimension by 64 columns.  Checkpoints
whose intermediate size is not a multiple of 64 (e.g. the 8608-wide
Gemma-4-31B AWQ checkpoint in sgl-project/sglang#43342) used to abort weight
loading in gptq_marlin_repack.  The scheme now zero-pads the last tile and
drops those columns after the GEMM.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.activation import SiluAndMul
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

    def _run_case(
        self,
        size_n: int,
        symmetric: bool,
        size_k: int = K,
        group_size: int = GROUP_SIZE,
        with_bias: bool = True,
    ) -> torch.Tensor:
        torch.manual_seed(0)
        device = "cuda"
        weight = torch.randn((size_n, size_k), dtype=torch.float32, device=device)
        q, scale, zp, dequant = _quantize_awq(weight, group_size, symmetric)

        layer = torch.nn.Module()
        scheme = CompressedTensorsWNA16(
            strategy="group",
            num_bits=4,
            group_size=group_size,
            symmetric=symmetric,
            actorder=None,
        )
        scheme.create_weights(
            layer,
            output_size=size_n,
            input_size=size_k,
            output_partition_sizes=[size_n],
            input_size_per_partition=size_k,
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

        x = torch.randn((M, size_k), dtype=torch.float16, device=device)
        bias = (
            torch.randn((size_n,), dtype=torch.float16, device=device)
            if with_bias
            else None
        )
        out = scheme.apply_weights(layer, x, bias)
        ref = x.float() @ dequant.float().T
        if bias is not None:
            ref = ref + bias.float()

        self.assertEqual(tuple(out.shape), (M, size_n))
        torch.testing.assert_close(out.float(), ref, atol=0.2, rtol=0.02)
        return out

    def test_n_not_multiple_of_64(self):
        self._run_case(8608, symmetric=False)

    def test_n_not_multiple_of_64_symmetric(self):
        self._run_case(8608, symmetric=True)

    def test_n_multiple_of_64(self):
        self._run_case(8640, symmetric=False)

    def test_bias_free_gate_up_activation(self):
        # A fused gate/up projection is consumed by SiluAndMul, which requires
        # contiguous input.  8624 is not a multiple of 64, so this 2 * 8624 wide
        # output is padded and narrowed again, and the narrowed view used to come
        # back non-contiguous.

        size_n = 2 * 8624
        out = self._run_case(size_n, symmetric=True, with_bias=False)
        self.assertTrue(out.is_contiguous())

        d = size_n // 2
        expected = F.silu(out[:, :d]) * out[:, d:]
        torch.testing.assert_close(SiluAndMul()(out), expected)

    def test_k_aligned_to_64_loads(self):
        # K = 448 only satisfies the 64-wide Marlin K tile; N = 896 is
        # 128-aligned, so the (thread_k, thread_n) = (64, 128) config covers it.
        self._run_case(896, symmetric=True, size_k=448, group_size=64)

    def test_shape_without_thread_config_rejected(self):
        # K = 448 needs a 128-aligned N, and 832 is only 64-aligned.
        with self.assertRaisesRegex(ValueError, "No Marlin thread config"):
            self._run_case(832, symmetric=True, size_k=448, group_size=64)


if __name__ == "__main__":
    unittest.main()
