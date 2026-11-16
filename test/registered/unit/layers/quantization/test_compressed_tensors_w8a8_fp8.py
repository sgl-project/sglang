"""CompressedTensorsLinearMethod.apply_into writes a block-FP8 linear into
caller-owned storage (RowParallelLinear's ``output_tensor``).

The DeepGEMM linear writes the caller's tensor directly; the Triton fallback
computes and copies. Either way the result must match ``apply`` bit for bit and
the returned tensor must be the one passed in.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=100, stage="base-b", runner_config="1-gpu-large")

BLOCK = [128, 128]
FP8_MAX = 448.0
# (M, N, K): N and K multiples of the (128, 128) weight block
SHAPES = [(5, 384, 896), (64, 512, 512), (512, 1024, 1024)]


def _layer(linear, n: int, k: int) -> SimpleNamespace:
    """A block-FP8 layer as the scheme sees it, with ``linear`` as its backend."""
    g = torch.Generator(device="cuda").manual_seed(n + k)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.05
    blocks = w.view(n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)) / FP8_MAX
    weight = (blocks / scale[:, None, :, None]).reshape(n, k).to(torch.float8_e4m3fn)
    scheme = object.__new__(CompressedTensorsW8A8Fp8)
    scheme.weight_block_size = BLOCK
    scheme.w8a8_block_fp8_linear = linear
    return SimpleNamespace(
        weight=weight,
        weight_scale=scale.contiguous(),
        input_scale=None,
        scheme=scheme,
        output_size=n,
    )


def _backends():
    backends = {"triton": fp8_utils.triton_w8a8_block_fp8_linear}
    if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and get_device_sm() in (90, 100, 103):
        backends["deep_gemm"] = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
    return backends


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestCompressedTensorsFp8ApplyInto(CustomTestCase):
    def setUp(self):
        self.method = object.__new__(CompressedTensorsLinearMethod)

    def _inputs(self, m: int, k: int):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        yield "bf16", x
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            # the fused all-reduce site's form: fp8 rows + row-major fp32 scales
            yield "prequant", sglang_per_token_group_quant_fp8(x, BLOCK[1])

    def test_matches_apply_and_returns_output(self):
        for name, linear in _backends().items():
            for m, n, k in SHAPES:
                layer = _layer(linear, n, k)
                for kind, x in self._inputs(m, k):
                    with self.subTest(backend=name, shape=(m, n, k), input=kind):
                        ref = self.method.apply(layer, x)
                        out = torch.full((m, n), float("nan"), device="cuda").bfloat16()
                        got = self.method.apply_into(layer, x, out)
                        self.assertEqual(got.data_ptr(), out.data_ptr())
                        self.assertTrue(torch.equal(out, ref))

    def test_deepgemm_writes_the_output_directly(self):
        if "deep_gemm" not in _backends():
            self.skipTest("DeepGEMM unavailable")
        m, n, k = SHAPES[-1]
        layer = _layer(_backends()["deep_gemm"], n, k)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        real = fp8_utils.w8a8_block_fp8_matmul_deepgemm
        with (
            mock.patch.object(
                fp8_utils, "w8a8_block_fp8_matmul_deepgemm", wraps=real
            ) as gemm,
            mock.patch.object(
                torch.Tensor, "copy_", side_effect=AssertionError("copied")
            ),
        ):
            self.method.apply_into(layer, x, out)
        self.assertEqual(gemm.call_args.kwargs["output"].data_ptr(), out.data_ptr())
        self.assertTrue(torch.equal(out, self.method.apply(layer, x)))

    def test_copies_when_the_output_cannot_be_written_directly(self):
        m, n, k = SHAPES[1]
        for name, linear in _backends().items():
            layer = _layer(linear, n, k)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = self.method.apply(layer, x)
            with self.subTest(backend=name, output="strided"):
                out = torch.empty(m, 2 * n, device="cuda", dtype=torch.bfloat16)[:, ::2]
                self.method.apply_into(layer, x, out)
                self.assertTrue(torch.equal(out, ref))
            with self.subTest(backend=name, output="float32"):
                out = torch.empty(m, n, device="cuda", dtype=torch.float32)
                self.method.apply_into(layer, x, out)
                self.assertTrue(torch.equal(out, ref.float()))

    def test_rejects_a_wrong_shaped_output(self):
        m, n, k = SHAPES[1]
        layer = _layer(fp8_utils.triton_w8a8_block_fp8_linear, n, k)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m + 1, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            self.method.apply_into(layer, x, out)


if __name__ == "__main__":
    unittest.main()
