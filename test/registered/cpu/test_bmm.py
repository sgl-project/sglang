import itertools
import unittest

import pytest

# TODO: use interface in cpu.py
import sgl_kernel  # noqa: F401
import torch
import torch.nn as nn
from torch._inductor import config as inductor_config

from sglang.srt.layers.quantization.fp8_utils import input_to_float8
from sglang.srt.model_executor import cpu_graph_runner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="extra-test-cpu-intel")

torch.manual_seed(1234)


@pytest.fixture
def fp8_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def fp8_rows_reference(x):
    x = x.float()
    scales = (x.abs().amax(dim=-1) / 448).clamp_min(torch.finfo(torch.float32).eps)
    return (x * scales.reciprocal().unsqueeze(-1)).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ), scales


@pytest.mark.parametrize(
    "m,k,n", [(1, 128, 512), (35, 512, 128), (96, 128, 512), (0, 512, 128)]
)
def test_fp8_w8a8_bmm(m, k, n, fp8_rng):
    b = 3
    x = torch.randn(m, b, k + 64, dtype=torch.bfloat16)[..., :k].transpose(0, 1)
    if m:
        x[:, 0].zero_()
    weight = (torch.randn(b, n, k) * 8).to(torch.float8_e4m3fn)
    scales = (torch.rand(b, n, k // 128) + 0.5) * 0.02
    packed = [
        torch.ops.sgl_kernel.float8_linear_prepack_cpu(w, s)
        for w, s in zip(weight, scales)
    ]
    packed_weight = torch.stack([p[0] for p in packed])
    packed_scales = torch.stack([p[1] for p in packed])
    storage = torch.full((m, b, n + 64), 73, dtype=torch.bfloat16)
    out = storage[..., :n].transpose(0, 1)
    xq, xs = fp8_rows_reference(x)
    expected = (
        torch.bmm(
            xq.float(),
            (weight.float() * scales.repeat_interleave(128, -1)).transpose(1, 2),
        )
        * xs.unsqueeze(-1)
    ).bfloat16()
    torch.ops.sgl_kernel.bmm_cpu(out, x, packed_weight, True, packed_scales)
    torch.testing.assert_close(out, expected, atol=0.003, rtol=0.01)
    assert torch.all(storage[..., n:] == 73)


@pytest.mark.parametrize(
    "invalid", ["scales", "dtype", "output", "unpacked", "missing"]
)
def test_fp8_w8a8_bmm_invalid(invalid, fp8_rng):
    x = torch.randn(2, 1, 128, dtype=torch.bfloat16)
    weight, scales = torch.ops.sgl_kernel.float8_linear_prepack_cpu(
        torch.ones(32, 128).to(torch.float8_e4m3fn), torch.ones(32, 1)
    )
    weight = weight.unsqueeze(0).repeat(2, 1, 1, 1, 1)
    scales = scales.unsqueeze(0).repeat(2, 1, 1, 1)
    out = torch.empty(2, 1, 32, dtype=torch.bfloat16)
    if invalid == "scales":
        scales = scales[:, :, :, :16].contiguous()
    elif invalid == "dtype":
        x = x.float()
    elif invalid == "output":
        out = out[:, :, :16]
    elif invalid == "missing":
        scales = None
    with pytest.raises(RuntimeError):
        torch.ops.sgl_kernel.bmm_cpu(out, x, weight, invalid != "unpacked", scales)


@pytest.mark.parametrize("m", [0, 1])
@pytest.mark.parametrize("cpp_wrapper", [False, True])
def test_fp8_w8a8_bmm_compile(m, cpp_wrapper, fp8_rng):
    weight, scales = torch.ops.sgl_kernel.float8_linear_prepack_cpu(
        torch.randn(128, 512).to(torch.float8_e4m3fn), torch.ones(128, 4)
    )
    weight = weight.unsqueeze(0).repeat(2, 1, 1, 1, 1)
    scales = scales.unsqueeze(0).repeat(2, 1, 1, 1)
    x = torch.randn(2, m, 512, dtype=torch.bfloat16)

    def forward(x):
        out = x.new_empty((x.shape[0], x.shape[1], 128))
        torch.ops.sgl_kernel.bmm_cpu(out, x, weight, True, scales)
        return out

    with inductor_config.patch(cpp_wrapper=True):
        cpu_graph_runner.register_fake_ops(tp_size=1)
    with inductor_config.patch(cpp_wrapper=cpp_wrapper):
        compiled = torch.compile(forward, backend="inductor", fullgraph=True)
        actual = compiled(x)
    torch.testing.assert_close(actual, forward(x), atol=0, rtol=0)


class Mod(nn.Module):
    def __init__(self, input_channel, output_channel, has_bias):
        super(Mod, self).__init__()
        self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

    def forward(self, x):
        return self.linear(x)


class TestBmm(CustomTestCase):
    M = [1, 2, 11, 111]
    N = [128 + 32, 512]
    K = [512 + 32, 128 + 32]
    B = [1, 16, 17]
    chunk = [True, False]

    def _get_bmm_inputs(self, B, M, N, K, chunk, dtype):
        if chunk:
            mat1 = (
                torch.randn(M, B, K + 64, dtype=dtype).narrow(2, 0, K).transpose_(0, 1)
            )
            mat2 = torch.randn(B, N, K, dtype=dtype).transpose_(1, 2)
            mat3 = (
                torch.randn(M, B, N + 64, dtype=dtype).narrow(2, 0, N).transpose_(0, 1)
            )
        else:
            mat1 = torch.randn(M, B, K, dtype=dtype).transpose_(0, 1)
            mat2 = torch.randn(B, N, K, dtype=dtype).transpose_(1, 2)
            mat3 = torch.randn(M, B, N, dtype=dtype).transpose_(0, 1)
        return mat1, mat2, mat3

    def _bf16_bmm(self, B, M, N, K, chunk, dtype=torch.bfloat16):
        mat1, mat2, mat3 = self._get_bmm_inputs(B, M, N, K, chunk, dtype)
        ref = torch.bmm(mat1, mat2)
        mat2_t = mat2.transpose_(1, 2)
        mat3.zero_()
        torch.ops.sgl_kernel.bmm_cpu(mat3, mat1, mat2, False, None)
        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, mat3, atol=atol, rtol=rtol)

        packed_B = torch.ops.sgl_kernel.convert_weight_packed(mat2_t)
        mat3.zero_()
        torch.ops.sgl_kernel.bmm_cpu(mat3, mat1, packed_B, True, None)
        torch.testing.assert_close(ref, mat3, atol=atol, rtol=rtol)

    def _fp8_bmm(self, B, M, N, K, chunk, dtype=torch.bfloat16):
        mat1, mat2, mat3 = self._get_bmm_inputs(B, M, N, K, chunk, dtype)
        mat2_q, mat2_s = input_to_float8(mat2)
        ref = torch.bmm(mat1, mat2_q.to(torch.bfloat16)) * mat2_s
        mat2_q_t = mat2_q.transpose_(1, 2).contiguous()
        mat3.zero_()
        atol = rtol = precision[ref.dtype]
        torch.ops.sgl_kernel.bmm_cpu(mat3, mat1, mat2_q_t, False, mat2_s)
        torch.testing.assert_close(ref, mat3, atol=atol, rtol=rtol)

        packed_B_q = torch.ops.sgl_kernel.convert_weight_packed(mat2_q_t)
        mat3.zero_()
        torch.ops.sgl_kernel.bmm_cpu(mat3, mat1, packed_B_q, True, mat2_s)
        torch.testing.assert_close(ref, mat3, atol=atol, rtol=rtol)

    def test_bmm(self):
        for params in itertools.product(
            self.B,
            self.M,
            self.N,
            self.K,
            self.chunk,
        ):
            with self.subTest(
                B=params[0],
                M=params[1],
                N=params[2],
                K=params[3],
                chunk=params[4],
            ):
                self._bf16_bmm(*params)
                self._fp8_bmm(*params)


if __name__ == "__main__":
    unittest.main()
