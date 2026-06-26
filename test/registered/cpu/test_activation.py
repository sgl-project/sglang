import itertools
import unittest

import torch
from utils import GeluAndMul, SiluAndMul, precision

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")
register_cpu_ci(est_time=10, suite="base-b-test-cpu-arm64")

torch.manual_seed(1234)


class TestActivation(CustomTestCase):
    M = [128, 129, 257]
    N = [22016, 22018]
    dtype = [torch.float16, torch.bfloat16]

    def _silu_and_mul_test(self, m, n, dtype):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        x = torch.randn([m, n], dtype=dtype)

        out = torch.ops.sgl_kernel.silu_and_mul_cpu(x)
        ref_out = SiluAndMul(x)

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def _gelu_and_mul_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)

        out = torch.ops.sgl_kernel.gelu_and_mul_cpu(x)
        ref_out = GeluAndMul(x, approximate="none")

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def _gelu_tanh_and_mul_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)

        out = torch.ops.sgl_kernel.gelu_tanh_and_mul_cpu(x)
        ref_out = GeluAndMul(x, approximate="tanh")

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def _fused_sigmoid_mul_test(self, m, num_heads, head_dim, dtype, gate_3d):
        x = torch.randn([m, num_heads * head_dim], dtype=dtype)
        if gate_3d:
            gate_storage = torch.randn([m, num_heads, head_dim * 2], dtype=dtype)
            gate = gate_storage[..., :head_dim]
            self.assertFalse(gate.is_contiguous())
        else:
            gate = torch.randn_like(x)

        gate_ref = gate.reshape(m, -1) if gate_3d else gate
        ref_out = x * torch.sigmoid(gate_ref)

        out = torch.ops.sgl_kernel.fused_sigmoid_mul_cpu(x, gate, False)
        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

        x_inplace = x.clone()
        ref_inplace = x_inplace * torch.sigmoid(gate_ref)
        out_inplace = torch.ops.sgl_kernel.fused_sigmoid_mul_cpu(x_inplace, gate, True)
        self.assertEqual(out_inplace.data_ptr(), x_inplace.data_ptr())
        torch.testing.assert_close(ref_inplace, x_inplace, atol=atol, rtol=rtol)

    def test_activation(self):
        for params in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=params[0], n=params[1], dtype=params[2]):
                self._silu_and_mul_test(*params)
                self._gelu_and_mul_test(*params)
                self._gelu_tanh_and_mul_test(*params)

    def test_fused_sigmoid_mul(self):
        for params in itertools.product(
            [1, 17, 128], [16], [256], self.dtype, [False, True]
        ):
            with self.subTest(
                m=params[0],
                num_heads=params[1],
                head_dim=params[2],
                dtype=params[3],
                gate_3d=params[4],
            ):
                self._fused_sigmoid_mul_test(*params)


if __name__ == "__main__":
    unittest.main()
