from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.layers.elementwise import fused_gate_sigmoid_mul_add


def _reference(hidden_states, gate_weight, shared_output, final_hidden_states):
    gate = hidden_states @ gate_weight
    final_hidden_states += torch.sigmoid(gate).unsqueeze(1) * shared_output


class TestFusedGateSigmoidMulAdd(unittest.TestCase):
    DTYPES = [torch.float16, torch.bfloat16]
    TOKEN_COUNTS = [1, 7, 32, 128, 1024]
    HIDDEN_DIMS = [2048, 3072, 4096, 5120]

    def _run_and_compare(self, num_tokens, hidden_dim, dtype, rtol=1e-2, atol=1e-2):
        torch.manual_seed(42)
        hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
        gate_weight = torch.randn(hidden_dim, dtype=dtype, device="cuda")
        shared_output = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
        final_ref = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
        final_test = final_ref.clone()

        _reference(hidden_states, gate_weight, shared_output, final_ref)
        fused_gate_sigmoid_mul_add(
            hidden_states, gate_weight, shared_output, final_test
        )

        torch.testing.assert_close(final_test, final_ref, rtol=rtol, atol=atol)

    def test_correctness_fp16(self):
        for n in self.TOKEN_COUNTS:
            for d in self.HIDDEN_DIMS:
                with self.subTest(num_tokens=n, hidden_dim=d, dtype="fp16"):
                    self._run_and_compare(n, d, torch.float16)

    def test_correctness_bf16(self):
        for n in self.TOKEN_COUNTS:
            for d in self.HIDDEN_DIMS:
                with self.subTest(num_tokens=n, hidden_dim=d, dtype="bf16"):
                    self._run_and_compare(n, d, torch.bfloat16, rtol=2e-2, atol=2e-2)

    def test_gate_near_zero(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                num_tokens, hidden_dim = 16, 2048
                hs = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
                gw = torch.zeros(hidden_dim, dtype=dtype, device="cuda")
                so = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
                f_ref = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
                f_test = f_ref.clone()

                _reference(hs, gw, so, f_ref)
                fused_gate_sigmoid_mul_add(hs, gw, so, f_test)

                torch.testing.assert_close(f_test, f_ref, rtol=1e-2, atol=1e-2)

    def test_inplace_semantics(self):
        num_tokens, hidden_dim = 32, 2048
        hs = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device="cuda")
        gw = torch.randn(hidden_dim, dtype=torch.float16, device="cuda")
        so = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device="cuda")
        fhs = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device="cuda")
        original_ptr = fhs.data_ptr()

        fused_gate_sigmoid_mul_add(hs, gw, so, fhs)

        self.assertEqual(fhs.data_ptr(), original_ptr)


if __name__ == "__main__":
    unittest.main()
