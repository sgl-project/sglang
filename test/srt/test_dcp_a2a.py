"""
Tests for DCP All-to-All communication backend.

Covers:
1. Config validation: --dcp-comm-backend choices and constraints
2. Triton LSE combine kernel correctness vs CPU reference
3. Both base-e and base-2 LSE conventions
4. Various DCP world sizes (N=2,4,8)
5. Edge cases: single shard, NaN/inf LSE values

Reference: vLLM PR #34883 tests/distributed/test_dcp_a2a.py
"""

import unittest

import torch

from sglang.srt.layers.attention.dcp_a2a import (
    _lse_weighted_combine_cpu,
    dcp_lse_combine_triton,
)


class TestDCPCommBackendConfig(unittest.TestCase):
    """Test --dcp-comm-backend config validation."""

    def test_valid_choices(self):
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(ServerArgs.dcp_comm_backend, "ag_rs")

    def test_field_exists(self):
        """Verify the dcp_comm_backend field exists on ServerArgs dataclass."""
        import dataclasses

        from sglang.srt.server_args import ServerArgs

        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("dcp_comm_backend", fields)


class TestLSECombineTritonVsCPU(unittest.TestCase):
    """Test Triton LSE combine kernel against CPU reference."""

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cls.device == "cpu":
            raise unittest.SkipTest("CUDA required for Triton kernel tests")

    def _run_combine_test(
        self, N, B, H_local, D, is_base_e, dtype=torch.bfloat16, atol=1e-2
    ):
        """Run Triton combine and compare against CPU reference."""
        torch.manual_seed(42)

        partial_outputs = torch.randn(N, B, H_local, D, device=self.device, dtype=dtype)
        if is_base_e:
            partial_lses = torch.randn(
                N, B, H_local, device=self.device, dtype=torch.float32
            )
        else:
            partial_lses = (
                torch.randn(N, B, H_local, device=self.device, dtype=torch.float32)
                * 5.0
            )

        cpu_result = _lse_weighted_combine_cpu(
            partial_outputs.cpu(),
            partial_lses.cpu(),
            is_lse_base_on_e=is_base_e,
        )

        triton_result, _ = dcp_lse_combine_triton(
            partial_outputs,
            partial_lses,
            is_lse_base_on_e=is_base_e,
            return_lse=False,
        )

        torch.testing.assert_close(
            triton_result.float().cpu(),
            cpu_result.float(),
            atol=atol,
            rtol=1e-2,
        )

    def test_n2_base_e(self):
        """N=2, base-e LSE (FlashAttention convention)."""
        self._run_combine_test(N=2, B=4, H_local=8, D=64, is_base_e=True)

    def test_n2_base_2(self):
        """N=2, base-2 LSE (FlashInfer convention)."""
        self._run_combine_test(N=2, B=4, H_local=8, D=64, is_base_e=False)

    def test_n4_base_2(self):
        """N=4, base-2 LSE."""
        self._run_combine_test(N=4, B=8, H_local=16, D=128, is_base_e=False)

    def test_n8_base_2(self):
        """N=8, base-2 LSE (typical DCP=8 config for DeepSeek-V2)."""
        self._run_combine_test(N=8, B=4, H_local=8, D=512, is_base_e=False)

    def test_n2_large_batch(self):
        """N=2, large batch."""
        self._run_combine_test(N=2, B=64, H_local=16, D=128, is_base_e=False)

    def test_n4_base_e_large_head_dim(self):
        """N=4, base-e, large head dimension (MLA kv_lora_rank=512)."""
        self._run_combine_test(N=4, B=8, H_local=8, D=512, is_base_e=True)

    def test_single_shard(self):
        """N=1 — should return input unchanged."""
        N, B, H_local, D = 1, 4, 8, 64
        partial_outputs = torch.randn(
            N, B, H_local, D, device=self.device, dtype=torch.bfloat16
        )
        partial_lses = torch.randn(
            N, B, H_local, device=self.device, dtype=torch.float32
        )

        triton_result, _ = dcp_lse_combine_triton(
            partial_outputs, partial_lses, is_lse_base_on_e=True
        )

        torch.testing.assert_close(
            triton_result.float().cpu(),
            partial_outputs.squeeze(0).float().cpu(),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_return_lse(self):
        """Verify return_lse=True produces valid global LSE."""
        N, B, H_local, D = 2, 4, 8, 64
        partial_outputs = torch.randn(
            N, B, H_local, D, device=self.device, dtype=torch.bfloat16
        )
        partial_lses = torch.randn(
            N, B, H_local, device=self.device, dtype=torch.float32
        )

        triton_result, triton_lse = dcp_lse_combine_triton(
            partial_outputs, partial_lses, is_lse_base_on_e=True, return_lse=True
        )

        self.assertIsNotNone(triton_lse)
        self.assertEqual(triton_lse.shape, (B, H_local))
        self.assertFalse(torch.isnan(triton_lse).any())


class TestLSECombineEdgeCases(unittest.TestCase):
    """Test edge cases for LSE combine."""

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cls.device == "cpu":
            raise unittest.SkipTest("CUDA required for Triton kernel tests")

    def test_one_shard_dominant(self):
        """One shard has much larger LSE — output should be close to that shard."""
        N, B, H_local, D = 2, 1, 1, 64
        partial_outputs = torch.randn(
            N, B, H_local, D, device=self.device, dtype=torch.bfloat16
        )
        partial_lses = torch.tensor(
            [[[100.0]], [[-100.0]]], device=self.device, dtype=torch.float32
        )

        triton_result, _ = dcp_lse_combine_triton(
            partial_outputs, partial_lses, is_lse_base_on_e=True
        )
        cpu_result = _lse_weighted_combine_cpu(
            partial_outputs.cpu(), partial_lses.cpu(), is_lse_base_on_e=True
        )

        torch.testing.assert_close(
            triton_result.float().cpu(),
            cpu_result.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            triton_result.float().cpu(),
            partial_outputs[0].float().cpu(),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_equal_lse(self):
        """Equal LSE across shards — output should be mean of outputs."""
        N, B, H_local, D = 2, 1, 1, 64
        partial_outputs = torch.randn(
            N, B, H_local, D, device=self.device, dtype=torch.bfloat16
        )
        partial_lses = torch.tensor(
            [[[5.0]], [[5.0]]], device=self.device, dtype=torch.float32
        )

        triton_result, _ = dcp_lse_combine_triton(
            partial_outputs, partial_lses, is_lse_base_on_e=True
        )
        expected = partial_outputs.float().mean(dim=0)

        torch.testing.assert_close(
            triton_result.float().cpu(),
            expected.cpu(),
            atol=1e-2,
            rtol=1e-2,
        )


class TestCPUReference(unittest.TestCase):
    """Test the CPU reference implementation independently."""

    def test_basic_combine(self):
        """Basic N=2 combine on CPU."""
        N, B, H, D = 2, 2, 4, 8
        outputs = torch.randn(N, B, H, D)
        lses = torch.randn(N, B, H)

        result = _lse_weighted_combine_cpu(outputs, lses, is_lse_base_on_e=True)
        self.assertEqual(result.shape, (B, H, D))
        self.assertFalse(torch.isnan(result).any())

    def test_base2_vs_base_e(self):
        """Base-2 and base-e should produce different results for same input."""
        N, B, H, D = 2, 2, 4, 8
        outputs = torch.randn(N, B, H, D)
        lses = torch.randn(N, B, H) * 3.0

        result_e = _lse_weighted_combine_cpu(outputs, lses, is_lse_base_on_e=True)
        result_2 = _lse_weighted_combine_cpu(outputs, lses, is_lse_base_on_e=False)

        self.assertFalse(torch.allclose(result_e, result_2, atol=1e-3))

    def test_nan_lse_handled(self):
        """NaN in LSE should not propagate to output."""
        N, B, H, D = 2, 1, 1, 8
        outputs = torch.randn(N, B, H, D)
        lses = torch.tensor([[[5.0]], [[float("nan")]]])

        result = _lse_weighted_combine_cpu(outputs, lses, is_lse_base_on_e=True)
        self.assertFalse(torch.isnan(result).any())

    def test_inf_lse_handled(self):
        """Inf in LSE should not propagate to output."""
        N, B, H, D = 2, 1, 1, 8
        outputs = torch.randn(N, B, H, D)
        lses = torch.tensor([[[5.0]], [[float("inf")]]])

        result = _lse_weighted_combine_cpu(outputs, lses, is_lse_base_on_e=True)
        self.assertFalse(torch.isnan(result).any())


if __name__ == "__main__":
    unittest.main()
