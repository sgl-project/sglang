"""Tests for DCP LSE combine kernels.

Covers:
1. Triton LSE combine kernel correctness vs CPU reference (base-e and base-2)
2. Various DCP world sizes (N=1,2,4,8)
3. Edge cases: single shard, dominant LSE, equal LSE, NaN/inf
4. return_lse mode
5. dcp_a2a_lse_reduce with pre-allocated CUDA graph buffers
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-large")


class TestLSECombineTritonVsCPU(CustomTestCase):
    """Test Triton LSE combine kernel against CPU reference."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for Triton kernel tests")
        cls.device = "cuda"

    def _run_combine_test(
        self, N, B, H_local, D, is_base_e, dtype=torch.bfloat16, atol=1e-2
    ):
        from sglang.srt.layers.attention.dcp_a2a import (
            _lse_weighted_combine_cpu,
            dcp_lse_combine_triton,
        )

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
        self._run_combine_test(N=2, B=4, H_local=8, D=64, is_base_e=True)

    def test_n2_base_2(self):
        self._run_combine_test(N=2, B=4, H_local=8, D=64, is_base_e=False)

    def test_n4_base_e(self):
        self._run_combine_test(N=4, B=8, H_local=16, D=128, is_base_e=True)

    def test_n4_base_2(self):
        self._run_combine_test(N=4, B=8, H_local=16, D=128, is_base_e=False)

    def test_n8_base_e(self):
        self._run_combine_test(N=8, B=4, H_local=8, D=128, is_base_e=True)

    def test_n8_base_2(self):
        self._run_combine_test(N=8, B=4, H_local=8, D=512, is_base_e=False)

    def test_n2_large_batch(self):
        self._run_combine_test(N=2, B=64, H_local=16, D=128, is_base_e=False)

    def test_n4_large_head_dim(self):
        self._run_combine_test(N=4, B=8, H_local=8, D=512, is_base_e=True)


class TestLSECombineSingleShard(CustomTestCase):
    """N=1 should return input unchanged."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def test_single_shard(self):
        from sglang.srt.layers.attention.dcp_a2a import dcp_lse_combine_triton

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


class TestLSECombineReturnLSE(CustomTestCase):
    """Verify return_lse=True produces valid global LSE."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def test_return_lse(self):
        from sglang.srt.layers.attention.dcp_a2a import dcp_lse_combine_triton

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


class TestLSECombineEdgeCases(CustomTestCase):
    """Test edge cases for LSE combine."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def test_one_shard_dominant(self):
        """One shard has much larger LSE -- output should be close to that shard."""
        from sglang.srt.layers.attention.dcp_a2a import (
            _lse_weighted_combine_cpu,
            dcp_lse_combine_triton,
        )

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
            triton_result.float().cpu(), cpu_result.float(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            triton_result.float().cpu(),
            partial_outputs[0].float().cpu(),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_equal_lse(self):
        """Equal LSE across shards -- output should be mean of outputs."""
        from sglang.srt.layers.attention.dcp_a2a import dcp_lse_combine_triton

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
            triton_result.float().cpu(), expected.cpu(), atol=1e-2, rtol=1e-2
        )


class TestCPUReference(CustomTestCase):
    """Test the CPU reference implementation independently."""

    def test_basic_combine(self):
        from sglang.srt.layers.attention.dcp_a2a import _lse_weighted_combine_cpu

        N, B, H, D = 2, 2, 4, 8
        outputs = torch.randn(N, B, H, D)
        lses = torch.randn(N, B, H)

        result = _lse_weighted_combine_cpu(outputs, lses, is_lse_base_on_e=True)
        self.assertEqual(result.shape, (B, H, D))
        self.assertFalse(torch.isnan(result).any())

    def test_base2_vs_base_e(self):
        from sglang.srt.layers.attention.dcp_a2a import _lse_weighted_combine_cpu

        N, B, H, D = 2, 2, 4, 8
        outputs = torch.randn(N, B, H, D)
        lses = torch.randn(N, B, H) * 3.0

        result_e = _lse_weighted_combine_cpu(outputs, lses, is_lse_base_on_e=True)
        result_2 = _lse_weighted_combine_cpu(outputs, lses, is_lse_base_on_e=False)

        self.assertFalse(torch.allclose(result_e, result_2, atol=1e-3))

    def test_nan_lse_handled(self):
        from sglang.srt.layers.attention.dcp_a2a import _lse_weighted_combine_cpu

        N, B, H, D = 2, 1, 1, 8
        outputs = torch.randn(N, B, H, D)
        lses = torch.tensor([[[5.0]], [[float("nan")]]])

        result = _lse_weighted_combine_cpu(outputs, lses, is_lse_base_on_e=True)
        self.assertFalse(torch.isnan(result).any())

    def test_inf_lse_handled(self):
        from sglang.srt.layers.attention.dcp_a2a import _lse_weighted_combine_cpu

        N, B, H, D = 2, 1, 1, 8
        outputs = torch.randn(N, B, H, D)
        lses = torch.tensor([[[5.0]], [[float("inf")]]])

        result = _lse_weighted_combine_cpu(outputs, lses, is_lse_base_on_e=True)
        self.assertFalse(torch.isnan(result).any())


class TestDCPA2AReduceWithCUDAGraphBuffers(CustomTestCase):
    """Test dcp_a2a_lse_reduce with pre-allocated CUDA graph buffers."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def _make_mock_group(self, world_size):
        group = MagicMock()
        group.world_size = world_size

        def identity_a2a(output, input_):
            output.copy_(input_)

        group.all_to_all_single = MagicMock(side_effect=identity_a2a)
        return group

    def _make_cuda_graph_buffers(self, N, max_bs, H_per_rank, D, lpd=2):
        """Create fused CUDA graph buffers matching dcp_a2a_lse_reduce API."""
        return {
            "send_combined": torch.empty(
                N, max_bs, H_per_rank, D + lpd, dtype=torch.bfloat16, device=self.device
            ),
            "recv_combined": torch.empty(
                N, max_bs, H_per_rank, D + lpd, dtype=torch.bfloat16, device=self.device
            ),
            "send_lse": torch.empty(
                N, max_bs, H_per_rank, dtype=torch.float32, device=self.device
            ),
            "recv_lse": torch.empty(
                N, max_bs, H_per_rank, dtype=torch.float32, device=self.device
            ),
        }

    def test_cuda_graph_buffers_same_as_dynamic(self):
        from sglang.srt.layers.attention.dcp_a2a import dcp_a2a_lse_reduce

        torch.manual_seed(123)
        N, B, H_per_rank, D = 2, 4, 8, 128
        H = H_per_rank * N
        max_bs = 16

        group = self._make_mock_group(N)

        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)

        result_dynamic = dcp_a2a_lse_reduce(
            attn_out.clone(), attn_lse.clone(), group, is_lse_base_on_e=True
        )

        cuda_graph_buffers = self._make_cuda_graph_buffers(N, max_bs, H_per_rank, D)

        result_graph = dcp_a2a_lse_reduce(
            attn_out.clone(),
            attn_lse.clone(),
            group,
            is_lse_base_on_e=True,
            cuda_graph_buffers=cuda_graph_buffers,
        )

        torch.testing.assert_close(
            result_graph.float().cpu(),
            result_dynamic.float().cpu(),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cuda_graph_buffers_n4(self):
        from sglang.srt.layers.attention.dcp_a2a import dcp_a2a_lse_reduce

        torch.manual_seed(456)
        N, B, H_per_rank, D = 4, 2, 4, 64
        H = H_per_rank * N
        max_bs = 8

        group = self._make_mock_group(N)

        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)

        result_dynamic = dcp_a2a_lse_reduce(
            attn_out.clone(), attn_lse.clone(), group, is_lse_base_on_e=True
        )

        cuda_graph_buffers = self._make_cuda_graph_buffers(N, max_bs, H_per_rank, D)

        result_graph = dcp_a2a_lse_reduce(
            attn_out.clone(),
            attn_lse.clone(),
            group,
            is_lse_base_on_e=True,
            cuda_graph_buffers=cuda_graph_buffers,
        )

        torch.testing.assert_close(
            result_graph.float().cpu(),
            result_dynamic.float().cpu(),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cuda_graph_buffers_partial_batch(self):
        """Buffer max_bs > actual B -- should correctly slice."""
        from sglang.srt.layers.attention.dcp_a2a import dcp_a2a_lse_reduce

        torch.manual_seed(789)
        N, B, H_per_rank, D = 2, 3, 8, 128
        H = H_per_rank * N
        max_bs = 32

        group = self._make_mock_group(N)

        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)

        cuda_graph_buffers = self._make_cuda_graph_buffers(N, max_bs, H_per_rank, D)

        result = dcp_a2a_lse_reduce(
            attn_out,
            attn_lse,
            group,
            is_lse_base_on_e=True,
            cuda_graph_buffers=cuda_graph_buffers,
        )

        self.assertEqual(result.shape, (B, H_per_rank, D))
        self.assertFalse(torch.isnan(result).any())

    def test_a2a_reduce_allocates_when_no_buffers(self):
        """Without cuda_graph_buffers, dcp_a2a_lse_reduce still works (eager mode)."""
        from sglang.srt.layers.attention.dcp_a2a import dcp_a2a_lse_reduce

        N, B, H_per_rank, D = 2, 4, 8, 64
        H = H_per_rank * N

        group = self._make_mock_group(N)

        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)

        result = dcp_a2a_lse_reduce(
            attn_out,
            attn_lse,
            group,
            is_lse_base_on_e=True,
            cuda_graph_buffers=None,
        )

        self.assertEqual(result.shape, (B, H_per_rank, D))
        self.assertFalse(torch.isnan(result).any())

    def test_buffers_have_fixed_data_ptrs(self):
        """Pre-allocated buffer data_ptr must not change -- required for graph replay."""
        from sglang.srt.layers.attention.dcp_a2a import dcp_a2a_lse_reduce

        N, B, H_per_rank, D = 2, 4, 8, 64
        H = H_per_rank * N
        max_bs = 16

        group = self._make_mock_group(N)

        buffers = self._make_cuda_graph_buffers(N, max_bs, H_per_rank, D)
        send_ptr = buffers["send_combined"].data_ptr()
        recv_ptr = buffers["recv_combined"].data_ptr()

        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)

        dcp_a2a_lse_reduce(
            attn_out,
            attn_lse,
            group,
            is_lse_base_on_e=True,
            cuda_graph_buffers=buffers,
        )

        self.assertEqual(buffers["send_combined"].data_ptr(), send_ptr)
        self.assertEqual(buffers["recv_combined"].data_ptr(), recv_ptr)


if __name__ == "__main__":
    unittest.main()
