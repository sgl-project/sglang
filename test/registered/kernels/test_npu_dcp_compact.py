"""Ascend coverage for the shared compact DCP Triton kernels."""

import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1, suite="base-a-test-1-npu-a2")

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def _npu_is_available() -> bool:
    return torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available()


@unittest.skipUnless(_npu_is_available(), "Ascend NPU is required")
class TestNpuDcpCompactKernels(CustomTestCase):
    device = "npu"

    def test_pack_preserves_bf16_output_and_fp32_lse_bits(self):
        from sglang.kernels.ops.attention.dcp_kernels import dcp_pack_a2a_send

        num_shards, batch_size, local_heads, head_dim = 2, 2, 6, 512
        total_heads = num_shards * local_heads
        output = torch.randn(
            batch_size,
            total_heads,
            head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        lse = torch.randn(
            batch_size, total_heads, device=self.device, dtype=torch.float32
        )
        packed = torch.empty(
            num_shards,
            batch_size,
            local_heads,
            head_dim + 2,
            device=self.device,
            dtype=torch.bfloat16,
        )

        dcp_pack_a2a_send(
            output,
            lse,
            packed[..., :head_dim],
            packed.view(torch.float32)[..., head_dim // 2],
        )

        expected_output = output.view(
            batch_size, num_shards, local_heads, head_dim
        ).permute(1, 0, 2, 3)
        expected_lse = lse.view(batch_size, num_shards, local_heads).permute(1, 0, 2)
        self.assertTrue(torch.equal(packed[..., :head_dim], expected_output))
        self.assertTrue(
            torch.equal(packed.view(torch.float32)[..., head_dim // 2], expected_lse)
        )

    def test_fused_unpack_merge_handles_empty_graph_row(self):
        from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton

        output = torch.randn(2, 2, 6, 512, device=self.device, dtype=torch.bfloat16)
        lse = torch.randn(2, 2, 6, device=self.device, dtype=torch.float32)
        lse[:, 0, 0] = float("-inf")

        merged, _ = dcp_lse_combine_triton(output, lse, is_lse_base_on_e=True)

        reference_weights = torch.softmax(lse.cpu(), dim=0)
        reference = (output.cpu().float() * reference_weights.unsqueeze(-1)).sum(dim=0)
        reference[0, 0] = 0
        torch.testing.assert_close(
            merged.cpu().float(), reference, atol=1e-2, rtol=1e-2
        )
        self.assertTrue(torch.equal(merged[0, 0], torch.zeros_like(merged[0, 0])))


if __name__ == "__main__":
    unittest.main()
