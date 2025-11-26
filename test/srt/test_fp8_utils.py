import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    _unpack_ue8m0_scale_for_triton,
    inverse_transform_scale_ue8m0,
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)
from sglang.test.test_utils import CustomTestCase


class TestInverseTransformScaleUe8m0(CustomTestCase):
    def test_round_trip(self):
        for _ in range(100):
            weight_bf16 = torch.randn(
                # DeepSeek V3 kv_b_proj
                (32768, 512),
                dtype=torch.bfloat16,
                device="cuda",
            )

            weight_block_size = [128, 128]

            qweight, sf_fp32_original = quant_weight_ue8m0(
                weight_bf16, weight_block_size=weight_block_size
            )
            mn = qweight.shape[-2]

            sf_packed_original = transform_scale_ue8m0(sf_fp32_original, mn=mn)
            sf_fp32_recreated = inverse_transform_scale_ue8m0(sf_packed_original, mn=mn)

            sf_packed_recreated = transform_scale_ue8m0(sf_fp32_recreated, mn=mn)

            assert torch.all(
                sf_packed_original == sf_packed_recreated
            ), f"{sf_packed_original=} {sf_packed_recreated}"
            assert torch.all(
                sf_fp32_original == sf_fp32_recreated
            ), f"{sf_fp32_original=} {sf_fp32_recreated}"


class TestUnpackUe8m0ScaleForTriton(CustomTestCase):
    """Test _unpack_ue8m0_scale_for_triton function.

    This function converts UE8M0 packed scale tensors back to float32 format
    for use with the triton kernel when DeepGemm falls back to triton.
    """

    def test_unpack_row_replicated_format(self):
        """Test unpacking when scales are row-replicated (N rows, not n_groups)."""
        # Simulate UE8M0 packed format with row replication
        # Weight shape: (N=1024, K=3072), block_size=[128, 128]
        N, K = 1024, 3072
        block_size = [128, 128]
        n_groups = N // block_size[0]  # 8
        k_groups = K // block_size[1]  # 24

        # Create original float32 scale in expected format (n_groups, k_groups)
        sf_fp32_original = torch.rand(
            n_groups, k_groups, dtype=torch.float32, device="cuda"
        )
        # Clamp to valid UE8M0 range (powers of 2)
        sf_fp32_original = (
            2.0 ** torch.randint(-127, 127, (n_groups, k_groups), device="cuda").float()
        )

        # Simulate the row-replicated packed format:
        # Replicate each row 128 times (block_n)
        sf_replicated = sf_fp32_original.repeat_interleave(
            block_size[0], dim=0
        )  # (N, k_groups)

        # Pad k dimension to multiple of 4
        k_padded = ((k_groups + 3) // 4) * 4
        sf_padded = torch.zeros(N, k_padded, dtype=torch.float32, device="cuda")
        sf_padded[:, :k_groups] = sf_replicated

        # Pack to UE8M0 int32 format
        # float32 -> extract exponent -> uint8 -> pack 4 into int32
        sf_u8 = ((sf_padded.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        sf_packed = sf_u8.view(N, k_padded // 4, 4).view(torch.uint8)
        # Reinterpret 4 bytes as int32
        sf_packed = (
            sf_packed.reshape(N, -1)
            .contiguous()
            .view(torch.uint8)
            .view(N, k_padded // 4 * 4)
        )
        sf_packed = sf_packed.view(N, -1, 4).view(torch.int32).squeeze(-1)

        # Unpack using our function
        sf_unpacked = _unpack_ue8m0_scale_for_triton(sf_packed, (N, K), block_size)

        # Verify shape
        self.assertEqual(sf_unpacked.shape, (n_groups, k_groups))
        self.assertEqual(sf_unpacked.dtype, torch.float32)

        # Verify values match original (within UE8M0 precision)
        torch.testing.assert_close(sf_unpacked, sf_fp32_original)

    def test_unpack_direct_format(self):
        """Test unpacking when scales are already in n_groups format (not replicated)."""
        # Weight shape: (N=1024, K=3072), block_size=[128, 128]
        N, K = 1024, 3072
        block_size = [128, 128]
        n_groups = N // block_size[0]  # 8
        k_groups = K // block_size[1]  # 24

        # Create original float32 scale in expected format (n_groups, k_groups)
        sf_fp32_original = (
            2.0 ** torch.randint(-127, 127, (n_groups, k_groups), device="cuda").float()
        )

        # Pad k dimension to multiple of 4
        k_padded = ((k_groups + 3) // 4) * 4
        sf_padded = torch.zeros(n_groups, k_padded, dtype=torch.float32, device="cuda")
        sf_padded[:, :k_groups] = sf_fp32_original

        # Pack to UE8M0 int32 format
        sf_u8 = ((sf_padded.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        sf_packed = sf_u8.view(n_groups, k_padded // 4, 4).view(torch.uint8)
        sf_packed = (
            sf_packed.reshape(n_groups, -1)
            .contiguous()
            .view(torch.uint8)
            .view(n_groups, k_padded // 4 * 4)
        )
        sf_packed = sf_packed.view(n_groups, -1, 4).view(torch.int32).squeeze(-1)

        # Unpack using our function
        sf_unpacked = _unpack_ue8m0_scale_for_triton(sf_packed, (N, K), block_size)

        # Verify shape
        self.assertEqual(sf_unpacked.shape, (n_groups, k_groups))
        self.assertEqual(sf_unpacked.dtype, torch.float32)

        # Verify values match original
        torch.testing.assert_close(sf_unpacked, sf_fp32_original)

    def test_unpack_non_aligned_k(self):
        """Test unpacking when K is not perfectly aligned to block_k."""
        # Weight shape: (N=1024, K=3072+64), block_size=[128, 128]
        # k_groups = ceil(3136/128) = 25
        N, K = 1024, 3136
        block_size = [128, 128]
        n_groups = N // block_size[0]  # 8
        k_groups = (K + block_size[1] - 1) // block_size[1]  # 25

        # Create original float32 scale in expected format (n_groups, k_groups)
        sf_fp32_original = (
            2.0 ** torch.randint(-127, 127, (n_groups, k_groups), device="cuda").float()
        )

        # Pad k dimension to multiple of 4
        k_padded = ((k_groups + 3) // 4) * 4  # 28
        sf_padded = torch.zeros(n_groups, k_padded, dtype=torch.float32, device="cuda")
        sf_padded[:, :k_groups] = sf_fp32_original

        # Pack to UE8M0 int32 format
        sf_u8 = ((sf_padded.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        sf_packed = sf_u8.view(n_groups, k_padded // 4, 4).view(torch.uint8)
        sf_packed = (
            sf_packed.reshape(n_groups, -1)
            .contiguous()
            .view(torch.uint8)
            .view(n_groups, k_padded // 4 * 4)
        )
        sf_packed = sf_packed.view(n_groups, -1, 4).view(torch.int32).squeeze(-1)

        # Unpack using our function
        sf_unpacked = _unpack_ue8m0_scale_for_triton(sf_packed, (N, K), block_size)

        # Verify shape - should crop to exact k_groups
        self.assertEqual(sf_unpacked.shape, (n_groups, k_groups))
        self.assertEqual(sf_unpacked.dtype, torch.float32)

        # Verify values match original
        torch.testing.assert_close(sf_unpacked, sf_fp32_original)


if __name__ == "__main__":
    unittest.main()
