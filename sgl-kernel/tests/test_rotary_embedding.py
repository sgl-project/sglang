from typing import Optional, Tuple

import torch
from vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding as VLLMRotaryEmbedding,
)


class SGLRotaryEmbedding(VLLMRotaryEmbedding):

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from sgl_kernel import batched_rotary_embedding, rotary_embedding

        self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            batched_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
                self.rotary_dim,
                offsets,
            )
        else:
            rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        return query, key


# Compare the output of SGLRotaryEmbedding's forward_cuda with VLLMRotaryEmbedding's forward_native


def test_rotary_embedding():
    # Test case 1: FP32
    def run_test(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        dtype,
        batch_size,
        seq_len,
        num_heads,
        test_name,
    ):
        print(f"\nRunning {test_name}...")
        # Initialize both implementations
        sgl_rope = SGLRotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        ).to("cuda")
        vllm_rope = VLLMRotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        ).to("cuda")

        # Regular forward pass
        positions = torch.arange(seq_len, device="cuda").repeat(batch_size)
        query = torch.randn(
            batch_size * seq_len, num_heads * head_size, device="cuda", dtype=dtype
        )
        key = torch.randn(
            batch_size * seq_len, num_heads * head_size, device="cuda", dtype=dtype
        )

        # Make copies for both implementations
        query_sgl = query.clone()
        key_sgl = key.clone()
        query_vllm = query.clone()
        key_vllm = key.clone()

        # Run both implementations
        query_sgl_out, key_sgl_out = sgl_rope.forward_cuda(
            positions, query_sgl, key_sgl
        )
        query_vllm_out, key_vllm_out = vllm_rope.forward_native(
            positions, query_vllm, key_vllm
        )

        # Compare outputs
        torch.testing.assert_close(query_sgl_out, query_vllm_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(key_sgl_out, key_vllm_out, rtol=1e-3, atol=1e-3)

        # Forward pass with offsets
        offsets = torch.randint(
            0, max_position // 2, (batch_size * seq_len,), device="cuda"
        )

        # Make copies for both implementations
        query_sgl = query.clone()
        key_sgl = key.clone()
        query_vllm = query.clone()
        key_vllm = key.clone()

        # Run both implementations with offsets
        query_sgl_out, key_sgl_out = sgl_rope.forward_cuda(
            positions, query_sgl, key_sgl, offsets
        )
        query_vllm_out, key_vllm_out = vllm_rope.forward_native(
            positions, query_vllm, key_vllm, offsets
        )

        # Compare outputs
        torch.testing.assert_close(query_sgl_out, query_vllm_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(key_sgl_out, key_vllm_out, rtol=1e-3, atol=1e-3)
        print(f"{test_name} passed!")

    # Test Case 1: FP32 with larger dimensions
    run_test(
        head_size=128,
        rotary_dim=64,
        max_position=4096,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        batch_size=4,
        seq_len=32,
        num_heads=8,
        test_name="FP32 Test",
    )

    # Test Case 2: BF16 with smaller dimensions
    run_test(
        head_size=64,
        rotary_dim=32,
        max_position=2048,
        base=8000,
        is_neox_style=True,
        dtype=torch.bfloat16,
        batch_size=2,
        seq_len=16,
        num_heads=4,
        test_name="BF16 Test",
    )


if __name__ == "__main__":
    test_rotary_embedding()
