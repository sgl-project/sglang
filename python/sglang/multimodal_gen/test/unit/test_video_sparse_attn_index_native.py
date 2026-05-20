import unittest

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import (
    VSA_TILE_SIZE,
    _use_index_native_vsa,
    _video_sparse_attn_index_native,
    torch_attention,
    triton_block_sparse_attn_forward,
    video_sparse_attn,
)


HAS_INDEX_NATIVE_VSA = (
    torch.cuda.is_available()
    and video_sparse_attn is not None
    and torch_attention is not None
    and triton_block_sparse_attn_forward is not None
)


class TestVideoSparseAttentionIndexNative(unittest.TestCase):
    def test_grad_path_uses_legacy_wrapper(self):
        query = torch.ones(1, requires_grad=True)
        with torch.enable_grad():
            self.assertFalse(_use_index_native_vsa(query, query, query, None))

    @unittest.skipUnless(HAS_INDEX_NATIVE_VSA, "vsa CUDA kernels are unavailable")
    def test_index_native_matches_legacy_wrapper(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        batch_size = 1
        num_heads = 2
        num_blocks = 3
        head_dim = 64
        block_elements = VSA_TILE_SIZE[0] * VSA_TILE_SIZE[1] * VSA_TILE_SIZE[2]
        seq_len = num_blocks * block_elements

        query = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        variable_block_sizes = torch.full(
            (num_blocks,), block_elements, device=device, dtype=torch.int32
        )

        with torch.inference_mode():
            expected = video_sparse_attn(
                query,
                key,
                value,
                variable_block_sizes=variable_block_sizes,
                topk=2,
                block_size=VSA_TILE_SIZE,
            )
            actual = _video_sparse_attn_index_native(
                query,
                key,
                value,
                variable_block_sizes=variable_block_sizes,
                topk=2,
                block_size=VSA_TILE_SIZE,
            )

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    @unittest.skipUnless(HAS_INDEX_NATIVE_VSA, "vsa CUDA kernels are unavailable")
    def test_index_native_matches_legacy_wrapper_with_compress_weight(self):
        torch.manual_seed(1)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        block_elements = VSA_TILE_SIZE[0] * VSA_TILE_SIZE[1] * VSA_TILE_SIZE[2]
        query = torch.randn(1, 2, block_elements * 2, 64, device=device, dtype=dtype)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        compress_weight = torch.rand_like(query)
        variable_block_sizes = torch.full(
            (2,), block_elements, device=device, dtype=torch.int32
        )

        with torch.inference_mode():
            expected = video_sparse_attn(
                query, key, value, variable_block_sizes, 1, VSA_TILE_SIZE, compress_weight
            )
            actual = _video_sparse_attn_index_native(
                query, key, value, variable_block_sizes, 1, VSA_TILE_SIZE, compress_weight
            )

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
