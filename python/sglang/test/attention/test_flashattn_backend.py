import unittest

import torch

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class MockModelRunner:
    model_config = type(
        "ModelConfig", (), {"context_len": 2048, "is_multimodal": False}
    )
    sliding_window_size = None
    req_to_token_pool = type("TokenPool", (), {"size": 32})

    def __init__(self, device="cuda"):
        self.device = device


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFlashAttentionBackend(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model_runner = MockModelRunner()
        self.backend = FlashAttentionBackend(self.model_runner)

        # Common test parameters
        self.batch_size = 2
        self.seq_len = 4
        self.num_heads = 2
        self.head_dim = 8

    def test_forward_extend(self):
        # Create mock inputs with correct shapes
        # Shape should be [batch_size * seq_len, num_heads, head_dim]
        q = torch.randn(
            self.batch_size * self.seq_len,
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        k = torch.randn(
            self.batch_size * self.seq_len,
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        v = torch.randn(
            self.batch_size * self.seq_len,
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device="cuda",
        )

        # Create attention layer
        layer = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_heads,
            layer_id=0,
        )

        # Create mock forward batch with all required arguments
        forward_batch = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(
                0, 100, (self.batch_size, self.seq_len), device="cuda"
            ),
            out_cache_loc=torch.arange(self.batch_size * self.seq_len, device="cuda"),
            seq_lens_sum=self.batch_size * self.seq_len,
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.arange(self.batch_size, device="cuda"),
            seq_lens=torch.tensor([self.seq_len] * self.batch_size, device="cuda"),
            extend_prefix_lens=torch.tensor([2] * self.batch_size, device="cuda"),
            extend_seq_lens=torch.tensor([2] * self.batch_size, device="cuda"),
            attn_backend=self.backend,
        )

        # Initialize KV cache pool with correct shapes
        kv_cache_size = self.batch_size * self.seq_len
        forward_batch.token_to_kv_pool = MHATokenToKVPool(
            size=kv_cache_size,
            page_size=1,  # only consider page=1 for unit test
            dtype=torch.float16,
            head_num=self.num_heads,
            head_dim=self.head_dim,
            layer_num=1,  # only consider layer=1 for unit test
            device="cuda",
            enable_memory_saver=False,
        )

        # Run forward_extend
        output = self.backend.forward_extend(q, k, v, layer, forward_batch)

        # Basic shape checks
        expected_shape = (
            self.batch_size * self.seq_len,
            self.num_heads * self.head_dim,
        )
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {output.shape}",
        )

        # Check output is not None and contains no NaN values
        self.assertEqual(
            torch.isnan(output).sum().item(), 0, "Output contains NaN values"
        )

    def test_forward_decode(self):
        # For decode, we only have one token per sequence
        decode_len = 1
        curr_seq_len = self.seq_len + decode_len
        # Create inputs for the current token
        q = torch.randn(
            self.batch_size * decode_len,
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        # k and v for current token (will be added to cache)
        k = torch.randn(
            self.batch_size * decode_len,
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        v = torch.randn(
            self.batch_size * decode_len,
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device="cuda",
        )

        # Create attention layer
        layer = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_heads,
            layer_id=0,
        )

        # Create mock req_to_token_pool
        class MockReqToTokenPool:
            def __init__(self, batch_size, seq_len, device):
                self.req_to_token = torch.arange(
                    batch_size * seq_len, device=device
                ).reshape(batch_size, seq_len)

        # Create mock forward batch with all required arguments for decode
        forward_batch = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(
                0, 100, (self.batch_size, decode_len), device="cuda"
            ),
            out_cache_loc=torch.arange(
                self.batch_size * self.seq_len,
                self.batch_size * curr_seq_len,
                device="cuda",
            ),
            seq_lens_sum=self.batch_size * curr_seq_len,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(self.batch_size, device="cuda"),
            seq_lens=torch.tensor([curr_seq_len] * self.batch_size, device="cuda"),
            attn_backend=self.backend,
        )

        # Add req_to_token_pool to forward_batch
        forward_batch.req_to_token_pool = MockReqToTokenPool(
            self.batch_size, curr_seq_len, "cuda"
        )

        # Initialize KV cache pool
        kv_cache_size = self.batch_size * (
            curr_seq_len
        )  # Include space for new token
        forward_batch.token_to_kv_pool = MHATokenToKVPool(
            size=kv_cache_size,
            page_size=1,
            dtype=torch.float16,
            head_num=self.num_heads,
            head_dim=self.head_dim,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )

        # Pre-fill the KV cache with some values
        # This simulates having previous tokens' KV pairs in cache
        cache_k = torch.randn(
            self.batch_size * self.seq_len,
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        cache_v = torch.randn(
            self.batch_size * self.seq_len,
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            torch.arange(self.batch_size * self.seq_len, device="cuda"),
            cache_k,
            cache_v,
            layer.k_scale,
            layer.v_scale,
        )

        # Run forward_decode
        output = self.backend.forward_decode(q, k, v, layer, forward_batch)

        # Basic shape checks - decode output should be [batch_size, num_heads * head_dim]
        expected_shape = (self.batch_size, self.num_heads * self.head_dim)
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {output.shape}",
        )

        # Check output is not None and contains no NaN values
        self.assertEqual(
            torch.isnan(output).sum().item(), 0, "Output contains NaN values"
        )


if __name__ == "__main__":
    unittest.main()
