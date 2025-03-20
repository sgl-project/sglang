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
        self.seq_len = 16
        self.num_heads = 8
        self.head_dim = 64

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
            req_pool_indices=torch.zeros(
                self.batch_size, dtype=torch.long, device="cuda"
            ),
            seq_lens=torch.tensor([self.seq_len] * self.batch_size, device="cuda"),
            extend_prefix_lens=torch.tensor([2] * self.batch_size, device="cuda"),
            extend_seq_lens=torch.tensor([2] * self.batch_size, device="cuda"),
            attn_backend=self.backend,
        )

        # Initialize KV cache pool with correct shapes
        kv_cache_size = self.batch_size * self.seq_len
        forward_batch.token_to_kv_pool = MHATokenToKVPool(
            size=kv_cache_size,
            page_size=self.seq_len,  # Changed to match seq_len
            dtype=torch.float16,
            head_num=self.num_heads,
            head_dim=self.head_dim,
            layer_num=1,
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

        # Check output dtype
        self.assertEqual(
            output.dtype,
            torch.float16,
            f"Expected dtype torch.float16, got {output.dtype}",
        )

        # Check output device
        self.assertEqual(
            output.device.type,
            "cuda",
            f"Expected device cuda, got {output.device.type}",
        )

        # Check output is not None and contains no NaN values
        self.assertEqual(
            torch.isnan(output).sum().item(), 0, "Output contains NaN values"
        )


if __name__ == "__main__":
    unittest.main()
