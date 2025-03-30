import unittest

import torch

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import CustomTestCase


class MockModelRunner:
    model_config = type(
        "ModelConfig", (), {"context_len": 2048, "is_multimodal": False}
    )
    sliding_window_size = None

    def __init__(self, device="cuda"):
        self.device = device
        # Create a proper req_to_token_pool with the req_to_token attribute
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": 160,  # a typical max_bs * max_context_len for cuda graph decode
                "req_to_token": torch.zeros(
                    160, 2048, dtype=torch.int32, device=device
                ),  # Add req_to_token attribute
            },
        )


class MockReqToTokenPool:
    def __init__(self, batch_size, seq_len, device):
        self.req_to_token = (
            torch.arange(batch_size * seq_len, device=device)
            .reshape(batch_size, seq_len)
            .to(torch.int32)
        )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFlashAttentionBackend(CustomTestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model_runner = MockModelRunner()
        self.backend = FlashAttentionBackend(self.model_runner)

        # Common test parameters
        self.batch_size = 2
        self.seq_len = 4
        self.num_heads = 2
        self.head_dim = 8
        self.device = "cuda"
        self.dtype = torch.float16

    def _create_attention_layer(self):
        """Helper method to create an attention layer."""
        return RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_heads,
            layer_id=0,
        )

    def _create_kv_pool(self, size):
        """Helper method to create a KV pool."""
        return MHATokenToKVPool(
            size=size,
            page_size=1,  # only consider page=1 for unit test
            dtype=self.dtype,
            head_num=self.num_heads,
            head_dim=self.head_dim,
            layer_num=1,  # only consider layer=1 for unit test
            device=self.device,
            enable_memory_saver=False,
        )

    def _create_qkv_tensors(self, tokens_len):
        """Helper method to create q, k, v tensors."""
        return (
            torch.randn(
                tokens_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            torch.randn(
                tokens_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            torch.randn(
                tokens_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            ),
        )

    def _verify_output(self, output, expected_shape):
        """Helper method to verify output."""
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {output.shape}",
        )
        self.assertEqual(output.dtype, self.dtype)
        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(
            torch.isnan(output).sum().item(), 0, "Output contains NaN values"
        )

    def test_forward_extend(self):
        """Test the standard extend operation."""
        # Create test inputs
        q, k, v = self._create_qkv_tensors(self.batch_size * self.seq_len)

        # Create attention layer
        layer = self._create_attention_layer()

        # Create forward batch
        forward_batch = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(
                0, 100, (self.batch_size, self.seq_len), device=self.device
            ),
            out_cache_loc=torch.arange(
                self.batch_size * self.seq_len, device=self.device
            ),
            seq_lens_sum=self.batch_size * self.seq_len,
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=torch.tensor([self.seq_len] * self.batch_size, device=self.device),
            # 0 prefix, 4 extend
            extend_prefix_lens=torch.tensor([0] * self.batch_size, device=self.device),
            extend_seq_lens=torch.tensor([4] * self.batch_size, device=self.device),
            attn_backend=self.backend,
        )

        # Add token pool and KV cache
        forward_batch.req_to_token_pool = MockReqToTokenPool(
            self.batch_size, self.seq_len, self.device
        )
        forward_batch.token_to_kv_pool = self._create_kv_pool(
            self.batch_size * self.seq_len
        )

        # Initialize forward metadata before running the attention
        self.backend.init_forward_metadata(forward_batch)

        # Run forward_extend
        output = self.backend.forward_extend(q, k, v, layer, forward_batch)

        # Verify output
        expected_shape = (
            self.batch_size * self.seq_len,
            self.num_heads * self.head_dim,
        )
        self._verify_output(output, expected_shape)

    def test_forward_decode(self):
        """Test the decode operation with cached tokens."""
        # For decode, we only have one token per sequence
        decode_len = 1
        curr_seq_len = self.seq_len + decode_len

        # Create test inputs
        q, k, v = self._create_qkv_tensors(self.batch_size * decode_len)

        # Create attention layer
        layer = self._create_attention_layer()

        # Create forward batch
        forward_batch = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(
                0, 100, (self.batch_size, decode_len), device=self.device
            ),
            out_cache_loc=torch.arange(
                self.batch_size * self.seq_len,
                self.batch_size * curr_seq_len,
                device=self.device,
            ),
            seq_lens_sum=self.batch_size * curr_seq_len,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=torch.tensor([curr_seq_len] * self.batch_size, device=self.device),
            attn_backend=self.backend,
        )

        # Add token pool and KV cache
        forward_batch.req_to_token_pool = MockReqToTokenPool(
            self.batch_size, curr_seq_len, self.device
        )
        forward_batch.token_to_kv_pool = self._create_kv_pool(
            self.batch_size * curr_seq_len
        )

        # Pre-fill KV cache
        cache_k, cache_v, _ = self._create_qkv_tensors(self.batch_size * self.seq_len)
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            torch.arange(self.batch_size * self.seq_len, device=self.device),
            cache_k,
            cache_v,
            layer.k_scale,
            layer.v_scale,
        )

        # Initialize forward metadata before running the attention
        self.backend.init_forward_metadata(forward_batch)

        # Run forward_decode
        output = self.backend.forward_decode(q, k, v, layer, forward_batch)

        # Verify output
        expected_shape = (self.batch_size, self.num_heads * self.head_dim)
        self._verify_output(output, expected_shape)

    def test_forward_extend_with_prefix(self):
        """Test extending from cached prefix tokens."""
        # Define prefix and extend lengths
        prefix_len = 2
        extend_len = 2
        total_len = prefix_len + extend_len

        # Create test inputs for the extend portion
        q, k, v = self._create_qkv_tensors(self.batch_size * extend_len)

        # Create attention layer
        layer = self._create_attention_layer()

        # Create forward batch
        forward_batch = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(
                0, 100, (self.batch_size, extend_len), device=self.device
            ),
            out_cache_loc=torch.arange(
                self.batch_size * prefix_len,
                self.batch_size * total_len,
                device=self.device,
            ),
            seq_lens_sum=self.batch_size * total_len,
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=torch.tensor([total_len] * self.batch_size, device=self.device),
            extend_prefix_lens=torch.tensor(
                [prefix_len] * self.batch_size, device=self.device
            ),
            extend_seq_lens=torch.tensor(
                [extend_len] * self.batch_size, device=self.device
            ),
            attn_backend=self.backend,
        )

        # Add token pool and KV cache
        forward_batch.req_to_token_pool = MockReqToTokenPool(
            self.batch_size, total_len, self.device
        )
        forward_batch.token_to_kv_pool = self._create_kv_pool(
            self.batch_size * total_len
        )

        # Pre-fill the KV cache for prefix with known values
        cache_k = torch.ones(
            self.batch_size * prefix_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        cache_v = (
            torch.ones(
                self.batch_size * prefix_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            * 2
        )

        # Set the prefix KV cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            torch.arange(self.batch_size * prefix_len, device=self.device),
            cache_k,
            cache_v,
            layer.k_scale,
            layer.v_scale,
        )

        # Initialize forward metadata before running the attention
        self.backend.init_forward_metadata(forward_batch)

        # Run forward_extend
        output = self.backend.forward_extend(q, k, v, layer, forward_batch)

        # Verify output
        expected_shape = (self.batch_size * extend_len, self.num_heads * self.head_dim)
        self._verify_output(output, expected_shape)


if __name__ == "__main__":
    unittest.main()
