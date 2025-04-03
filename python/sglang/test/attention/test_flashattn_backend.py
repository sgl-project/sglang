import unittest

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import CustomTestCase


class MockModelRunner:
    def __init__(
        self,
        attention_arch=AttentionArch.MHA,
        context_len=2048,
        page_size=1,
        is_multimodal=False,
        device="cuda",
    ):
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": context_len,
                "is_multimodal": is_multimodal,
                "attention_arch": attention_arch,
            },
        )
        self.sliding_window_size = None
        self.device = device
        batch_size = 160
        # Create a proper req_to_token_pool with the req_to_token attribute
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                # A typical max_bs * max_context_len for cuda graph decode
                "size": batch_size,
                # Add req_to_token attribute
                "req_to_token": torch.zeros(
                    batch_size, context_len, dtype=torch.int32, device=device
                ),
            },
        )
        self.page_size = page_size
        from sglang.srt.model_executor.model_runner import ServerArgs

        self.server_args = ServerArgs(model_path="fake_model_path")


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
        # Test parameters
        self.batch_size = 2
        self.seq_len = 256
        self.num_heads = 2
        self.head_dim = 8
        self.device = "cuda"
        self.dtype = torch.float16

        # Initialize model runner and backend
        self.model_runner = MockModelRunner(attention_arch=AttentionArch.MHA)
        self.backend = FlashAttentionBackend(self.model_runner)

        # Reference backend for comparison
        self.ref_backend = TorchNativeAttnBackend(self.model_runner)

        # Set number of heads for reference backend comparison
        self.model_runner.model_config.num_attention_heads = self.num_heads

    def _create_attention_layer(self):
        """Create attention layer for testing."""
        return RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_heads,
            layer_id=0,
        )

    def _create_kv_pool(self, size, page_size=1):
        """Create KV pool for testing."""
        return MHATokenToKVPool(
            size=size,
            page_size=page_size,
            dtype=self.dtype,
            head_num=self.num_heads,
            head_dim=self.head_dim,
            layer_num=1,  # only consider layer=1 for unit test
            device=self.device,
            enable_memory_saver=False,
        )

    def _create_qkv_tensors(self, tokens_len):
        """Create q, k, v tensors for testing."""
        shape = (tokens_len, self.num_heads, self.head_dim)
        return (
            torch.randn(shape, dtype=self.dtype, device=self.device),
            torch.randn(shape, dtype=self.dtype, device=self.device),
            torch.randn(shape, dtype=self.dtype, device=self.device),
        )

    def _run_reference_forward(
        self, mode, q, k, v, layer, forward_batch, expected_shape
    ):
        """Run reference forward pass using native backend."""
        if mode == ForwardMode.EXTEND:
            output = self.ref_backend.forward_extend(q, k, v, layer, forward_batch)
        else:  # ForwardMode.DECODE
            output = self.ref_backend.forward_decode(q, k, v, layer, forward_batch)
        return output.view(expected_shape)

    def _verify_output(self, output, expected_shape, output_ref=None):
        """Verify output tensor shape, dtype, and values."""
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

        if output_ref is not None:
            if not torch.allclose(output, output_ref, atol=1e-1, rtol=0.0):
                diff = torch.abs(output - output_ref)
                max_diff = torch.max(diff)
                max_idx = torch.argmax(diff.view(-1))
                flat_output = output.view(-1)
                flat_ref = output_ref.view(-1)
                print(
                    f"Output is not close to the reference output. Max diff: {max_diff}"
                )
                print(
                    f"Max diff at index {max_idx}: output={flat_output[max_idx]}, reference={flat_ref[max_idx]}"
                )
                raise AssertionError(
                    "Attention output is not close to the torch native backend output"
                )

    def _create_forward_batch(self, mode, q_len=None, prefix_len=0, page_size=1):
        """Create a forward batch for testing based on mode and lengths."""
        self.model_runner.page_size = page_size

        # Default to self.seq_len if not specified
        q_len = q_len or self.seq_len

        if mode == ForwardMode.EXTEND:
            total_len = prefix_len + q_len
            out_cache_start = prefix_len * self.batch_size
            out_cache_end = total_len * self.batch_size

            forward_batch = ForwardBatch(
                batch_size=self.batch_size,
                input_ids=torch.randint(
                    0, 100, (self.batch_size, q_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    out_cache_start, out_cache_end, device=self.device
                ),
                seq_lens_sum=self.batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(self.batch_size, device=self.device),
                seq_lens=torch.tensor(
                    [total_len] * self.batch_size, device=self.device
                ),
                seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
                extend_prefix_lens=torch.tensor(
                    [prefix_len] * self.batch_size, device=self.device
                ),
                extend_prefix_lens_cpu=torch.tensor(
                    [prefix_len] * self.batch_size, device="cpu"
                ),
                extend_seq_lens=torch.tensor(
                    [q_len] * self.batch_size, device=self.device
                ),
                extend_seq_lens_cpu=torch.tensor(
                    [q_len] * self.batch_size, device="cpu"
                ),
                attn_backend=self.backend,
            )
            kv_pool_size = self.batch_size * total_len

        else:  # ForwardMode.DECODE
            decode_len = q_len  # typically 1 for decode mode
            total_len = self.seq_len + decode_len
            out_cache_start = self.batch_size * self.seq_len
            out_cache_end = self.batch_size * total_len

            forward_batch = ForwardBatch(
                batch_size=self.batch_size,
                input_ids=torch.randint(
                    0, 100, (self.batch_size, decode_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    out_cache_start, out_cache_end, device=self.device
                ),
                seq_lens_sum=self.batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(self.batch_size, device=self.device),
                seq_lens=torch.tensor(
                    [total_len] * self.batch_size, device=self.device
                ),
                seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
                attn_backend=self.backend,
            )

            # Calculate KV pool size divisible by page_size
            if page_size > 1:
                kv_pool_size = self.batch_size * ((total_len // page_size) * page_size)
                if kv_pool_size < total_len * self.batch_size:
                    kv_pool_size += page_size
            else:
                kv_pool_size = self.batch_size * total_len

        # Add token pool
        forward_batch.req_to_token_pool = MockReqToTokenPool(
            self.batch_size, total_len, self.device
        )

        # Add KV cache
        forward_batch.token_to_kv_pool = self._create_kv_pool(
            kv_pool_size, page_size=page_size
        )

        return forward_batch

    def _setup_kv_cache(self, forward_batch, layer, cache_len):
        """Set up KV cache with prefix tokens."""
        if cache_len <= 0:
            return

        # Create constant values for the prefix cache for easy debugging
        cache_k = torch.ones(
            self.batch_size * cache_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        cache_v = (
            torch.ones(
                self.batch_size * cache_len,
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
            torch.arange(self.batch_size * cache_len, device=self.device),
            cache_k,
            cache_v,
            layer.k_scale,
            layer.v_scale,
        )

    def _run_attention_test(self, mode, q_len, prefix_len=0, page_size=1):
        """
            Run an attention test with the specified parameters.
        Args:
            mode: ForwardMode.EXTEND or ForwardMode.DECODE
            q_len: Length of the query sequence. For decode mode, q_len is 1.
            prefix_len: Length of the prefix sequence for extend mode
            page_size: Page size for the KV cache
        """
        layer = self._create_attention_layer()

        # Create forward batch and set up
        forward_batch = self._create_forward_batch(mode, q_len, prefix_len, page_size)

        # Create QKV tensors for the input
        q, k, v = self._create_qkv_tensors(self.batch_size * q_len)

        # KV cache for prefixed extend is prefix_len
        # KV cache for decode is same as seq_len
        # No KV cache for extend without prefix
        if mode == ForwardMode.EXTEND:
            self._setup_kv_cache(forward_batch, layer, prefix_len)
        else:
            self._setup_kv_cache(forward_batch, layer, self.seq_len)

        self.backend.init_forward_metadata(forward_batch)

        if mode == ForwardMode.EXTEND:
            expected_shape = (
                self.batch_size * q_len,
                self.num_heads * self.head_dim,
            )
        else:
            expected_shape = (self.batch_size, self.num_heads * self.head_dim)

        if mode == ForwardMode.EXTEND:
            output = self.backend.forward_extend(q, k, v, layer, forward_batch)
        else:
            output = self.backend.forward_decode(q, k, v, layer, forward_batch)

        output_ref = self._run_reference_forward(
            mode, q, k, v, layer, forward_batch, expected_shape
        )

        self._verify_output(output, expected_shape, output_ref)

        return output

    def test_forward_extend(self):
        """Test the standard extend operation."""
        self._run_attention_test(ForwardMode.EXTEND, q_len=self.seq_len)

    def test_forward_decode(self):
        """Test the decode operation with cached tokens."""
        self._run_attention_test(ForwardMode.DECODE, q_len=1)

    def test_forward_extend_with_prefix(self):
        """Test extending from cached prefix tokens."""
        prefix_len = self.seq_len // 2
        extend_len = self.seq_len - prefix_len
        self._run_attention_test(
            ForwardMode.EXTEND, q_len=extend_len, prefix_len=prefix_len
        )

    def test_forward_extend_with_page_size_greater_than_1(self):
        """Test extending from cached prefix tokens with page size greater than 1."""
        self._run_attention_test(ForwardMode.EXTEND, q_len=self.seq_len, page_size=64)

    def test_forward_decode_with_page_size_greater_than_1(self):
        """Test decode operation with page size greater than 1."""
        self._run_attention_test(ForwardMode.DECODE, q_len=1, page_size=64)


if __name__ == "__main__":
    unittest.main()
