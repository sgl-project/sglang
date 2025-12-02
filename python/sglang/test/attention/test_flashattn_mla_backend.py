import unittest

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import CustomTestCase


class MockModelRunner:
    def __init__(
        self,
        kv_lora_rank,
        qk_rope_head_dim,
    ):
        attention_arch = AttentionArch.MLA
        self.device = "cuda"
        self.dtype = torch.float16
        self.is_hybrid_swa = False
        context_len = 2048
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": context_len,
                "attention_arch": attention_arch,
            },
        )
        self.sliding_window_size = None
        # Add server_args attribute
        self.server_args = type(
            "ServerArgs",
            (),
            {
                "kv_cache_dtype": torch.float16,
                "speculative_eagle_topk": None,
                "speculative_num_draft_tokens": 0,
                "enable_deterministic_inference": False,
            },
        )
        self.kv_cache_dtype = self.server_args.kv_cache_dtype

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
                    batch_size, context_len, dtype=torch.int32, device=self.device
                ),
            },
        )
        self.page_size = 1
        max_total_num_tokens = batch_size * context_len
        self.token_to_kv_pool = MLATokenToKVPool(
            size=max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=1,  # only consider layer=1 for unit test
            device=self.device,
            enable_memory_saver=False,
        )


class MockReqToTokenPool:
    def __init__(self, batch_size, seq_len, device):
        self.req_to_token = (
            torch.arange(batch_size * seq_len, device=device)
            .reshape(batch_size, seq_len)
            .to(torch.int32)
        )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFlashAttentionMLABackend(CustomTestCase):
    def setUp(self):
        # MLA with different V headdim requires Hopper architecture (compute capability >= 9.0)
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] < 9:
                self.skipTest(
                    f"MLA requires Hopper GPU (compute capability >= 9.0), "
                    f"but found compute capability {compute_capability[0]}.{compute_capability[1]}"
                )

        # Test parameters
        self.batch_size = 2
        self.seq_len = 360
        self.num_heads = 2
        self.device = "cuda"
        self.dtype = torch.float16
        self.kv_lora_rank = 512
        self.q_lora_rank = 128
        self.qk_rope_head_dim = 64
        self.qk_head_dim = self.qk_rope_head_dim + self.kv_lora_rank
        # Assume no rope scaling
        self.scaling = self.qk_head_dim**-0.5
        # Initialize model runner and backend
        self._init_model_runner()
        self.backend = FlashAttentionBackend(self.model_runner)
        self.ref_backend = TorchNativeAttnBackend(self.model_runner)
        self.num_local_heads = 2

    def _init_model_runner(self):
        self.model_runner = MockModelRunner(
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )

    def _create_attention_layer(self):
        """Create attention layer for testing."""
        self.attn_mqa = RadixAttention(
            num_heads=self.num_local_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            scaling=self.scaling,
            num_kv_heads=1,
            layer_id=0,
            v_head_dim=self.kv_lora_rank,
            prefix="attn_mqa",
        )
        return self.attn_mqa

    def _run_reference_forward(
        self, mode, q, k, v, layer, forward_batch, expected_shape
    ):
        """Run reference forward pass using native backend."""
        if mode == ForwardMode.EXTEND:
            output = self.ref_backend.forward_extend(q, k, v, layer, forward_batch)
        else:  # ForwardMode.DECODE
            output = self.ref_backend.forward_decode(q, k, v, layer, forward_batch)
        return output.view(expected_shape)

    def _verify_output(self, output, expected_shape):
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

    def _create_forward_batch(self, mode, q_len=None, prefix_len=0):
        """Create a forward batch for testing based on mode and lengths."""
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

        # Add token pool from model runner to forward batch
        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool

        # Add KV cache from model runner to forward batch
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool

        return forward_batch

    def _setup_kv_cache(self, forward_batch, layer, cache_len):
        """Set up KV cache with prefix tokens."""
        if cache_len <= 0:
            return

        # For MLA, create separate nope and rope caches
        cache_k_nope = torch.ones(
            self.batch_size * cache_len,
            1,  # latent cache has only one head in MQA
            self.kv_lora_rank,
            dtype=self.dtype,
            device=self.device,
        )

        cache_k_rope = torch.ones(
            self.batch_size * cache_len,
            1,  # latent cache has only one head in MQA
            self.qk_rope_head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        # Set the prefix KV cache using MLA-specific method
        forward_batch.token_to_kv_pool.set_mla_kv_buffer(
            layer,
            torch.arange(self.batch_size * cache_len, device=self.device),
            cache_k_nope,
            cache_k_rope,
        )

    def _run_attention_test(self, mode, q_len, prefix_len=0):
        """
            Run an attention test with the specified parameters.
        Args:
            mode: ForwardMode.EXTEND or ForwardMode.DECODE
            q_len: Length of the query sequence. For decode mode, q_len is 1.
            prefix_len: Length of the prefix sequence for extend mode
        """
        layer = self._create_attention_layer()

        # Create forward batch and set up
        forward_batch = self._create_forward_batch(mode, q_len, prefix_len)

        # Create q, kv_compressed for testing
        q_shape = (self.batch_size * q_len, self.num_heads, self.qk_head_dim)
        kv_shape = (self.batch_size * q_len, self.qk_head_dim)
        q = torch.randn(q_shape, dtype=self.dtype, device=self.device)
        kv_compressed = torch.randn(kv_shape, dtype=self.dtype, device=self.device)

        # For MLA, split kv_compressed into k_nope and k_rope
        # k_nope has dimension kv_lora_rank, k_rope has dimension qk_rope_head_dim
        k_nope = kv_compressed[:, : self.kv_lora_rank]
        k_rope = kv_compressed[:, self.kv_lora_rank :]

        # k_nope needs to be unsqueezed for the num_heads dimension
        k = k_nope.unsqueeze(1)
        # k_rope also needs to be unsqueezed
        k_rope = k_rope.unsqueeze(1)

        # v is not used for mqa
        v = torch.randn((1), dtype=self.dtype, device=self.device)

        self._setup_kv_cache(forward_batch, layer, prefix_len)

        self.backend.init_forward_metadata(forward_batch)

        expected_shape = (
            self.batch_size * q_len,
            self.num_heads * self.kv_lora_rank,
        )

        if mode == ForwardMode.EXTEND:
            output = self.backend.forward_extend(
                q, k, v, layer, forward_batch, k_rope=k_rope
            )
        else:
            output = self.backend.forward_decode(
                q, k, v, layer, forward_batch, k_rope=k_rope
            )

        self._verify_output(output, expected_shape)
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


if __name__ == "__main__":
    unittest.main()
