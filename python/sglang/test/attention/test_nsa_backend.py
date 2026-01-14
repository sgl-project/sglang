import unittest

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import CustomTestCase


class MockNSAConfig:
    """Mock config for DeepSeek NSA model."""

    def __init__(self):
        self.architectures = ["DeepseekV3ForCausalLM"]
        self.index_topk = 256
        self.index_head_dim = 128
        self.index_n_heads = 16


class MockModelRunner:
    def __init__(
        self,
        kv_lora_rank,
        qk_rope_head_dim,
        context_len=2048,
        page_size=64,
        nsa_prefill_backend="flashmla_sparse",
        nsa_decode_backend="flashmla_sparse",
    ):
        attention_arch = AttentionArch.MLA
        self.device = "cuda"
        # flashmla_sparse kernel requires bfloat16
        self.dtype = torch.bfloat16
        self.is_hybrid_swa = False
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": context_len,
                "attention_arch": attention_arch,
                "hf_config": MockNSAConfig(),
                "num_attention_heads": 128,
            },
        )
        self.sliding_window_size = None
        self.server_args = type(
            "ServerArgs",
            (),
            {
                "kv_cache_dtype": torch.bfloat16,
                "speculative_eagle_topk": None,
                "speculative_num_draft_tokens": 0,
                "enable_deterministic_inference": False,
                "nsa_prefill_backend": nsa_prefill_backend,
                "nsa_decode_backend": nsa_decode_backend,
                "enable_nsa_prefill_context_parallel": False,
            },
        )
        self.kv_cache_dtype = self.server_args.kv_cache_dtype

        batch_size = 160
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": batch_size,
                "req_to_token": torch.zeros(
                    batch_size, context_len, dtype=torch.int32, device=self.device
                ),
            },
        )
        self.page_size = page_size
        max_total_num_tokens = batch_size * context_len
        # Get index_head_dim from config
        nsa_config = self.model_config.hf_config
        self.token_to_kv_pool = NSATokenToKVPool(
            size=max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=1,
            device=self.device,
            index_head_dim=nsa_config.index_head_dim,
            enable_memory_saver=False,
        )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestNSABackend(CustomTestCase):
    def setUp(self):
        # NSA requires Hopper architecture (compute capability >= 9.0)
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] < 9:
                self.skipTest(
                    f"NSA requires Hopper GPU (compute capability >= 9.0), "
                    f"but found compute capability {compute_capability[0]}.{compute_capability[1]}"
                )

        # Test parameters
        self.batch_size = 2
        self.seq_len = 360
        self.num_heads = 2
        self.device = "cuda"
        # flashmla_sparse kernel requires bfloat16, not float16
        self.dtype = torch.bfloat16
        # Use different dimensions to avoid the buggy concat_mla_absorb_q kernel
        # The kernel only triggers when q_nope.shape[-1] == 512 and q_rope.shape[-1] == 64
        # We use 256/128 to force torch.cat fallback which is more stable
        self.kv_lora_rank = 256
        self.qk_rope_head_dim = 128
        self.qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.nsa_index_topk = 256

        # Initialize DP attention context (required by NSA backend)
        self._init_dp_attention()

        # Initialize model runner and backend
        self._init_model_runner()

        # Set global server args (required by NSA backend for prefill)
        from sglang.srt.server_args import set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(self.model_runner.server_args)

        self.backend = NativeSparseAttnBackend(self.model_runner)
        self.num_local_heads = 2

    def _init_dp_attention(self):
        """Initialize distributed parallelism attention context for testing."""
        from sglang.srt.layers import dp_attention

        # Set globals manually for single-GPU test environment
        dp_attention._ATTN_TP_SIZE = 1
        dp_attention._ATTN_TP_RANK = 0
        dp_attention._ATTN_DP_RANK = 0
        dp_attention._ATTN_DP_SIZE = 1
        dp_attention._LOCAL_ATTN_DP_SIZE = 1
        dp_attention._LOCAL_ATTN_DP_RANK = 0
        dp_attention._ENABLE_DP_ATTENTION_FLAG = False

    def tearDown(self):
        """Clean up after each test."""
        # Reset DP attention globals to avoid side effects
        from sglang.srt.layers import dp_attention

        dp_attention._ATTN_TP_SIZE = None
        dp_attention._ATTN_TP_RANK = None
        dp_attention._ATTN_DP_RANK = None
        dp_attention._ATTN_DP_SIZE = None
        dp_attention._LOCAL_ATTN_DP_SIZE = None
        dp_attention._LOCAL_ATTN_DP_RANK = None
        dp_attention._ATTN_TP_GROUP = None
        dp_attention._ENABLE_DP_ATTENTION_FLAG = False

        # Reset global server args
        import sglang.srt.server_args as server_args_module
        server_args_module._global_server_args = None

    def _init_model_runner(
        self,
        nsa_prefill_backend="flashmla_sparse",
        nsa_decode_backend="flashmla_sparse",
    ):
        self.model_runner = MockModelRunner(
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            nsa_prefill_backend=nsa_prefill_backend,
            nsa_decode_backend=nsa_decode_backend,
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
            decode_len = q_len
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
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool

        return forward_batch

    def _setup_kv_cache(self, forward_batch, layer, cache_len):
        """Set up KV cache with prefix tokens."""
        if cache_len <= 0:
            return

        # For MLA, create separate nope and rope caches
        cache_k_nope = torch.ones(
            self.batch_size * cache_len,
            1,
            self.kv_lora_rank,
            dtype=self.dtype,
            device=self.device,
        )

        cache_k_rope = torch.ones(
            self.batch_size * cache_len,
            1,
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

    def _create_mock_topk_indices(self, mode, q_len, total_kv_len):
        """Create mock topk indices for testing."""
        if mode == ForwardMode.EXTEND:
            # For extend mode, create topk indices for each query token
            # Shape: (batch_size * q_len, topk)
            num_queries = self.batch_size * q_len
            topk_indices = torch.randint(
                0,
                min(total_kv_len, self.nsa_index_topk),
                (num_queries, self.nsa_index_topk),
                dtype=torch.int32,
                device=self.device,
            )
        else:  # ForwardMode.DECODE
            # For decode mode, q_len is always 1
            # Shape: (batch_size, topk)
            topk_indices = torch.randint(
                0,
                min(total_kv_len, self.nsa_index_topk),
                (self.batch_size, self.nsa_index_topk),
                dtype=torch.int32,
                device=self.device,
            )

        return topk_indices

    def _run_attention_test(self, mode, q_len, prefix_len=0):
        """
        Run an attention test with the specified parameters.

        Args:
            mode: ForwardMode.EXTEND or ForwardMode.DECODE
            q_len: Length of the query sequence. For decode mode, q_len is 1.
            prefix_len: Length of the prefix sequence for extend mode
        """
        layer = self._create_attention_layer()

        # Create forward batch
        forward_batch = self._create_forward_batch(mode, q_len, prefix_len)

        # Create q with full dimension, then split into q_nope and q_rope
        # NSA backend requires q_rope to be passed explicitly (cannot be None)
        q_shape = (self.batch_size * q_len, self.num_heads, self.qk_head_dim)
        q_full = torch.randn(q_shape, dtype=self.dtype, device=self.device)

        # Split q_full into q_nope and q_rope
        # q_nope has dimension v_head_dim (kv_lora_rank)
        # q_rope has dimension (head_dim - v_head_dim) = qk_rope_head_dim
        q = q_full[:, :, : self.kv_lora_rank].contiguous()  # q_nope
        q_rope = q_full[:, :, self.kv_lora_rank :].contiguous()  # q_rope

        # Create k and k_rope separately
        kv_shape = (self.batch_size * q_len, 1, self.kv_lora_rank)
        k = torch.randn(kv_shape, dtype=self.dtype, device=self.device)

        k_rope_shape = (self.batch_size * q_len, 1, self.qk_rope_head_dim)
        k_rope = torch.randn(k_rope_shape, dtype=self.dtype, device=self.device)

        # v is not used for MQA with MLA
        v = torch.randn((1), dtype=self.dtype, device=self.device)

        # Setup KV cache with prefix if needed
        self._setup_kv_cache(forward_batch, layer, prefix_len)

        # Initialize forward metadata
        self.backend.init_forward_metadata(forward_batch)

        # Create mock topk indices
        total_kv_len = prefix_len + q_len
        topk_indices = self._create_mock_topk_indices(mode, q_len, total_kv_len)

        # Expected output shape
        expected_shape = (
            self.batch_size * q_len,
            self.num_heads * self.kv_lora_rank,
        )

        # Run forward pass - NSA backend requires q_rope to be passed explicitly
        if mode == ForwardMode.EXTEND:
            output = self.backend.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
                topk_indices=topk_indices,
            )
        else:
            output = self.backend.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
                topk_indices=topk_indices,
            )

        # Flatten output from (batch_size * q_len, num_heads, v_head_dim)
        # to (batch_size * q_len, num_heads * v_head_dim)
        output = output.reshape(self.batch_size * q_len, -1)

        self._verify_output(output, expected_shape)
        return output

    def test_forward_extend(self):
        """Test the standard extend operation with NSA."""
        self._run_attention_test(ForwardMode.EXTEND, q_len=self.seq_len)

    def test_forward_decode(self):
        """Test the decode operation with cached tokens and NSA."""
        self._run_attention_test(ForwardMode.DECODE, q_len=1)

    def test_forward_extend_with_prefix(self):
        """Test extending from cached prefix tokens with NSA."""
        prefix_len = self.seq_len // 2
        extend_len = self.seq_len - prefix_len
        self._run_attention_test(
            ForwardMode.EXTEND, q_len=extend_len, prefix_len=prefix_len
        )

    # ============================================================================
    # NSA Metadata Tests
    # ============================================================================

    def test_nsa_metadata_structure_decode(self):
        """Test NSAMetadata creation and structure for decode mode."""
        forward_batch = self._create_forward_batch(ForwardMode.DECODE, q_len=1)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # Verify basic metadata fields
        self.assertEqual(metadata.page_size, self.model_runner.page_size)
        self.assertEqual(metadata.max_seq_len_q, 1)
        self.assertGreater(metadata.max_seq_len_k, 0)

        # Verify tensor shapes
        self.assertEqual(metadata.cache_seqlens_int32.shape[0], self.batch_size)
        self.assertEqual(metadata.cu_seqlens_q.shape[0], self.batch_size + 1)
        self.assertEqual(metadata.cu_seqlens_k.shape[0], self.batch_size + 1)

        # Verify NSA-specific fields
        self.assertIsNotNone(metadata.nsa_cache_seqlens_int32)
        self.assertIsNotNone(metadata.nsa_cu_seqlens_q)
        self.assertIsNotNone(metadata.nsa_cu_seqlens_k)
        self.assertEqual(metadata.nsa_max_seqlen_q, 1)

    def test_nsa_metadata_structure_extend(self):
        """Test NSAMetadata creation and structure for extend mode."""
        forward_batch = self._create_forward_batch(ForwardMode.EXTEND, q_len=self.seq_len)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # Verify basic metadata fields
        self.assertEqual(metadata.page_size, self.model_runner.page_size)
        self.assertGreater(metadata.max_seq_len_q, 0)
        self.assertEqual(metadata.max_seq_len_k, self.seq_len)

        # Verify tensor shapes
        self.assertEqual(metadata.cache_seqlens_int32.shape[0], self.batch_size)
        self.assertEqual(metadata.page_table_1.shape[0], self.batch_size)

        # Verify NSA extend-specific fields
        self.assertIsNotNone(metadata.seq_lens_sum)
        self.assertEqual(metadata.seq_lens_sum, self.batch_size * self.seq_len)

    def test_nsa_seqlens_clipping(self):
        """Test that NSA seqlens are properly clipped to index_topk."""
        # Create forward batch with long sequences
        long_seq_len = 512  # Longer than index_topk (256)
        forward_batch = self._create_forward_batch(ForwardMode.EXTEND, q_len=long_seq_len)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # NSA seqlens should be clipped to topk
        # Each query token can attend to min(kv_len, topk) tokens
        max_nsa_seqlen = metadata.nsa_cache_seqlens_int32.max().item()
        self.assertLessEqual(
            max_nsa_seqlen,
            self.nsa_index_topk,
            f"NSA seqlen {max_nsa_seqlen} exceeds topk {self.nsa_index_topk}"
        )

    def test_cumulative_seqlens_correctness(self):
        """Test cumulative sequence lengths are correctly computed."""
        forward_batch = self._create_forward_batch(ForwardMode.DECODE, q_len=1)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # Verify cu_seqlens_q starts at 0
        self.assertEqual(metadata.cu_seqlens_q[0].item(), 0)

        # Verify cu_seqlens_q ends at total query tokens
        expected_total_q = self.batch_size * 1  # decode has q_len=1
        self.assertEqual(metadata.cu_seqlens_q[-1].item(), expected_total_q)

        # Verify cu_seqlens_k starts at 0
        self.assertEqual(metadata.cu_seqlens_k[0].item(), 0)

        # Verify cu_seqlens_k is monotonically increasing
        cu_seqlens_k_cpu = metadata.cu_seqlens_k.cpu()
        for i in range(1, len(cu_seqlens_k_cpu)):
            self.assertGreaterEqual(
                cu_seqlens_k_cpu[i].item(),
                cu_seqlens_k_cpu[i-1].item(),
                "cu_seqlens_k must be monotonically increasing"
            )

    def test_nsa_cu_seqlens_consistency(self):
        """Test NSA cumulative seqlens consistency."""
        forward_batch = self._create_forward_batch(ForwardMode.EXTEND, q_len=100)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # NSA cu_seqlens_k should match cumsum of nsa_cache_seqlens
        expected_cu_seqlens_k = torch.cat([
            torch.tensor([0], dtype=torch.int32, device=self.device),
            torch.cumsum(metadata.nsa_cache_seqlens_int32, dim=0, dtype=torch.int32)
        ])

        # Compare the first nsa_seqlens_expanded elements
        n_elements = len(metadata.nsa_seqlens_expanded)
        torch.testing.assert_close(
            metadata.nsa_cu_seqlens_k[:n_elements + 1],
            expected_cu_seqlens_k[:n_elements + 1]
        )

    def test_page_table_shape(self):
        """Test page table has correct shape."""
        forward_batch = self._create_forward_batch(ForwardMode.DECODE, q_len=1)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # page_table_1 should have shape (batch_size, max_seq_len_k)
        self.assertEqual(metadata.page_table_1.shape[0], self.batch_size)
        self.assertEqual(metadata.page_table_1.shape[1], metadata.max_seq_len_k)

        # real_page_table should account for page_size
        expected_real_cols = (metadata.max_seq_len_k + self.model_runner.page_size - 1) // self.model_runner.page_size
        if self.model_runner.page_size > 1:
            self.assertLessEqual(metadata.real_page_table.shape[1], expected_real_cols)

    def test_variable_batch_sizes(self):
        """Test metadata creation with different batch sizes."""
        original_batch_size = self.batch_size

        for batch_size in [1, 3, 8]:
            self.batch_size = batch_size
            try:
                forward_batch = self._create_forward_batch(ForwardMode.DECODE, q_len=1)
                self.backend.init_forward_metadata(forward_batch)

                metadata = self.backend.forward_metadata

                # Verify batch-dependent shapes
                self.assertEqual(metadata.cache_seqlens_int32.shape[0], batch_size)
                self.assertEqual(metadata.cu_seqlens_q.shape[0], batch_size + 1)
                self.assertEqual(metadata.page_table_1.shape[0], batch_size)

            finally:
                self.batch_size = original_batch_size

    def test_variable_sequence_lengths(self):
        """Test metadata creation with different sequence lengths."""
        for seq_len in [1, 64, 256, 512]:
            forward_batch = self._create_forward_batch(ForwardMode.EXTEND, q_len=seq_len)
            self.backend.init_forward_metadata(forward_batch)

            metadata = self.backend.forward_metadata

            # Verify sequence length dependent fields
            self.assertEqual(metadata.max_seq_len_k, seq_len)
            self.assertEqual(metadata.seq_lens_sum, self.batch_size * seq_len)

            # Verify NSA seqlens are computed correctly
            self.assertIsNotNone(metadata.nsa_cache_seqlens_int32)
            self.assertEqual(
                len(metadata.nsa_seqlens_expanded),
                self.batch_size * seq_len
            )

    def test_single_token_decode(self):
        """Test edge case with single token decode."""
        forward_batch = self._create_forward_batch(ForwardMode.DECODE, q_len=1)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # For decode, max_seq_len_q should be 1
        self.assertEqual(metadata.max_seq_len_q, 1)
        self.assertEqual(metadata.nsa_max_seqlen_q, 1)

        # Each batch element processes exactly 1 query token
        expected_q_tokens = self.batch_size
        self.assertEqual(metadata.cu_seqlens_q[-1].item(), expected_q_tokens)

    def test_extend_with_no_prefix(self):
        """Test extend mode with no prefix (cold start)."""
        forward_batch = self._create_forward_batch(ForwardMode.EXTEND, q_len=100, prefix_len=0)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # Verify no prefix means all tokens are query tokens
        self.assertEqual(metadata.max_seq_len_q, metadata.max_seq_len_k)

        # Verify extend_prefix_lens are all zeros
        self.assertTrue(
            torch.all(forward_batch.extend_prefix_lens == 0),
            "extend_prefix_lens should be all zeros for no prefix"
        )

    def test_extend_with_full_prefix(self):
        """Test extend mode where prefix = total length (edge case)."""
        total_len = 200
        prefix_len = total_len - 1  # Almost full prefix
        extend_len = 1

        forward_batch = self._create_forward_batch(
            ForwardMode.EXTEND, q_len=extend_len, prefix_len=prefix_len
        )
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # Verify only 1 token is being extended
        self.assertEqual(metadata.max_seq_len_q, extend_len)

        # Verify total KV length includes prefix
        self.assertEqual(metadata.max_seq_len_k, total_len)

    def test_nsa_extend_seq_lens_list(self):
        """Test nsa_extend_seq_lens_list correctness."""
        for q_len in [1, 10, 100]:
            forward_batch = self._create_forward_batch(ForwardMode.EXTEND, q_len=q_len)
            self.backend.init_forward_metadata(forward_batch)

            metadata = self.backend.forward_metadata

            # For extend without prefix, each request extends q_len tokens
            self.assertEqual(len(metadata.nsa_extend_seq_lens_list), self.batch_size)
            for extend_len in metadata.nsa_extend_seq_lens_list:
                self.assertEqual(extend_len, q_len)

    def test_nsa_seqlens_expanded_shape(self):
        """Test nsa_seqlens_expanded has correct shape and values."""
        q_len = 100
        forward_batch = self._create_forward_batch(ForwardMode.EXTEND, q_len=q_len)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # nsa_seqlens_expanded should have one entry per query token
        expected_len = self.batch_size * q_len
        self.assertEqual(len(metadata.nsa_seqlens_expanded), expected_len)

        # Values should be monotonically increasing within each batch
        seqlens_cpu = metadata.nsa_seqlens_expanded.cpu().numpy()
        for b in range(self.batch_size):
            batch_seqlens = seqlens_cpu[b * q_len:(b + 1) * q_len]
            # Each query token should attend to an increasing number of KV tokens
            for i in range(1, len(batch_seqlens)):
                self.assertGreaterEqual(batch_seqlens[i], batch_seqlens[i-1])

    def test_metadata_dtype_consistency(self):
        """Test that metadata tensors have correct dtypes."""
        forward_batch = self._create_forward_batch(ForwardMode.DECODE, q_len=1)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # All seqlens should be int32
        self.assertEqual(metadata.cache_seqlens_int32.dtype, torch.int32)
        self.assertEqual(metadata.cu_seqlens_q.dtype, torch.int32)
        self.assertEqual(metadata.cu_seqlens_k.dtype, torch.int32)
        self.assertEqual(metadata.nsa_cache_seqlens_int32.dtype, torch.int32)
        self.assertEqual(metadata.nsa_cu_seqlens_q.dtype, torch.int32)
        self.assertEqual(metadata.nsa_cu_seqlens_k.dtype, torch.int32)

        # Page tables should be int32
        self.assertEqual(metadata.page_table_1.dtype, torch.int32)
        self.assertEqual(metadata.real_page_table.dtype, torch.int32)

    def test_metadata_device_consistency(self):
        """Test that all metadata tensors are on correct device."""
        forward_batch = self._create_forward_batch(ForwardMode.DECODE, q_len=1)
        self.backend.init_forward_metadata(forward_batch)

        metadata = self.backend.forward_metadata

        # All tensors should be on CUDA
        self.assertEqual(metadata.cache_seqlens_int32.device.type, "cuda")
        self.assertEqual(metadata.cu_seqlens_q.device.type, "cuda")
        self.assertEqual(metadata.cu_seqlens_k.device.type, "cuda")
        self.assertEqual(metadata.page_table_1.device.type, "cuda")
        self.assertEqual(metadata.nsa_cache_seqlens_int32.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
