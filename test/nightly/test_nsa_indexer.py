import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, suite="nightly-1-gpu", nightly=True)

from sglang.srt.layers import dp_attention as _dp_attn

# Patch DP-attention globals before importing backends
_dp_attn.get_attention_tp_size = lambda: 1  # TP size = 1 for unit test

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.nsa.nsa_indexer import (
    BaseIndexerMetadata,
    Indexer,
    rotate_activation,
)
from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.linear import LinearBase
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

# Global configuration for all indexer tests
DEFAULT_CONFIG = {
    "device": "cuda",
    "dtype": torch.bfloat16,
    "kv_cache_dtype": torch.float8_e4m3fn,
    "context_len": 2048,
    "max_bs": 64,
    "hidden_size": 5120,
    "index_n_heads": 1,
    "index_head_dim": 128,
    "rope_head_dim": 64,
    "index_topk": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "max_position_embeddings": 163840,
    "rope_theta": 10000.0,
    "layer_id": 0,
    "page_size": 64,
}


class MockIndexerMetadata(BaseIndexerMetadata):
    """Mock implementation of BaseIndexerMetadata for testing."""

    def __init__(self, batch_size, seq_lens, page_table=None):
        self.batch_size = batch_size
        self.seq_lens = seq_lens
        self.page_table = page_table
        self.device = "cuda"

    def get_seqlens_int32(self) -> torch.Tensor:
        """Return: (batch_size,) int32 tensor"""
        return torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device)

    def get_page_table_64(self) -> torch.Tensor:
        """Return: (batch_size, num_blocks) int32, page table with page size 64."""
        if self.page_table is not None:
            return self.page_table
        # Create a simple page table for testing
        max_seq_len = max(self.seq_lens)
        num_blocks = (max_seq_len + 63) // 64  # Round up to page size 64
        page_table = torch.zeros(
            (self.batch_size, num_blocks), dtype=torch.int32, device=self.device
        )
        for i in range(self.batch_size):
            # Simple linear mapping: block i maps to page i
            num_blocks_needed = (self.seq_lens[i] + 63) // 64
            page_table[i, :num_blocks_needed] = torch.arange(
                num_blocks_needed, device=self.device
            )
        return page_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        """Return: (sum_extend_seq_len,) int32 tensor"""
        # For extend mode, each new token attends to progressively more tokens
        # For a sequence being extended from position 0 to seq_len, token i attends to i+1 tokens
        result = []
        for seq_len in self.seq_lens:
            result.extend(range(1, seq_len + 1))
        return torch.tensor(result, dtype=torch.int32, device=self.device)

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits.
        For testing, just return the topk indices.
        """
        return torch.topk(logits, k=topk, dim=-1).indices


class MockModelRunner:
    def __init__(self, config=None):
        self.device = "cuda"
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.dtype = self.config["dtype"]
        self.kv_cache_dtype = self.config["kv_cache_dtype"]
        self.is_hybrid_swa = False

        # Model configuration
        attention_arch = AttentionArch.MLA
        max_context_len = self.config["context_len"]
        max_batch_size = self.config["max_bs"]

        # Create mock hf_config for NSA - instantiate it as an object, not a type
        hf_config = type(
            "HfConfig",
            (),
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "index_topk": self.config["index_topk"],
                "index_head_dim": self.config["index_head_dim"],
                "index_n_heads": self.config["index_n_heads"],
            },
        )()

        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": max_context_len,
                "is_multimodal": False,
                "attention_arch": attention_arch,
                "num_attention_heads": 128,
                "kv_lora_rank": self.config["kv_lora_rank"],
                "qk_rope_head_dim": self.config["qk_rope_head_dim"],
                "hf_config": hf_config,
            },
        )()

        self.sliding_window_size = None
        self.page_size = self.config["page_size"]

        # Create req_to_token_pool
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": max_batch_size,
                "req_to_token": torch.zeros(
                    max_batch_size,
                    max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            },
        )()

        # Create NSATokenToKVPool
        max_total_num_tokens = max_batch_size * max_context_len
        self.token_to_kv_pool = NSATokenToKVPool(
            size=max_total_num_tokens,
            page_size=self.config["page_size"],
            dtype=self.config["kv_cache_dtype"],
            kv_lora_rank=self.config["kv_lora_rank"],
            qk_rope_head_dim=self.config["qk_rope_head_dim"],
            layer_num=1,
            device=self.device,
            index_head_dim=self.config["index_head_dim"],
            enable_memory_saver=False,
        )

        # Required by backend with NSA-specific attributes
        self.server_args = type(
            "ServerArgs",
            (),
            {
                "kv_cache_dtype": "auto",
                "speculative_eagle_topk": None,
                "speculative_num_draft_tokens": 0,
                "enable_deterministic_inference": False,
                "nsa_prefill_backend": "flashmla_sparse",
                "nsa_decode_backend": "fa3",
            },
        )()


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestNSAIndexer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        """Set up global server args for testing."""
        server_args = ServerArgs(model_path="dummy")
        server_args.enable_dp_attention = False
        server_args.nsa_prefill_backend = "flashmla_sparse"
        server_args.nsa_decode_backend = "flashmla_sparse"
        set_global_server_args_for_scheduler(server_args)

        # Check GPU capability for FP8
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            cls.supports_fp8 = compute_capability[0] >= 9  # Hopper or newer

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        pass

    def setUp(self):
        # Test parameters
        self.batch_size = 2
        self.seq_len = 128
        self.config = DEFAULT_CONFIG.copy()
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def _init_model_runner(self, config_override=None):
        """Initialize model runner with optional config override."""
        config = self.config.copy()
        if config_override:
            config.update(config_override)
        self.model_runner = MockModelRunner(config)
        self.backend = NativeSparseAttnBackend(self.model_runner)

    def _create_indexer(self, **kwargs):
        """Create an Indexer instance with default parameters."""
        params = {
            "hidden_size": self.config["hidden_size"],
            "index_n_heads": self.config["index_n_heads"],
            "index_head_dim": self.config["index_head_dim"],
            "rope_head_dim": self.config["rope_head_dim"],
            "index_topk": self.config["index_topk"],
            "q_lora_rank": self.config["q_lora_rank"],
            "max_position_embeddings": self.config["max_position_embeddings"],
            "rope_theta": self.config["rope_theta"],
            "layer_id": self.config["layer_id"],
            "scale_fmt": "ue8m0",
            "block_size": 128,
            "quant_config": None,  # No quantization for testing
        }
        params.update(kwargs)

        torch.set_default_dtype(self.dtype)
        indexer = Indexer(**params)
        # Move indexer to CUDA device
        indexer = indexer.to(device=self.device)

        # Convert linear layer weights to bfloat16 (but preserve LayerNorm's float32
        # and weights_proj's float32 - it uses params_dtype=torch.float32 in production)
        # Need to recursively convert LinearBase submodules (like ReplicatedLinear)
        for name, module in indexer.named_modules():
            # Check for LinearBase (parent of ReplicatedLinear) but exclude LayerNorm
            # Also exclude weights_proj which uses float32 params in production
            if isinstance(module, LinearBase) and not isinstance(module, LayerNorm):
                if "weights_proj" not in name:
                    module.to(dtype=self.dtype)

        return indexer

    def _create_forward_batch(
        self, mode, batch_size=None, seq_len=None, extend_len=None
    ):
        """Create a forward batch for testing."""
        batch_size = batch_size or self.batch_size
        seq_len = seq_len or self.seq_len

        if mode == ForwardMode.EXTEND:
            q_len = extend_len or seq_len
            total_len = seq_len

            forward_batch = ForwardBatch(
                batch_size=batch_size,
                input_ids=torch.randint(
                    0, 100, (batch_size, q_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    batch_size * (total_len - q_len),
                    batch_size * total_len,
                    device=self.device,
                ),
                seq_lens_sum=batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(batch_size, device=self.device),
                seq_lens=torch.tensor([total_len] * batch_size, device=self.device),
                seq_lens_cpu=torch.tensor([total_len] * batch_size, device="cpu"),
                extend_prefix_lens=torch.tensor(
                    [total_len - q_len] * batch_size, device=self.device
                ),
                extend_prefix_lens_cpu=torch.tensor(
                    [total_len - q_len] * batch_size, device="cpu"
                ),
                extend_seq_lens=torch.tensor([q_len] * batch_size, device=self.device),
                extend_seq_lens_cpu=torch.tensor([q_len] * batch_size, device="cpu"),
                attn_backend=self.backend,
            )
        else:  # ForwardMode.DECODE
            decode_len = 1
            total_len = seq_len + decode_len

            forward_batch = ForwardBatch(
                batch_size=batch_size,
                input_ids=torch.randint(
                    0, 100, (batch_size, decode_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    batch_size * seq_len, batch_size * total_len, device=self.device
                ),
                seq_lens_sum=batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(batch_size, device=self.device),
                seq_lens=torch.tensor([total_len] * batch_size, device=self.device),
                seq_lens_cpu=torch.tensor([total_len] * batch_size, device="cpu"),
                attn_backend=self.backend,
            )

        # Add token pools
        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool

        # Mock write to req_to_token_pool
        page_size = self.model_runner.page_size
        for i in range(batch_size):
            seq_length = total_len
            for j in range(seq_length):
                self.model_runner.req_to_token_pool.req_to_token[i, j] = (
                    i * seq_length + j + page_size
                )

        return forward_batch

    def _verify_topk_output(self, topk_indices, batch_size, q_len, topk):
        """Verify the topk indices output shape and basic properties."""
        self.assertIsNotNone(topk_indices)
        self.assertEqual(topk_indices.device.type, "cuda")

        # Check shape - should be (total_q_len, topk_padded)
        # where topk_padded is aligned to 2048
        self.assertEqual(len(topk_indices.shape), 2)
        self.assertEqual(topk_indices.shape[0], batch_size * q_len)

        # Check that topk is padded to at least topk
        self.assertGreaterEqual(topk_indices.shape[1], topk)

        # Check for padding values (-1)
        has_padding = (topk_indices == -1).any()
        self.assertTrue(
            has_padding or topk_indices.shape[1] == topk,
            "Output should have padding or exact topk size",
        )

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def test_indexer_basic_creation(self, mock_deep_gemm):
        """Test basic indexer creation and initialization."""
        mock_deep_gemm.get_num_sms.return_value = 132

        indexer = self._create_indexer()

        self.assertEqual(indexer.hidden_size, self.config["hidden_size"])
        self.assertEqual(indexer.n_heads, self.config["index_n_heads"])
        self.assertEqual(indexer.head_dim, self.config["index_head_dim"])
        self.assertEqual(indexer.rope_head_dim, self.config["rope_head_dim"])
        self.assertEqual(indexer.index_topk, self.config["index_topk"])
        self.assertEqual(indexer.layer_id, self.config["layer_id"])

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    @patch("sglang.srt.layers.attention.nsa.triton_kernel.act_quant")
    def test_forward_extend_mode(self, mock_act_quant, mock_deep_gemm):
        """Test indexer forward pass in extend mode."""
        if not self.supports_fp8:
            self.skipTest("FP8 requires Hopper GPU or newer")

        # Setup mocks
        mock_deep_gemm.get_num_sms.return_value = 132
        mock_deep_gemm.get_paged_mqa_logits_metadata.return_value = MagicMock()

        def mock_quant(x, *args, **kwargs):
            # Return FP8 tensor and scale
            return x.to(torch.float8_e4m3fn), torch.ones(
                x.shape[0], dtype=torch.float32, device=x.device
            )

        mock_act_quant.side_effect = mock_quant

        # Mock deep_gemm.fp8_mqa_logits to return logits (ragged path)
        def mock_mqa_logits(q, kv, weights, ks, ke, *args, **kwargs):
            # q shape: (sum_extend_seq_len, ...), return logits for each query token
            num_queries = q.shape[0]
            # kv is a tuple (k_fp8, k_scale), get total number of keys from k_fp8
            k_fp8, k_scale = kv
            max_kv_len = k_fp8.shape[0]  # Total keys across all batches (k_offset)
            return torch.randn(
                num_queries, max_kv_len, dtype=torch.float32, device="cuda"
            )

        mock_deep_gemm.fp8_mqa_logits.side_effect = mock_mqa_logits

        # Also mock the paged version for completeness
        def mock_paged_mqa_logits(q, kv, weights, *args, **kwargs):
            batch_size = q.shape[0]
            seq_len = 128
            return torch.randn(batch_size, seq_len, dtype=torch.float32, device="cuda")

        mock_deep_gemm.fp8_paged_mqa_logits.side_effect = mock_paged_mqa_logits

        self._init_model_runner()

        indexer = self._create_indexer()
        forward_batch = self._create_forward_batch(ForwardMode.EXTEND)

        # Create input tensors
        total_tokens = self.batch_size * self.seq_len
        hidden_states = torch.randn(
            total_tokens,
            self.config["hidden_size"],
            dtype=self.dtype,
            device=self.device,
        )
        q_lora = torch.randn(
            total_tokens,
            self.config["q_lora_rank"],
            dtype=self.dtype,
            device=self.device,
        )
        positions = torch.arange(total_tokens, device=self.device)

        # Run forward pass
        with patch.object(
            self.backend,
            "get_indexer_metadata",
            return_value=MockIndexerMetadata(
                self.batch_size, [self.seq_len] * self.batch_size
            ),
        ):
            topk_indices = indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.config["layer_id"],
            )

        # Verify output
        self._verify_topk_output(
            topk_indices, self.batch_size, self.seq_len, self.config["index_topk"]
        )

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    @patch("sglang.srt.layers.attention.nsa.triton_kernel.act_quant")
    def test_forward_decode_mode(self, mock_act_quant, mock_deep_gemm):
        """Test indexer forward pass in decode mode."""
        if not self.supports_fp8:
            self.skipTest("FP8 requires Hopper GPU or newer")

        # Setup mocks
        mock_deep_gemm.get_num_sms.return_value = 132
        mock_deep_gemm.get_paged_mqa_logits_metadata.return_value = MagicMock()

        def mock_quant(x, *args, **kwargs):
            return x.to(torch.float8_e4m3fn), torch.ones(
                x.shape[0], dtype=torch.float32, device=x.device
            )

        mock_act_quant.side_effect = mock_quant

        def mock_paged_mqa_logits(q, kv, weights, *args, **kwargs):
            batch_size = q.shape[0]
            seq_len = 128
            return torch.randn(batch_size, seq_len, dtype=torch.float32, device="cuda")

        mock_deep_gemm.fp8_paged_mqa_logits.side_effect = mock_paged_mqa_logits

        self._init_model_runner()

        indexer = self._create_indexer()
        forward_batch = self._create_forward_batch(ForwardMode.DECODE)

        # Create input tensors for decode (batch_size tokens only)
        hidden_states = torch.randn(
            self.batch_size,
            self.config["hidden_size"],
            dtype=self.dtype,
            device=self.device,
        )
        q_lora = torch.randn(
            self.batch_size,
            self.config["q_lora_rank"],
            dtype=self.dtype,
            device=self.device,
        )
        positions = torch.arange(self.batch_size, device=self.device)

        # Run forward pass
        with patch.object(
            self.backend,
            "get_indexer_metadata",
            return_value=MockIndexerMetadata(
                self.batch_size, [self.seq_len + 1] * self.batch_size
            ),
        ):
            topk_indices = indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.config["layer_id"],
            )

        # Verify output - decode mode has q_len=1
        self._verify_topk_output(
            topk_indices, self.batch_size, 1, self.config["index_topk"]
        )

    def test_rotate_activation(self):
        """Test the Hadamard transform (rotate_activation) function."""
        # Test with power-of-2 hidden size
        hidden_size = 128
        x = torch.randn(16, hidden_size, dtype=torch.bfloat16, device=self.device)

        try:
            output = rotate_activation(x)
            self.assertEqual(output.shape, x.shape)
            self.assertEqual(output.dtype, torch.bfloat16)
        except ImportError:
            self.skipTest("sgl_kernel not available for hadamard_transform")

    def test_rotate_activation_invalid_size(self):
        """Test that rotate_activation fails with non-power-of-2 size."""
        # Test with non-power-of-2 hidden size
        hidden_size = 129  # Not a power of 2
        x = torch.randn(16, hidden_size, dtype=torch.bfloat16, device=self.device)

        with self.assertRaises(AssertionError):
            rotate_activation(x)

    def test_indexer_metadata_interface(self):
        """Test the BaseIndexerMetadata interface implementation."""
        batch_size = 4
        seq_lens = [64, 128, 96, 112]

        metadata = MockIndexerMetadata(batch_size, seq_lens)

        # Test get_seqlens_int32
        seqlens = metadata.get_seqlens_int32()
        self.assertEqual(seqlens.shape, (batch_size,))
        self.assertEqual(seqlens.dtype, torch.int32)
        self.assertTrue(torch.all(seqlens == torch.tensor(seq_lens, device="cuda")))

        # Test get_page_table_64
        page_table = metadata.get_page_table_64()
        self.assertEqual(len(page_table.shape), 2)
        self.assertEqual(page_table.shape[0], batch_size)
        self.assertEqual(page_table.dtype, torch.int32)

        # Test topk_transform
        logits = torch.randn(batch_size, 128, device="cuda")
        topk = 64
        topk_indices = metadata.topk_transform(logits, topk)
        self.assertEqual(topk_indices.shape, (batch_size, topk))

    # TODO: enable this test after indexer accuracy aligned
    # @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    # def test_indexer_with_different_topk(self, mock_deep_gemm):
    #     """Test indexer with different topk values."""
    #     mock_deep_gemm.get_num_sms.return_value = 132

    #     for topk in [32, 64, 128]:
    #         with self.subTest(topk=topk):
    #             indexer = self._create_indexer(index_topk=topk)
    #             self.assertEqual(indexer.index_topk, topk)

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def test_indexer_with_fused_wk(self, mock_deep_gemm):
        """Test indexer creation with fused wk and weights projection."""
        mock_deep_gemm.get_num_sms.return_value = 132

        # Note: fuse_wk_and_weights_proj feature is not currently implemented
        # This test verifies basic indexer creation still works
        indexer = self._create_indexer()
        self.assertIsNotNone(indexer)

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def test_indexer_with_alt_stream(self, mock_deep_gemm):
        """Test indexer creation with alternative CUDA stream."""
        mock_deep_gemm.get_num_sms.return_value = 132

        alt_stream = torch.cuda.Stream()
        indexer = self._create_indexer(alt_stream=alt_stream)
        self.assertEqual(indexer.alt_stream, alt_stream)

    def test_shape_sanity_checks(self):
        """Test various shape combinations for consistency."""
        test_configs = [
            {"batch_size": 1, "seq_len": 64},
            {"batch_size": 4, "seq_len": 128},
            {"batch_size": 8, "seq_len": 256},
        ]

        for config in test_configs:
            with self.subTest(**config):
                batch_size = config["batch_size"]
                seq_len = config["seq_len"]

                # Test metadata shapes
                metadata = MockIndexerMetadata(batch_size, [seq_len] * batch_size)

                seqlens = metadata.get_seqlens_int32()
                self.assertEqual(seqlens.shape, (batch_size,))

                page_table = metadata.get_page_table_64()
                expected_blocks = (seq_len + 63) // 64
                self.assertEqual(page_table.shape[0], batch_size)
                self.assertGreaterEqual(page_table.shape[1], expected_blocks)


if __name__ == "__main__":
    unittest.main()
