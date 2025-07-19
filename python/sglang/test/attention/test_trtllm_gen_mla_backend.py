"""
Clean test suite for TRTLLM MLA backend.

This test file provides comprehensive testing for the TRTLLM MLA (Multi-Head Latent Attention) backend:

1. test_basic_functionality: Basic smoke test with minimal setup
2. test_decode_output_match: Compares TRTLLM MLA output against FlashInfer MLA reference 
   across different batch sizes and sequence lengths
3. test_different_page_sizes: Tests consistency across different page sizes
4. test_forward_decode_shape_sanity: Shape and sanity checks across various configurations

The tests use unittest with subTest for parameterized testing, following the sglang test patterns.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.layers import dp_attention as _dp_attn

# Patch DP-attention globals before importing backends
_dp_attn.get_attention_tp_size = lambda: 1  # TP size = 1 for unit test

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.trtllm_gen_mla_backend import TRTLLMGENMLABackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase


class MockModelRunner:
    """Minimal fake ModelRunner for testing MLA backends."""

    def __init__(self, page_size: int):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.kv_cache_dtype = torch.bfloat16
        self.page_size = page_size

        # Model-config stub with MLA attributes
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": 2048,
                "attention_arch": AttentionArch.MLA,
                "num_attention_heads": 128,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 512,
                "scaling": 1.0 / ((128 + 64) ** 0.5),
                "get_num_kv_heads": staticmethod(lambda _: 1),
            },
        )

        # Req-to-token pool
        max_bs = 64
        max_ctx = self.model_config.context_len
        req_to_token = torch.arange(
            max_bs * max_ctx, dtype=torch.int32, device=self.device
        ).reshape(max_bs, max_ctx)
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": max_bs,
                "req_to_token": req_to_token,
            },
        )

        # KV-token pool (MLA)
        self.token_to_kv_pool = MLATokenToKVPool(
            size=max_bs * max_ctx,
            page_size=page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
        )


def compare_outputs(trtllm_out, reference_out, tolerance=1e-2):
    """Compare outputs with detailed analysis."""

    # Basic checks
    assert (
        trtllm_out.shape == reference_out.shape
    ), f"Shape mismatch: {trtllm_out.shape} vs {reference_out.shape}"
    assert (
        trtllm_out.dtype == reference_out.dtype
    ), f"Dtype mismatch: {trtllm_out.dtype} vs {reference_out.dtype}"

    # Check for NaN/Inf
    assert not torch.isnan(trtllm_out).any(), "TRTLLM output contains NaN"
    assert not torch.isnan(reference_out).any(), "Reference output contains NaN"
    assert not torch.isinf(trtllm_out).any(), "TRTLLM output contains Inf"
    assert not torch.isinf(reference_out).any(), "Reference output contains Inf"

    # Element-wise differences
    diff = (trtllm_out - reference_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check numerical equivalence
    all_close = torch.allclose(
        trtllm_out, reference_out, rtol=tolerance, atol=tolerance
    )

    if not all_close:
        print(
            f"Comparison failed: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, tolerance={tolerance}"
        )
        # Find top differences for debugging
        flat_diff = diff.flatten()
        top_diff_indices = torch.topk(flat_diff, k=min(5, flat_diff.numel())).indices
        print("Top 5 differences:")
        for i, idx in enumerate(top_diff_indices):
            idx_tuple = np.unravel_index(idx.cpu().numpy(), trtllm_out.shape)
            trt_val = trtllm_out[idx_tuple].item()
            ref_val = reference_out[idx_tuple].item()
            print(
                f"  [{idx_tuple}]: TRTLLM={trt_val:.6f}, Reference={ref_val:.6f}, diff={abs(trt_val-ref_val):.6f}"
            )

    return all_close


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer required",
)
class TestTRTLLMMLAClean(CustomTestCase):
    """Test TRTLLM MLA backend against FlashInfer MLA backend (reference)."""

    def setUp(self):
        """Setup test fixtures."""
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.page_size = 32

        # Create model runner and backends
        self.model_runner_trtllm = MockModelRunner(self.page_size)
        self.model_runner_reference = MockModelRunner(self.page_size)

        self.trtllm_backend = TRTLLMGENMLABackend(self.model_runner_trtllm)
        self.reference_backend = FlashInferMLAAttnBackend(self.model_runner_reference)

        # Create RadixAttention layer
        self.layer = RadixAttention(
            num_heads=128,
            head_dim=512 + 64,  # kv_lora_rank + qk_rope_head_dim
            scaling=self.model_runner_trtllm.model_config.scaling,
            num_kv_heads=1,
            layer_id=0,
            v_head_dim=512,
            prefix="attn_mqa",
        )

    def _create_qkv_tensors(self, batch_size):
        """Create Q, K, V tensors for testing."""
        head_dim = 512 + 64  # kv_lora_rank + qk_rope_head_dim
        q = torch.randn(
            (batch_size, 128, head_dim), dtype=self.dtype, device=self.device
        )
        k = torch.randn((batch_size, 1, head_dim), dtype=self.dtype, device=self.device)
        v = torch.randn((batch_size, 1, 512), dtype=self.dtype, device=self.device)
        return q, k, v

    def _create_forward_batch(self, batch_size, seq_lens, backend, model_runner):
        """Create a forward batch for the given backend."""
        fb = ForwardBatch(
            batch_size=batch_size,
            input_ids=torch.randint(0, 100, (batch_size, 1), device=self.device),
            out_cache_loc=torch.arange(batch_size, device=self.device),
            seq_lens_sum=int(seq_lens.sum().item()),
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(batch_size, device=self.device),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            attn_backend=backend,
        )
        fb.req_to_token_pool = model_runner.req_to_token_pool
        fb.token_to_kv_pool = model_runner.token_to_kv_pool
        return fb

    def _populate_kv_cache(self, batch_size, seq_lens, model_runners):
        """Populate KV cache with identical data for both backends."""
        torch.manual_seed(42)  # Fixed seed for reproducible cache

        for model_runner in model_runners:
            torch.manual_seed(42)  # Reset seed for each backend
            for i in range(batch_size):
                seq_len = int(seq_lens[i].item())
                for token_idx in range(seq_len - 1):
                    # Create random K components for MLA
                    cache_k_nope = torch.randn(
                        (1, 128), dtype=self.dtype, device=self.device
                    )
                    cache_k_rope = torch.randn(
                        (1, 64), dtype=self.dtype, device=self.device
                    )

                    # Calculate cache location
                    cache_loc = model_runner.req_to_token_pool.req_to_token[
                        i, token_idx
                    ]

                    # Save to KV cache
                    model_runner.token_to_kv_pool.set_mla_kv_buffer(
                        self.layer,
                        cache_loc.unsqueeze(0),
                        cache_k_nope.squeeze(0),
                        cache_k_rope.squeeze(0),
                    )

    def test_decode_output_match(self):
        """Test that TRTLLM and FlashInfer MLA backends produce matching outputs."""
        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 64),
            (4, 64),
            (16, 64),
            (32, 64),
            (1, 128),
            (4, 128),
            (16, 128),
            (32, 128),
            (1, 256),
            (4, 256),
            (16, 256),
            (32, 256),
        ]

        for batch_size, max_seq_len in test_cases:
            with self.subTest(batch_size=batch_size, max_seq_len=max_seq_len):
                # Create identical sequence lengths for both backends
                torch.manual_seed(42)
                seq_lens = torch.randint(
                    1, max_seq_len, (batch_size,), device=self.device
                )
                seq_lens[0] = max_seq_len  # Ensure at least one max length

                # Create forward batches with identical inputs
                fb_trtllm = self._create_forward_batch(
                    batch_size,
                    seq_lens.clone(),
                    self.trtllm_backend,
                    self.model_runner_trtllm,
                )
                fb_reference = self._create_forward_batch(
                    batch_size,
                    seq_lens.clone(),
                    self.reference_backend,
                    self.model_runner_reference,
                )

                # Initialize metadata for both backends
                self.trtllm_backend.init_forward_metadata(fb_trtllm)
                self.reference_backend.init_forward_metadata(fb_reference)

                # Populate both KV caches identically
                self._populate_kv_cache(
                    batch_size,
                    seq_lens,
                    [self.model_runner_trtllm, self.model_runner_reference],
                )

                # Create Q, K, V tensors for current decode step
                torch.manual_seed(123)  # Fixed seed for Q, K, V
                q, k, v = self._create_qkv_tensors(batch_size)

                # Run forward decode on both backends
                out_trtllm = self.trtllm_backend.forward_decode(
                    q.clone(), k.clone(), v, self.layer, fb_trtllm
                )
                out_reference = self.reference_backend.forward_decode(
                    q.clone(), k.clone(), v.clone(), self.layer, fb_reference
                )

                # Compare outputs
                comparison_passed = compare_outputs(
                    out_trtllm, out_reference, tolerance=1e-2
                )

                self.assertTrue(
                    comparison_passed,
                    f"TRTLLM and Reference outputs differ beyond tolerance. "
                    f"batch_size={batch_size}, max_seq_len={max_seq_len}, "
                    f"Max diff: {(out_trtllm - out_reference).abs().max().item()}",
                )

    def test_different_page_sizes(self):
        """Test output consistency across different page sizes."""
        page_sizes = [32, 64]
        batch_size = 8
        max_seq_len = 128

        for page_size in page_sizes:
            with self.subTest(page_size=page_size):
                # Create model runner with specific page size
                model_runner = MockModelRunner(page_size)
                backend = TRTLLMGENMLABackend(model_runner)

                # Create sequence lengths
                torch.manual_seed(42)
                seq_lens = torch.randint(
                    1, max_seq_len, (batch_size,), device=self.device
                )
                seq_lens[0] = max_seq_len

                # Create forward batch
                fb = self._create_forward_batch(
                    batch_size, seq_lens, backend, model_runner
                )
                backend.init_forward_metadata(fb)

                # Populate KV cache
                self._populate_kv_cache(batch_size, seq_lens, [model_runner])

                # Create Q, K, V tensors
                torch.manual_seed(123)
                q, k, v = self._create_qkv_tensors(batch_size)

                # Run forward decode
                output = backend.forward_decode(q, k, v, self.layer, fb)

                # Basic checks
                expected_shape = (batch_size, 128 * 512)  # num_heads * v_head_dim
                self.assertEqual(
                    output.shape,
                    expected_shape,
                    f"Output shape mismatch: {output.shape} vs {expected_shape}",
                )
                self.assertFalse(torch.isnan(output).any(), "Output contains NaN")
                self.assertFalse(torch.isinf(output).any(), "Output contains Inf")

    def test_basic_functionality(self):
        """Test basic functionality with minimal setup."""
        batch_size = 2
        max_seq_len = 32

        # Create sequence lengths
        seq_lens = torch.tensor([max_seq_len, max_seq_len // 2], device=self.device)

        # Create forward batch
        fb = self._create_forward_batch(
            batch_size, seq_lens, self.trtllm_backend, self.model_runner_trtllm
        )
        self.trtllm_backend.init_forward_metadata(fb)

        # Populate KV cache
        self._populate_kv_cache(batch_size, seq_lens, [self.model_runner_trtllm])

        # Create Q, K, V tensors
        q, k, v = self._create_qkv_tensors(batch_size)

        # Run forward decode
        output = self.trtllm_backend.forward_decode(q, k, v, self.layer, fb)

        # Basic checks
        expected_shape = (batch_size, 128 * 512)  # num_heads * v_head_dim
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, self.dtype)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_forward_decode_shape_sanity(self):
        """Smoke test decode across several page sizes and batch configurations."""
        # Test configurations similar to the original test
        test_configs = [
            (16, 512, 32),  # batch_size, seq_len, page_size
            (16, 512, 64),
            (8, 256, 32),
            (4, 128, 32),
            (1, 64, 32),
            (32, 1024, 64),
        ]

        for batch_size, seq_len, page_size in test_configs:
            with self.subTest(
                batch_size=batch_size, seq_len=seq_len, page_size=page_size
            ):
                # Create model runner with specific page size
                model_runner = MockModelRunner(page_size)
                backend = TRTLLMGENMLABackend(model_runner)

                # Random seq lens (ensure one matches max)
                torch.manual_seed(42)
                seq_lens = torch.randint(1, seq_len, (batch_size,), device=self.device)
                seq_lens[0] = seq_len

                # Create forward batch
                fb = ForwardBatch(
                    batch_size=batch_size,
                    input_ids=torch.randint(
                        0, 100, (batch_size, 1), device=self.device
                    ),
                    out_cache_loc=torch.arange(batch_size, device=self.device),
                    seq_lens_sum=int(seq_lens.sum().item()),
                    forward_mode=ForwardMode.DECODE,
                    req_pool_indices=torch.arange(batch_size, device=self.device),
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens.cpu(),
                    attn_backend=backend,
                )
                fb.req_to_token_pool = model_runner.req_to_token_pool
                fb.token_to_kv_pool = model_runner.token_to_kv_pool

                backend.init_forward_metadata(fb)

                # Create Q, K, V tensors
                head_dim = 512 + 64  # kv_lora_rank + qk_rope_head_dim
                q = torch.randn(
                    (batch_size, 128, head_dim), dtype=self.dtype, device=self.device
                )
                k = torch.randn(
                    (batch_size, 1, head_dim), dtype=self.dtype, device=self.device
                )
                v = None  # TRTLLM MLA decode kernel ignores v

                # Create layer
                layer = RadixAttention(
                    num_heads=128,
                    head_dim=512 + 64,
                    scaling=model_runner.model_config.scaling,
                    num_kv_heads=1,
                    layer_id=0,
                    v_head_dim=512,
                    prefix="attn_mqa",
                )

                # Run forward decode
                output = backend.forward_decode(q, k, v, layer, fb)

                # Shape and sanity checks
                expected_shape = (batch_size, 128 * 512)  # num_heads * v_head_dim
                self.assertEqual(
                    output.shape,
                    expected_shape,
                    f"Output shape mismatch for config (bs={batch_size}, seq_len={seq_len}, page_size={page_size})",
                )
                self.assertEqual(output.dtype, self.dtype)
                self.assertEqual(output.device.type, "cuda")
                self.assertFalse(
                    torch.isnan(output).any(),
                    f"Output contains NaN for config (bs={batch_size}, seq_len={seq_len}, page_size={page_size})",
                )
                self.assertFalse(
                    torch.isinf(output).any(),
                    f"Output contains Inf for config (bs={batch_size}, seq_len={seq_len}, page_size={page_size})",
                )


if __name__ == "__main__":
    unittest.main()
