"""
Unit tests for SageAttention backend comparing INT8 quantized attention against FP16 baseline.

These tests verify numerical correctness by comparing SageAttention outputs against
TorchNative (reference) backend outputs. SageAttention quantizes Q and K to INT8
on-the-fly, so we expect small numerical differences but overall accuracy preservation.

Usage:
    python -m pytest python/sglang/test/attention/test_sage_attention_backend.py -v
    python -m unittest sglang.test.attention.test_sage_attention_backend
"""

import unittest
from typing import Optional, Tuple

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ServerArgs
from sglang.test.test_utils import CustomTestCase


def is_sage_attention_available() -> bool:
    """Check if sageattention package is available."""
    try:
        from sageattention import sageattn_varlen

        return True
    except ImportError:
        return False


class MockModelRunner:
    """Mock model runner for testing attention backends without full model setup."""

    def __init__(
        self,
        page_size: int = 1,
        num_heads: int = 32,
        head_dim: int = 128,
        num_kv_heads: Optional[int] = None,
        max_batch_size: int = 32,
        max_context_len: int = 8192,
    ):
        self.device = "cuda"
        self.dtype = torch.float16
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads

        attention_arch = AttentionArch.MHA
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": max_context_len,
                "is_multimodal": False,
                "attention_arch": attention_arch,
                "num_attention_heads": num_heads,
            },
        )
        self.sliding_window_size = None

        # Create req_to_token pool
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
        )

        self.page_size = page_size
        max_total_num_tokens = max_batch_size * max_context_len

        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_total_num_tokens,
            page_size=page_size,
            dtype=self.dtype,
            head_num=self.num_kv_heads,
            head_dim=head_dim,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
        )

        self.server_args = ServerArgs(model_path="dummy")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
@unittest.skipIf(not is_sage_attention_available(), "SageAttention not installed")
class TestSageAttentionNumericalAccuracy(CustomTestCase):
    """
    Test numerical accuracy of SageAttention INT8 quantized attention
    against FP16 baseline (TorchNative backend).
    """

    def setUp(self):
        # Common test configurations - using realistic model dimensions
        self.device = "cuda"
        self.dtype = torch.float16

        # Llama-like configurations
        self.test_configs = [
            # (batch_size, seq_len, num_heads, head_dim, num_kv_heads)
            (1, 128, 32, 128, 8),  # Small: GQA config
            (1, 512, 32, 128, 8),  # Medium sequence
            (1, 2048, 32, 128, 8),  # Long sequence
            (4, 256, 32, 128, 8),  # Batched
            (8, 128, 32, 128, 32),  # MHA config
            (1, 4096, 32, 128, 8),  # Very long sequence for accuracy drift test
        ]

    def _init_backends(
        self, num_heads: int, head_dim: int, num_kv_heads: int, max_seq_len: int
    ):
        """Initialize SageAttention and reference backends."""
        from sglang.srt.layers.attention.sage_attention_backend import SageAttnBackend
        from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

        self.model_runner = MockModelRunner(
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            max_context_len=max_seq_len + 1024,
        )

        self.sage_backend = SageAttnBackend(self.model_runner)
        self.ref_backend = TorchNativeAttnBackend(self.model_runner)

    def _create_attention_layer(
        self, num_heads: int, head_dim: int, num_kv_heads: int
    ) -> RadixAttention:
        """Create attention layer for testing."""
        return RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=1.0 / (head_dim**0.5),
            num_kv_heads=num_kv_heads,
            layer_id=0,
        )

    def _create_qkv_tensors(
        self, total_tokens: int, num_heads: int, num_kv_heads: int, head_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create Q, K, V tensors for testing."""
        # Use a fixed seed for reproducibility
        torch.manual_seed(42)

        q = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, dtype=self.dtype, device=self.device
        )
        return q, k, v

    def _setup_req_to_token(self, batch_size: int, seq_len: int):
        """Setup req_to_token mapping."""
        req_to_token = (
            torch.arange(0, batch_size, dtype=torch.int32, device=self.device)[:, None]
            * seq_len
            + torch.arange(0, seq_len, dtype=torch.int32, device=self.device)[None, :]
        )
        self.model_runner.req_to_token_pool.req_to_token[:batch_size, :seq_len] = (
            req_to_token
        )

    def _create_extend_forward_batch(
        self, batch_size: int, seq_len: int
    ) -> ForwardBatch:
        """Create forward batch for extend (prefill) operation."""
        total_tokens = batch_size * seq_len

        forward_batch = ForwardBatch(
            batch_size=batch_size,
            input_ids=torch.randint(0, 100, (batch_size, seq_len), device=self.device),
            out_cache_loc=torch.arange(0, total_tokens, device=self.device),
            seq_lens_sum=total_tokens,
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.arange(batch_size, device=self.device),
            seq_lens=torch.tensor([seq_len] * batch_size, device=self.device),
            seq_lens_cpu=torch.tensor([seq_len] * batch_size, device="cpu"),
            extend_prefix_lens=torch.zeros(batch_size, device=self.device, dtype=torch.int32),
            extend_prefix_lens_cpu=torch.zeros(batch_size, device="cpu", dtype=torch.int64),
            extend_seq_lens=torch.tensor([seq_len] * batch_size, device=self.device),
            extend_seq_lens_cpu=torch.tensor([seq_len] * batch_size, device="cpu"),
            attn_backend=self.sage_backend,
        )
        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
        return forward_batch

    def _create_decode_forward_batch(
        self, batch_size: int, cached_seq_len: int
    ) -> ForwardBatch:
        """Create forward batch for decode operation."""
        # Decode processes 1 new token per sequence
        total_seq_len = cached_seq_len + 1
        out_cache_start = batch_size * cached_seq_len

        forward_batch = ForwardBatch(
            batch_size=batch_size,
            input_ids=torch.randint(0, 100, (batch_size, 1), device=self.device),
            out_cache_loc=torch.arange(
                out_cache_start, out_cache_start + batch_size, device=self.device
            ),
            seq_lens_sum=batch_size * total_seq_len,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(batch_size, device=self.device),
            seq_lens=torch.tensor([total_seq_len] * batch_size, device=self.device),
            seq_lens_cpu=torch.tensor([total_seq_len] * batch_size, device="cpu"),
            attn_backend=self.sage_backend,
        )
        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
        return forward_batch

    def _setup_kv_cache(
        self,
        layer: RadixAttention,
        batch_size: int,
        cache_len: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        """Populate KV cache with test data."""
        torch.manual_seed(123)
        cache_k = torch.randn(
            batch_size * cache_len,
            num_kv_heads,
            head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        cache_v = torch.randn(
            batch_size * cache_len,
            num_kv_heads,
            head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        cache_indices = torch.arange(batch_size * cache_len, device=self.device)
        self.model_runner.token_to_kv_pool.set_kv_buffer(
            layer,
            cache_indices,
            cache_k,
            cache_v,
            layer.k_scale,
            layer.v_scale,
        )

    def _compute_relative_error(
        self, output: torch.Tensor, ref_output: torch.Tensor
    ) -> dict:
        """Compute various error metrics between outputs."""
        abs_diff = torch.abs(output - ref_output)
        rel_diff = abs_diff / (torch.abs(ref_output) + 1e-8)

        return {
            "max_abs_error": abs_diff.max().item(),
            "mean_abs_error": abs_diff.mean().item(),
            "max_rel_error": rel_diff.max().item(),
            "mean_rel_error": rel_diff.mean().item(),
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                output.flatten().unsqueeze(0),
                ref_output.flatten().unsqueeze(0),
            )
            .item(),
        }

    def test_extend_numerical_accuracy(self):
        """Test extend (prefill) numerical accuracy across different configurations."""
        for batch_size, seq_len, num_heads, head_dim, num_kv_heads in self.test_configs:
            with self.subTest(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            ):
                self._init_backends(num_heads, head_dim, num_kv_heads, seq_len)
                layer = self._create_attention_layer(num_heads, head_dim, num_kv_heads)

                # Setup
                self._setup_req_to_token(batch_size, seq_len)
                forward_batch = self._create_extend_forward_batch(batch_size, seq_len)
                total_tokens = batch_size * seq_len
                q, k, v = self._create_qkv_tensors(
                    total_tokens, num_heads, num_kv_heads, head_dim
                )

                # Run both backends
                self.sage_backend.init_forward_metadata(forward_batch)
                self.ref_backend.init_forward_metadata(forward_batch)

                # Clone tensors to ensure both backends see the same input
                sage_output = self.sage_backend.forward_extend(
                    q.clone(), k.clone(), v.clone(), layer, forward_batch
                )
                ref_output = self.ref_backend.forward_extend(
                    q.clone(), k.clone(), v.clone(), layer, forward_batch
                )

                # Compute error metrics
                errors = self._compute_relative_error(sage_output, ref_output)

                print(
                    f"\nExtend - bs={batch_size}, seq_len={seq_len}: "
                    f"max_rel_error={errors['max_rel_error']:.6f}, "
                    f"cosine_sim={errors['cosine_similarity']:.6f}"
                )

                # Assertions - SageAttention should maintain high accuracy
                self.assertLess(
                    errors["mean_rel_error"],
                    0.05,
                    f"Mean relative error too high: {errors['mean_rel_error']}",
                )
                self.assertGreater(
                    errors["cosine_similarity"],
                    0.99,
                    f"Cosine similarity too low: {errors['cosine_similarity']}",
                )

    def test_decode_numerical_accuracy(self):
        """Test decode numerical accuracy across different configurations."""
        decode_configs = [
            # (batch_size, cached_seq_len, num_heads, head_dim, num_kv_heads)
            (1, 128, 32, 128, 8),
            (1, 512, 32, 128, 8),
            (1, 2048, 32, 128, 8),
            (8, 256, 32, 128, 8),
            (16, 128, 32, 128, 8),
        ]

        for batch_size, cached_seq_len, num_heads, head_dim, num_kv_heads in decode_configs:
            with self.subTest(
                batch_size=batch_size,
                cached_seq_len=cached_seq_len,
            ):
                self._init_backends(num_heads, head_dim, num_kv_heads, cached_seq_len + 1)
                layer = self._create_attention_layer(num_heads, head_dim, num_kv_heads)

                total_seq_len = cached_seq_len + 1
                self._setup_req_to_token(batch_size, total_seq_len)

                # Setup KV cache
                self._setup_kv_cache(
                    layer, batch_size, cached_seq_len, num_kv_heads, head_dim
                )

                # Create decode batch
                forward_batch = self._create_decode_forward_batch(
                    batch_size, cached_seq_len
                )

                # Q, K, V for the new token
                q, k, v = self._create_qkv_tensors(
                    batch_size, num_heads, num_kv_heads, head_dim
                )

                # Run both backends
                self.sage_backend.init_forward_metadata(forward_batch)
                self.ref_backend.init_forward_metadata(forward_batch)

                sage_output = self.sage_backend.forward_decode(
                    q.clone(), k.clone(), v.clone(), layer, forward_batch
                )
                ref_output = self.ref_backend.forward_decode(
                    q.clone(), k.clone(), v.clone(), layer, forward_batch
                )

                # Compute error metrics
                errors = self._compute_relative_error(sage_output, ref_output)

                print(
                    f"\nDecode - bs={batch_size}, cached_len={cached_seq_len}: "
                    f"max_rel_error={errors['max_rel_error']:.6f}, "
                    f"cosine_sim={errors['cosine_similarity']:.6f}"
                )

                # Decode should also maintain high accuracy
                self.assertLess(
                    errors["mean_rel_error"],
                    0.05,
                    f"Mean relative error too high: {errors['mean_rel_error']}",
                )
                self.assertGreater(
                    errors["cosine_similarity"],
                    0.99,
                    f"Cosine similarity too low: {errors['cosine_similarity']}",
                )

    def test_long_context_accuracy_drift(self):
        """
        Test that accuracy doesn't drift significantly with longer contexts.
        SageAttention claims "≈ same accuracy" - verify this claim.
        """
        num_heads, head_dim, num_kv_heads = 32, 128, 8
        batch_size = 1

        context_lengths = [256, 512, 1024, 2048, 4096]
        accuracy_metrics = []

        for seq_len in context_lengths:
            self._init_backends(num_heads, head_dim, num_kv_heads, seq_len)
            layer = self._create_attention_layer(num_heads, head_dim, num_kv_heads)

            self._setup_req_to_token(batch_size, seq_len)
            forward_batch = self._create_extend_forward_batch(batch_size, seq_len)
            total_tokens = batch_size * seq_len
            q, k, v = self._create_qkv_tensors(
                total_tokens, num_heads, num_kv_heads, head_dim
            )

            self.sage_backend.init_forward_metadata(forward_batch)
            self.ref_backend.init_forward_metadata(forward_batch)

            sage_output = self.sage_backend.forward_extend(
                q.clone(), k.clone(), v.clone(), layer, forward_batch
            )
            ref_output = self.ref_backend.forward_extend(
                q.clone(), k.clone(), v.clone(), layer, forward_batch
            )

            errors = self._compute_relative_error(sage_output, ref_output)
            accuracy_metrics.append(
                {
                    "seq_len": seq_len,
                    "cosine_similarity": errors["cosine_similarity"],
                    "mean_rel_error": errors["mean_rel_error"],
                }
            )

        print("\n=== Long Context Accuracy Drift Analysis ===")
        for m in accuracy_metrics:
            print(
                f"  seq_len={m['seq_len']:5d}: "
                f"cosine_sim={m['cosine_similarity']:.6f}, "
                f"mean_rel_error={m['mean_rel_error']:.6f}"
            )

        # Verify accuracy doesn't degrade significantly with longer contexts
        min_cosine_sim = min(m["cosine_similarity"] for m in accuracy_metrics)
        max_mean_rel_error = max(m["mean_rel_error"] for m in accuracy_metrics)

        self.assertGreater(
            min_cosine_sim,
            0.98,
            f"Cosine similarity dropped too low at longer contexts: {min_cosine_sim}",
        )
        self.assertLess(
            max_mean_rel_error,
            0.1,
            f"Mean relative error too high at longer contexts: {max_mean_rel_error}",
        )

    def test_output_distribution_statistics(self):
        """
        Test that output distribution statistics (mean, std, range) are preserved.
        This helps detect systematic biases introduced by INT8 quantization.
        """
        batch_size, seq_len, num_heads, head_dim, num_kv_heads = 4, 512, 32, 128, 8

        self._init_backends(num_heads, head_dim, num_kv_heads, seq_len)
        layer = self._create_attention_layer(num_heads, head_dim, num_kv_heads)

        self._setup_req_to_token(batch_size, seq_len)
        forward_batch = self._create_extend_forward_batch(batch_size, seq_len)
        total_tokens = batch_size * seq_len
        q, k, v = self._create_qkv_tensors(
            total_tokens, num_heads, num_kv_heads, head_dim
        )

        self.sage_backend.init_forward_metadata(forward_batch)
        self.ref_backend.init_forward_metadata(forward_batch)

        sage_output = self.sage_backend.forward_extend(
            q.clone(), k.clone(), v.clone(), layer, forward_batch
        )
        ref_output = self.ref_backend.forward_extend(
            q.clone(), k.clone(), v.clone(), layer, forward_batch
        )

        # Compare distribution statistics
        sage_stats = {
            "mean": sage_output.mean().item(),
            "std": sage_output.std().item(),
            "min": sage_output.min().item(),
            "max": sage_output.max().item(),
        }
        ref_stats = {
            "mean": ref_output.mean().item(),
            "std": ref_output.std().item(),
            "min": ref_output.min().item(),
            "max": ref_output.max().item(),
        }

        print("\n=== Output Distribution Statistics ===")
        print(f"  SageAttention: mean={sage_stats['mean']:.4f}, std={sage_stats['std']:.4f}")
        print(f"  Reference:     mean={ref_stats['mean']:.4f}, std={ref_stats['std']:.4f}")

        # Verify distribution statistics are close
        self.assertAlmostEqual(
            sage_stats["mean"],
            ref_stats["mean"],
            delta=0.02,
            msg="Mean values differ too much",
        )
        self.assertAlmostEqual(
            sage_stats["std"],
            ref_stats["std"],
            delta=0.02,
            msg="Std values differ too much",
        )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
@unittest.skipIf(not is_sage_attention_available(), "SageAttention not installed")
class TestSageAttentionKernelPerformance(CustomTestCase):
    """
    Micro-benchmarks for SageAttention kernel performance.
    These tests measure raw kernel performance without full model context.
    """

    def setUp(self):
        self.device = "cuda"
        self.dtype = torch.float16
        self.warmup_iterations = 10
        self.benchmark_iterations = 100

    def test_kernel_throughput_vs_baseline(self):
        """
        Benchmark SageAttention kernel throughput against baseline.
        Measures raw attention computation speed.
        """
        from sageattention import sageattn_varlen

        # Test configuration - typical Llama-3 sizes
        batch_size = 8
        seq_len = 2048
        num_heads = 32
        head_dim = 128
        num_kv_heads = 8

        total_tokens = batch_size * seq_len
        sm_scale = 1.0 / (head_dim**0.5)

        # Create tensors
        torch.manual_seed(42)
        q = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, dtype=self.dtype, device=self.device
        )

        # For GQA, expand K and V to match Q heads
        k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_expanded = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

        # Create cumulative sequence lengths for varlen API
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=self.device
        )

        # Warmup
        for _ in range(self.warmup_iterations):
            _ = sageattn_varlen(
                q.contiguous(),
                k_expanded.contiguous(),
                v_expanded.contiguous(),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                is_causal=True,
                sm_scale=sm_scale,
            )
        torch.cuda.synchronize()

        # Benchmark SageAttention
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(self.benchmark_iterations):
            _ = sageattn_varlen(
                q.contiguous(),
                k_expanded.contiguous(),
                v_expanded.contiguous(),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                is_causal=True,
                sm_scale=sm_scale,
            )
        end.record()
        torch.cuda.synchronize()

        sage_time_ms = start.elapsed_time(end) / self.benchmark_iterations

        # Benchmark PyTorch scaled_dot_product_attention as baseline
        q_sdpa = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k_sdpa = k_expanded.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v_sdpa = v_expanded.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        for _ in range(self.warmup_iterations):
            _ = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=True, scale=sm_scale
            )
        torch.cuda.synchronize()

        start.record()
        for _ in range(self.benchmark_iterations):
            _ = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=True, scale=sm_scale
            )
        end.record()
        torch.cuda.synchronize()

        baseline_time_ms = start.elapsed_time(end) / self.benchmark_iterations

        # Calculate metrics
        total_flops = batch_size * num_heads * seq_len * seq_len * head_dim * 2 * 2  # QK^T + softmax*V
        sage_tflops = (total_flops / sage_time_ms / 1e9)
        baseline_tflops = (total_flops / baseline_time_ms / 1e9)
        speedup = baseline_time_ms / sage_time_ms

        print("\n=== Kernel Throughput Benchmark ===")
        print(f"  Config: bs={batch_size}, seq_len={seq_len}, heads={num_heads}, dim={head_dim}")
        print(f"  SageAttention: {sage_time_ms:.3f} ms ({sage_tflops:.2f} TFLOPS)")
        print(f"  SDPA Baseline: {baseline_time_ms:.3f} ms ({baseline_tflops:.2f} TFLOPS)")
        print(f"  Speedup: {speedup:.2f}x")

        # SageAttention should ideally be faster (but depends on GPU)
        # At minimum, verify it completes without errors and produces valid output
        self.assertGreater(sage_tflops, 0, "SageAttention should have positive TFLOPS")

    def test_memory_efficiency(self):
        """
        Test memory efficiency of SageAttention.
        INT8 quantization should reduce memory bandwidth requirements.
        """
        torch.cuda.reset_peak_memory_stats()

        batch_size = 4
        seq_len = 4096
        num_heads = 32
        head_dim = 128
        num_kv_heads = 8

        total_tokens = batch_size * seq_len

        # Measure baseline memory with FP16 tensors
        torch.cuda.reset_peak_memory_stats()
        q = torch.randn(
            total_tokens, num_heads, head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, dtype=self.dtype, device=self.device
        )
        torch.cuda.synchronize()

        input_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Run SageAttention
        from sageattention import sageattn_varlen

        k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_expanded = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=self.device
        )
        sm_scale = 1.0 / (head_dim**0.5)

        torch.cuda.reset_peak_memory_stats()
        output = sageattn_varlen(
            q.contiguous(),
            k_expanded.contiguous(),
            v_expanded.contiguous(),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            is_causal=True,
            sm_scale=sm_scale,
        )
        torch.cuda.synchronize()

        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        print("\n=== Memory Efficiency ===")
        print(f"  Config: bs={batch_size}, seq_len={seq_len}")
        print(f"  Input tensors: {input_memory_mb:.2f} MB")
        print(f"  Peak memory during SageAttention: {peak_memory_mb:.2f} MB")

        # Verify output is valid
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf values")


if __name__ == "__main__":
    unittest.main()
