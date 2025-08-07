import math
import unittest

import numpy as np
import torch

from sglang.srt.layers import dp_attention as _dp_attn

# Patch DP-attention globals before importing backends
# TODO: change the interface of both trtllm_mla and flashinfer backends to take tp_size as an argument instead of patching
_dp_attn.get_attention_tp_size = lambda: 1  # TP size = 1 for unit test

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
)
from sglang.srt.layers.attention.utils import TRITON_PAD_NUM_PAGE_PER_BLOCK
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase

# Global configuration for all tests
DEFAULT_CONFIG = {
    "device": "cuda",
    "dtype": torch.bfloat16,
    "kv_cache_dtype": torch.bfloat16,
    "context_len": 2048,
    "max_bs": 64,
    "tolerance": 1e-2,
    "seed_cache": 42,
    "seed_qkv": 123,
    # MLA model config (TRTLLM MLA has fixed constraints)
    "num_attention_heads": 128,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 512,
    "num_kv_heads": 1,
    "layer_id": 0,
}

# Centralized test cases for different test scenarios
TEST_CASES = {
    "basic_functionality": [
        {
            "name": "single",
            "batch_size": 1,
            "max_seq_len": 32,
            "page_size": 32,
            "description": "Minimal smoke test",
        },
        {
            "name": "batch",
            "batch_size": 32,
            "max_seq_len": 128,
            "page_size": 32,
            "description": "Medium-scale batch",
        },
    ],
    "decode_output_match": [
        {
            "name": "single",
            "batch_size": 1,
            "max_seq_len": 64,
            "page_size": 32,
            "description": "Single vs reference",
        },
        {
            "name": "batch",
            "batch_size": 32,
            "max_seq_len": 64,
            "page_size": 32,
            "description": "Batch vs reference",
        },
    ],
    "page_size_consistency": [
        # Only 32 and 64 supported for now in flashinfer TRTLLM-GEN MLA kernel
        {
            "name": "page_32",
            "batch_size": 8,
            "max_seq_len": 128,
            "page_size": 32,
            "description": "32-token pages",
        },
        {
            "name": "page_64",
            "batch_size": 8,
            "max_seq_len": 128,
            "page_size": 64,
            "description": "64-token pages",
        },
    ],
    "shape_sanity_tests": [
        {
            "name": "basic",
            "batch_size": 1,
            "max_seq_len": 128,
            "page_size": 32,
            "description": "Single sequence",
        },
        {
            "name": "basic_different_pagesize",
            "batch_size": 1,
            "max_seq_len": 128,
            "page_size": 64,
            "description": "Different page size",
        },
        {
            "name": "batch",
            "batch_size": 8,
            "max_seq_len": 128,
            "page_size": 32,
            "description": "Batch shapes",
        },
    ],
    "metadata_tests": [
        {
            "name": "single_sequence",
            "batch_size": 1,
            "max_seq_len": 64,
            "page_size": 32,
            "description": "Single sequence metadata",
        },
        {
            "name": "batch_mixed_lengths",
            "batch_size": 8,
            "max_seq_len": 128,
            "page_size": 32,
            "description": "Mixed sequence lengths",
        },
        {
            "name": "large_batch",
            "batch_size": 32,
            "max_seq_len": 256,
            "page_size": 64,
            "description": "Large batch stress test",
        },
        {
            "name": "edge_case_short",
            "batch_size": 4,
            "max_seq_len": 16,
            "page_size": 32,
            "description": "Sub-page sequences",
        },
    ],
}


class MockModelRunner:
    """Minimal fake ModelRunner for testing MLA backends."""

    def __init__(self, config):
        self.device = config["device"]
        self.dtype = config["dtype"]
        self.kv_cache_dtype = config["kv_cache_dtype"]
        self.page_size = config["page_size"]

        # Model-config stub with MLA attributes
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": config["context_len"],
                "attention_arch": AttentionArch.MLA,
                "num_attention_heads": config["num_attention_heads"],
                "kv_lora_rank": config["kv_lora_rank"],
                "qk_nope_head_dim": config["qk_nope_head_dim"],
                "qk_rope_head_dim": config["qk_rope_head_dim"],
                "v_head_dim": config["v_head_dim"],
                "scaling": 1.0
                / ((config["qk_nope_head_dim"] + config["qk_rope_head_dim"]) ** 0.5),
                "get_num_kv_heads": staticmethod(lambda _: config["num_kv_heads"]),
            },
        )

        # Req-to-token pool
        max_bs = config["max_bs"]
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
            page_size=config["page_size"],
            dtype=self.kv_cache_dtype,
            kv_lora_rank=config["kv_lora_rank"],
            qk_rope_head_dim=config["qk_rope_head_dim"],
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
class TestTRTLLMMLA(CustomTestCase):
    """Test suite for TRTLLM MLA backend with centralized configuration."""

    def _merge_config(self, test_case):
        """Merge test case with default configuration."""
        config = DEFAULT_CONFIG.copy()
        config.update(test_case)
        return config

    def _create_model_components(self, config):
        """Create model runners, backends, and layer for testing."""
        # Create model runners
        model_runner_trtllm = MockModelRunner(config)
        model_runner_reference = MockModelRunner(config)

        # Create backends
        trtllm_backend = TRTLLMMLABackend(model_runner_trtllm)
        reference_backend = FlashInferMLAAttnBackend(model_runner_reference)

        # Create RadixAttention layer
        layer = RadixAttention(
            num_heads=config["num_attention_heads"],
            head_dim=config["kv_lora_rank"] + config["qk_rope_head_dim"],
            scaling=model_runner_trtllm.model_config.scaling,
            num_kv_heads=config["num_kv_heads"],
            layer_id=config["layer_id"],
            v_head_dim=config["v_head_dim"],
            prefix="attn_mqa",
        )

        return (
            model_runner_trtllm,
            model_runner_reference,
            trtllm_backend,
            reference_backend,
            layer,
        )

    def _create_qkv_tensors(self, batch_size, config):
        """Create Q, K, V tensors for testing."""
        head_dim = config["kv_lora_rank"] + config["qk_rope_head_dim"]
        device = config["device"]
        dtype = config["dtype"]

        q = torch.randn(
            (batch_size, config["num_attention_heads"], head_dim),
            dtype=dtype,
            device=device,
        )
        k = torch.randn(
            (batch_size, config["num_kv_heads"], head_dim), dtype=dtype, device=device
        )
        v = torch.randn(
            (batch_size, config["num_kv_heads"], config["v_head_dim"]),
            dtype=dtype,
            device=device,
        )
        return q, k, v

    def _create_forward_batch(
        self, batch_size, seq_lens, backend, model_runner, config
    ):
        """Create a forward batch for the given backend."""
        fb = ForwardBatch(
            batch_size=batch_size,
            input_ids=torch.randint(0, 100, (batch_size, 1), device=config["device"]),
            out_cache_loc=torch.arange(batch_size, device=config["device"]),
            seq_lens_sum=int(seq_lens.sum().item()),
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(batch_size, device=config["device"]),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            attn_backend=backend,
        )
        fb.req_to_token_pool = model_runner.req_to_token_pool
        fb.token_to_kv_pool = model_runner.token_to_kv_pool
        return fb

    def _populate_kv_cache(self, batch_size, seq_lens, model_runners, layer, config):
        """Populate KV cache with identical data for both backends."""
        torch.manual_seed(config["seed_cache"])  # Fixed seed for reproducible cache

        for model_runner in model_runners:
            torch.manual_seed(config["seed_cache"])  # Reset seed for each backend
            for i in range(batch_size):
                seq_len = int(seq_lens[i].item())
                for token_idx in range(seq_len - 1):
                    # Create random K components for MLA
                    cache_k_nope = torch.randn(
                        (1, config["qk_nope_head_dim"]),
                        dtype=config["dtype"],
                        device=config["device"],
                    )
                    cache_k_rope = torch.randn(
                        (1, config["qk_rope_head_dim"]),
                        dtype=config["dtype"],
                        device=config["device"],
                    )

                    # Calculate cache location
                    cache_loc = model_runner.req_to_token_pool.req_to_token[
                        i, token_idx
                    ]

                    # Save to KV cache
                    model_runner.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc.unsqueeze(0),
                        cache_k_nope.squeeze(0),
                        cache_k_rope.squeeze(0),
                    )

    def test_basic_functionality(self):
        """Test basic functionality with minimal setup."""
        print(f"\nRunning basic functionality tests...")

        for test_case in TEST_CASES["basic_functionality"]:
            with self.subTest(test_case=test_case["name"]):
                print(f"  Testing {test_case['name']}: {test_case['description']}")

                config = self._merge_config(test_case)
                batch_size = config["batch_size"]
                max_seq_len = config["max_seq_len"]

                # Create components
                model_runner_trtllm, _, trtllm_backend, _, layer = (
                    self._create_model_components(config)
                )

                # Create sequence lengths - properly handle different batch sizes
                if batch_size == 2:
                    seq_lens = torch.tensor(
                        [max_seq_len, max_seq_len // 2], device=config["device"]
                    )
                else:
                    # For larger batch sizes, create varied sequence lengths
                    torch.manual_seed(config["seed_cache"])
                    seq_lens = torch.randint(
                        max_seq_len // 2,
                        max_seq_len + 1,
                        (batch_size,),
                        device=config["device"],
                    )
                    seq_lens[0] = max_seq_len  # Ensure at least one max length

                # Create forward batch
                fb = self._create_forward_batch(
                    batch_size, seq_lens, trtllm_backend, model_runner_trtllm, config
                )
                trtllm_backend.init_forward_metadata(fb)

                # Populate KV cache
                self._populate_kv_cache(
                    batch_size, seq_lens, [model_runner_trtllm], layer, config
                )

                # Create Q, K, V tensors
                torch.manual_seed(config["seed_qkv"])
                q, k, v = self._create_qkv_tensors(batch_size, config)

                # Run forward decode
                output = trtllm_backend.forward_decode(q, k, v, layer, fb)

                # Basic checks
                expected_shape = (
                    batch_size,
                    config["num_attention_heads"] * config["v_head_dim"],
                )
                self.assertEqual(output.shape, expected_shape)
                self.assertEqual(output.dtype, config["dtype"])
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_decode_output_match(self):
        """Test that TRTLLM and FlashInfer MLA backends produce matching outputs."""
        print(f"\nRunning decode output matching tests...")

        for test_case in TEST_CASES["decode_output_match"]:
            with self.subTest(test_case=test_case["name"]):
                print(f"  Testing {test_case['name']}: {test_case['description']}")

                config = self._merge_config(test_case)
                batch_size = config["batch_size"]
                max_seq_len = config["max_seq_len"]

                # Create components
                (
                    model_runner_trtllm,
                    model_runner_reference,
                    trtllm_backend,
                    reference_backend,
                    layer,
                ) = self._create_model_components(config)

                # Create identical sequence lengths for both backends
                torch.manual_seed(config["seed_cache"])
                seq_lens = torch.randint(
                    1, max_seq_len, (batch_size,), device=config["device"]
                )
                seq_lens[0] = max_seq_len  # Ensure at least one max length

                # Create forward batches with identical inputs
                fb_trtllm = self._create_forward_batch(
                    batch_size,
                    seq_lens.clone(),
                    trtllm_backend,
                    model_runner_trtllm,
                    config,
                )
                fb_reference = self._create_forward_batch(
                    batch_size,
                    seq_lens.clone(),
                    reference_backend,
                    model_runner_reference,
                    config,
                )

                # Initialize metadata for both backends
                trtllm_backend.init_forward_metadata(fb_trtllm)
                reference_backend.init_forward_metadata(fb_reference)

                # Populate both KV caches identically
                self._populate_kv_cache(
                    batch_size,
                    seq_lens,
                    [model_runner_trtllm, model_runner_reference],
                    layer,
                    config,
                )

                # Create Q, K, V tensors for current decode step
                torch.manual_seed(config["seed_qkv"])
                q, k, v = self._create_qkv_tensors(batch_size, config)

                # Run forward decode on both backends
                out_trtllm = trtllm_backend.forward_decode(
                    q.clone(), k.clone(), v.clone(), layer, fb_trtllm
                )
                out_reference = reference_backend.forward_decode(
                    q.clone(), k.clone(), v.clone(), layer, fb_reference
                )

                # Compare outputs
                comparison_passed = compare_outputs(
                    out_trtllm, out_reference, tolerance=config["tolerance"]
                )

                self.assertTrue(
                    comparison_passed,
                    f"TRTLLM and Reference outputs differ beyond tolerance. "
                    f"Config: {test_case['name']}, "
                    f"Max diff: {(out_trtllm - out_reference).abs().max().item()}",
                )

    def test_page_size_consistency(self):
        """Test output consistency across different page sizes."""
        print(f"\nRunning page size consistency tests...")

        for test_case in TEST_CASES["page_size_consistency"]:
            with self.subTest(test_case=test_case["name"]):
                print(f"  Testing {test_case['name']}: {test_case['description']}")

                config = self._merge_config(test_case)
                batch_size = config["batch_size"]
                max_seq_len = config["max_seq_len"]

                # Create components
                model_runner, _, backend, _, layer = self._create_model_components(
                    config
                )

                # Create sequence lengths
                torch.manual_seed(config["seed_cache"])
                seq_lens = torch.randint(
                    1, max_seq_len, (batch_size,), device=config["device"]
                )
                seq_lens[0] = max_seq_len

                # Create forward batch
                fb = self._create_forward_batch(
                    batch_size, seq_lens, backend, model_runner, config
                )
                backend.init_forward_metadata(fb)

                # Populate KV cache
                self._populate_kv_cache(
                    batch_size, seq_lens, [model_runner], layer, config
                )

                # Create Q, K, V tensors
                torch.manual_seed(config["seed_qkv"])
                q, k, v = self._create_qkv_tensors(batch_size, config)

                # Run forward decode
                output = backend.forward_decode(q, k, v, layer, fb)

                expected_shape = (
                    batch_size,
                    config["num_attention_heads"] * config["v_head_dim"],
                )
                self.assertEqual(
                    output.shape,
                    expected_shape,
                    f"Output shape mismatch: {output.shape} vs {expected_shape}",
                )
                self.assertFalse(torch.isnan(output).any(), "Output contains NaN")
                self.assertFalse(torch.isinf(output).any(), "Output contains Inf")

    def test_shape_sanity(self):
        """Smoke test decode across several configurations."""
        print(f"\nRunning shape sanity tests...")

        for test_case in TEST_CASES["shape_sanity_tests"]:
            with self.subTest(test_case=test_case["name"]):
                print(f"  Testing {test_case['name']}: {test_case['description']}")

                config = self._merge_config(test_case)
                batch_size = config["batch_size"]
                max_seq_len = config["max_seq_len"]

                model_runner, _, backend, _, layer = self._create_model_components(
                    config
                )

                # Random seq lens (ensure one matches max)
                torch.manual_seed(config["seed_cache"])
                seq_lens = torch.randint(
                    1, max_seq_len, (batch_size,), device=config["device"]
                )
                seq_lens[0] = max_seq_len

                fb = self._create_forward_batch(
                    batch_size, seq_lens, backend, model_runner, config
                )
                backend.init_forward_metadata(fb)

                # Create Q, K, V tensors
                torch.manual_seed(config["seed_qkv"])
                head_dim = config["kv_lora_rank"] + config["qk_rope_head_dim"]
                q = torch.randn(
                    (batch_size, config["num_attention_heads"], head_dim),
                    dtype=config["dtype"],
                    device=config["device"],
                )
                k = torch.randn(
                    (batch_size, config["num_kv_heads"], head_dim),
                    dtype=config["dtype"],
                    device=config["device"],
                )
                v = None

                # Run forward decode
                output = backend.forward_decode(q, k, v, layer, fb)

                # Shape and sanity checks
                expected_shape = (
                    batch_size,
                    config["num_attention_heads"] * config["v_head_dim"],
                )
                self.assertEqual(
                    output.shape,
                    expected_shape,
                    f"Output shape mismatch for {test_case['name']}",
                )
                self.assertEqual(output.dtype, config["dtype"])
                self.assertEqual(output.device.type, "cuda")
                self.assertFalse(
                    torch.isnan(output).any(),
                    f"Output contains NaN for {test_case['name']}",
                )
                self.assertFalse(
                    torch.isinf(output).any(),
                    f"Output contains Inf for {test_case['name']}",
                )

    def test_metadata_initialization(self):
        """Test TRTLLM MLA metadata initialization and structure."""
        print(f"\nRunning metadata initialization tests...")

        for test_case in TEST_CASES["metadata_tests"]:
            with self.subTest(test_case=test_case["name"]):
                print(f"  Testing {test_case['name']}: {test_case['description']}")

                config = self._merge_config(test_case)
                batch_size = config["batch_size"]
                max_seq_len = config["max_seq_len"]

                # Create components
                model_runner, _, backend, _, layer = self._create_model_components(
                    config
                )

                # Create varied sequence lengths
                torch.manual_seed(config["seed_cache"])
                if batch_size == 1:
                    seq_lens = torch.tensor([max_seq_len], device=config["device"])
                else:
                    seq_lens = torch.randint(
                        max(1, max_seq_len // 4),
                        max_seq_len + 1,
                        (batch_size,),
                        device=config["device"],
                    )
                    seq_lens[0] = max_seq_len  # Ensure at least one max length

                # Create forward batch
                fb = self._create_forward_batch(
                    batch_size, seq_lens, backend, model_runner, config
                )

                # Initialize metadata
                backend.init_forward_metadata(fb)

                # Verify metadata exists
                self.assertIsNotNone(backend.forward_metadata)
                self.assertIsInstance(backend.forward_metadata, TRTLLMMLADecodeMetadata)

                # Test metadata structure
                metadata = backend.forward_metadata
                self.assertIsNotNone(
                    metadata.workspace, "Workspace should be allocated"
                )
                self.assertIsNotNone(
                    metadata.block_kv_indices, "Block KV indices should be created"
                )

                # Test workspace properties
                self.assertEqual(metadata.workspace.device.type, "cuda")
                self.assertEqual(metadata.workspace.dtype, torch.int8)
                self.assertGreater(
                    metadata.workspace.numel(), 0, "Workspace should have non-zero size"
                )

                # Test block KV indices properties
                self.assertEqual(metadata.block_kv_indices.device.type, "cuda")
                self.assertEqual(metadata.block_kv_indices.dtype, torch.int32)
                self.assertEqual(metadata.block_kv_indices.shape[0], batch_size)

                # Verify block indices are valid (>= -1, since -1 is padding)
                self.assertTrue(
                    (metadata.block_kv_indices >= -1).all(),
                    "All block indices should be >= -1 (with -1 as padding)",
                )

    def test_metadata_block_calculation(self):
        """Test block count calculation logic."""
        print(f"\nRunning metadata block calculation tests...")

        test_scenarios = [
            {"seq_len": 31, "page_size": 32, "expected_min_blocks": 1},
            {"seq_len": 32, "page_size": 32, "expected_min_blocks": 1},
            {"seq_len": 33, "page_size": 32, "expected_min_blocks": 2},
            {"seq_len": 128, "page_size": 32, "expected_min_blocks": 4},
            {"seq_len": 128, "page_size": 64, "expected_min_blocks": 2},
        ]

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario):
                config = self._merge_config(
                    {
                        "batch_size": 1,
                        "max_seq_len": scenario["seq_len"],
                        "page_size": scenario["page_size"],
                    }
                )

                model_runner, _, backend, _, _ = self._create_model_components(config)

                # Test internal block calculation
                calculated_blocks = backend._calc_padded_blocks(scenario["seq_len"])

                # Should be at least the minimum required
                self.assertGreaterEqual(
                    calculated_blocks,
                    scenario["expected_min_blocks"],
                    f"Calculated blocks ({calculated_blocks}) should be >= minimum required ({scenario['expected_min_blocks']})",
                )

                # Should satisfy page_size constraint
                total_tokens = calculated_blocks * scenario["page_size"]
                self.assertGreaterEqual(
                    total_tokens,
                    scenario["seq_len"],
                    f"Total tokens ({total_tokens}) should cover sequence length ({scenario['seq_len']})",
                )

                # Should satisfy TRT-LLM and Triton constraints
                trtllm_constraint = 128 // scenario["page_size"]
                constraint_lcm = math.lcm(
                    trtllm_constraint, TRITON_PAD_NUM_PAGE_PER_BLOCK
                )
                self.assertEqual(
                    calculated_blocks % constraint_lcm,
                    0,
                    f"Block count should be multiple of LCM of constraints ({constraint_lcm})",
                )

    def test_metadata_kv_indices_correctness(self):
        """Test KV indices creation and correctness."""
        print(f"\nRunning KV indices correctness tests...")

        for test_case in TEST_CASES["metadata_tests"][
            :2
        ]:  # Test subset for performance
            with self.subTest(test_case=test_case["name"]):
                print(f"  Testing {test_case['name']}: {test_case['description']}")

                config = self._merge_config(test_case)
                batch_size = config["batch_size"]
                max_seq_len = config["max_seq_len"]

                model_runner, _, backend, _, layer = self._create_model_components(
                    config
                )

                # Create known sequence lengths
                torch.manual_seed(config["seed_cache"])
                if batch_size == 1:
                    seq_lens = torch.tensor([max_seq_len], device=config["device"])
                else:
                    seq_lens = torch.randint(
                        max_seq_len // 2,
                        max_seq_len + 1,
                        (batch_size,),
                        device=config["device"],
                    )

                fb = self._create_forward_batch(
                    batch_size, seq_lens, backend, model_runner, config
                )

                # Populate some KV cache to have valid indices
                self._populate_kv_cache(
                    batch_size, seq_lens, [model_runner], layer, config
                )

                # Initialize metadata
                backend.init_forward_metadata(fb)
                metadata = backend.forward_metadata

                # Verify KV indices structure
                block_kv_indices = metadata.block_kv_indices

                for i in range(batch_size):
                    seq_len = seq_lens[i].item()
                    expected_blocks = backend._calc_padded_blocks(seq_len)

                    # Count valid (non -1) indices for this sequence
                    valid_indices = (block_kv_indices[i] >= 0).sum().item()

                    # Should have at least enough blocks for the sequence
                    min_required_blocks = (seq_len + config["page_size"] - 1) // config[
                        "page_size"
                    ]
                    self.assertGreaterEqual(
                        valid_indices,
                        min_required_blocks,
                        f"Sequence {i} should have at least {min_required_blocks} valid blocks, got {valid_indices}",
                    )

                    # Verify indices are within valid range
                    valid_block_indices = block_kv_indices[i][block_kv_indices[i] >= 0]
                    if len(valid_block_indices) > 0:
                        max_possible_blocks = (
                            model_runner.token_to_kv_pool.size // config["page_size"]
                        )
                        self.assertTrue(
                            (valid_block_indices < max_possible_blocks).all(),
                            f"All block indices should be < {max_possible_blocks}",
                        )

    def test_metadata_cuda_graph_compatibility(self):
        """Test metadata compatibility with CUDA graph capture/replay."""
        print(f"\nRunning CUDA graph compatibility tests...")

        config = self._merge_config(
            {"batch_size": 4, "max_seq_len": 64, "page_size": 32}
        )

        model_runner, _, backend, _, layer = self._create_model_components(config)
        batch_size = config["batch_size"]

        # Initialize CUDA graph state
        backend.init_cuda_graph_state(
            max_bs=batch_size, max_num_tokens=config["max_seq_len"] * batch_size
        )

        # Verify CUDA graph buffers are allocated
        self.assertIsNotNone(backend.cuda_graph_kv_indices)
        self.assertIsNotNone(backend.cuda_graph_workspace)

        # Test capture metadata
        seq_lens = torch.full(
            (batch_size,), config["max_seq_len"], device=config["device"]
        )
        req_pool_indices = torch.arange(batch_size, device=config["device"])

        backend.init_forward_metadata_capture_cuda_graph(
            bs=batch_size,
            num_tokens=batch_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )

        # Verify capture metadata
        self.assertIn(batch_size, backend.decode_cuda_graph_metadata)
        capture_metadata = backend.decode_cuda_graph_metadata[batch_size]

        self.assertIsNotNone(capture_metadata.workspace)
        self.assertIsNotNone(capture_metadata.block_kv_indices)

        # Test replay with different sequence lengths
        new_seq_lens = torch.randint(
            config["max_seq_len"] // 2,
            config["max_seq_len"] + 1,
            (batch_size,),
            device=config["device"],
        )

        backend.init_forward_metadata_replay_cuda_graph(
            bs=batch_size,
            req_pool_indices=req_pool_indices,
            seq_lens=new_seq_lens,
            seq_lens_sum=new_seq_lens.sum().item(),
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=new_seq_lens.cpu(),
        )

        # Verify replay updated the metadata
        replay_metadata = backend.forward_metadata
        self.assertIsNotNone(replay_metadata)
        self.assertEqual(
            replay_metadata.workspace.data_ptr(), capture_metadata.workspace.data_ptr()
        )

    def test_metadata_consistency_across_calls(self):
        """Test metadata consistency across multiple forward calls."""
        print(f"\nRunning metadata consistency tests...")

        config = self._merge_config(
            {"batch_size": 2, "max_seq_len": 64, "page_size": 32}
        )

        model_runner, _, backend, _, layer = self._create_model_components(config)

        # First call
        seq_lens_1 = torch.tensor([32, 48], device=config["device"])
        fb_1 = self._create_forward_batch(
            config["batch_size"], seq_lens_1, backend, model_runner, config
        )
        backend.init_forward_metadata(fb_1)
        metadata_1 = backend.forward_metadata

        # Second call with same sequence lengths
        seq_lens_2 = torch.tensor([32, 48], device=config["device"])
        fb_2 = self._create_forward_batch(
            config["batch_size"], seq_lens_2, backend, model_runner, config
        )
        backend.init_forward_metadata(fb_2)
        metadata_2 = backend.forward_metadata

        # Metadata structure should be consistent
        self.assertEqual(metadata_1.workspace.shape, metadata_2.workspace.shape)
        self.assertEqual(
            metadata_1.block_kv_indices.shape, metadata_2.block_kv_indices.shape
        )

        # Third call with different sequence lengths
        seq_lens_3 = torch.tensor([16, 64], device=config["device"])
        fb_3 = self._create_forward_batch(
            config["batch_size"], seq_lens_3, backend, model_runner, config
        )
        backend.init_forward_metadata(fb_3)
        metadata_3 = backend.forward_metadata

        # Should still have valid structure
        self.assertIsNotNone(metadata_3.workspace)
        self.assertIsNotNone(metadata_3.block_kv_indices)
        self.assertEqual(metadata_3.block_kv_indices.shape[0], config["batch_size"])


if __name__ == "__main__":
    unittest.main()
