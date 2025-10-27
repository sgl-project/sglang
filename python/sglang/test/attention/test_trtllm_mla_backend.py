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
from sglang.srt.layers.attention.utils import get_num_page_per_block_flashmla
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
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
    "tp_q_head_num": 128,
    "tp_k_head_num": 128,
    "prefill_head_dim": 192,
    "prefill_v_head_dim": 128,
}

ROPE_BASE = 10000
ROPE_SCALING_CONFIG = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn",
    "rope_type": "deepseek_yarn",
}


def build_rotary_emb(config, device=None):
    from sglang.srt.layers.rotary_embedding import get_rope_wrapper

    dev = device or config["device"]
    rope_scaling = config.get("rope_scaling", ROPE_SCALING_CONFIG)
    rotary = get_rope_wrapper(
        head_size=config["qk_rope_head_dim"],
        rotary_dim=config["qk_rope_head_dim"],
        max_position=config["context_len"],
        base=ROPE_BASE,
        rope_scaling=rope_scaling,
        is_neox_style=False,
        device=dev,
    )
    rotary.cos_sin_cache = rotary.cos_sin_cache.to(dev)
    return rotary


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
    "output_match": [
        {
            "name": "single_fp16",
            "batch_size": 1,
            "max_seq_len": 64,
            "page_size": 32,
            "description": "Single FP16 vs reference",
        },
        # {
        #     "name": "single_fp8",
        #     "batch_size": 1,
        #     "max_seq_len": 64,
        #     "page_size": 64,
        #     "tolerance": 1e-1,
        #     "kv_cache_dtype": torch.float8_e4m3fn,
        #     "description": "Single FP8 vs reference",
        # },
        {
            "name": "batch_fp16",
            "batch_size": 32,
            "max_seq_len": 64,
            "page_size": 32,
            "description": "Batch FP16 vs reference",
        },
        # {
        #     "name": "batch_fp8",
        #     "batch_size": 32,
        #     "max_seq_len": 64,
        #     "page_size": 64,
        #     "tolerance": 1e-1,
        #     "kv_cache_dtype": torch.float8_e4m3fn,
        #     "description": "Batch FP8 vs reference",
        # },
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

        # Server args stub - needed by attention backends
        self.server_args = get_global_server_args()

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

    @classmethod
    def setUpClass(cls):
        """Set up global server args for testing."""
        server_args = ServerArgs(model_path="dummy")
        server_args.enable_dp_attention = False
        set_global_server_args_for_scheduler(server_args)

    @classmethod
    def tearDownClass(cls):
        pass

    def _merge_config(self, test_case):
        """Merge test case with default configuration."""
        config = DEFAULT_CONFIG.copy()
        config.update(test_case)
        return config

    def _create_model_components(self, config, is_prefill=False):
        """Create model runners, backends, and layer for testing."""
        # Create model runners
        model_runner_trtllm = MockModelRunner(config)
        model_runner_reference = MockModelRunner(config)

        # Create backends
        trtllm_backend = TRTLLMMLABackend(model_runner_trtllm)
        reference_backend = FlashInferMLAAttnBackend(model_runner_reference)

        head_dim = (
            config["kv_lora_rank"] + config["qk_rope_head_dim"]
            if not is_prefill
            else config["prefill_head_dim"]
        )
        v_head_dim = (
            config["v_head_dim"] if not is_prefill else config["prefill_v_head_dim"]
        )

        # Create RadixAttention layer
        layer = RadixAttention(
            num_heads=config["num_attention_heads"],
            head_dim=head_dim,
            scaling=model_runner_trtllm.model_config.scaling,
            num_kv_heads=config["num_kv_heads"],
            layer_id=config["layer_id"],
            v_head_dim=v_head_dim,
            prefix="attn_mqa",
        )

        return (
            model_runner_trtllm,
            model_runner_reference,
            trtllm_backend,
            reference_backend,
            layer,
        )

    def _create_qkv_tensors(self, batch_size, config, dtype_override=None):
        """Create Q, K, V random tensors for given batch size with separate MLA components.

        Args:
            batch_size: Batch size.
            config: Configuration dict with model dims and device.
            dtype_override: Optional torch dtype to override config["dtype"].

        Returns:
            Tuple of (q_nope, q_rope, k_nope, k_rope, v, cos_sin_cache)
        """
        device = config["device"]
        target_dtype = dtype_override or config["dtype"]

        # Create separate nope and rope components for Q
        q_nope = torch.randn(
            (batch_size, config["num_attention_heads"], config["kv_lora_rank"]),
            dtype=config["dtype"],
            device=device,
        )
        q_rope = torch.randn(
            (batch_size, config["num_attention_heads"], config["qk_rope_head_dim"]),
            dtype=config["dtype"],
            device=device,
        )

        # Create separate nope and rope components for K
        k_nope = torch.randn(
            (batch_size, config["num_kv_heads"], config["kv_lora_rank"]),
            dtype=config["dtype"],
            device=device,
        )
        k_rope = torch.randn(
            (batch_size, config["num_kv_heads"], config["qk_rope_head_dim"]),
            dtype=config["dtype"],
            device=device,
        )

        # V tensor (unchanged)
        v = torch.randn(
            (batch_size, config["num_kv_heads"], config["v_head_dim"]),
            dtype=config["dtype"],
            device=device,
        )

        return q_nope, q_rope, k_nope, k_rope, v

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

        # Add position information for RoPE
        fb.positions = torch.arange(batch_size, device=config["device"])

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
                        (1, config["kv_lora_rank"]),
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

                # Create Q, K, V tensors with separate MLA components
                torch.manual_seed(config["seed_qkv"])
                q_nope, q_rope, k_nope, k_rope, v = self._create_qkv_tensors(
                    batch_size, config
                )

                # Run forward decode with separate MLA components
                output = trtllm_backend.forward_decode(
                    q_nope, k_nope, None, layer, fb, q_rope=q_rope, k_rope=k_rope
                )

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

        for test_case in TEST_CASES["output_match"]:
            with self.subTest(test_case=test_case["name"]):
                print(f"  Testing {test_case['name']}: {test_case['description']}")

                config = self._merge_config(test_case)
                batch_size = config["batch_size"]
                max_seq_len = config["max_seq_len"]
                use_fp8 = config["kv_cache_dtype"] == torch.float8_e4m3fn

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

                q_nope_ref, q_rope_ref, k_nope_ref, k_rope_ref, v_ref = (
                    self._create_qkv_tensors(batch_size, config)
                )
                q_nope_trt, q_rope_trt, k_nope_trt, k_rope_trt, v_trt = (
                    q_nope_ref.clone(),
                    q_rope_ref.clone(),
                    k_nope_ref.clone(),
                    k_rope_ref.clone(),
                    v_ref.clone(),
                )
                tolerance = config["tolerance"]

                extra_args = {}
                if use_fp8:
                    # TRT kernel applies RoPE + FP8 quantization internally
                    # pre-apply RoPE on the reference (FlashInfer) path here so
                    # both paths share the same rope params/cache while keeping
                    # the TRT path unrotated.
                    rotary_emb = build_rotary_emb(config)
                    q_rope_ref, k_rope_ref = rotary_emb(
                        fb_reference.positions, q_rope_ref, k_rope_ref
                    )
                    extra_args = {
                        "cos_sin_cache": rotary_emb.cos_sin_cache,
                        "is_neox": rotary_emb.is_neox_style,
                    }

                    dtype = q_rope_ref.dtype
                    q_rope_ref = q_rope_ref.to(torch.float8_e4m3fn).to(dtype)
                    q_nope_ref = q_nope_ref.to(torch.float8_e4m3fn).to(dtype)
                    k_rope_ref = k_rope_ref.to(torch.float8_e4m3fn).to(dtype)
                    k_nope_ref = k_nope_ref.to(torch.float8_e4m3fn).to(dtype)

                # Run forward decode on both backends
                out_trtllm = trtllm_backend.forward_decode(
                    q_nope_trt,
                    k_nope_trt,
                    None,
                    layer,
                    fb_trtllm,
                    q_rope=q_rope_trt,
                    k_rope=k_rope_trt,
                    **extra_args,
                )

                # Reference backend should also take separate components, not concatenated
                out_reference = reference_backend.forward_decode(
                    q_nope_ref,
                    k_nope_ref,
                    v_ref,
                    layer,
                    fb_reference,
                    q_rope=q_rope_ref,
                    k_rope=k_rope_ref,
                )

                # Compare outputs
                comparison_passed = compare_outputs(
                    out_trtllm, out_reference, tolerance=tolerance
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

                # Create Q, K, V tensors with separate MLA components
                torch.manual_seed(config["seed_qkv"])
                q_nope, q_rope, k_nope, k_rope, v = self._create_qkv_tensors(
                    batch_size, config
                )

                # Run forward decode with separate MLA components
                output = backend.forward_decode(
                    q_nope, k_nope, None, layer, fb, q_rope=q_rope, k_rope=k_rope
                )

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

                # Create Q, K, V tensors with separate MLA components
                torch.manual_seed(config["seed_qkv"])
                q_nope = torch.randn(
                    (batch_size, config["num_attention_heads"], config["kv_lora_rank"]),
                    dtype=config["dtype"],
                    device=config["device"],
                )
                k_nope = torch.randn(
                    (batch_size, config["num_kv_heads"], config["kv_lora_rank"]),
                    dtype=config["dtype"],
                    device=config["device"],
                )
                q_rope = torch.randn(
                    (
                        batch_size,
                        config["num_attention_heads"],
                        config["qk_rope_head_dim"],
                    ),
                    dtype=config["dtype"],
                    device=config["device"],
                )
                k_rope = torch.randn(
                    (batch_size, config["num_kv_heads"], config["qk_rope_head_dim"]),
                    dtype=config["dtype"],
                    device=config["device"],
                )
                v = None  # Test with None v

                # Run forward decode
                output = backend.forward_decode(
                    q_nope, k_nope, v, layer, fb, q_rope=q_rope, k_rope=k_rope
                )

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
                self.assertIsNotNone(backend.forward_decode_metadata)
                self.assertIsInstance(
                    backend.forward_decode_metadata, TRTLLMMLADecodeMetadata
                )

                # Test metadata structure
                metadata = backend.forward_decode_metadata
                self.assertIsNotNone(
                    metadata.block_kv_indices, "Block KV indices should be created"
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
                triton_constraint = get_num_page_per_block_flashmla(
                    scenario["page_size"]
                )
                constraint_lcm = math.lcm(trtllm_constraint, triton_constraint)
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
                metadata = backend.forward_decode_metadata

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
        self.assertIsNotNone(backend.decode_cuda_graph_kv_indices)

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
        replay_metadata = backend.forward_decode_metadata
        self.assertIsNotNone(replay_metadata)

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
        metadata_1 = backend.forward_decode_metadata

        # Second call with same sequence lengths
        seq_lens_2 = torch.tensor([32, 48], device=config["device"])
        fb_2 = self._create_forward_batch(
            config["batch_size"], seq_lens_2, backend, model_runner, config
        )
        backend.init_forward_metadata(fb_2)
        metadata_2 = backend.forward_decode_metadata

        # Metadata structure should be consistent
        self.assertEqual(
            metadata_1.block_kv_indices.shape, metadata_2.block_kv_indices.shape
        )

        # Third call with different sequence lengths
        seq_lens_3 = torch.tensor([16, 64], device=config["device"])
        fb_3 = self._create_forward_batch(
            config["batch_size"], seq_lens_3, backend, model_runner, config
        )
        backend.init_forward_metadata(fb_3)
        metadata_3 = backend.forward_decode_metadata

        # Should still have valid structure
        self.assertIsNotNone(metadata_3.block_kv_indices)
        self.assertEqual(metadata_3.block_kv_indices.shape[0], config["batch_size"])

    def test_prefill_output_match_self_attention(self):
        """Test prefill (forward) behavior of TRTLLM MLA backend vs reference."""
        print(f"\nRunning prefill output tests...")

        for test_case in TEST_CASES["output_match"][:2]:  # Just a subset for speed
            with self.subTest(test_case=test_case["name"]):
                print(
                    f"Prefill Testing {test_case['name']}: {test_case['description']}"
                )

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
                ) = self._create_model_components(config, is_prefill=True)

                # Prefill uses full sequences
                seq_lens = torch.full(
                    (batch_size,), max_seq_len, device=config["device"]
                )

                def _create_forward_batch_prefill(
                    batch_size,
                    seq_lens,
                    extend_prefix_lens,
                    backend,
                    model_runner,
                    config,
                ):
                    """Create a forward batch for the given backend."""

                    fb = ForwardBatch(
                        batch_size=batch_size,
                        input_ids=torch.randint(
                            0, 100, (batch_size, 1), device=config["device"]
                        ),
                        out_cache_loc=torch.arange(batch_size, device=config["device"]),
                        seq_lens_sum=int(seq_lens.sum().item()),
                        extend_prefix_lens=extend_prefix_lens,
                        extend_prefix_lens_cpu=extend_prefix_lens.cpu().int().tolist(),
                        extend_seq_lens_cpu=(seq_lens - extend_prefix_lens)
                        .cpu()
                        .int()
                        .tolist(),
                        forward_mode=ForwardMode.EXTEND,
                        req_pool_indices=torch.arange(
                            batch_size, device=config["device"]
                        ),
                        seq_lens=seq_lens,
                        seq_lens_cpu=seq_lens.cpu(),
                        attn_attend_prefix_cache=False,
                        mha_return_lse=False,
                        attn_backend=backend,
                    )
                    fb.req_to_token_pool = model_runner.req_to_token_pool
                    fb.token_to_kv_pool = model_runner.token_to_kv_pool

                    # Add position information for RoPE
                    fb.positions = torch.arange(batch_size, device=config["device"])

                    return fb

                # Create forward batches
                fb_trtllm = _create_forward_batch_prefill(
                    batch_size,
                    seq_lens.clone(),
                    torch.zeros(batch_size, device=config["device"], dtype=torch.int32),
                    trtllm_backend,
                    model_runner_trtllm,
                    config,
                )
                fb_reference = _create_forward_batch_prefill(
                    batch_size,
                    seq_lens.clone(),
                    torch.zeros(batch_size, device=config["device"], dtype=torch.int32),
                    reference_backend,
                    model_runner_reference,
                    config,
                )

                # Initialize metadata for both backends
                trtllm_backend.init_forward_metadata(fb_trtllm)
                reference_backend.init_forward_metadata(fb_reference)

                # Create Q, K, V tensors for prefill
                torch.manual_seed(config["seed_qkv"])

                def _create_qkv_tensors_prefill(
                    batch_size, seq_len, config, dtype_override=None
                ):
                    """Create Q, K, V tensors for prefill, using config for head_num and head_dim."""
                    device = config["device"]
                    dtype = dtype_override or config["dtype"]

                    total_tokens = batch_size * seq_len

                    tp_q_head_num = config["tp_q_head_num"]
                    tp_k_head_num = config["tp_k_head_num"]
                    head_dim = config["prefill_head_dim"]
                    v_head_dim = config["prefill_v_head_dim"]

                    q = torch.randn(
                        (total_tokens, tp_q_head_num * head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    k = torch.randn(
                        (total_tokens, tp_k_head_num * head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    v = torch.randn(
                        (total_tokens, tp_k_head_num * v_head_dim),
                        dtype=dtype,
                        device=device,
                    )

                    # Reshape as requested
                    q = q.view(-1, tp_q_head_num, head_dim)
                    k = k.view(-1, tp_k_head_num, head_dim)
                    v = v.view(-1, tp_k_head_num, v_head_dim)

                    return q, k, v

                q, k, v = _create_qkv_tensors_prefill(batch_size, max_seq_len, config)
                # Run prefill on both backends
                out_trtllm = trtllm_backend.forward_extend(
                    q, k, v, layer, fb_trtllm, False
                ).view(-1, layer.tp_q_head_num * layer.v_head_dim)
                out_reference = reference_backend.forward_extend(
                    q, k, v, layer, fb_reference, False
                )

                tolerance = config.get("tolerance", 1e-2)
                comparison_passed = compare_outputs(
                    out_trtllm, out_reference, tolerance=tolerance
                )
                self.assertTrue(
                    comparison_passed,
                    f"TRTLLM and Reference prefill outputs differ beyond tolerance. "
                    f"Config: {test_case['name']}, "
                    f"Max diff: {(out_trtllm - out_reference).abs().max().item()}",
                )

    def test_draft_extend_padding_unpadding_kernels(self):
        """Test TRTLLM MLA Triton kernels: pad_draft_extend_query_kernel and unpad_draft_extend_output_kernel."""

        # Import the kernels
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            pad_draft_extend_query_kernel,
            unpad_draft_extend_output_kernel,
        )

        def _create_test_data(
            self, batch_size, max_seq_len, num_heads, head_dim, dtype=torch.float32
        ):
            """Create test data for kernel testing."""
            device = torch.device("cuda")

            # Create sequence lengths (varying lengths for each batch)
            seq_lens = torch.randint(
                1, max_seq_len + 1, (batch_size,), device=device, dtype=torch.int32
            )

            # Create cumulative sequence lengths
            cum_seq_lens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
            cum_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)

            # Create input query tensor (flattened format)
            total_tokens = cum_seq_lens[-1].item()
            q_input = torch.randn(
                total_tokens, num_heads, head_dim, device=device, dtype=dtype
            )

            # Create padded query tensor (batch format)
            padded_q = torch.zeros(
                batch_size, max_seq_len, num_heads, head_dim, device=device, dtype=dtype
            )

            return q_input, padded_q, seq_lens, cum_seq_lens

        def _create_test_output_data(
            self,
            batch_size,
            token_per_batch,
            tp_q_head_num,
            v_head_dim,
            dtype=torch.float32,
        ):
            """Create test data for unpad kernel testing."""
            device = torch.device("cuda")

            # Create accept lengths (varying lengths for each batch)
            accept_lengths = torch.randint(
                1, token_per_batch + 1, (batch_size,), device=device, dtype=torch.int32
            )

            # Create cumulative accept lengths
            cum_accept_lengths = torch.zeros(
                batch_size + 1, device=device, dtype=torch.int32
            )
            cum_accept_lengths[1:] = torch.cumsum(accept_lengths, dim=0)

            # Create raw output tensor (batch format)
            raw_out = torch.randn(
                batch_size,
                token_per_batch,
                tp_q_head_num,
                v_head_dim,
                device=device,
                dtype=dtype,
            )

            # Create output tensor (flattened format)
            total_tokens = cum_accept_lengths[-1].item()
            output = torch.empty(
                total_tokens, tp_q_head_num, v_head_dim, device=device, dtype=dtype
            )

            return raw_out, output, accept_lengths, cum_accept_lengths

        # Test 1: pad_draft_extend_query_kernel basic functionality
        with self.subTest(test="pad_kernel_basic"):
            batch_size = 4
            max_seq_len = 8
            num_heads = 16
            head_dim = 64

            q_input, padded_q, seq_lens, cum_seq_lens = _create_test_data(
                self, batch_size, max_seq_len, num_heads, head_dim
            )

            # Launch kernel
            BLOCK_SIZE = 64
            grid = (batch_size * max_seq_len,)

            pad_draft_extend_query_kernel[grid](
                q_ptr=q_input,
                padded_q_ptr=padded_q,
                seq_lens_q_ptr=seq_lens,
                cumsum_ptr=cum_seq_lens,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            # Verify the padding worked correctly
            for i in range(batch_size):
                seq_len = seq_lens[i].item()

                # Check that valid positions are copied correctly
                for pos in range(seq_len):
                    input_start = cum_seq_lens[i].item()
                    input_pos = input_start + pos

                    # Compare input and output for valid positions
                    input_data = q_input[input_pos]
                    output_data = padded_q[i, pos]

                    torch.testing.assert_close(
                        input_data, output_data, rtol=1e-5, atol=1e-6
                    )

                # Check that invalid positions are zero
                for pos in range(seq_len, max_seq_len):
                    output_data = padded_q[i, pos]
                    self.assertTrue(
                        torch.allclose(output_data, torch.zeros_like(output_data)),
                        f"Position {pos} in batch {i} should be zero",
                    )

        # Test 2: unpad_draft_extend_output_kernel basic functionality
        with self.subTest(test="unpad_kernel_basic"):
            batch_size = 4
            token_per_batch = 8
            tp_q_head_num = 16
            v_head_dim = 64

            raw_out, output, accept_lengths, cum_accept_lengths = (
                _create_test_output_data(
                    self, batch_size, token_per_batch, tp_q_head_num, v_head_dim
                )
            )

            # Launch kernel
            BLOCK_SIZE = 64
            grid = (batch_size * token_per_batch,)

            unpad_draft_extend_output_kernel[grid](
                raw_out_ptr=raw_out,
                output_ptr=output,
                accept_length_ptr=accept_lengths,
                cumsum_ptr=cum_accept_lengths,
                batch_size=batch_size,
                token_per_batch=token_per_batch,
                tp_q_head_num=tp_q_head_num,
                v_head_dim=v_head_dim,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            # Verify the unpadding worked correctly
            for i in range(batch_size):
                accept_len = accept_lengths[i].item()
                output_start = cum_accept_lengths[i].item()

                # Check that valid positions are copied correctly
                for pos in range(accept_len):
                    input_data = raw_out[i, pos]
                    output_data = output[output_start + pos]

                    torch.testing.assert_close(
                        input_data, output_data, rtol=1e-5, atol=1e-6
                    )


if __name__ == "__main__":
    unittest.main()
