"""
DFlash speculative decoding tests.

Tests correctness, batch generation, and acceptance rate for DFlash.
Following the EAGLE test pattern from test_eagle_infer_a.py.

Unit tests (TestDFlashModelImport, TestDFlashBaseComponents, TestDFlashKVCacheManager,
TestDFlashInfoClasses) can run without models.

Engine tests (TestDFlashEngine) require:
- Qwen/Qwen3-4B model
- z-lab/Qwen3-4B-DFlash-b16 draft model
"""

import os
import unittest

import torch

import sglang as sgl
from sglang.test.test_utils import CustomTestCase


# Skip engine tests if models are not available
SKIP_ENGINE_TESTS = os.environ.get("SKIP_DFLASH_ENGINE_TESTS", "0") == "1"


@unittest.skipIf(SKIP_ENGINE_TESTS, "Skipping engine tests (SKIP_DFLASH_ENGINE_TESTS=1)")
class TestDFlashEngine(CustomTestCase):
    """Test DFlash speculative decoding engine.
    
    Requires Qwen/Qwen3-4B and z-lab/Qwen3-4B-DFlash-b16 models.
    Set SKIP_DFLASH_ENGINE_TESTS=1 to skip.
    """

    BASE_CONFIG = {
        "model_path": "Qwen/Qwen3-4B",
        "speculative_draft_model_path": "z-lab/Qwen3-4B-DFlash-b16",
        "speculative_algorithm": "DFLASH",
        "speculative_dflash_block_size": 16,
        "mem_fraction_static": 0.5,
        "disable_radix_cache": True,
        "disable_cuda_graph": True,
    }

    THRESHOLDS = {
        "accept_len": 3.0,  # Expected minimum acceptance length
        "batch_avg_accept_len": 2.5,
    }

    @classmethod
    def setUpClass(cls):
        """Set up reference output from non-speculative baseline."""
        cls.prompt = "Today is a sunny day and I like"
        cls.sampling_params = {"temperature": 0, "max_new_tokens": 32}

        # Get reference output from non-speculative engine
        ref_engine = sgl.Engine(
            model_path=cls.BASE_CONFIG["model_path"],
            mem_fraction_static=0.5,
            disable_cuda_graph=True,
        )
        cls.ref_output = ref_engine.generate(cls.prompt, cls.sampling_params)["text"]
        ref_engine.shutdown()

    def test_single_correctness(self):
        """Test that DFlash output matches non-speculative baseline."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            output = engine.generate(self.prompt, self.sampling_params)["text"]
            print(f"DFlash output: {output}")
            print(f"Reference output: {self.ref_output}")
            self.assertEqual(output, self.ref_output)
        finally:
            engine.shutdown()

    def test_batch_generation(self):
        """Test batch generation with multiple prompts."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]
            params = {"temperature": 0, "max_new_tokens": 50}

            outputs = engine.generate(prompts, params)
            for prompt, output in zip(prompts, outputs):
                print(f"Prompt: {prompt}")
                print(f"Generated: {output['text']}")
                print("-" * 40)

            # Verify we got outputs for all prompts
            self.assertEqual(len(outputs), len(prompts))

            # Check acceptance length from server info
            server_info = engine.get_server_info()
            avg_spec_accept_length = server_info["internal_states"][0].get(
                "avg_spec_accept_length", 0
            )
            print(f"Average spec accept length: {avg_spec_accept_length}")
            self.assertGreater(
                avg_spec_accept_length, self.THRESHOLDS["batch_avg_accept_len"]
            )
        finally:
            engine.shutdown()

    def test_acceptance_length(self):
        """Test that acceptance length meets threshold."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            # Use a longer prompt to get meaningful acceptance stats
            prompt = "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:"
            params = {"temperature": 0, "max_new_tokens": 256}

            output = engine.generate(prompt, params)

            # Calculate acceptance length from meta info
            if "spec_verify_ct" in output["meta_info"]:
                acc_length = (
                    output["meta_info"]["completion_tokens"]
                    / output["meta_info"]["spec_verify_ct"]
                )
            else:
                acc_length = 1.0

            print(f"Acceptance length: {acc_length:.2f}")
            print(f"Completion tokens: {output['meta_info'].get('completion_tokens', 0)}")
            print(f"Verify count: {output['meta_info'].get('spec_verify_ct', 0)}")

            self.assertGreater(acc_length, self.THRESHOLDS["accept_len"])
        finally:
            engine.shutdown()


class TestDFlashModelImport(CustomTestCase):
    """Test DFlash model imports and registry."""

    def test_model_import(self):
        """Test that DFlash model can be imported."""
        from sglang.srt.models.qwen3_dflash import Qwen3ForCausalLMDFlash

        self.assertIsNotNone(Qwen3ForCausalLMDFlash)

    def test_model_alias_import(self):
        """Test that DFlashDraftModel alias exists for backward compatibility."""
        from sglang.srt.models.qwen3_dflash import DFlashDraftModel, Qwen3ForCausalLMDFlash

        self.assertIsNotNone(DFlashDraftModel)
        self.assertIs(DFlashDraftModel, Qwen3ForCausalLMDFlash)

    def test_registry_resolution(self):
        """Test that model resolves from registry."""
        from sglang.srt.models.registry import ModelRegistry
        from sglang.srt.models.qwen3_dflash import Qwen3ForCausalLMDFlash

        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            "Qwen3ForCausalLMDFlash"
        )
        self.assertIsNotNone(model_cls)
        self.assertEqual(resolved_arch, "Qwen3ForCausalLMDFlash")
        self.assertIs(model_cls, Qwen3ForCausalLMDFlash)

    def test_dflash_worker_import(self):
        """Test that DFlashWorker can be imported."""
        from sglang.srt.speculative.dflash_worker import DFlashWorker

        self.assertIsNotNone(DFlashWorker)

    def test_dflash_info_import(self):
        """Test that DFlash info classes can be imported."""
        from sglang.srt.speculative.dflash_info import (
            DFlashDraftInput,
            DFlashVerifyInput,
        )

        self.assertIsNotNone(DFlashDraftInput)
        self.assertIsNotNone(DFlashVerifyInput)

    def test_spec_algorithm_registration(self):
        """Test that DFLASH is registered in SpeculativeAlgorithm."""
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertTrue(hasattr(SpeculativeAlgorithm, "DFLASH"))
        spec_algo = SpeculativeAlgorithm.DFLASH
        self.assertTrue(spec_algo.is_dflash())
        self.assertTrue(spec_algo.supports_spec_v2())


class TestDFlashBaseComponents(CustomTestCase):
    """Test DFlash base component imports and inheritance."""

    def test_base_components_import(self):
        """Test that base components can be imported."""
        from sglang.srt.models.dflash import (
            DFlashAttentionBase,
            DFlashDecoderLayerBase,
            DFlashModelBase,
            RMSNorm3D,
            apply_rotary_pos_emb,
            build_target_layer_ids,
            rotate_half,
        )

        self.assertIsNotNone(DFlashAttentionBase)
        self.assertIsNotNone(DFlashDecoderLayerBase)
        self.assertIsNotNone(DFlashModelBase)
        self.assertIsNotNone(RMSNorm3D)

    def test_qwen3_inherits_from_base(self):
        """Test that Qwen3 components inherit from base classes."""
        from sglang.srt.models.dflash import (
            DFlashAttentionBase,
            DFlashDecoderLayerBase,
            DFlashModelBase,
        )
        from sglang.srt.models.qwen3_dflash import (
            Qwen3DFlashAttention,
            Qwen3DFlashDecoderLayer,
            Qwen3DFlashModel,
        )

        self.assertTrue(issubclass(Qwen3DFlashAttention, DFlashAttentionBase))
        self.assertTrue(issubclass(Qwen3DFlashDecoderLayer, DFlashDecoderLayerBase))
        self.assertTrue(issubclass(Qwen3DFlashModel, DFlashModelBase))

    def test_build_target_layer_ids(self):
        """Test target layer ID computation."""
        from sglang.srt.models.dflash import build_target_layer_ids

        # Single draft layer -> middle of target
        ids = build_target_layer_ids(28, 1)
        self.assertEqual(ids, [14])

        # Multiple draft layers -> distributed evenly
        ids = build_target_layer_ids(28, 3)
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids[0], 1)  # Start
        self.assertEqual(ids[-1], 25)  # End

    def test_rmsnorm3d(self):
        """Test RMSNorm3D works with 3D tensors."""
        from sglang.srt.models.dflash import RMSNorm3D

        norm = RMSNorm3D(64, eps=1e-6)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)


class TestDFlashKVCacheManager(CustomTestCase):
    """Test DFlashKVCacheManager functionality."""

    def test_kv_cache_manager_creation(self):
        """Test KV cache manager instantiation."""
        from sglang.srt.models.dflash import DFlashKVCacheManager

        kv_mgr = DFlashKVCacheManager(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            max_seq_len=1024,
            device="cpu",
            dtype=torch.float32,
        )
        self.assertEqual(kv_mgr.num_layers, 4)
        self.assertEqual(kv_mgr.num_kv_heads, 8)

    def test_kv_cache_update(self):
        """Test KV cache update and retrieval."""
        from sglang.srt.models.dflash import DFlashKVCacheManager

        kv_mgr = DFlashKVCacheManager(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            max_seq_len=512,
            device="cpu",
            dtype=torch.float32,
        )

        # Initial update
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        all_k, all_v = kv_mgr.update("req1", layer_idx=0, k=k, v=v)
        self.assertEqual(all_k.shape[2], 10)
        self.assertEqual(kv_mgr.get_seq_length("req1"), 10)

        # Second update - should concatenate
        k2 = torch.randn(1, 4, 5, 32)
        v2 = torch.randn(1, 4, 5, 32)
        all_k2, all_v2 = kv_mgr.update("req1", layer_idx=0, k=k2, v=v2)
        self.assertEqual(all_k2.shape[2], 15)

    def test_kv_cache_crop(self):
        """Test KV cache cropping."""
        from sglang.srt.models.dflash import DFlashKVCacheManager

        kv_mgr = DFlashKVCacheManager(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            max_seq_len=512,
            device="cpu",
            dtype=torch.float32,
        )

        k = torch.randn(1, 4, 20, 32)
        v = torch.randn(1, 4, 20, 32)
        kv_mgr.update("req1", layer_idx=0, k=k, v=v)

        # Crop to 10
        kv_mgr.crop("req1", 10)
        self.assertEqual(kv_mgr.get_seq_length("req1"), 10)

    def test_kv_cache_clear(self):
        """Test KV cache clearing."""
        from sglang.srt.models.dflash import DFlashKVCacheManager

        kv_mgr = DFlashKVCacheManager(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            max_seq_len=512,
            device="cpu",
            dtype=torch.float32,
        )

        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        kv_mgr.update("req1", layer_idx=0, k=k, v=v)

        kv_mgr.clear("req1")
        self.assertEqual(kv_mgr.get_seq_length("req1"), 0)

    def test_multiple_requests(self):
        """Test KV cache with multiple requests."""
        from sglang.srt.models.dflash import DFlashKVCacheManager

        kv_mgr = DFlashKVCacheManager(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            max_seq_len=512,
            device="cpu",
            dtype=torch.float32,
        )

        # Two different requests
        k1 = torch.randn(1, 4, 10, 32)
        v1 = torch.randn(1, 4, 10, 32)
        k2 = torch.randn(1, 4, 15, 32)
        v2 = torch.randn(1, 4, 15, 32)

        kv_mgr.update("req1", layer_idx=0, k=k1, v=v1)
        kv_mgr.update("req2", layer_idx=0, k=k2, v=v2)

        self.assertEqual(kv_mgr.get_seq_length("req1"), 10)
        self.assertEqual(kv_mgr.get_seq_length("req2"), 15)


class TestDFlashInfoClasses(CustomTestCase):
    """Test DFlash info class functionality."""

    def test_draft_input_creation(self):
        """Test DFlashDraftInput creation."""
        from sglang.srt.speculative.dflash_info import DFlashDraftInput

        draft_input = DFlashDraftInput(
            hidden_states=torch.randn(1, 10, 4096),
            verified_id=torch.tensor([0]),
            block_size=16,
        )
        self.assertEqual(draft_input.block_size, 16)
        self.assertIsNotNone(draft_input.hidden_states)

    def test_verify_input_creation(self):
        """Test DFlashVerifyInput creation."""
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        verify_input = DFlashVerifyInput(
            draft_token=torch.randint(0, 1000, (16,)),
            positions=torch.arange(16),
            block_size=16,
        )
        self.assertEqual(verify_input.draft_token_num, 16)
        self.assertEqual(len(verify_input.draft_token), 16)


class TestDFlashFlashAttention(CustomTestCase):
    """Test DFlash Flash Attention implementation."""

    def test_flash_attention_import(self):
        """Test that flash attention module can be imported."""
        from sglang.srt.layers.attention.dflash_attention import (
            DFlashAttentionMetadata,
            can_use_dflash_flash_attention,
            dflash_attention,
            dflash_eager_attention,
            dflash_flash_attention,
        )

        self.assertIsNotNone(DFlashAttentionMetadata)
        self.assertIsNotNone(dflash_attention)
        self.assertIsNotNone(dflash_eager_attention)
        self.assertIsNotNone(dflash_flash_attention)
        self.assertIsNotNone(can_use_dflash_flash_attention)

    def test_metadata_uniform_batch(self):
        """Test DFlashAttentionMetadata creation for uniform batch."""
        from sglang.srt.layers.attention.dflash_attention import (
            DFlashAttentionMetadata,
        )

        metadata = DFlashAttentionMetadata.from_uniform_batch(
            batch_size=4,
            q_len=8,
            kv_len=24,
            device=torch.device("cpu"),
        )

        self.assertEqual(metadata.max_seqlen_q, 8)
        self.assertEqual(metadata.max_seqlen_kv, 24)
        self.assertEqual(len(metadata.cu_seqlens_q), 5)  # batch_size + 1
        self.assertEqual(len(metadata.cu_seqlens_kv), 5)
        self.assertEqual(metadata.cu_seqlens_q[-1].item(), 32)  # 4 * 8
        self.assertEqual(metadata.cu_seqlens_kv[-1].item(), 96)  # 4 * 24

    def test_metadata_variable_batch(self):
        """Test DFlashAttentionMetadata creation for variable batch."""
        from sglang.srt.layers.attention.dflash_attention import (
            DFlashAttentionMetadata,
        )

        q_lens = torch.tensor([8, 8, 8])
        kv_lens = torch.tensor([20, 30, 25])

        metadata = DFlashAttentionMetadata.from_variable_batch(
            batch_size=3,
            q_lens=q_lens,
            kv_lens=kv_lens,
            device=torch.device("cpu"),
        )

        self.assertEqual(metadata.max_seqlen_q, 8)
        self.assertEqual(metadata.max_seqlen_kv, 30)
        self.assertEqual(metadata.cu_seqlens_q[-1].item(), 24)  # 8+8+8
        self.assertEqual(metadata.cu_seqlens_kv[-1].item(), 75)  # 20+30+25

    def test_eager_attention_basic(self):
        """Test eager attention with basic input."""
        from sglang.srt.layers.attention.dflash_attention import dflash_eager_attention

        batch_size, n_heads, q_len, head_dim = 2, 8, 8, 64
        kv_len = 24
        n_kv_heads = 4

        q = torch.randn(batch_size, n_heads, q_len, head_dim)
        k = torch.randn(batch_size, n_kv_heads, kv_len, head_dim)
        v = torch.randn(batch_size, n_kv_heads, kv_len, head_dim)

        output = dflash_eager_attention(
            q=q,
            k=k,
            v=v,
            softmax_scale=1.0 / (head_dim ** 0.5),
            num_kv_groups=n_heads // n_kv_heads,
        )

        self.assertEqual(output.shape, (batch_size, n_heads, q_len, head_dim))

    def test_eager_attention_gqa(self):
        """Test eager attention with GQA (grouped query attention)."""
        from sglang.srt.layers.attention.dflash_attention import dflash_eager_attention

        batch_size, n_heads, q_len, head_dim = 1, 16, 8, 64
        kv_len = 32
        n_kv_heads = 4
        num_kv_groups = n_heads // n_kv_heads

        q = torch.randn(batch_size, n_heads, q_len, head_dim)
        k = torch.randn(batch_size, n_kv_heads, kv_len, head_dim)
        v = torch.randn(batch_size, n_kv_heads, kv_len, head_dim)

        output = dflash_eager_attention(
            q=q,
            k=k,
            v=v,
            softmax_scale=1.0 / (head_dim ** 0.5),
            num_kv_groups=num_kv_groups,
        )

        self.assertEqual(output.shape, (batch_size, n_heads, q_len, head_dim))

    def test_dflash_attention_cpu_fallback(self):
        """Test that dflash_attention falls back to eager on CPU."""
        from sglang.srt.layers.attention.dflash_attention import dflash_attention

        batch_size, n_heads, q_len, head_dim = 1, 8, 4, 32
        kv_len = 12
        n_kv_heads = 4

        # Eager format: [batch, heads, seq, dim]
        q = torch.randn(batch_size, n_heads, q_len, head_dim)
        k = torch.randn(batch_size, n_kv_heads, kv_len, head_dim)
        v = torch.randn(batch_size, n_kv_heads, kv_len, head_dim)

        output = dflash_attention(
            q=q,
            k=k,
            v=v,
            softmax_scale=1.0 / (head_dim ** 0.5),
            num_kv_groups=n_heads // n_kv_heads,
            use_flash_attention=True,  # Should still fall back on CPU
        )

        self.assertEqual(output.shape, (batch_size, n_heads, q_len, head_dim))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_flash_attention_correctness(self):
        """Test that Flash Attention matches eager attention output."""
        from sglang.srt.layers.attention.dflash_attention import (
            DFlashAttentionMetadata,
            can_use_dflash_flash_attention,
            dflash_eager_attention,
            dflash_flash_attention,
        )

        if not can_use_dflash_flash_attention():
            self.skipTest("Flash Attention not available")

        batch_size, n_heads, q_len, head_dim = 2, 8, 8, 64
        kv_len = 24
        softmax_scale = 1.0 / (head_dim ** 0.5)

        # Create test tensors on CUDA with bf16 (Flash Attention requires fp16/bf16)
        q = torch.randn(batch_size, q_len, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch_size, kv_len, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch_size, kv_len, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)

        # Flash attention
        metadata = DFlashAttentionMetadata.from_uniform_batch(
            batch_size=batch_size,
            q_len=q_len,
            kv_len=kv_len,
            device=torch.device("cuda"),
        )
        flash_output = dflash_flash_attention(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            metadata=metadata,
        )

        # Eager attention (needs transposed format)
        q_eager = q.transpose(1, 2)  # [batch, heads, q_len, dim]
        k_eager = k.transpose(1, 2)
        v_eager = v.transpose(1, 2)
        eager_output = dflash_eager_attention(
            q=q_eager,
            k=k_eager,
            v=v_eager,
            softmax_scale=softmax_scale,
            num_kv_groups=1,
        )
        eager_output = eager_output.transpose(1, 2)  # Back to [batch, seq, heads, dim]

        # Compare outputs
        self.assertEqual(flash_output.shape, eager_output.shape)
        torch.testing.assert_close(
            flash_output, eager_output, rtol=1e-2, atol=1e-2
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_flash_attention_gqa_correctness(self):
        """Test Flash Attention with GQA matches eager."""
        from sglang.srt.layers.attention.dflash_attention import (
            DFlashAttentionMetadata,
            can_use_dflash_flash_attention,
            dflash_eager_attention,
            dflash_flash_attention,
        )

        if not can_use_dflash_flash_attention():
            self.skipTest("Flash Attention not available")

        batch_size, n_heads, q_len, head_dim = 1, 16, 8, 64
        kv_len = 32
        n_kv_heads = 4
        num_kv_groups = n_heads // n_kv_heads
        softmax_scale = 1.0 / (head_dim ** 0.5)

        # Create test tensors on CUDA with bf16 (Flash Attention requires fp16/bf16)
        q = torch.randn(batch_size, q_len, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch_size, kv_len, n_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch_size, kv_len, n_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)

        # Flash attention
        metadata = DFlashAttentionMetadata.from_uniform_batch(
            batch_size=batch_size,
            q_len=q_len,
            kv_len=kv_len,
            device=torch.device("cuda"),
        )
        flash_output = dflash_flash_attention(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            metadata=metadata,
        )

        # Eager attention (needs transposed format and GQA expansion)
        q_eager = q.transpose(1, 2)  # [batch, heads, q_len, dim]
        k_eager = k.transpose(1, 2)
        v_eager = v.transpose(1, 2)
        eager_output = dflash_eager_attention(
            q=q_eager,
            k=k_eager,
            v=v_eager,
            softmax_scale=softmax_scale,
            num_kv_groups=num_kv_groups,
        )
        eager_output = eager_output.transpose(1, 2)

        # Compare outputs
        self.assertEqual(flash_output.shape, eager_output.shape)
        torch.testing.assert_close(
            flash_output, eager_output, rtol=1e-2, atol=1e-2
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_flash_attention_with_kv_cache_manager(self):
        """Test Flash Attention works with DFlashKVCacheManager."""
        from sglang.srt.layers.attention.dflash_attention import (
            can_use_dflash_flash_attention,
        )
        from sglang.srt.models.dflash import DFlashKVCacheManager

        if not can_use_dflash_flash_attention():
            self.skipTest("Flash Attention not available")

        # Create a simple config-like object
        class MockConfig:
            hidden_size = 256
            num_attention_heads = 8
            num_key_value_heads = 4
            head_dim = 32
            rms_norm_eps = 1e-6
            attention_bias = False
            intermediate_size = 512

        config = MockConfig()

        # Import and create attention layer
        from sglang.srt.models.qwen3_dflash import Qwen3DFlashAttention

        attn = Qwen3DFlashAttention(config, layer_idx=0).cuda().to(torch.bfloat16)

        # Create KV cache manager
        kv_mgr = DFlashKVCacheManager(
            num_layers=1,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seq_len=256,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Create inputs
        batch_size, q_len, ctx_len = 1, 8, 16
        hidden_states = torch.randn(
            batch_size, q_len, config.hidden_size,
            device="cuda", dtype=torch.bfloat16
        )
        target_hidden = torch.randn(
            batch_size, ctx_len, config.hidden_size,
            device="cuda", dtype=torch.bfloat16
        )

        # Create position embeddings (simplified)
        seq_len = ctx_len + q_len
        cos = torch.randn(batch_size, seq_len, config.head_dim, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(batch_size, seq_len, config.head_dim, device="cuda", dtype=torch.bfloat16)
        position_embeddings = (cos, sin)

        # Forward with flash attention + KV cache
        output = attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_embeddings=position_embeddings,
            use_cache=True,
            kv_cache_manager=kv_mgr,
            request_id="test_req",
            use_flash_attention=True,
        )

        # Verify output shape
        self.assertEqual(output.shape, (batch_size, q_len, config.hidden_size))

        # Verify KV cache was updated
        self.assertEqual(kv_mgr.get_seq_length("test_req"), ctx_len + q_len)


if __name__ == "__main__":
    unittest.main()

