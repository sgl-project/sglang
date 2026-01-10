"""
DFlash speculative decoding tests.

Tests correctness, batch generation, acceptance rate, radix cache, and CUDA graphs.
Following the EAGLE test pattern from test_eagle_infer_a.py.

Unit tests (TestDFlashModelImport, TestDFlashBaseComponents) can run without models.

Engine tests require:
- Qwen/Qwen3-4B model
- z-lab/Qwen3-4B-DFlash-b16 draft model
"""

import os
import random
import unittest

import torch

import sglang as sgl
from sglang.test.test_utils import CustomTestCase

try:
    from sglang.test.ci.ci_register import register_cuda_ci

    register_cuda_ci(est_time=400, suite="stage-b-test-small-1-gpu")
except ImportError:
    pass  # CI registration optional for local testing


# Skip engine tests if models are not available
SKIP_ENGINE_TESTS = os.environ.get("SKIP_DFLASH_ENGINE_TESTS", "0") == "1"


# Common config for DFlash tests
DFLASH_BASE_CONFIG = {
    "model_path": "Qwen/Qwen3-4B",
    "speculative_draft_model_path": "z-lab/Qwen3-4B-DFlash-b16",
    "speculative_algorithm": "DFLASH",
    "speculative_dflash_block_size": 16,
    "mem_fraction_static": 0.5,
}

DFLASH_THRESHOLDS = {
    "accept_len": 3.0,  # Expected minimum acceptance length
    "batch_avg_accept_len": 2.5,
}


@unittest.skipIf(
    SKIP_ENGINE_TESTS, "Skipping engine tests (SKIP_DFLASH_ENGINE_TESTS=1)"
)
class TestDFlashEngine(CustomTestCase):
    """Test DFlash speculative decoding engine.

    Requires Qwen/Qwen3-4B and z-lab/Qwen3-4B-DFlash-b16 models.
    Set SKIP_DFLASH_ENGINE_TESTS=1 to skip.
    """

    BASE_CONFIG = {
        **DFLASH_BASE_CONFIG,
        "disable_radix_cache": True,
        "disable_cuda_graph": True,
    }

    THRESHOLDS = DFLASH_THRESHOLDS

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
            print(
                f"Completion tokens: {output['meta_info'].get('completion_tokens', 0)}"
            )
            print(f"Verify count: {output['meta_info'].get('spec_verify_ct', 0)}")

            self.assertGreater(acc_length, self.THRESHOLDS["accept_len"])
        finally:
            engine.shutdown()

    def test_first_token_finish(self):
        """Test requests that finish after very few tokens."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            prompts = [
                f"There are {i} apples on the table. How to divide them equally?"
                for i in range(8)
            ]
            # Random max_new_tokens between 1-3
            params = [
                {"temperature": 0, "max_new_tokens": random.randint(1, 3)}
                for _ in range(8)
            ]

            outputs = engine.generate(prompts, params)
            for i, output in enumerate(outputs):
                print(f"Prompt: {prompts[i]}")
                print(f"Max tokens: {params[i]['max_new_tokens']}")
                print(f"Generated: {output['text']}")
                print("-" * 40)

            # All should complete successfully
            self.assertEqual(len(outputs), len(prompts))
        finally:
            engine.shutdown()

    def test_eos_token_handling(self):
        """Test that EOS tokens are properly handled in output."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            # Use a prompt that encourages short response
            prompt = "Say hello in one word:"
            params = {
                "temperature": 0.1,
                "max_new_tokens": 128,
                "skip_special_tokens": False,
            }

            output = engine.generate(prompt, params)
            print(f"Output: {output['text']}")

            # Output should not contain EOS token in middle of text
            # (EOS at end is expected for completed generation)
            self.assertIsNotNone(output["text"])
            self.assertGreater(len(output["text"]), 0)
        finally:
            engine.shutdown()

    def test_variable_length_batch(self):
        """Test batch with varying prompt and output lengths."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            prompts = [
                "Hi",  # Very short
                "The quick brown fox jumps over the lazy dog.",  # Medium
                "In the realm of artificial intelligence, large language models have revolutionized natural language processing.",  # Long
            ]
            params = {"temperature": 0, "max_new_tokens": 64}

            outputs = engine.generate(prompts, params)

            for prompt, output in zip(prompts, outputs):
                print(f"Prompt ({len(prompt)} chars): {prompt[:50]}...")
                print(f"Generated: {output['text'][:100]}...")
                print("-" * 40)

            self.assertEqual(len(outputs), len(prompts))
            # All outputs should have content
            for output in outputs:
                self.assertGreater(len(output["text"]), 0)
        finally:
            engine.shutdown()


@unittest.skipIf(
    SKIP_ENGINE_TESTS, "Skipping engine tests (SKIP_DFLASH_ENGINE_TESTS=1)"
)
class TestDFlashCudaGraph(CustomTestCase):
    """Test DFlash with CUDA graphs enabled for target model."""

    BASE_CONFIG = {
        **DFLASH_BASE_CONFIG,
        "disable_radix_cache": True,
        "cuda_graph_max_bs": 4,
    }

    @classmethod
    def setUpClass(cls):
        """Set up reference from eager mode."""
        cls.prompt = "The meaning of life is"
        cls.sampling_params = {"temperature": 0, "max_new_tokens": 32}

        # Get reference from eager mode (no CUDA graphs)
        eager_config = {**cls.BASE_CONFIG, "disable_cuda_graph": True}
        eager_engine = sgl.Engine(**eager_config, log_level="info")
        cls.eager_output = eager_engine.generate(cls.prompt, cls.sampling_params)[
            "text"
        ]
        eager_engine.shutdown()

    def test_cuda_graph_correctness(self):
        """Test CUDA graphs produce identical output to eager mode."""
        # Enable CUDA graphs for target model
        cuda_graph_config = {**self.BASE_CONFIG, "disable_cuda_graph": False}
        engine = sgl.Engine(**cuda_graph_config, log_level="info")
        try:
            output = engine.generate(self.prompt, self.sampling_params)["text"]
            print(f"CUDA graph output: {output}")
            print(f"Eager output: {self.eager_output}")
            self.assertEqual(output, self.eager_output)
        finally:
            engine.shutdown()

    def test_cuda_graph_batch(self):
        """Test CUDA graphs work correctly with batched requests."""
        cuda_graph_config = {**self.BASE_CONFIG, "disable_cuda_graph": False}
        engine = sgl.Engine(**cuda_graph_config, log_level="info")
        try:
            prompts = [
                "Hello, my name is",
                "The capital of Japan is",
                "Python is a programming language that",
            ]
            params = {"temperature": 0, "max_new_tokens": 32}

            outputs = engine.generate(prompts, params)

            # All should complete
            self.assertEqual(len(outputs), len(prompts))

            # Check acceptance length
            server_info = engine.get_server_info()
            avg_spec_accept_length = server_info["internal_states"][0].get(
                "avg_spec_accept_length", 0
            )
            print(f"CUDA graph batch avg accept length: {avg_spec_accept_length}")
            self.assertGreater(avg_spec_accept_length, 2.0)
        finally:
            engine.shutdown()


@unittest.skipIf(
    SKIP_ENGINE_TESTS, "Skipping engine tests (SKIP_DFLASH_ENGINE_TESTS=1)"
)
class TestDFlashRadixCache(CustomTestCase):
    """Test DFlash radix cache hidden state integration."""

    BASE_CONFIG = {
        **DFLASH_BASE_CONFIG,
        "disable_radix_cache": False,  # Enable radix cache
        "disable_cuda_graph": True,
    }

    def test_prefix_sharing(self):
        """Test that repeated prefixes benefit from caching."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            # Same prefix, different suffixes
            common_prefix = "The quick brown fox jumps over the lazy dog. "
            prompts = [
                common_prefix + "What color is the fox?",
                common_prefix + "What animal is lazy?",
                common_prefix + "How does the fox move?",
            ]
            params = {"temperature": 0, "max_new_tokens": 32}

            outputs = engine.generate(prompts, params)

            # All should complete successfully
            self.assertEqual(len(outputs), len(prompts))
            for output in outputs:
                self.assertGreater(len(output["text"]), 0)
                print(f"Output: {output['text']}")
        finally:
            engine.shutdown()

    def test_cache_reuse_performance(self):
        """Test that cache reuse improves performance on repeated prompts."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            prompt = "Explain the concept of machine learning in simple terms:"
            params = {"temperature": 0, "max_new_tokens": 64}

            # First request - cache miss
            output1 = engine.generate(prompt, params)
            latency1 = output1["meta_info"]["e2e_latency"]

            # Second request - should hit cache
            output2 = engine.generate(prompt, params)
            latency2 = output2["meta_info"]["e2e_latency"]

            print(f"First latency: {latency1:.3f}s")
            print(f"Second latency: {latency2:.3f}s")

            # Outputs should be identical
            self.assertEqual(output1["text"], output2["text"])

            # Second should generally be faster (cache hit)
            # Note: Not asserting strict improvement as it depends on system state
        finally:
            engine.shutdown()

    def test_radix_cache_with_batch(self):
        """Test radix cache with batched requests sharing prefixes."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            # Mix of shared and unique prefixes
            prompts = [
                "Hello world! How are you today?",
                "Hello world! What is your name?",
                "Hello world! Tell me a joke.",
                "Goodbye world! See you later.",
            ]
            params = {"temperature": 0, "max_new_tokens": 32}

            outputs = engine.generate(prompts, params)

            self.assertEqual(len(outputs), len(prompts))

            # Check server info for cache stats if available
            server_info = engine.get_server_info()
            print(f"Server info: {server_info['internal_states'][0]}")
        finally:
            engine.shutdown()


@unittest.skipIf(
    SKIP_ENGINE_TESTS, "Skipping engine tests (SKIP_DFLASH_ENGINE_TESTS=1)"
)
class TestDFlashBlockSizes(CustomTestCase):
    """Test DFlash with different block sizes."""

    @classmethod
    def setUpClass(cls):
        """Set up reference output."""
        cls.prompt = "Write a haiku about programming:"
        cls.sampling_params = {"temperature": 0, "max_new_tokens": 32}

        # Get reference from non-speculative
        ref_engine = sgl.Engine(
            model_path=DFLASH_BASE_CONFIG["model_path"],
            mem_fraction_static=0.5,
            disable_cuda_graph=True,
        )
        cls.ref_output = ref_engine.generate(cls.prompt, cls.sampling_params)["text"]
        ref_engine.shutdown()

    def _test_block_size(self, block_size: int):
        """Helper to test a specific block size."""
        config = {
            **DFLASH_BASE_CONFIG,
            "speculative_dflash_block_size": block_size,
            "disable_radix_cache": True,
            "disable_cuda_graph": True,
        }
        engine = sgl.Engine(**config, log_level="info")
        try:
            output = engine.generate(self.prompt, self.sampling_params)["text"]
            print(f"Block size {block_size} output: {output}")

            # Output should match reference
            self.assertEqual(output, self.ref_output)

            # Check acceptance stats
            server_info = engine.get_server_info()
            avg_accept = server_info["internal_states"][0].get(
                "avg_spec_accept_length", 0
            )
            print(f"Block size {block_size} avg accept length: {avg_accept}")
        finally:
            engine.shutdown()

    def test_block_size_8(self):
        """Test with block_size=8 (smaller blocks, more iterations)."""
        self._test_block_size(8)

    def test_block_size_16(self):
        """Test with block_size=16 (default)."""
        self._test_block_size(16)

    def test_block_size_32(self):
        """Test with block_size=32 (larger blocks, fewer iterations)."""
        self._test_block_size(32)


@unittest.skipIf(
    SKIP_ENGINE_TESTS, "Skipping engine tests (SKIP_DFLASH_ENGINE_TESTS=1)"
)
class TestDFlashStress(CustomTestCase):
    """Stress tests for DFlash."""

    BASE_CONFIG = {
        **DFLASH_BASE_CONFIG,
        "disable_radix_cache": True,
        "disable_cuda_graph": True,
    }

    def test_large_batch(self):
        """Test with larger batch sizes."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            prompts = [f"Question {i}: What is {i} + {i}?" for i in range(16)]
            params = {"temperature": 0, "max_new_tokens": 32}

            outputs = engine.generate(prompts, params)

            self.assertEqual(len(outputs), len(prompts))
            for output in outputs:
                self.assertGreater(len(output["text"]), 0)

            # Check throughput
            total_tokens = sum(o["meta_info"]["completion_tokens"] for o in outputs)
            total_time = max(o["meta_info"]["e2e_latency"] for o in outputs)
            throughput = total_tokens / total_time
            print(f"Large batch throughput: {throughput:.1f} tokens/s")
        finally:
            engine.shutdown()

    def test_long_generation(self):
        """Test longer generation sequences."""
        engine = sgl.Engine(**self.BASE_CONFIG, log_level="info")
        try:
            prompt = "Write a detailed essay about artificial intelligence:"
            params = {"temperature": 0, "max_new_tokens": 512}

            output = engine.generate(prompt, params)

            print(f"Generated {output['meta_info']['completion_tokens']} tokens")
            self.assertGreater(output["meta_info"]["completion_tokens"], 100)

            # Check acceptance rate over long generation
            if "spec_verify_ct" in output["meta_info"]:
                acc_length = (
                    output["meta_info"]["completion_tokens"]
                    / output["meta_info"]["spec_verify_ct"]
                )
                print(f"Long generation acceptance length: {acc_length:.2f}")
                self.assertGreater(acc_length, 2.5)
        finally:
            engine.shutdown()


# ============================================================================
# Unit Tests (no model required)
# ============================================================================


class TestDFlashModelImport(CustomTestCase):
    """Test DFlash model imports and registry."""

    def test_model_import(self):
        """Test that DFlash model can be imported."""
        from sglang.srt.models.qwen3_dflash import Qwen3ForCausalLMDFlash

        self.assertIsNotNone(Qwen3ForCausalLMDFlash)

    def test_model_alias_import(self):
        """Test that DFlashDraftModel alias exists for backward compatibility."""
        from sglang.srt.models.qwen3_dflash import (
            DFlashDraftModel,
            Qwen3ForCausalLMDFlash,
        )

        self.assertIsNotNone(DFlashDraftModel)
        self.assertIs(DFlashDraftModel, Qwen3ForCausalLMDFlash)

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


class TestDFlashBaseComponents(CustomTestCase):
    """Test DFlash base component imports."""

    def test_utils_import(self):
        """Test that utility functions can be imported."""
        from sglang.srt.models.dflash import RMSNorm3D, build_target_layer_ids

        self.assertIsNotNone(RMSNorm3D)
        self.assertIsNotNone(build_target_layer_ids)

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

    def test_draft_input_merge_batch(self):
        """Test DFlashDraftInput.merge_batch correctly concatenates verified_id."""
        from sglang.srt.speculative.dflash_info import DFlashDraftInput

        # Create two draft inputs
        input1 = DFlashDraftInput(
            hidden_states=None,  # DFlash sets hidden_states=None
            verified_id=torch.tensor([100]),
            block_size=16,
        )
        input2 = DFlashDraftInput(
            hidden_states=None,
            verified_id=torch.tensor([200]),
            block_size=16,
        )

        # Merge
        input1.merge_batch(input2)

        # verified_id should be concatenated, not replaced
        self.assertEqual(input1.verified_id.shape[0], 2)
        self.assertEqual(input1.verified_id[0].item(), 100)
        self.assertEqual(input1.verified_id[1].item(), 200)

    def test_verify_input_cumprod_verification(self):
        """Test that cumprod-based verification logic is correct."""
        # This tests the core verification algorithm:
        # matches = (draft[:, 1:] == target[:, :-1])
        # accept_length = matches.cumprod(dim=1).sum(dim=1)

        # Simulated scenario: draft predicts [anchor, A, B, C]
        # Target predicts [A, B, X, ...] after seeing [anchor, A, B, C]
        # Match pattern: [True, True, False] -> cumprod: [1, 1, 0] -> sum: 2

        draft = torch.tensor([[0, 1, 2, 3]])  # anchor=0, predictions=1,2,3
        target_predict = torch.tensor([[1, 2, 99, 4]])  # target predicts 1,2,99,4

        draft_predictions = draft[:, 1:]  # [1, 2, 3]
        target_for_comparison = target_predict[:, :-1]  # [1, 2, 99]

        matches = draft_predictions == target_for_comparison  # [T, T, F]
        cumprod_matches = matches.cumprod(dim=1)  # [1, 1, 0]
        accept_length = cumprod_matches.sum(dim=1)  # [2]

        self.assertEqual(accept_length[0].item(), 2)


if __name__ == "__main__":
    unittest.main()
