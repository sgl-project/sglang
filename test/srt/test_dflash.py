"""
DFlash speculative decoding tests.

Tests correctness, batch generation, and acceptance rate for DFlash.
Following the EAGLE test pattern from test_eagle_infer_a.py.

Unit tests (TestDFlashModelImport, TestDFlashBaseComponents) can run without models.

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
        from sglang.srt.models.dflash import (
            RMSNorm3D,
            build_target_layer_ids,
        )

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


if __name__ == "__main__":
    unittest.main()
