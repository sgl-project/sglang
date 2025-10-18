"""
Test forward_split_prefill functionality.

Usage:
python3 -m unittest test_forward_split_prefill.TestForwardSplitPrefill
or
python3 test_forward_split_prefill.py
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)


class TestForwardSplitPrefill(CustomTestCase):
    """Test cases for forward_split_prefill functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        cls.model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.tp_size = 1
        cls.device = device_type

        # Initialize server args
        cls.server_args = ServerArgs(
            model_path=cls.model_path,
            tokenizer_path=cls.model_path,
            host="127.0.0.1",
            disable_cuda_graph=True,  # Disable CUDA graph for testing split prefill
            disable_hybrid_swa_memory=True,
            port=30000,
            tp_size=cls.tp_size,
            mem_fraction_static=0.8,
            trust_remote_code=True,
        )

        cls.port_args = PortArgs.init_new(cls.server_args)

        # Load model and tokenizer
        cls.model_config = ModelConfig.from_server_args(cls.server_args)
        cls.model_runner = ModelRunner(
            model_config=cls.model_config,
            mem_fraction_static=cls.server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=cls.tp_size,
            pp_rank=0,
            pp_size=1,
            nccl_port=cls.port_args.nccl_port,
            server_args=cls.server_args,
        )

        cls.tokenizer = get_tokenizer(
            cls.server_args.tokenizer_path,
            tokenizer_mode=cls.server_args.tokenizer_mode,
            trust_remote_code=cls.server_args.trust_remote_code,
        )

        print(
            f"Test with model: {cls.model_path}, num_hidden_layers: {cls.model_config.num_hidden_layers}"
        )

    def prepare_test_batch(self, batch_size=2, input_len=128, is_split_prefill=True):
        """Prepare a test batch for split prefill testing."""
        # Create synthetic input
        input_ids = np.random.randint(10, 1000, (batch_size, input_len), dtype=np.int32)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_new_tokens=8,
        )

        reqs = []
        for i in range(batch_size):
            req = Req(
                rid=i,
                origin_input_text="",
                origin_input_ids=list(input_ids[i]),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)

        # Create dummy tree_cache for tests (no prefix caching, just allocation)
        dummy_tree_cache = SimpleNamespace(
            page_size=1,
            device=self.model_runner.device,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
        )

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=dummy_tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            enable_custom_logit_processor=False,
        )
        if is_split_prefill:
            batch.prepare_for_split_prefill()
        else:
            batch.prepare_for_extend()

        # Create forward batch
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

        return forward_batch

    def test_split_prefill_functionality(self):
        """Test that split prefill can complete successfully."""
        print("\n=== Testing split prefill functionality ===")

        forward_batch = self.prepare_test_batch(batch_size=2, input_len=64)

        # Reset split index
        forward_batch.split_index = 0

        # Test split prefill in chunks
        num_layers = self.model_config.num_hidden_layers
        chunk_size = max(1, num_layers // 4)  # Split into 4 chunks

        results = []
        split_count = 0

        while forward_batch.split_index < num_layers:
            print(
                f"Processing split {split_count}, split_index: {forward_batch.split_index}"
            )

            result = self.model_runner.forward_split_prefill(
                forward_batch=forward_batch,
                reinit_attn_backend=(split_count == 0),
                forward_count=chunk_size,
            )

            results.append(result)
            split_count += 1

            # Verify split_index is updated correctly
            expected_next_index = min(split_count * chunk_size, num_layers)
            self.assertEqual(forward_batch.split_index, expected_next_index)

        # The last result should contain logits
        self.assertIsNotNone(results[-1], "Final split should return logits")
        print(f"Split prefill completed in {split_count} splits")

    def test_split_prefill_vs_normal_prefill(self):
        """Test that split prefill produces the same results as normal prefill."""
        print("\n=== Testing split prefill vs normal prefill consistency ===")

        forward_batch_normal = self.prepare_test_batch(
            batch_size=2, input_len=128, is_split_prefill=False
        )
        forward_batch_split = self.prepare_test_batch(
            batch_size=2, input_len=128, is_split_prefill=True
        )

        # Ensure same input
        forward_batch_split.input_ids = forward_batch_normal.input_ids.clone()
        forward_batch_split.positions = forward_batch_normal.positions.clone()

        # Method 1: Normal extend (prefill)
        print("Running normal extend (prefill)...")
        normal_result = self.model_runner.forward_extend(forward_batch_normal)

        # Method 2: Split prefill
        print("Running split prefill...")
        num_layers = self.model_config.num_hidden_layers
        chunk_size = max(1, num_layers // 3)  # Split into 3 chunks

        split_result = None

        while forward_batch_split.split_index < num_layers:
            result = self.model_runner.forward_split_prefill(
                forward_batch=forward_batch_split,
                forward_count=chunk_size,
            )
            if result is not None:
                split_result = result

        # Compare results
        self.assertIsNotNone(normal_result, "Normal prefill should return result")
        self.assertIsNotNone(split_result, "Split prefill should return result")

        # Compare logits shapes
        self.assertEqual(
            normal_result.next_token_logits.shape,
            split_result.next_token_logits.shape,
            "Logits shapes should match",
        )

        # Compare logits values (should be very close due to same computation)
        # Use a larger tolerance for numerical differences in split computation
        torch.testing.assert_close(
            normal_result.next_token_logits,
            split_result.next_token_logits,
            rtol=1e-3,
            atol=1e-3,
            msg="Split prefill and normal prefill should produce similar logits",
        )

        print("✓ Split prefill and normal prefill produce consistent results")

    def test_split_prefill_different_chunk_sizes(self):
        """Test split prefill with different chunk sizes."""
        print("\n=== Testing split prefill with different chunk sizes ===")

        num_layers = self.model_config.num_hidden_layers
        chunk_sizes = [1, 2, max(1, num_layers // 2), num_layers]

        # Prepare identical batches for each test
        base_batch = self.prepare_test_batch(batch_size=1, input_len=16)
        base_input_ids = base_batch.input_ids.clone()
        base_positions = base_batch.positions.clone()

        results = []

        for chunk_size in chunk_sizes:
            if chunk_size > num_layers:
                continue

            print(f"Testing chunk size: {chunk_size}")

            # Prepare fresh batch
            forward_batch = self.prepare_test_batch(batch_size=1, input_len=16)
            forward_batch.input_ids = base_input_ids.clone()
            forward_batch.positions = base_positions.clone()
            forward_batch.split_index = 0

            # Run split prefill
            split_result = None

            while forward_batch.split_index < num_layers:
                result = self.model_runner.forward_split_prefill(
                    forward_batch=forward_batch,
                    forward_count=chunk_size,
                )
                if result is not None:
                    split_result = result

            self.assertIsNotNone(
                split_result,
                f"Split prefill should succeed with chunk_size={chunk_size}",
            )
            results.append(split_result)

        # Compare all results should be identical (same input, same computation)
        if len(results) > 1:
            for i, result in enumerate(results[1:], 1):
                torch.testing.assert_close(
                    results[0].next_token_logits,
                    result.next_token_logits,
                    rtol=1e-3,
                    atol=1e-3,
                    msg=f"Results with different chunk sizes should be identical (chunk_size {chunk_sizes[i]})",
                )

        print("✓ All chunk sizes produce consistent results")

    def test_split_prefill_edge_cases(self):
        """Test edge cases for split prefill."""
        print("\n=== Testing split prefill edge cases ===")

        # Test with single layer chunks
        forward_batch = self.prepare_test_batch(batch_size=1, input_len=8)

        # Process one layer at a time
        num_layers = self.model_config.num_hidden_layers
        for layer_idx in range(num_layers):
            result = self.model_runner.forward_split_prefill(
                forward_batch=forward_batch,
                reinit_attn_backend=(layer_idx == 0),
                forward_count=1,  # One layer at a time
            )

            if layer_idx == num_layers - 1:
                # Last layer should return result
                self.assertIsNotNone(result, "Last layer should return logits")
            else:
                # Intermediate layers should return None
                self.assertIsNone(result, f"Layer {layer_idx} should return None")

        print("✓ Single layer processing works correctly")


if __name__ == "__main__":
    unittest.main()
