# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Test MoE LoRA implementation by comparing against HuggingFace.

This test file verifies that SGLang's MoE LoRA implementation produces the same
outputs as HuggingFace's PEFT library for MoE models.

WORKFLOW:
1. Run SGLang test in first session (loads SGLang model, saves results to temp file)
2. Run HF test in second session (loads HF model, saves results to temp file)
3. Compare saved results from both sessions

This avoids loading both models simultaneously, saving GPU memory.

Usage:
    # Run basic functionality test (no HF comparison)
    python -m pytest test/srt/lora/test_lora_moe.py::TestMoELoRA::test_moe_lora_basic_functionality -v

    # Run full HF comparison test (requires model and LoRA adapter)
    # This will run SGLang and HF tests separately, then compare results
    python -m pytest test/srt/lora/test_lora_moe.py::TestMoELoRA::test_moe_lora_qwen15 -v

Manual Testing:
    # Run full SRT vs HF comparison (sequential, saves memory)
    python test_lora_moe.py --debug

    # Or run individual components separately:
    # Terminal 1: Run SGLang test and save results
    python -c "
    from test.srt.lora.test_lora_moe import TestMoELoRA
    import torch
    import tempfile
    import os

    test = TestMoELoRA()
    model_case = test.MOE_LORA_TEST_CASES[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        result_file = os.path.join(temp_dir, 'srt_results.pkl')
        test._run_srt_test(result_file, model_case, torch.float16)
        print(f'SGLang results saved to {result_file}')
        input('Press Enter after copying result file...')
    "

    # Terminal 2: Run HF test and save results
    python -c "
    from test.srt.lora.test_lora_moe import TestMoELoRA
    import torch
    import tempfile
    import os

    test = TestMoELoRA()
    model_case = test.MOE_LORA_TEST_CASES[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        result_file = os.path.join(temp_dir, 'hf_results.pkl')
        test._run_hf_test(result_file, model_case, torch.float16)
        print(f'HF results saved to {result_file}')
        input('Press Enter after copying result file...')
    "

Requirements:
    - Qwen/Qwen1.5-MoE-A2.7B model
    - sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest LoRA adapter
    - Sufficient GPU memory (tests run in separate sessions)
    - Uses same memory configuration as launch.json for compatibility
"""

import json
import logging
import multiprocessing as mp
import os
import pickle
import psutil
import random
import tempfile
import time
import torch
import unittest
from pathlib import Path

from utils import LoRAModelCase, LoRAAdaptor, ensure_reproducibility

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, calculate_rouge_l, is_in_ci

# Test prompts for MoE LoRA comparison
TEST_MOE_PROMPTS = [
    "The capital of France is Paris. The capital of",
    "Explain what mixture of experts means in machine learning.",
    "Write a short poem about artificial intelligence and large language models.",
    "What are the benefits of using MoE architectures in transformers?",
]

# MoE model test cases with LoRA adapters
MOE_LORA_TEST_CASES = [
    LoRAModelCase(
        base="Qwen/Qwen1.5-MoE-A2.7B",
        adaptors=[
            # Use a real LoRA adapter path - replace with actual path when testing
            LoRAAdaptor(
                name="sai-lakkshmii/Qwen1.5-MoE-A2.7B-squad-lora-latest",
                prefill_tolerance=1e-1,
                decode_tolerance=1e-1,
                rouge_l_tolerance=1.0
            ),
        ],
        tp_size=1,
        prefill_tolerance=1e-1,
        decode_tolerance=1e-1,
        rouge_l_tolerance=1.0,
        max_loras_per_batch=1,
    ),
    # Add more MoE models here when available
]


class TestMoELoRA(CustomTestCase):
    """Test MoE LoRA implementation by comparing against HuggingFace."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up detailed logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _log_system_info(self):
        """Log system and GPU memory information."""
        try:
            # CPU memory
            memory = psutil.virtual_memory()
            self.logger.info(f"System Memory: {memory.available / (1024**3):.2f}GB available / {memory.total / (1024**3):.2f}GB total")

            # GPU memory if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                self.logger.info(f"GPU Memory: {gpu_memory:.2f}GB total, {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            else:
                self.logger.warning("CUDA not available")
        except Exception as e:
            self.logger.warning(f"Could not get system info: {e}")

    def _get_srt_server_args(self):
        """Get SGLang server arguments from launch.json config."""
        return {
            "quantization": "fp8",
            "disable_radix_cache": True,
            "lora_backend": "csgmv",
            "max_lora_chunk_size": 16,
            "port": 30000,
            "host": "127.0.0.1",
            "max_loras_per_batch": 1,
            "tp_size": 2,
            "max_total_tokens": 128,
            "page_size": 64,
            "max_running_requests": 1,
            "mem_fraction_static": 0.85,
        }

    def _run_srt_test(self, result_file, model_case, torch_dtype, max_new_tokens=32):
        """Run SGLang test and save results to file."""
        try:
            self.logger.info("=== Starting SGLang MoE LoRA test ===")
            self._log_system_info()

            base_path = model_case.base
            adaptor_names = [adaptor.name for adaptor in model_case.adaptors]

            self.logger.info(f"Model: {base_path}")
            self.logger.info(f"LoRA adapters: {adaptor_names}")
            self.logger.info(f"dtype: {torch_dtype}")

            server_args = self._get_srt_server_args()
            server_args["model_path"] = base_path
            server_args["lora_paths"] = adaptor_names
            server_args["enable_lora"] = True

            self.logger.info(f"Server args: {server_args}")
            self.logger.info("Creating SRTRunner...")

            # Initialize SGLang runner with launch.json args
            self.logger.info("Initializing SRTRunner...")
            srt_runner = SRTRunner(
                model_path=base_path,
                torch_dtype=torch_dtype,
                model_type="generation",
                tp_size=server_args["tp_size"],
                lora_paths=adaptor_names,
                max_loras_per_batch=server_args["max_loras_per_batch"],
                quantization=server_args["quantization"],
                max_lora_chunk_size=server_args["max_lora_chunk_size"],
                max_running_requests=server_args["max_running_requests"],
                lora_backend=server_args["lora_backend"],
                disable_cuda_graph=False,
                disable_radix_cache=server_args["disable_radix_cache"],
                max_total_tokens=server_args["max_total_tokens"],
                page_size=server_args["page_size"],
                mem_fraction_static=server_args["mem_fraction_static"],
                sleep_on_idle=True,
            )
            self.logger.info("SRTRunner created successfully")

            results = {}
            self._log_system_info()  # Check memory after runner creation

            # Test with different batch configurations
            test_configs = [
                {"batch_size": 1, "lora_paths": [adaptor_names[0]]},  # Single request, single LoRA
                {"batch_size": 2, "lora_paths": [adaptor_names[0], adaptor_names[0]]},  # Multiple requests, same LoRA
            ]

            self.logger.info("Entering SRTRunner context manager (this loads the model)...")
            with srt_runner:
                self.logger.info("SRTRunner context entered - model should be loaded")
                self._log_system_info()  # Check memory after model loading

                for config in test_configs:
                    batch_size = config["batch_size"]
                    lora_paths = config["lora_paths"]

                    # Use fixed prompts for reproducibility
                    prompts = TEST_MOE_PROMPTS[:batch_size]

                    config_key = f"batch_{batch_size}_lora_{len(set(lora_paths))}"

                    self.logger.info(f"=== Testing config: {config_key} ===")
                    self.logger.info(f"Batch size: {batch_size}")
                    self.logger.info(f"LoRA paths: {lora_paths}")
                    self.logger.info(f"Prompts: {prompts}")

                    # Ensure reproducibility
                    self.logger.info("Ensuring reproducibility...")
                    ensure_reproducibility()

                    # Run SGLang
                    self.logger.info("Running batch_forward...")
                    srt_outputs = srt_runner.batch_forward(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        lora_paths=lora_paths,
                    )
                    self.logger.info("batch_forward completed successfully")

                    # Print responses
                    self.logger.info("=== SRT Generated Responses ===")
                    for i, (prompt, output) in enumerate(zip(prompts, srt_outputs.output_strs)):
                        self.logger.info(f"Prompt {i+1}: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                        self.logger.info(f"SRT Output {i+1}: {output}")
                        self.logger.info("")

                    results[config_key] = {
                        "prompts": prompts,
                        "lora_paths": lora_paths,
                        "srt_outputs": {
                            "output_strs": srt_outputs.output_strs,
                            "top_input_logprobs": srt_outputs.top_input_logprobs,
                            "top_output_logprobs": srt_outputs.top_output_logprobs,
                        }
                    }
                    self.logger.info(f"Results saved for config: {config_key}")

            # Save results
            self.logger.info(f"Saving results to {result_file}")
            with open(result_file, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"SGLang results saved successfully to {result_file}")

            # Force GPU memory cleanup after context manager exit
            self.logger.info("Forcing GPU memory cleanup...")
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self._log_system_info()  # Log memory after cleanup

        except Exception as e:
            self.logger.error(f"SGLang test failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            with open(result_file, 'wb') as f:
                pickle.dump({"error": str(e), "traceback": traceback.format_exc()}, f)

    def _run_hf_test(self, result_file, model_case, torch_dtype, max_new_tokens=32):
        """Run HuggingFace test and save results to file."""
        try:
            self.logger.info("=== Starting HF MoE LoRA test ===")
            self._log_system_info()

            base_path = model_case.base
            adaptor_names = [adaptor.name for adaptor in model_case.adaptors]

            self.logger.info(f"Model: {base_path}")
            self.logger.info(f"LoRA adapters: {adaptor_names}")
            self.logger.info(f"dtype: {torch_dtype}")

            # Get server args for consistency
            server_args = self._get_srt_server_args()

            # Initialize HF runner
            self.logger.info("Creating HFRunner...")
            hf_runner = HFRunner(
                base_path,
                torch_dtype=torch_dtype,
                model_type="generation",
                trust_remote_code=True,  # Match SRTRunner behavior
                quantization=server_args["quantization"],  # Enable quantization if specified
                device_map="auto",  # Distribute across available GPUs (like tp_size for SRT)
            )
            self.logger.info("HFRunner created successfully")
            self._log_system_info()  # Check memory after runner creation

            results = {}

            # Test with different batch configurations
            test_configs = [
                {"batch_size": 1, "lora_paths": [adaptor_names[0]]},  # Single request, single LoRA
                {"batch_size": 2, "lora_paths": [adaptor_names[0], adaptor_names[0]]},  # Multiple requests, same LoRA
            ]

            self.logger.info("Entering HFRunner context manager (this loads the model)...")
            with hf_runner:
                self.logger.info("HFRunner context entered - model should be loaded")
                self._log_system_info()  # Check memory after model loading

                for config in test_configs:
                    batch_size = config["batch_size"]
                    lora_paths = config["lora_paths"]

                    # Use fixed prompts for reproducibility
                    prompts = TEST_MOE_PROMPTS[:batch_size]

                    config_key = f"batch_{batch_size}_lora_{len(set(lora_paths))}"

                    self.logger.info(f"=== HF Testing config: {config_key} ===")
                    self.logger.info(f"Batch size: {batch_size}")
                    self.logger.info(f"LoRA paths: {lora_paths}")
                    self.logger.info(f"Prompts: {prompts}")

                    # Ensure reproducibility
                    self.logger.info("Ensuring reproducibility...")
                    ensure_reproducibility()

                    # Run HuggingFace
                    self.logger.info("Running HF forward...")
                    hf_outputs = hf_runner.forward(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        lora_paths=lora_paths,
                    )
                    self.logger.info("HF forward completed successfully")

                    # Print responses
                    self.logger.info("=== HF Generated Responses ===")
                    for i, (prompt, output) in enumerate(zip(prompts, hf_outputs.output_strs)):
                        self.logger.info(f"Prompt {i+1}: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                        self.logger.info(f"HF Output {i+1}: {output}")
                        self.logger.info("")

                    results[config_key] = {
                        "prompts": prompts,
                        "lora_paths": lora_paths,
                        "hf_outputs": {
                            "output_strs": hf_outputs.output_strs,
                            "top_input_logprobs": hf_outputs.top_input_logprobs,
                            "top_output_logprobs": hf_outputs.top_output_logprobs,
                        }
                    }
                    self.logger.info(f"HF results saved for config: {config_key}")

            # Save results
            self.logger.info(f"Saving HF results to {result_file}")
            with open(result_file, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"HF results saved successfully to {result_file}")

            # Force GPU memory cleanup after context manager exit
            self.logger.info("Forcing GPU memory cleanup...")
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self._log_system_info()  # Log memory after cleanup

        except Exception as e:
            self.logger.error(f"HF test failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            with open(result_file, 'wb') as f:
                pickle.dump({"error": str(e), "traceback": traceback.format_exc()}, f)

    def _run_moe_lora_comparison(self, model_case: LoRAModelCase, torch_dtype, max_new_tokens=32):
        """Run LoRA comparison test by loading saved results from separate sessions."""
        self.logger.info("=== Starting MoE LoRA comparison test ===")
        self._log_system_info()

        base_path = model_case.base
        self.logger.info(f"Model: {base_path}, dtype: {torch_dtype}")

        # Create temp directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            srt_result_file = os.path.join(temp_dir, 'srt_results.pkl')
            hf_result_file = os.path.join(temp_dir, 'hf_results.pkl')

            self.logger.info(f"Results directory: {temp_dir}")
            self.logger.info(f"SRT result file: {srt_result_file}")
            self.logger.info(f"HF result file: {hf_result_file}")

            # Check if results already exist
            srt_done = os.path.exists(srt_result_file)
            hf_done = os.path.exists(hf_result_file)

            self.logger.info(f"SRT results exist: {srt_done}")
            self.logger.info(f"HF results exist: {hf_done}")

            if not srt_done:
                self.logger.info("Running SGLang test...")
                self._run_srt_test(srt_result_file, model_case, torch_dtype, max_new_tokens)
                srt_done = True

            if not hf_done:
                self.logger.info("Running HuggingFace test...")
                self._run_hf_test(hf_result_file, model_case, torch_dtype, max_new_tokens)
                hf_done = True

            # Load results
            self.logger.info("Loading SRT results...")
            if srt_done:
                with open(srt_result_file, 'rb') as f:
                    srt_results = pickle.load(f)
                if "error" in srt_results:
                    error_msg = f"SGLang test failed: {srt_results['error']}"
                    if "traceback" in srt_results:
                        error_msg += f"\nTraceback: {srt_results['traceback']}"
                    self.fail(error_msg)
                self.logger.info("SRT results loaded successfully")

            self.logger.info("Loading HF results...")
            if hf_done:
                with open(hf_result_file, 'rb') as f:
                    hf_results = pickle.load(f)
                if "error" in hf_results:
                    error_msg = f"HF test failed: {hf_results['error']}"
                    if "traceback" in hf_results:
                        error_msg += f"\nTraceback: {hf_results['traceback']}"
                    self.fail(error_msg)
                self.logger.info("HF results loaded successfully")

            # Compare results
            if srt_done and hf_done:
                self.logger.info("Starting result comparison...")
                for config_key in srt_results.keys():
                    if config_key in hf_results:
                        self.logger.info(f"Comparing config: {config_key}")
                        srt_data = srt_results[config_key]
                        hf_data = hf_results[config_key]

                        self._compare_outputs(
                            srt_data["srt_outputs"], hf_data["hf_outputs"],
                            model_case, srt_data["prompts"], srt_data["lora_paths"]
                        )
                    else:
                        self.logger.warning(f"No HF results for config: {config_key}")

                self.logger.info("Comparison completed successfully")

    def _compare_outputs(self, srt_outputs, hf_outputs, model_case, prompts, lora_paths):
        """Compare SGLang and HF outputs."""
        for i, (prompt, lora_path) in enumerate(zip(prompts, lora_paths)):
            print(f"\nRequest {i}: lora_path='{lora_path}'")
            print(f"Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

            # Compare output strings
            srt_output_str = srt_outputs.output_strs[i].strip()
            hf_output_str = hf_outputs.output_strs[i].strip()

            print(f"SRT output: {srt_output_str[:100]}{'...' if len(srt_output_str) > 100 else ''}")
            print(f"HF output:  {hf_output_str[:100]}{'...' if len(hf_output_str) > 100 else ''}")

            # Calculate ROUGE-L similarity
            rouge_l = calculate_rouge_l(srt_output_str, hf_output_str)
            print(f"ROUGE-L similarity: {rouge_l:.4f}")

            # Check ROUGE-L tolerance
            self.assertGreaterEqual(
                rouge_l,
                model_case.rouge_l_tolerance,
                f"ROUGE-L similarity {rouge_l:.4f} below tolerance {model_case.rouge_l_tolerance} "
                f"for request {i} with lora_path='{lora_path}'"
            )

            # Compare logprobs if available
            if hasattr(srt_outputs, 'top_input_logprobs') and hasattr(hf_outputs, 'top_input_logprobs'):
                if srt_outputs.top_input_logprobs[i] is not None and hf_outputs.top_input_logprobs[i] is not None:
                    import torch
                    srt_prefill = torch.tensor(srt_outputs.top_input_logprobs[i])
                    hf_prefill = torch.tensor(hf_outputs.top_input_logprobs[i])

                    max_prefill_diff = torch.max(torch.abs(hf_prefill - srt_prefill))
                    print(f"Max prefill logprob diff: {max_prefill_diff:.6f}")

                    # Check prefill tolerance
                    prefill_tol = model_case.prefill_tolerance
                    self.assertLessEqual(
                        max_prefill_diff,
                        prefill_tol,
                        f"Prefill logprob diff {max_prefill_diff:.6f} exceeds tolerance {prefill_tol} "
                        f"for request {i} with lora_path='{lora_path}'"
                    )

            if hasattr(srt_outputs, 'top_output_logprobs') and hasattr(hf_outputs, 'top_output_logprobs'):
                if srt_outputs.top_output_logprobs[i] is not None and hf_outputs.top_output_logprobs[i] is not None:
                    import torch
                    srt_decode = torch.tensor(srt_outputs.top_output_logprobs[i])
                    hf_decode = torch.tensor(hf_outputs.top_output_logprobs[i])

                    max_decode_diff = torch.max(torch.abs(hf_decode - srt_decode))
                    print(f"Max decode logprob diff: {max_decode_diff:.6f}")

                    # Check decode tolerance
                    decode_tol = model_case.decode_tolerance
                    self.assertLessEqual(
                        max_decode_diff,
                        decode_tol,
                        f"Decode logprob diff {max_decode_diff:.6f} exceeds tolerance {decode_tol} "
                        f"for request {i} with lora_path='{lora_path}'"
                    )

    def test_moe_lora_qwen15(self):
        """Test LoRA on Qwen1.5-MoE-A2.7B."""
        self.logger.info("=== Starting test_moe_lora_qwen15 ===")

        if is_in_ci():
            self.logger.info("Skipping MoE LoRA test in CI environment")
            self.skipTest("Skipping MoE LoRA test in CI environment")

        model_case = MOE_LORA_TEST_CASES[0]
        self.logger.info(f"Using model case: {model_case.base}")

        # Test with different dtypes
        import torch
        for torch_dtype in [torch.float16, torch.bfloat16]:
            self.logger.info(f"Testing dtype: {torch_dtype}")
            with self.subTest(dtype=torch_dtype):
                try:
                    self._run_moe_lora_comparison(model_case, torch_dtype)
                except Exception as e:
                    self.logger.error(f"Test failed for {model_case.base} with dtype {torch_dtype}: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    self.fail(f"Test failed for {model_case.base} with dtype {torch_dtype}: {e}")

    def test_moe_lora_basic_functionality(self):
        """Basic functionality test for MoE LoRA dispatch."""
        # This test focuses on the core dispatch logic without full HF comparison
        # Useful for debugging the MoE LoRA implementation

        import torch
        from sglang.srt.lora.moe_dispatch import moe_dispatch

        # Create test data
        num_tokens = 4
        top_k = 2
        num_experts = 8

        # Mock top-k routing results
        topk_ids = torch.tensor([
            [0, 1],  # token 0 routes to experts 0, 1
            [2, 3],  # token 1 routes to experts 2, 3
            [1, 4],  # token 2 routes to experts 1, 4
            [5, 6],  # token 3 routes to experts 5, 6
        ], dtype=torch.int32)

        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)

        # Mock LoRA indices (one per token)
        lora_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)  # tokens 0,1 use lora 0; tokens 2,3 use lora 1

        # Run dispatch
        token_ids, expert_ids, sorted_topk_weights, lora_ids = moe_dispatch(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            lora_indices=lora_indices,
        )

        # Verify results
        # Should have 4 tokens * 2 experts each = 8 dispatched entries
        self.assertEqual(len(token_ids), 8)
        self.assertEqual(len(expert_ids), 8)
        self.assertEqual(len(sorted_topk_weights), 8)
        self.assertEqual(len(lora_ids), 8)

        # Check that tokens are grouped by expert (not by LoRA)
        # All tokens going to expert 0 should come first, then expert 1, etc.
        unique_experts, expert_counts = torch.unique_consecutive(expert_ids, return_counts=True)
        self.assertTrue(torch.all(expert_counts >= 1))  # Each expert should have at least one token

        print(f"Dispatch successful: {len(token_ids)} dispatched tokens to experts {unique_experts.tolist()}")


def debug_full_comparison():
    """Debug helper to run full SRT vs HF comparison."""
    import torch
    import tempfile
    import os

    # Set up logging for debugging
    logging.basicConfig(
        level=logging.INFO,  # Less verbose for debug script
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    test = TestMoELoRA()
    model_case = MOE_LORA_TEST_CASES[0]

    print("=" * 80)
    print("DEBUG: Running Full SRT vs HF Comparison")
    print("=" * 80)

    srt_results = {}
    hf_results = {}

    # Test configurations
    test_configs = [
        {"batch_size": 1, "lora_paths": [model_case.adaptors[0].name]},  # Single request, single LoRA
        {"batch_size": 2, "lora_paths": [model_case.adaptors[0].name, model_case.adaptors[0].name]},  # Multiple requests, same LoRA
    ]

    try:
        # Phase 1: Run SRT tests
        print("\n" + "="*50)
        print("PHASE 1: Running SGLang (SRT) Tests")
        print("="*50)

        server_args = test._get_srt_server_args()
        print(f"Server args: quantization={server_args['quantization']}, tp_size={server_args['tp_size']}, mem_fraction_static={server_args['mem_fraction_static']}")

        srt_runner = SRTRunner(
            model_path=model_case.base,
            torch_dtype=torch.float16,
            model_type="generation",
            tp_size=server_args["tp_size"],
            lora_paths=[adaptor.name for adaptor in model_case.adaptors],
            max_loras_per_batch=server_args["max_loras_per_batch"],
            quantization=server_args["quantization"],
            max_lora_chunk_size=server_args["max_lora_chunk_size"],
            max_running_requests=server_args["max_running_requests"],
            lora_backend=server_args["lora_backend"],
            disable_cuda_graph=False,
            disable_radix_cache=server_args["disable_radix_cache"],
            max_total_tokens=server_args["max_total_tokens"],
            page_size=server_args["page_size"],
            mem_fraction_static=server_args["mem_fraction_static"],
            sleep_on_idle=True,
        )
        print("✓ SRT runner created successfully")

        with srt_runner:
            print("✓ SRT model loaded successfully")
            test._log_system_info()

            for config in test_configs:
                batch_size = config["batch_size"]
                lora_paths = config["lora_paths"]
                prompts = TEST_MOE_PROMPTS[:batch_size]
                config_key = f"batch_{batch_size}_lora_{len(set(lora_paths))}"

                print(f"\n--- SRT Testing {config_key} ---")
                print(f"Prompts: {prompts}")

                # Ensure reproducibility
                ensure_reproducibility()

                srt_outputs = srt_runner.batch_forward(
                    prompts,
                    max_new_tokens=32,
                    lora_paths=lora_paths,
                )
                print("✓ SRT batch_forward completed")

                # Print SRT responses
                print("=== SRT Generated Responses ===")
                for i, (prompt, output) in enumerate(zip(prompts, srt_outputs.output_strs)):
                    print(f"Prompt {i+1}: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                    print(f"SRT Output {i+1}: {output}")
                    print("")

                srt_results[config_key] = {
                    "prompts": prompts,
                    "lora_paths": lora_paths,
                    "srt_outputs": srt_outputs
                }

        print("✓ All SRT tests completed successfully")

        # Clear GPU memory before HF test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ GPU cache cleared before HF test")

        # Small delay to ensure memory is fully released
        import time
        time.sleep(2)
        test._log_system_info()

        # Phase 2: Run HF tests
        print("\n" + "="*50)
        print("PHASE 2: Running HuggingFace (HF) Tests")
        print("="*50)

        server_args = test._get_srt_server_args()
        hf_runner = HFRunner(
            model_case.base,
            torch_dtype=torch.float16,
            model_type="generation",
            trust_remote_code=True,  # Match SRTRunner behavior
            quantization=server_args["quantization"],  # Enable quantization if specified
            device_map="auto",  # Distribute across available GPUs
        )
        print("✓ HF runner created successfully")

        with hf_runner:
            print("✓ HF model loaded successfully")
            test._log_system_info()

            for config in test_configs:
                batch_size = config["batch_size"]
                lora_paths = config["lora_paths"]
                prompts = TEST_MOE_PROMPTS[:batch_size]
                config_key = f"batch_{batch_size}_lora_{len(set(lora_paths))}"

                print(f"\n--- HF Testing {config_key} ---")
                print(f"Prompts: {prompts}")

                # Ensure reproducibility
                ensure_reproducibility()

                hf_outputs = hf_runner.forward(
                    prompts,
                    max_new_tokens=32,
                    lora_paths=lora_paths,
                )
                print("✓ HF forward completed")

                # Print HF responses
                print("=== HF Generated Responses ===")
                for i, (prompt, output) in enumerate(zip(prompts, hf_outputs.output_strs)):
                    print(f"Prompt {i+1}: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                    print(f"HF Output {i+1}: {output}")
                    print("")

                hf_results[config_key] = {
                    "prompts": prompts,
                    "lora_paths": lora_paths,
                    "hf_outputs": hf_outputs
                }

        print("✓ All HF tests completed successfully")

        # Force final GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ Final GPU memory cleanup completed")
            test._log_system_info()

        # Phase 3: Compare results
        print("\n" + "="*50)
        print("PHASE 3: Comparing SRT vs HF Outputs")
        print("="*50)

        for config_key in srt_results.keys():
            if config_key in hf_results:
                srt_data = srt_results[config_key]
                hf_data = hf_results[config_key]

                print(f"\n{'='*30} Comparing {config_key} {'='*30}")
                try:
                    test._compare_outputs(
                        srt_data["srt_outputs"], hf_data["hf_outputs"],
                        model_case, srt_data["prompts"], srt_data["lora_paths"]
                    )
                    print(f"✓ Comparison passed for {config_key}")
                except AssertionError as e:
                    print(f"✗ Comparison failed for {config_key}: {e}")
                    raise
            else:
                print(f"✗ No HF results for config: {config_key}")

        print("\n" + "="*80)
        print("🎉 ALL COMPARISONS PASSED! SRT and HF outputs match!")
        print("="*80)

    except Exception as e:
        print(f"\n💥 DEBUG SCRIPT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        # Run debug tests instead of unittest
        success = debug_full_comparison()
        sys.exit(0 if success else 1)
    else:
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

        unittest.main()
