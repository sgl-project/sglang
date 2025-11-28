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

Usage:
    python -m pytest test/srt/lora/test_lora_moe.py -v
    python test_lora_moe.py --debug
"""

import logging
import multiprocessing as mp
import pickle
import tempfile
import os
import torch
import unittest

from utils import LoRAModelCase, LoRAAdaptor, ensure_reproducibility
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, calculate_rouge_l, is_in_ci

TEST_PROMPTS = [
    "The capital of France is Paris. The capital of",
    "Explain what mixture of experts means in machine learning.",
]

MOE_LORA_TEST_CASES = [
    LoRAModelCase(
        base="Qwen/Qwen1.5-MoE-A2.7B",
        adaptors=[
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
]


class TestMoELoRA(CustomTestCase):
    """Test MoE LoRA implementation by comparing against HuggingFace."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _get_server_args(self):
        """Get SGLang server arguments."""
        return {
            "quantization": "fp8",
            "disable_radix_cache": True,
            "lora_backend": "csgmv",
            "max_lora_chunk_size": 16,
            "max_loras_per_batch": 1,
            "tp_size": 2,
            "max_total_tokens": 128,
            "page_size": 64,
            "max_running_requests": 1,
            "mem_fraction_static": 0.85,
        }

    def _run_srt_test(self, result_file, model_case, torch_dtype, max_new_tokens=32):
        """Run SGLang test and save results."""
        try:
            server_args = self._get_server_args()
            adaptor_names = [a.name for a in model_case.adaptors]

            srt_runner = SRTRunner(
                model_path=model_case.base,
                torch_dtype=torch_dtype,
                model_type="generation",
                tp_size=server_args["tp_size"],
                lora_paths=adaptor_names,
                max_loras_per_batch=server_args["max_loras_per_batch"],
                quantization=server_args["quantization"],
                max_lora_chunk_size=server_args["max_lora_chunk_size"],
                max_running_requests=server_args["max_running_requests"],
                lora_backend=server_args["lora_backend"],
                disable_radix_cache=server_args["disable_radix_cache"],
                max_total_tokens=server_args["max_total_tokens"],
                page_size=server_args["page_size"],
                mem_fraction_static=server_args["mem_fraction_static"],
            )

            results = {}
            with srt_runner:
                for batch_size in [1, 2]:
                    prompts = TEST_PROMPTS[:batch_size]
                    lora_paths = [adaptor_names[0]] * batch_size
                    ensure_reproducibility()

                    outputs = srt_runner.batch_forward(prompts, max_new_tokens=max_new_tokens, lora_paths=lora_paths)
                    results[f"batch_{batch_size}"] = {
                        "prompts": prompts,
                        "lora_paths": lora_paths,
                        "outputs": {
                            "output_strs": outputs.output_strs,
                            "top_input_logprobs": outputs.top_input_logprobs,
                            "top_output_logprobs": outputs.top_output_logprobs,
                        }
                    }

            with open(result_file, 'wb') as f:
                pickle.dump(results, f)

            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            with open(result_file, 'wb') as f:
                pickle.dump({"error": str(e), "traceback": traceback.format_exc()}, f)

    def _run_hf_test(self, result_file, model_case, torch_dtype, max_new_tokens=32):
        """Run HuggingFace test and save results."""
        try:
            server_args = self._get_server_args()
            adaptor_names = [a.name for a in model_case.adaptors]

            hf_runner = HFRunner(
                model_case.base,
                torch_dtype=torch_dtype,
                model_type="generation",
                trust_remote_code=True,
                quantization=server_args["quantization"],
                device_map="auto",
            )

            results = {}
            with hf_runner:
                for batch_size in [1, 2]:
                    prompts = TEST_PROMPTS[:batch_size]
                    lora_paths = [adaptor_names[0]] * batch_size
                    ensure_reproducibility()

                    outputs = hf_runner.forward(prompts, max_new_tokens=max_new_tokens, lora_paths=lora_paths)
                    results[f"batch_{batch_size}"] = {
                        "prompts": prompts,
                        "lora_paths": lora_paths,
                        "outputs": {
                            "output_strs": outputs.output_strs,
                            "top_input_logprobs": outputs.top_input_logprobs,
                            "top_output_logprobs": outputs.top_output_logprobs,
                        }
                    }

            with open(result_file, 'wb') as f:
                pickle.dump(results, f)

            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            with open(result_file, 'wb') as f:
                pickle.dump({"error": str(e), "traceback": traceback.format_exc()}, f)

    def _compare_outputs(self, srt_outputs, hf_outputs, model_case, prompts, lora_paths):
        """Compare SGLang and HF outputs."""
        for i, (prompt, lora_path) in enumerate(zip(prompts, lora_paths)):
            srt_str = srt_outputs["output_strs"][i].strip()
            hf_str = hf_outputs["output_strs"][i].strip()

            rouge_l = calculate_rouge_l(srt_str, hf_str)
            self.assertGreaterEqual(rouge_l, model_case.rouge_l_tolerance,
                f"ROUGE-L {rouge_l:.4f} below tolerance for request {i}")

    def test_moe_lora_qwen15(self):
        """Test LoRA on Qwen1.5-MoE-A2.7B."""
        if is_in_ci():
            self.skipTest("Skipping MoE LoRA test in CI")

        model_case = MOE_LORA_TEST_CASES[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            srt_file = os.path.join(temp_dir, 'srt.pkl')
            hf_file = os.path.join(temp_dir, 'hf.pkl')

            self._run_srt_test(srt_file, model_case, torch.float16)
            self._run_hf_test(hf_file, model_case, torch.float16)

            with open(srt_file, 'rb') as f:
                srt_results = pickle.load(f)
            with open(hf_file, 'rb') as f:
                hf_results = pickle.load(f)

            if "error" in srt_results:
                self.fail(f"SRT failed: {srt_results['error']}")
            if "error" in hf_results:
                self.fail(f"HF failed: {hf_results['error']}")

            for key in srt_results:
                if key in hf_results:
                    self._compare_outputs(
                        srt_results[key]["outputs"],
                        hf_results[key]["outputs"],
                        model_case,
                        srt_results[key]["prompts"],
                        srt_results[key]["lora_paths"]
                    )

    def test_moe_lora_basic_functionality(self):
        """Basic functionality test for MoE LoRA dispatch."""
        from sglang.srt.lora.moe_dispatch import moe_dispatch

        topk_ids = torch.tensor([[0, 1], [2, 3], [1, 4], [5, 6]], dtype=torch.int32)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        lora_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

        token_ids, expert_ids, weights = moe_dispatch(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            lora_indices=lora_indices,
            num_experts=8,
            num_loras=2,
        )

        self.assertEqual(len(token_ids), 8)
        self.assertEqual(len(expert_ids), 8)
        self.assertEqual(len(weights), 8)


def debug_full_comparison():
    """Debug helper to run full SRT vs HF comparison."""
    test = TestMoELoRA()
    model_case = MOE_LORA_TEST_CASES[0]
    server_args = test._get_server_args()
    adaptor_names = [a.name for a in model_case.adaptors]

    srt_results, hf_results = {}, {}

    try:
        # SRT tests
        print("Running SGLang tests...")
        srt_runner = SRTRunner(
            model_path=model_case.base,
            torch_dtype=torch.float16,
            model_type="generation",
            tp_size=server_args["tp_size"],
            lora_paths=adaptor_names,
            max_loras_per_batch=server_args["max_loras_per_batch"],
            quantization=server_args["quantization"],
            max_lora_chunk_size=server_args["max_lora_chunk_size"],
            max_running_requests=server_args["max_running_requests"],
            lora_backend=server_args["lora_backend"],
            disable_radix_cache=server_args["disable_radix_cache"],
            max_total_tokens=server_args["max_total_tokens"],
            page_size=server_args["page_size"],
            mem_fraction_static=server_args["mem_fraction_static"],
        )

        with srt_runner:
            for batch_size in [1, 2]:
                prompts = TEST_PROMPTS[:batch_size]
                lora_paths = [adaptor_names[0]] * batch_size
                ensure_reproducibility()
                outputs = srt_runner.batch_forward(prompts, max_new_tokens=32, lora_paths=lora_paths)
                srt_results[f"batch_{batch_size}"] = {"prompts": prompts, "lora_paths": lora_paths, "outputs": outputs}
                for i, out in enumerate(outputs.output_strs):
                    print(f"SRT [{batch_size}] {i}: {out}")

        torch.cuda.empty_cache()

        # HF tests
        print("\nRunning HuggingFace tests...")
        hf_runner = HFRunner(
            model_case.base,
            torch_dtype=torch.float16,
            model_type="generation",
            trust_remote_code=True,
            quantization=server_args["quantization"],
            device_map="auto",
        )

        with hf_runner:
            for batch_size in [1, 2]:
                prompts = TEST_PROMPTS[:batch_size]
                lora_paths = [adaptor_names[0]] * batch_size
                ensure_reproducibility()
                outputs = hf_runner.forward(prompts, max_new_tokens=32, lora_paths=lora_paths)
                hf_results[f"batch_{batch_size}"] = {"prompts": prompts, "lora_paths": lora_paths, "outputs": outputs}
                for i, out in enumerate(outputs.output_strs):
                    print(f"HF [{batch_size}] {i}: {out}")

        torch.cuda.empty_cache()

        # Compare
        print("\nComparing outputs...")
        for key in srt_results:
            srt_data, hf_data = srt_results[key], hf_results[key]
            for i in range(len(srt_data["prompts"])):
                rouge_l = calculate_rouge_l(
                    srt_data["outputs"].output_strs[i],
                    hf_data["outputs"].output_strs[i]
                )
                print(f"{key} request {i}: ROUGE-L = {rouge_l:.4f}")

        print("\nAll comparisons completed!")
        return True

    except Exception as e:
        import traceback
        print(f"Failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        success = debug_full_comparison()
        sys.exit(0 if success else 1)
    else:
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        unittest.main()
