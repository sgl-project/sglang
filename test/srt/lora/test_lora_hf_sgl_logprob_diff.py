# Copyright 2023-2024 SGLang Team
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
Test to compare log probabilities between HuggingFace+LoRA and SGLang+LoRA.

This test:
1. Runs SGLang with LoRA and collects log probabilities
2. Runs HuggingFace with LoRA and collects log probabilities
3. Compares the differences (max and mean) between the two implementations
4. Uses unittest framework for easy integration with test suites

Usage:
    python test_lora_hf_sgl_logprob_diff.py
    or
    python -m unittest test_lora_hf_sgl_logprob_diff
"""

import multiprocessing as mp
import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add sglang to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import HFRunner, SRTRunner

register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)

from sglang.test.test_utils import (
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    CustomTestCase,
    is_in_ci,
)

# Test configuration constants
LORA_BACKEND = "triton"
DISABLE_CUDA_GRAPH = False
LORA_TARGET_MODULES = None
LOGPROB_THRESHOLD = 1e-01

# Default test prompts
DEFAULT_TEST_PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]

# Formatting constants
DIVIDER_WIDTH = 80
SECTION_CHAR = "="
SUBSECTION_CHAR = "-"


def print_section_header(title: str):
    """Print a major section header."""
    print("\n" + SECTION_CHAR * DIVIDER_WIDTH)
    print(title)
    print(SECTION_CHAR * DIVIDER_WIDTH)


def print_subsection_header(title: str):
    """Print a subsection header."""
    print(f"\n{SUBSECTION_CHAR * 40}")
    print(f"{title}")
    print(SUBSECTION_CHAR * 40)


def print_config_info(title: str, config: Dict[str, Any]):
    """Print configuration information in a consistent format."""
    print_section_header(title)
    for key, value in config.items():
        print(f"  {key}: {value}")


def compare_logprobs_for_type(
    sglang_logprobs: torch.Tensor, hf_logprobs: torch.Tensor, logprob_type: str
) -> Dict[str, Any]:
    """
    Compare logprobs for a specific type (prefill or decode).

    Args:
        sglang_logprobs: SGLang log probabilities
        hf_logprobs: HuggingFace log probabilities
        logprob_type: Type of logprobs ("prefill" or "decode")

    Returns:
        Dictionary containing comparison statistics
    """
    diff = torch.abs(sglang_logprobs - hf_logprobs)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    shape = list(sglang_logprobs.shape)
    matches_threshold = max_diff < LOGPROB_THRESHOLD

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "shape": shape,
        "matches_threshold": matches_threshold,
        "type": logprob_type,
    }


def print_logprob_comparison(comparison: Dict[str, Any]):
    """Print logprob comparison results in a consistent format."""
    logprob_type = comparison["type"].capitalize()
    print(f"\n{logprob_type} logprobs:")
    print(f"  Shape:           {comparison['shape']}")
    print(f"  Max difference:  {comparison['max_diff']:.6e}")
    print(f"  Mean difference: {comparison['mean_diff']:.6e}")

    status = "PASS" if comparison["matches_threshold"] else "FAIL"
    print(f"  Status:          {status} (threshold: {LOGPROB_THRESHOLD:.0e})")


def compare_output_strings(
    sglang_output: str, hf_output: str, max_display_len: int = 200
) -> Dict[str, Any]:
    """
    Compare output strings between SGLang and HuggingFace.

    Args:
        sglang_output: SGLang generated text
        hf_output: HuggingFace generated text
        max_display_len: Maximum length for display

    Returns:
        Dictionary containing comparison results
    """
    outputs_match = sglang_output.strip() == hf_output.strip()

    # Truncate for display if needed
    sglang_display = (
        sglang_output[:max_display_len]
        if len(sglang_output) > max_display_len
        else sglang_output
    )
    hf_display = (
        hf_output[:max_display_len] if len(hf_output) > max_display_len else hf_output
    )

    return {
        "match": outputs_match,
        "sglang_output": sglang_output,
        "hf_output": hf_output,
        "sglang_display": sglang_display,
        "hf_display": hf_display,
    }


def print_output_comparison(comparison: Dict[str, Any]):
    """Print output string comparison in a consistent format."""
    print(f"\nOutput strings:")
    status = "MATCH" if comparison["match"] else "DIFFER"
    print(f"  Status:      {status}")
    print(f"  SGLang:      {comparison['sglang_display']}")
    print(f"  HuggingFace: {comparison['hf_display']}")


def prepare_lora_paths_per_prompt(
    lora_paths: List[str], num_prompts: int
) -> List[Optional[str]]:
    """
    Prepare LoRA paths for each prompt by cycling through available LoRAs.

    Args:
        lora_paths: List of available LoRA adapter paths
        num_prompts: Number of prompts to generate LoRA paths for

    Returns:
        List of LoRA paths (one per prompt), or None values if no LoRAs
    """
    if not lora_paths:
        return [None] * num_prompts

    return [lora_paths[i % len(lora_paths)] for i in range(num_prompts)]


def run_sglang_with_lora(
    model_path: str,
    lora_paths: List[str],
    prompts: List[str],
    max_new_tokens: int,
    torch_dtype: torch.dtype,
    lora_backend: str,
    port: int,
    disable_cuda_graph: bool,
    lora_target_modules: Optional[List[str]],
    tp_size: int,
) -> Dict[str, Any]:
    """Run SGLang with LoRA and return log probabilities."""
    config = {
        "Model": model_path,
        "LoRA paths": lora_paths,
        "LoRA backend": lora_backend,
        "Disable CUDA graph": disable_cuda_graph,
        "Port": port,
        "Number of prompts": len(prompts),
        "Tensor parallel size": tp_size,
    }
    print_config_info("Running SGLang with LoRA", config)

    lora_paths_per_prompt = prepare_lora_paths_per_prompt(lora_paths, len(prompts))

    with SRTRunner(
        model_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        tp_size=tp_size,
        lora_paths=lora_paths,
        max_loras_per_batch=len(lora_paths) if lora_paths else 1,
        lora_backend=lora_backend,
        disable_cuda_graph=disable_cuda_graph,
        disable_radix_cache=True,
        port=port,
        mem_fraction_static=0.88,
        lora_target_modules=lora_target_modules,
    ) as srt_runner:
        srt_outputs = srt_runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
            lora_paths=lora_paths_per_prompt,
        )

    return {
        "top_input_logprobs": srt_outputs.top_input_logprobs,
        "top_output_logprobs": srt_outputs.top_output_logprobs,
        "output_strs": srt_outputs.output_strs,
        "lora_paths": lora_paths_per_prompt,
    }


def run_hf_with_lora(
    model_path: str,
    lora_paths: List[str],
    prompts: List[str],
    max_new_tokens: int,
    torch_dtype: torch.dtype,
) -> Dict[str, Any]:
    """Run HuggingFace with LoRA and return log probabilities."""
    config = {
        "Model": model_path,
        "LoRA paths": lora_paths,
        "Number of prompts": len(prompts),
    }
    print_config_info("Running HuggingFace with LoRA", config)

    lora_paths_per_prompt = prepare_lora_paths_per_prompt(lora_paths, len(prompts))

    with HFRunner(
        model_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        patch_model_do_sample_false=True,
    ) as hf_runner:
        hf_outputs = hf_runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
            lora_paths=lora_paths_per_prompt,
        )

    return {
        "top_input_logprobs": hf_outputs.top_input_logprobs,
        "top_output_logprobs": hf_outputs.top_output_logprobs,
        "output_strs": hf_outputs.output_strs,
        "lora_paths": lora_paths_per_prompt,
    }


def compare_single_prompt(
    prompt_idx: int,
    sglang_data: Dict[str, Any],
    hf_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare logprobs and outputs for a single prompt.

    Args:
        prompt_idx: Index of the prompt being compared
        sglang_data: SGLang results data
        hf_data: HuggingFace results data

    Returns:
        Dictionary containing all comparison results
    """
    print_subsection_header(f"Prompt {prompt_idx + 1}")
    print(f"LoRA adapter: {sglang_data['lora_paths'][prompt_idx]}")

    result = {
        "prompt_idx": prompt_idx,
        "lora_path": sglang_data["lora_paths"][prompt_idx],
    }

    # Compare prefill (input) logprobs
    sglang_prefill = torch.tensor(sglang_data["top_input_logprobs"][prompt_idx])
    hf_prefill = torch.tensor(hf_data["top_input_logprobs"][prompt_idx])
    prefill_comparison = compare_logprobs_for_type(
        sglang_prefill, hf_prefill, "prefill"
    )
    print_logprob_comparison(prefill_comparison)

    # Store prefill results
    result["prefill_max_diff"] = prefill_comparison["max_diff"]
    result["prefill_mean_diff"] = prefill_comparison["mean_diff"]
    result["prefill_shape"] = prefill_comparison["shape"]
    result["prefill_logprob_match"] = prefill_comparison["matches_threshold"]

    # Compare decode (output) logprobs
    sglang_decode = torch.tensor(sglang_data["top_output_logprobs"][prompt_idx])
    hf_decode = torch.tensor(hf_data["top_output_logprobs"][prompt_idx])
    decode_comparison = compare_logprobs_for_type(sglang_decode, hf_decode, "decode")
    print_logprob_comparison(decode_comparison)

    # Store decode results
    result["decode_max_diff"] = decode_comparison["max_diff"]
    result["decode_mean_diff"] = decode_comparison["mean_diff"]
    result["decode_shape"] = decode_comparison["shape"]
    result["decode_logprob_match"] = decode_comparison["matches_threshold"]

    # Overall logprob match
    result["overall_logprob_match"] = (
        prefill_comparison["matches_threshold"]
        and decode_comparison["matches_threshold"]
    )

    # Compare output strings
    sglang_output = sglang_data["output_strs"][prompt_idx]
    hf_output = hf_data["output_strs"][prompt_idx]
    output_comparison = compare_output_strings(sglang_output, hf_output)
    print_output_comparison(output_comparison)

    # Store output results
    result["outputs_match"] = output_comparison["match"]
    result["sglang_output"] = output_comparison["sglang_output"]
    result["hf_output"] = output_comparison["hf_output"]

    return result


def print_overall_statistics(results: List[Dict[str, Any]]):
    """Print overall statistics across all prompts."""
    print_section_header("Overall Statistics")

    # Gather statistics
    prefill_max_diffs = [r["prefill_max_diff"] for r in results]
    prefill_mean_diffs = [r["prefill_mean_diff"] for r in results]
    decode_max_diffs = [r["decode_max_diff"] for r in results]
    decode_mean_diffs = [r["decode_mean_diff"] for r in results]

    # Print logprob statistics
    print("\nLogprob Differences:")
    print(f"  Prefill:")
    print(f"    Max of max:   {max(prefill_max_diffs):.6e}")
    print(f"    Mean of max:  {np.mean(prefill_max_diffs):.6e}")
    print(f"    Mean of mean: {np.mean(prefill_mean_diffs):.6e}")

    print(f"  Decode:")
    print(f"    Max of max:   {max(decode_max_diffs):.6e}")
    print(f"    Mean of max:  {np.mean(decode_max_diffs):.6e}")
    print(f"    Mean of mean: {np.mean(decode_mean_diffs):.6e}")

    # Print match statistics
    num_prompts = len(results)
    logprob_match_count = sum(r["overall_logprob_match"] for r in results)
    prefill_match_count = sum(r["prefill_logprob_match"] for r in results)
    decode_match_count = sum(r["decode_logprob_match"] for r in results)
    outputs_match_count = sum(r["outputs_match"] for r in results)

    print(f"\nLogprob Statistics (threshold: {LOGPROB_THRESHOLD:.0e}):")
    overall_status = "PASSED" if logprob_match_count == num_prompts else "FAILED"
    print(f"  Overall logprob: {logprob_match_count}/{num_prompts} {overall_status}")
    print(f"  Prefill logprob: {prefill_match_count}/{num_prompts}")
    print(f"  Decode logprob:  {decode_match_count}/{num_prompts}")

    print(f"\nString Statistics:")
    print(f"  Output strings:  {outputs_match_count}/{num_prompts}")

    # Return overall stats for saving
    return {
        "logprob_differences": {
            "prefill": {
                "max_of_max_diffs": max(prefill_max_diffs),
                "mean_of_max_diffs": float(np.mean(prefill_max_diffs)),
                "mean_of_mean_diffs": float(np.mean(prefill_mean_diffs)),
            },
            "decode": {
                "max_of_max_diffs": max(decode_max_diffs),
                "mean_of_max_diffs": float(np.mean(decode_max_diffs)),
                "mean_of_mean_diffs": float(np.mean(decode_mean_diffs)),
            },
        },
        "match_statistics": {
            "overall_logprob_match_rate": logprob_match_count / num_prompts,
            "prefill_logprob_match_rate": prefill_match_count / num_prompts,
            "decode_logprob_match_rate": decode_match_count / num_prompts,
            "outputs_match_rate": outputs_match_count / num_prompts,
        },
    }


def compare_logprobs(
    sglang_logprobs: Dict[str, Any], hf_logprobs: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compare log probabilities and compute statistics."""
    print_section_header("Comparing Log Probabilities")

    results = []
    num_prompts = len(sglang_logprobs["top_input_logprobs"])

    for i in range(num_prompts):
        result = compare_single_prompt(i, sglang_logprobs, hf_logprobs)
        results.append(result)

    overall_stats = print_overall_statistics(results)

    return results, overall_stats


class TestLoRAHFSGLLogprobDifference(CustomTestCase):
    """
    Test case to compare log probabilities between HuggingFace+LoRA and SGLang+LoRA.
    """

    def _run_comparison_test(
        self,
        model_path: str,
        lora_paths: List[str],
        prompts: List[str],
        max_new_tokens: int = 32,
        torch_dtype: torch.dtype = torch.float16,
        lora_backend: str = LORA_BACKEND,
        port: int = DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
        disable_cuda_graph: bool = DISABLE_CUDA_GRAPH,
        lora_target_modules: Optional[List[str]] = LORA_TARGET_MODULES,
        tp_size: int = 1,
    ):
        """
        Run comparison test between SGLang and HuggingFace with LoRA.
        """
        print_section_header(f"Testing {model_path} with LoRA adapters")

        # Step 1: Run SGLang with LoRA
        sglang_logprobs = run_sglang_with_lora(
            model_path=model_path,
            lora_paths=lora_paths,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            torch_dtype=torch_dtype,
            lora_backend=lora_backend,
            port=port,
            disable_cuda_graph=disable_cuda_graph,
            lora_target_modules=lora_target_modules,
            tp_size=tp_size,
        )

        # Clear GPU memory
        print("\nClearing GPU memory...")
        torch.cuda.empty_cache()

        # Step 2: Run HuggingFace with LoRA
        hf_logprobs = run_hf_with_lora(
            model_path=model_path,
            lora_paths=lora_paths,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            torch_dtype=torch_dtype,
        )

        # Step 3: Compare log probabilities
        results, overall_stats = compare_logprobs(sglang_logprobs, hf_logprobs)

        # Assert that all prompts pass the threshold
        for result in results:
            self.assertTrue(
                result["prefill_logprob_match"],
                f"Prefill logprob mismatch for prompt {result['prompt_idx']} "
                f"(max_diff={result['prefill_max_diff']:.6e}, threshold={LOGPROB_THRESHOLD:.0e})",
            )
            self.assertTrue(
                result["decode_logprob_match"],
                f"Decode logprob mismatch for prompt {result['prompt_idx']} "
                f"(max_diff={result['decode_max_diff']:.6e}, threshold={LOGPROB_THRESHOLD:.0e})",
            )

        print_section_header("Test completed successfully!")

        return results, overall_stats

    def test_lora_logprob_comparison_basic(self):
        """
        Basic test comparing HF and SGLang LoRA logprobs with small model.
        """
        # Use a smaller model and shorter prompts for CI
        if is_in_ci():
            self.skipTest("Skipping in CI environment - requires large models")

        model_path = "meta-llama/Llama-2-7b-hf"
        lora_paths = ["yushengsu/sglang_lora_logprob_diff_without_tuning"]
        prompts = DEFAULT_TEST_PROMPTS[:2]  # Use fewer prompts for faster testing

        self._run_comparison_test(
            model_path=model_path,
            lora_paths=lora_paths,
            prompts=prompts,
            max_new_tokens=32,
        )

    def test_lora_logprob_comparison_full(self):
        """
        Full test comparing HF and SGLang LoRA logprobs with all prompts.
        """
        if is_in_ci():
            self.skipTest("Skipping in CI environment - requires large models")

        model_path = "meta-llama/Llama-2-7b-hf"
        lora_paths = ["yushengsu/sglang_lora_logprob_diff_without_tuning"]
        prompts = DEFAULT_TEST_PROMPTS

        self._run_comparison_test(
            model_path=model_path,
            lora_paths=lora_paths,
            prompts=prompts,
            max_new_tokens=32,
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
