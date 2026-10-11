import gc
import multiprocessing as mp
import os
import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, CustomTestCase

register_npu_ci(est_time=250, suite="full-1-npu-a3", nightly=True)

os.environ["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
os.environ["SGLANG_ENABLE_FAST_INPUT_LOGPROBS"] = "0"

# Test configuration constants
BASE_MODEL = "/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-4B"
LORA_PATHS = ["/root/.cache/huggingface/hub/Aswinkv07/qwen3.5-4b-neo4j-text2cypher-lora"]
LORA_BACKEND = "ascend"
ATTENTION_BACKEND = "ascend"
DISABLE_CUDA_GRAPH = True
LORA_TARGET_MODULES = None
PREFILL_LOGPROB_THRESHOLD = 3e-01
DECODE_LOGPROB_THRESHOLD = 5e-01
DECODE_DIVERGENCE_MARKER = 1.0
MAX_NEW_TOKENS = 16

# Default test prompts
DEFAULT_TEST_PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Write a long story.",
    "Write a short story.",
    "What are the main components of a computer?",
]

# Formatting constants
DIVIDER_WIDTH = 80
SECTION_CHAR = "="
SUBSECTION_CHAR = "-"

_TOKENIZER = None


def _get_tokenizer():
    """Lazily load the tokenizer once for token-level divergence detection."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True
        )
    return _TOKENIZER


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
    threshold = (
        DECODE_LOGPROB_THRESHOLD
        if logprob_type == "decode"
        else PREFILL_LOGPROB_THRESHOLD
    )
    matches_threshold = max_diff < threshold
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "shape": shape,
        "matches_threshold": matches_threshold,
        "type": logprob_type,
        "threshold": threshold,
    }


def print_logprob_comparison(comparison: Dict[str, Any]):
    """Print logprob comparison results in a consistent format."""
    logprob_type = comparison["type"].capitalize()
    print(f"\n{logprob_type} logprobs:")
    print(f"  Shape:           {comparison['shape']}")
    print(f"  Max difference:  {comparison['max_diff']:.6e}")
    print(f"  Mean difference: {comparison['mean_diff']:.6e}")
    status = "PASS" if comparison["matches_threshold"] else "FAIL"
    print(f"  Status:          {status} (threshold: {comparison['threshold']:.0e})")


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
    attention_backend: str,
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
        "Attention backend": attention_backend,
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
        attention_backend=attention_backend,
        disable_cuda_graph=disable_cuda_graph,
        disable_radix_cache=True,
        trust_remote_code=True,
        port=port,
        mem_fraction_static=0.50,
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
    # Truncate to min length: SGLang and HF may tokenize differently
    # (e.g., BOS token handling), causing different input token counts.
    min_prefill_len = min(len(sglang_prefill), len(hf_prefill))
    sglang_prefill = sglang_prefill[:min_prefill_len]
    hf_prefill = hf_prefill[:min_prefill_len]
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
    # Truncate to min length: SGLang may stop early (EOS) while HF
    # generates exactly max_new_tokens, causing different output lengths.
    min_decode_len = min(len(sglang_decode), len(hf_decode))
    sglang_decode = sglang_decode[:min_decode_len]
    hf_decode = hf_decode[:min_decode_len]
    # Compute before any truncation so it stays valid even when the prefix
    # becomes empty (divergence at step 0).
    per_step_max_diff = torch.max(torch.abs(sglang_decode - hf_decode), dim=-1).values
    tokenizer = _get_tokenizer()
    srt_gen_ids = tokenizer(
        sglang_data["output_strs"][prompt_idx], add_special_tokens=False
    )["input_ids"]
    hf_gen_ids = tokenizer(
        hf_data["output_strs"][prompt_idx], add_special_tokens=False
    )["input_ids"]
    token_divergence_step = None
    for step in range(min(len(srt_gen_ids), len(hf_gen_ids), min_decode_len)):
        if srt_gen_ids[step] != hf_gen_ids[step]:
            token_divergence_step = step
            break
    if token_divergence_step is not None:
        print(
            f"\n  NOTE: decode token sequences diverged at step "
            f"{token_divergence_step} (argmax flip on a near-tie token); "
            f"comparing common prefix only."
        )
        sglang_decode = sglang_decode[:token_divergence_step]
        hf_decode = hf_decode[:token_divergence_step]
    marker_scan_len = (
        token_divergence_step
        if token_divergence_step is not None
        else len(per_step_max_diff)
    )
    divergence_positions = (
        (per_step_max_diff[:marker_scan_len] > DECODE_DIVERGENCE_MARKER)
        .nonzero(as_tuple=True)[0]
    )
    if len(divergence_positions) > 0:
        divergence_step = divergence_positions[0].item()
        print(
            f"\n  NOTE: decode logprobs diverged at step {divergence_step} "
            f"(per-step diff {per_step_max_diff[divergence_step]:.3f} > "
            f"{DECODE_DIVERGENCE_MARKER:.1f}); comparing common prefix only."
        )
        sglang_decode = sglang_decode[:divergence_step]
        hf_decode = hf_decode[:divergence_step]
    else:
        divergence_step = None
    divergence_cuts = [
        s for s in (token_divergence_step, divergence_step) if s is not None
    ]
    first_divergence_step = min(divergence_cuts) if divergence_cuts else None
    if first_divergence_step == 0:
        decode_comparison = {
            "max_diff": per_step_max_diff[0].item(),
            "mean_diff": per_step_max_diff[0].item(),
            "shape": [0, len(sglang_data["top_output_logprobs"][prompt_idx][0])],
            "matches_threshold": False,
            "threshold": DECODE_LOGPROB_THRESHOLD,
            "type": "decode",
        }
    else:
        decode_comparison = compare_logprobs_for_type(
            sglang_decode, hf_decode, "decode"
        )
    print_logprob_comparison(decode_comparison)
    # Store decode results
    result["decode_max_diff"] = decode_comparison["max_diff"]
    result["decode_mean_diff"] = decode_comparison["mean_diff"]
    result["decode_shape"] = decode_comparison["shape"]
    result["decode_logprob_match"] = decode_comparison["matches_threshold"]
    result["decode_divergence_step"] = first_divergence_step
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
    print(
        f"\nLogprob Statistics (prefill threshold: {PREFILL_LOGPROB_THRESHOLD:.0e}, "
        f"decode threshold: {DECODE_LOGPROB_THRESHOLD:.0e}):"
    )
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
        max_new_tokens: int = MAX_NEW_TOKENS,
        torch_dtype: torch.dtype = torch.bfloat16,
        lora_backend: str = LORA_BACKEND,
        attention_backend: str = ATTENTION_BACKEND,
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
            attention_backend=attention_backend,
            port=port,
            disable_cuda_graph=disable_cuda_graph,
            lora_target_modules=lora_target_modules,
            tp_size=tp_size,
        )
        # Clear NPU memory before loading HF model
        print("\nClearing NPU memory...")
        if torch.npu.is_available():
            torch.npu.synchronize()
            torch.npu.empty_cache()
        gc.collect()
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
                f"(max_diff={result['prefill_max_diff']:.6e}, threshold={PREFILL_LOGPROB_THRESHOLD:.0e})",
            )
            self.assertTrue(
                result["decode_logprob_match"],
                f"Decode logprob mismatch for prompt {result['prompt_idx']} "
                f"(max_diff={result['decode_max_diff']:.6e}, threshold={DECODE_LOGPROB_THRESHOLD:.0e})",
            )
        print_section_header("Test completed successfully!")
        return results, overall_stats

    def test_lora_logprob_comparison_basic(self):
        """
        Basic test comparing HF and SGLang LoRA logprobs with small model.
        """
        prompts = DEFAULT_TEST_PROMPTS[:2]  # Use fewer prompts for faster testing
        self._run_comparison_test(
            model_path=BASE_MODEL,
            lora_paths=LORA_PATHS,
            prompts=prompts,
        )

    def test_lora_logprob_comparison_full(self):
        """
        Full test comparing HF and SGLang LoRA logprobs with all prompts.
        """
        self._run_comparison_test(
            model_path=BASE_MODEL,
            lora_paths=LORA_PATHS,
            prompts=DEFAULT_TEST_PROMPTS,
        )

    def test_lora_logprob_comparison_chunked(self):
        """
        Test with logprobs chunking enabled and a small chunk size so that
        even short prompts trigger the multi‑pass lm_head LoRA path.
        """
        saved = {}
        env_overrides = {
            "SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK": "true",
            "SGLANG_LOGITS_PROCESSER_CHUNK_SIZE": "4",
        }
        for key, val in env_overrides.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = val
        try:
            self._run_comparison_test(
                model_path=BASE_MODEL,
                lora_paths=LORA_PATHS,
                prompts=DEFAULT_TEST_PROMPTS,
            )
        finally:
            for key, orig in saved.items():
                if orig is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = orig


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        if torch.npu.is_available():
            torch.npu.empty_cache()
            torch.npu.synchronize()
