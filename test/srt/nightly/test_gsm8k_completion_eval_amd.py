"""
AMD GSM8K Completion Evaluation Test

This test uses the completion-based gsm8k benchmark (few-shot prompting)
which works with base models that don't have chat templates.

This complements test_gsm8k_eval_amd.py which uses mgsm_en (chat completions)
for instruction-tuned models.

Base models tested here:
- GPT-OSS series (lmsys/gpt-oss-*-bf16)
- GROK series (lmzheng/grok-1, xai-org/grok-2, amd/grok-1-W4A8KV8)

Model groups are selected via AMD_TEST_MODEL_GROUP environment variable:
- "gpt-oss" (default): GPT-OSS models only (nightly-amd-8-gpu)
- "grok1-fp8": GROK1-FP8 only (nightly-amd-8-gpu-grok1-fp8)
- "grok2": GROK2.5 only (nightly-amd-8-gpu-grok2)
- "grok1-in4": GROK1-IN4 only (nightly-amd-8-gpu-grok1-in4)
- "all": All models

Reference: benchmark/gsm8k/bench_sglang.py
"""

import ast
import os
import re
import subprocess
import time
import unittest
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# HuggingFace Hub for model cache checking and download progress
try:
    from huggingface_hub import HfFileSystem, snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("[WARNING] huggingface_hub not available - model cache checking disabled")

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999


@dataclass
class BaseModelConfig:
    """Configuration for a base model to test."""

    model_path: str
    tp_size: int = 8
    accuracy_threshold: float = 0.50
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    tokenizer_path: Optional[str] = None
    timeout: Optional[int] = None  # Custom timeout for server launch (seconds)

    def __post_init__(self):
        if self.other_args is None:
            self.other_args = []
        if self.env_vars is None:
            self.env_vars = {}


# =============================================================================
# MODEL GROUPS - Each group runs on a separate 8-GPU runner
# =============================================================================

# Group 1: GPT-OSS models (cached on upstream CI)
# Runner: nightly-amd-8-gpu
AMD_GPT_OSS_MODELS = [
    # GPT-OSS-20B - smaller model, run first for faster feedback
    BaseModelConfig(
        model_path="lmsys/gpt-oss-20b-bf16",
        tp_size=8,
        accuracy_threshold=0.50,
        other_args=[
            "--chunked-prefill-size",
            "130172",
            "--max-running-requests",
            "128",
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "triton",
            "--trust-remote-code",
        ],
        env_vars={"SGLANG_USE_AITER": "0"},
    ),
    # GPT-OSS-120B - large model, needs longer timeout
    BaseModelConfig(
        model_path="lmsys/gpt-oss-120b-bf16",
        tp_size=8,
        accuracy_threshold=0.82,
        timeout=900,  # 15 minutes for 120B model
        other_args=[
            "--chunked-prefill-size",
            "130172",
            "--max-running-requests",
            "128",
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "triton",
            "--trust-remote-code",
        ],
        env_vars={"SGLANG_USE_AITER": "0"},
    ),
]

# Group 2: GROK1-FP8 only (lmzheng/grok-1)
# Runner: nightly-amd-8-gpu-grok1-fp8
AMD_GROK1_FP8_MODELS = [
    # GROK1-FP8 - cached on upstream CI, verified accuracy: 0.860
    BaseModelConfig(
        model_path="lmzheng/grok-1",
        tp_size=8,
        accuracy_threshold=0.80,
        timeout=3600,  # 1 hour for kernel compilation
        tokenizer_path="Xenova/grok-1-tokenizer",
        other_args=[
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "RCCL_MSCCL_ENABLE": "0",
            "SGLANG_USE_AITER": "1",
            "SGLANG_INT4_WEIGHT": "0",
        },
    ),
]

# Group 3: GROK2.5 only (xai-org/grok-2)
# Runner: nightly-amd-8-gpu-grok2
AMD_GROK2_MODELS = [
    # GROK2.5 (grok-2) - latest GROK model
    BaseModelConfig(
        model_path="xai-org/grok-2",
        tp_size=8,
        accuracy_threshold=0.915,
        timeout=3600,  # 1 hour for download + kernel compilation
        tokenizer_path="alvarobartt/grok-2-tokenizer",
        other_args=[
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "RCCL_MSCCL_ENABLE": "0",
            "SGLANG_USE_AITER": "1",
            "SGLANG_INT4_WEIGHT": "0",
        },
    ),
]

# Group 4: GROK1-IN4 only (amd/grok-1-W4A8KV8)
# Runner: nightly-amd-8-gpu-grok1-in4
AMD_GROK1_IN4_MODELS = [
    # GROK1-IN4 - INT4 quantized version
    BaseModelConfig(
        model_path="amd/grok-1-W4A8KV8",
        tp_size=8,
        accuracy_threshold=0.80,
        timeout=3600,  # 1 hour for download + kernel compilation
        tokenizer_path="Xenova/grok-1-tokenizer",
        other_args=[
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "RCCL_MSCCL_ENABLE": "0",
            "SGLANG_USE_AITER": "1",
            "SGLANG_INT4_WEIGHT": "1",
        },
    ),
]


def get_model_group() -> str:
    """Get the model group to test from environment variable."""
    return os.environ.get("AMD_TEST_MODEL_GROUP", "gpt-oss")


def get_models_for_group(group: str) -> List[BaseModelConfig]:
    """Get the list of models for a given group."""
    if group == "gpt-oss":
        return AMD_GPT_OSS_MODELS
    elif group == "grok1-fp8":
        return AMD_GROK1_FP8_MODELS
    elif group == "grok2":
        return AMD_GROK2_MODELS
    elif group == "grok1-in4":
        return AMD_GROK1_IN4_MODELS
    elif group == "all":
        return (
            AMD_GPT_OSS_MODELS
            + AMD_GROK1_FP8_MODELS
            + AMD_GROK2_MODELS
            + AMD_GROK1_IN4_MODELS
        )
    else:
        print(f"[WARNING] Unknown model group '{group}', using 'gpt-oss'")
        return AMD_GPT_OSS_MODELS


# =============================================================================
# MODEL CACHE AND DOWNLOAD UTILITIES
# =============================================================================


def check_local_cache(model_path: str) -> Tuple[bool, str]:
    """
    Check if model is cached locally.

    Returns:
        Tuple of (is_cached, cache_path_or_message)
    """
    # Check common HF cache locations
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/sgl-data/hf-cache/hub",
        "/home/runner/sgl-data/hf-cache",
    ]

    # Convert model_path to cache directory format (org--model)
    cache_name = f"models--{model_path.replace('/', '--')}"

    for cache_dir in cache_dirs:
        cache_path = os.path.join(cache_dir, cache_name)
        if os.path.exists(cache_path):
            # Check if there are snapshots
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
                return True, cache_path

    return False, f"Not found in: {', '.join(cache_dirs)}"


def check_hf_repo_access(model_path: str) -> Tuple[bool, str]:
    """
    Check if HuggingFace repository is accessible.

    Returns:
        Tuple of (is_accessible, message)
    """
    if not HF_HUB_AVAILABLE:
        return True, "huggingface_hub not available, skipping access check"

    try:
        fs = HfFileSystem()
        # Try to list files in the repo
        files = fs.ls(model_path, detail=False)
        if files:
            return True, f"Repository accessible ({len(files)} files)"
        else:
            return False, "Repository exists but is empty"
    except GatedRepoError:
        return False, "GATED REPO - requires authentication/approval"
    except RepositoryNotFoundError:
        return False, "REPO NOT FOUND on HuggingFace"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return False, f"AUTH ERROR - may need HF_TOKEN: {error_msg[:100]}"
        elif "404" in error_msg:
            return False, f"NOT FOUND: {error_msg[:100]}"
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            return False, f"NETWORK ERROR: {error_msg[:100]}"
        else:
            return False, f"ERROR: {error_msg[:100]}"


def log_model_status(config: BaseModelConfig) -> Tuple[bool, str]:
    """
    Log detailed model availability status.

    Returns:
        Tuple of (is_available, status_message)
    """
    model_path = config.model_path
    print(f"\n📦 Checking model: {model_path}")
    print("-" * 50)

    # Check local cache first
    is_cached, cache_msg = check_local_cache(model_path)
    if is_cached:
        print(f"  ✅ LOCAL CACHE: Found at {cache_msg}")
        return True, f"Cached locally at {cache_msg}"
    else:
        print(f"  ⚠️  LOCAL CACHE: {cache_msg}")

    # Check HF repo access
    is_accessible, access_msg = check_hf_repo_access(model_path)
    if is_accessible:
        print(f"  ✅ HF ACCESS: {access_msg}")
        print(f"  📥 Model will be downloaded from HuggingFace (this may take a while)")
        return True, f"Will download from HF: {access_msg}"
    else:
        print(f"  ❌ HF ACCESS: {access_msg}")
        return False, access_msg

    # Also check tokenizer if specified
    if config.tokenizer_path:
        tok_cached, tok_msg = check_local_cache(config.tokenizer_path)
        if tok_cached:
            print(f"  ✅ TOKENIZER CACHE: Found at {tok_msg}")
        else:
            tok_accessible, tok_access_msg = check_hf_repo_access(config.tokenizer_path)
            if tok_accessible:
                print(f"  ✅ TOKENIZER HF: {tok_access_msg}")
            else:
                print(f"  ⚠️  TOKENIZER: {tok_access_msg}")

    return is_accessible, access_msg


def download_model_with_progress(
    model_path: str, timeout: int = 3600
) -> Tuple[bool, str]:
    """
    Download model with progress logging.

    Returns:
        Tuple of (success, message)
    """
    if not HF_HUB_AVAILABLE:
        return True, "huggingface_hub not available, skipping pre-download"

    print(f"\n📥 Pre-downloading model: {model_path}")
    print(f"   Timeout: {timeout}s ({timeout/60:.0f} minutes)")
    print("-" * 50)

    start_time = time.time()

    try:
        # Use snapshot_download which shows progress
        local_dir = snapshot_download(
            repo_id=model_path,
            local_files_only=False,
            resume_download=True,
        )
        elapsed = time.time() - start_time
        print(f"  ✅ Download complete in {elapsed:.1f}s")
        print(f"  📁 Location: {local_dir}")
        return True, f"Downloaded to {local_dir}"

    except GatedRepoError:
        return False, "GATED REPO - requires authentication/approval"
    except RepositoryNotFoundError:
        return False, "REPO NOT FOUND on HuggingFace"
    except Exception as e:
        error_msg = str(e)
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            return False, f"TIMEOUT after {elapsed:.0f}s: {error_msg[:100]}"
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            return False, f"NETWORK ERROR after {elapsed:.0f}s: {error_msg[:100]}"
        else:
            return False, f"ERROR after {elapsed:.0f}s: {error_msg[:100]}"


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================


def get_one_example(lines, i, include_answer):
    """Format a single GSM8K example."""
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    """Get k few-shot examples for prompting."""
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    """Extract numerical answer from response."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def run_gsm8k_benchmark(
    base_url: str,
    num_questions: int = 200,
    num_shots: int = 5,
    parallel: int = 64,
) -> Tuple[float, float, float]:
    """
    Run GSM8K few-shot completion benchmark.

    Returns:
        Tuple of (accuracy, invalid_rate, latency)
    """
    import sglang as sgl
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

    # Download and load data
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

    # Construct prompts
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    # Define sglang function
    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"]
        )

    # Set backend
    backend = RuntimeEndpoint(base_url)
    sgl.set_default_backend(backend)

    # Run benchmark
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=0,
        num_threads=parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    # Extract predictions
    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))

    # Compute metrics
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    return float(acc), float(invalid), float(latency)


def popen_launch_server_for_base_model(
    base_url: str,
    config: BaseModelConfig,
) -> "subprocess.Popen":
    """Launch server for a base model with appropriate configuration."""
    # Build environment - start with current env and add config-specific vars
    env = os.environ.copy()
    for key, value in config.env_vars.items():
        env[key] = value
        print(f"Setting env: {key}={value}")

    # Build other_args
    other_args = list(config.other_args)
    other_args.extend(["--tp", str(config.tp_size)])
    other_args.extend(["--log-level-http", "warning"])

    if config.tokenizer_path:
        other_args.extend(["--tokenizer-path", config.tokenizer_path])

    # Use custom timeout if provided, otherwise use default
    timeout = config.timeout if config.timeout else DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    process = popen_launch_server(
        model=config.model_path,
        base_url=base_url,
        timeout=timeout,
        other_args=other_args,
        env=env,  # Pass environment explicitly
    )
    return process


class TestNightlyGsm8kCompletionEvalAMD(unittest.TestCase):
    """
    AMD GSM8K Completion Evaluation Test

    Tests base models using few-shot completion benchmark.
    This is different from mgsm_en which uses chat completions.

    Model group is selected via AMD_TEST_MODEL_GROUP env var:
    - "gpt-oss": GPT-OSS models only (default)
    - "grok": GROK1-FP8 and GROK2.5
    - "grok-in4": GROK1-IN4 only
    - "all": All models
    """

    @classmethod
    def setUpClass(cls):
        # Get model group from environment
        cls.model_group = get_model_group()
        cls.models = get_models_for_group(cls.model_group)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

        print(f"\n{'='*60}")
        print(f"AMD GSM8K Completion Evaluation Test")
        print(f"{'='*60}")
        print(f"Model group: {cls.model_group}")
        print(f"Models to test: {len(cls.models)}")
        for m in cls.models:
            print(f"  - {m.model_path}")
        print(f"Questions per model: {cls.num_questions}")
        print(f"{'='*60}\n")

    def test_gsm8k_completion_all_models(self):
        """Test all configured base models with GSM8K completion benchmark."""
        all_results = []
        summary = f"### Model Group: {self.model_group}\n\n"
        summary += (
            "| Model | TP | Accuracy | Threshold | Invalid | Latency | Status |\n"
        )
        summary += (
            "| ----- | -- | -------- | --------- | ------- | ------- | ------ |\n"
        )

        for config in self.models:
            with self.subTest(model=config.model_path):
                print(f"\n{'='*60}")
                print(f"Testing: {config.model_path} (TP={config.tp_size})")
                print(f"{'='*60}")

                error_message = None
                acc, invalid, latency = None, None, None
                skipped = False

                # Check model availability with detailed logging
                is_available, status_msg = log_model_status(config)

                if not is_available:
                    print(f"\n❌ MODEL NOT AVAILABLE: {status_msg}")
                    print(f"⏭️ SKIPPING: {config.model_path}")
                    status = f"⏭️ SKIP ({status_msg[:20]}...)"
                    skipped = True
                    all_results.append(
                        {
                            "model": config.model_path,
                            "tp_size": config.tp_size,
                            "accuracy": None,
                            "threshold": config.accuracy_threshold,
                            "invalid": None,
                            "latency": None,
                            "passed": True,  # Don't count as failure
                            "skipped": True,
                            "error": status_msg,
                        }
                    )
                else:
                    try:
                        # Launch server
                        print(f"\n🚀 Launching server for {config.model_path}...")
                        process = popen_launch_server_for_base_model(
                            self.base_url, config
                        )

                        try:
                            # Run benchmark
                            print(
                                f"📊 Running GSM8K benchmark ({self.num_questions} questions)..."
                            )
                            acc, invalid, latency = run_gsm8k_benchmark(
                                self.base_url,
                                num_questions=self.num_questions,
                                num_shots=5,
                                parallel=64,
                            )

                            print(f"\n📈 Results for {config.model_path}:")
                            print(
                                f"   Accuracy: {acc:.3f} (threshold: {config.accuracy_threshold})"
                            )
                            print(f"   Invalid: {invalid:.3f}")
                            print(f"   Latency: {latency:.1f}s")

                            passed = acc >= config.accuracy_threshold
                            status = "✅ PASS" if passed else "❌ FAIL"

                            if passed:
                                print(f"   Status: ✅ PASSED")
                            else:
                                print(f"   Status: ❌ FAILED (below threshold)")

                            all_results.append(
                                {
                                    "model": config.model_path,
                                    "tp_size": config.tp_size,
                                    "accuracy": acc,
                                    "threshold": config.accuracy_threshold,
                                    "invalid": invalid,
                                    "latency": latency,
                                    "passed": passed,
                                    "skipped": False,
                                    "error": None,
                                }
                            )

                        except Exception as e:
                            error_message = str(e)
                            print(f"\n❌ Error during benchmark: {error_message}")
                            status = "❌ ERROR"
                            all_results.append(
                                {
                                    "model": config.model_path,
                                    "tp_size": config.tp_size,
                                    "accuracy": None,
                                    "threshold": config.accuracy_threshold,
                                    "invalid": None,
                                    "latency": None,
                                    "passed": False,
                                    "skipped": False,
                                    "error": error_message,
                                }
                            )

                        finally:
                            print(f"\n🛑 Stopping server for {config.model_path}...")
                            kill_process_tree(process.pid)

                    except Exception as e:
                        error_message = str(e)
                        print(f"\n❌ Error launching server: {error_message}")
                        status = "❌ ERROR"
                        all_results.append(
                            {
                                "model": config.model_path,
                                "tp_size": config.tp_size,
                                "accuracy": None,
                                "threshold": config.accuracy_threshold,
                                "invalid": None,
                                "latency": None,
                                "passed": False,
                                "skipped": False,
                                "error": error_message,
                            }
                        )

                # Add to summary
                acc_str = f"{acc:.3f}" if acc is not None else "N/A"
                invalid_str = f"{invalid:.3f}" if invalid is not None else "N/A"
                latency_str = f"{latency:.1f}s" if latency is not None else "N/A"
                summary += f"| {config.model_path} | {config.tp_size} | {acc_str} | {config.accuracy_threshold} | {invalid_str} | {latency_str} | {status} |\n"

        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY - Model Group: {self.model_group}")
        print(f"{'='*60}")
        print(summary)

        # Write GitHub step summary
        if is_in_ci():
            write_github_step_summary(
                f"### TestNightlyGsm8kCompletionEvalAMD ({self.model_group})\n{summary}"
            )

        # Check for failures (exclude skipped models)
        failed_models = [
            r for r in all_results if not r["passed"] and not r.get("skipped", False)
        ]
        skipped_models = [r for r in all_results if r.get("skipped", False)]
        passed_models = [
            r for r in all_results if r["passed"] and not r.get("skipped", False)
        ]

        print(f"\n📊 Final Statistics:")
        print(f"   Passed: {len(passed_models)}")
        print(f"   Failed: {len(failed_models)}")
        print(f"   Skipped: {len(skipped_models)}")

        if skipped_models:
            print(f"\n⏭️ Skipped models (not available):")
            for r in skipped_models:
                print(f"   - {r['model']}: {r['error']}")

        if failed_models:
            failure_msg = "\n".join(
                [
                    f"- {r['model']}: accuracy={r['accuracy']}, threshold={r['threshold']}, error={r['error']}"
                    for r in failed_models
                ]
            )
            raise AssertionError(f"The following models failed:\n{failure_msg}")


if __name__ == "__main__":
    unittest.main()
