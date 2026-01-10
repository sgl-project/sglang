"""
MI35x GSM8K Completion Evaluation Test (8-GPU)

This test uses the completion-based gsm8k benchmark (few-shot prompting)
for MI35x-specific models that differ from MI300X configurations.

MI35x-specific models:
- GPT-OSS series: Uses openai/gpt-oss-* (not lmsys/gpt-oss-*-bf16)
- DeepSeek-R1-0528: Same model as MI300X (MXFP4 only used for perf tests)

Model groups are selected via AMD_TEST_MODEL_GROUP environment variable:
- "gpt-oss" (default): GPT-OSS models with MI35x paths
- "deepseek-r1": DeepSeek-R1-0528 basic + MTP (same as MI300X)
- "deepseek-r1-dp-tc": DeepSeek-R1-0528 DP + TC (same as MI300X)
- "deepseek-r1-all": All DeepSeek-R1-0528 variants (basic, MTP, DP, TC)

Registry: nightly-amd-8-gpu-mi35x suite (8-GPU tests on MI35x)
"""

import ast
import os

# Set HF cache to /data2/models/ for MI35x so HF models download there
os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")
import re
import subprocess
import time
import unittest
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# HuggingFace Hub for model cache checking and download progress
try:
    from huggingface_hub import HfFileSystem
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("[WARNING] huggingface_hub not available - model cache checking disabled")

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file, read_jsonl

# Register for AMD CI - MI35x 8-GPU GSM8K completion tests (~120 min)
register_amd_ci(est_time=7200, suite="nightly-amd-8-gpu-mi35x", nightly=True)

INVALID = -9999999


@dataclass
class BaseModelConfig:
    """Configuration for a base model to test."""

    model_path: str  # HuggingFace model ID (e.g., "amd/DeepSeek-R1-MXFP4-Preview")
    tp_size: int = 8
    accuracy_threshold: float = 0.50
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    tokenizer_path: Optional[str] = None
    timeout: Optional[int] = None
    local_path: Optional[str] = None  # Preferred local path (checked first before HF)
    variant: Optional[str] = (
        None  # Test variant name (e.g., "basic", "MTP", "DP", "TC")
    )

    def __post_init__(self):
        if self.other_args is None:
            self.other_args = []
        if self.env_vars is None:
            self.env_vars = {}

    def get_effective_model_path(self) -> str:
        """Return local_path if it exists, otherwise model_path (HF ID)."""
        if self.local_path and os.path.exists(self.local_path):
            return self.local_path
        return self.model_path

    def get_display_name(self) -> str:
        """Return display name for logs/summary (model + variant if set)."""
        if self.variant:
            return f"{self.model_path} ({self.variant})"
        return self.model_path


# =============================================================================
# MI35x MODEL GROUPS - Different from MI300X configurations
# =============================================================================

# Group 1: GPT-OSS models (MI35x uses openai/* paths, not lmsys/*)
MI35X_GPT_OSS_MODELS = [
    # GPT-OSS-20B - MI35x specific path
    BaseModelConfig(
        model_path="openai/gpt-oss-20b",
        tp_size=8,
        accuracy_threshold=0.47,
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
        env_vars={"SGLANG_USE_AITER": "1"},
    ),
    # GPT-OSS-120B - MI35x specific path
    BaseModelConfig(
        model_path="openai/gpt-oss-120b",
        tp_size=8,
        accuracy_threshold=0.79,
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
        env_vars={"SGLANG_USE_AITER": "1"},
    ),
]

# Group 2: DeepSeek-R1-0528 basic + MTP (same model as MI300X for consistency)
# Runner: nightly-test-8-gpu-mi35x-deepseek-r1
# Note: MXFP4 variant only used for perf tests (test_deepseek_r1_mxfp4_perf.py)
MI35X_DEEPSEEK_R1_MODELS = [
    # DeepSeek-R1-0528 basic - reasoning model, ~80GB per GPU
    BaseModelConfig(
        model_path="deepseek-ai/DeepSeek-R1-0528",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=3600,  # 1 hour for large model
        variant="basic",
        other_args=[
            "--attention-backend",
            "aiter",
            "--chunked-prefill-size",
            "131072",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "SGLANG_USE_AITER": "1",
        },
    ),
    # DeepSeek-R1-0528 with MTP (EAGLE speculative decoding)
    BaseModelConfig(
        model_path="deepseek-ai/DeepSeek-R1-0528",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=3600,
        variant="MTP",
        other_args=[
            "--chunked-prefill-size",
            "131072",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--trust-remote-code",
        ],
        env_vars={
            "SGLANG_USE_AITER": "1",
        },
    ),
]

# Group 3: DeepSeek-R1-0528 with DP + TC (requires ROCm 7.0+)
# Runner: nightly-test-8-gpu-mi35x-deepseek-r1-dp-tc
MI35X_DEEPSEEK_R1_DP_TC_MODELS = [
    # DeepSeek-R1-0528 with DP attention
    BaseModelConfig(
        model_path="deepseek-ai/DeepSeek-R1-0528",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=3600,
        variant="DP",
        other_args=[
            "--chunked-prefill-size",
            "131072",
            "--dp-size",
            "8",
            "--enable-dp-attention",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "SGLANG_USE_ROCM700A": "1",
            "SGLANG_USE_AITER": "1",
        },
    ),
    # DeepSeek-R1-0528 with torch compile
    BaseModelConfig(
        model_path="deepseek-ai/DeepSeek-R1-0528",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=7200,  # 2 hours for compilation
        variant="TC",
        other_args=[
            "--chunked-prefill-size",
            "131072",
            "--mem-fraction-static",
            "0.70",
            "--cuda-graph-max-bs",
            "8",
            "--enable-torch-compile",
            "--disable-cuda-graph",
            "--trust-remote-code",
        ],
        env_vars={
            "SGLANG_USE_ROCM700A": "1",
            "SGLANG_USE_AITER": "1",
        },
    ),
]


def get_model_group() -> str:
    """Get the model group to test from environment variable."""
    return os.environ.get("AMD_TEST_MODEL_GROUP", "gpt-oss")


def get_models_for_group(group: str) -> List[BaseModelConfig]:
    """Get the list of models for a given group.

    Note: DeepSeek-R1-MXFP4 is only used for perf tests, not accuracy tests.
    See test_deepseek_r1_mxfp4_perf.py for MXFP4 perf tests.
    """
    if group == "gpt-oss":
        return MI35X_GPT_OSS_MODELS
    elif group == "deepseek-r1":
        return MI35X_DEEPSEEK_R1_MODELS
    elif group == "deepseek-r1-dp-tc":
        return MI35X_DEEPSEEK_R1_DP_TC_MODELS
    elif group == "deepseek-r1-all":
        # All DeepSeek-R1-0528 variants: basic, MTP, DP, TC
        return MI35X_DEEPSEEK_R1_MODELS + MI35X_DEEPSEEK_R1_DP_TC_MODELS
    elif group == "all":
        return (
            MI35X_GPT_OSS_MODELS
            + MI35X_DEEPSEEK_R1_MODELS
            + MI35X_DEEPSEEK_R1_DP_TC_MODELS
        )
    else:
        print(f"[WARNING] Unknown model group '{group}', using 'gpt-oss'")
        return MI35X_GPT_OSS_MODELS


# =============================================================================
# MODEL CACHE AND DOWNLOAD UTILITIES
# =============================================================================


def check_local_cache(model_path: str) -> Tuple[bool, str]:
    """
    Check if model is cached locally.

    Returns:
        Tuple of (is_cached, cache_path_or_message)
    """
    # Check common HF cache locations for MI35x
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/data2/models/huggingface/hub",
        os.environ.get("HF_HUB_CACHE", ""),
    ]
    cache_dirs = [d for d in cache_dirs if d]  # Remove empty

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


def log_model_status(config: "BaseModelConfig") -> Tuple[bool, str]:
    """
    Log detailed model availability status.

    Checks in order:
    1. local_path (if specified) - preferred local path
    2. model_path as local path (if starts with /)
    3. model_path as HF model ID - check cache then HF access

    Returns:
        Tuple of (is_available, status_message)
    """
    model_path = config.model_path
    local_path = config.local_path

    print(f"\n📦 Checking model: {model_path}")
    if local_path:
        print(f"   (preferred local: {local_path})")
    print("-" * 50)

    # Step 1: Check preferred local_path first (if specified)
    if local_path:
        if os.path.exists(local_path):
            print(f"  ✅ LOCAL PATH: Found at {local_path}")
            return True, f"Local path exists at {local_path}"
        else:
            print(f"  ⚠️  LOCAL PATH: Not found at {local_path}, trying HF fallback...")

    # Step 2: For absolute paths (starting with /), check if exists
    if model_path.startswith("/"):
        if os.path.exists(model_path):
            print(f"  ✅ LOCAL PATH: Found at {model_path}")
            return True, f"Local path exists at {model_path}"
        else:
            print(f"  ❌ LOCAL PATH: Not found at {model_path}")
            return False, f"Local path not found at {model_path}"

    # Step 3: For HF model IDs, check local cache first
    is_cached, cache_msg = check_local_cache(model_path)
    if is_cached:
        print(f"  ✅ LOCAL CACHE: Found at {cache_msg}")
        return True, f"Cached locally at {cache_msg}"
    else:
        print(f"  ⚠️  LOCAL CACHE: {cache_msg}")

    # Step 4: Check HF repo access (will download if accessible)
    is_accessible, access_msg = check_hf_repo_access(model_path)
    if is_accessible:
        print(f"  ✅ HF ACCESS: {access_msg}")
        print(
            f"  📥 Model will be downloaded from HuggingFace to {os.environ.get('HF_HOME', '~/.cache/huggingface')}"
        )
        return True, f"Will download from HF: {access_msg}"
    else:
        print(f"  ❌ HF ACCESS: {access_msg}")
        return False, access_msg


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
    """Run GSM8K few-shot completion benchmark."""
    import sglang as sgl
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"]
        )

    backend = RuntimeEndpoint(base_url)
    sgl.set_default_backend(backend)

    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=0,
        num_threads=parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))

    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    return float(acc), float(invalid), float(latency)


def popen_launch_server_for_base_model(
    base_url: str,
    config: BaseModelConfig,
) -> "subprocess.Popen":
    """Launch server for a base model with appropriate configuration."""
    env = os.environ.copy()
    for key, value in config.env_vars.items():
        env[key] = value
        print(f"Setting env: {key}={value}")

    other_args = list(config.other_args)
    other_args.extend(["--tp", str(config.tp_size)])
    other_args.extend(["--log-level-http", "warning"])

    if config.tokenizer_path:
        other_args.extend(["--tokenizer-path", config.tokenizer_path])

    timeout = config.timeout if config.timeout else DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    # Use effective model path (local if exists, else HF model ID)
    effective_model_path = config.get_effective_model_path()
    print(f"Using model path: {effective_model_path}")

    process = popen_launch_server(
        model=effective_model_path,
        base_url=base_url,
        timeout=timeout,
        other_args=other_args,
        env=env,
    )
    return process


class TestMI35xGsm8kCompletionEval(unittest.TestCase):
    """MI35x GSM8K Completion Evaluation Test (8-GPU)

    Tests MI35x-specific base models using few-shot completion benchmark.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_group = get_model_group()
        cls.models = get_models_for_group(cls.model_group)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

        print(f"\n{'='*60}")
        print(f"MI35x GSM8K Completion Evaluation Test (8-GPU)")
        print(f"{'='*60}")
        print(f"Model group: {cls.model_group}")
        print(f"Models to test: {len(cls.models)}")
        for m in cls.models:
            print(f"  - {m.model_path}")
        print(f"Questions per model: {cls.num_questions}")
        print(f"{'='*60}\n")

    def test_gsm8k_completion_all_models(self):
        """Test all configured MI35x models with GSM8K completion benchmark."""
        all_results = []
        total_test_start = time.time()

        summary = f"### MI35x Model Group: {self.model_group}\n\n"
        summary += (
            "| Model | TP | Accuracy | Threshold | Startup | Bench | Total | Status |\n"
        )
        summary += (
            "| ----- | -- | -------- | --------- | ------- | ----- | ----- | ------ |\n"
        )

        for config in self.models:
            display_name = config.get_display_name()
            with self.subTest(model=display_name):
                print(f"\n{'='*60}")
                print(f"Testing: {display_name} (TP={config.tp_size})")
                print(f"{'='*60}")

                error_message = None
                acc, invalid, latency = None, None, None
                startup_time, bench_time, total_time = None, None, None
                model_start = time.time()

                # Check model availability with detailed logging
                is_available, status_msg = log_model_status(config)

                if not is_available:
                    print(f"\n❌ MODEL NOT AVAILABLE: {status_msg}")
                    print(f"⏭️ SKIPPING: {display_name}")
                    status = "⏭️ SKIP"
                    all_results.append(
                        {
                            "model": display_name,
                            "tp_size": config.tp_size,
                            "accuracy": None,
                            "threshold": config.accuracy_threshold,
                            "passed": True,
                            "skipped": True,
                            "error": status_msg,
                        }
                    )
                else:
                    try:
                        print(f"\n🚀 Launching server for {display_name}...")
                        server_start = time.time()
                        process = popen_launch_server_for_base_model(
                            self.base_url, config
                        )
                        startup_time = time.time() - server_start
                        print(f"⏱️  Server startup: {startup_time:.1f}s")

                        try:
                            print(
                                f"📊 Running GSM8K benchmark ({self.num_questions} questions)..."
                            )
                            bench_start = time.time()
                            for attempt in range(3):
                                try:
                                    acc, invalid, latency = run_gsm8k_benchmark(
                                        self.base_url,
                                        num_questions=self.num_questions,
                                        num_shots=5,
                                        parallel=64,
                                    )
                                    print(
                                        f"   Attempt {attempt + 1}: accuracy={acc:.3f}"
                                    )
                                    if acc >= config.accuracy_threshold:
                                        break
                                except Exception as e:
                                    print(f"   Attempt {attempt + 1} failed: {e}")
                                    if attempt == 2:
                                        raise
                            bench_time = time.time() - bench_start
                            total_time = time.time() - model_start

                            passed = acc >= config.accuracy_threshold
                            status = "✅ PASS" if passed else "❌ FAIL"

                            print(
                                f"\n📈 Results: accuracy={acc:.3f} (threshold: {config.accuracy_threshold})"
                            )
                            print(f"⏱️  Total: {total_time:.1f}s")

                            all_results.append(
                                {
                                    "model": display_name,
                                    "tp_size": config.tp_size,
                                    "accuracy": acc,
                                    "threshold": config.accuracy_threshold,
                                    "startup_time": startup_time,
                                    "bench_time": bench_time,
                                    "total_time": total_time,
                                    "passed": passed,
                                    "skipped": False,
                                    "error": None,
                                }
                            )

                        except Exception as e:
                            error_message = str(e)
                            total_time = time.time() - model_start
                            print(f"\n❌ Error: {error_message}")
                            status = "❌ ERROR"
                            all_results.append(
                                {
                                    "model": display_name,
                                    "tp_size": config.tp_size,
                                    "accuracy": None,
                                    "threshold": config.accuracy_threshold,
                                    "passed": False,
                                    "skipped": False,
                                    "error": error_message,
                                }
                            )

                        finally:
                            print(f"\n🛑 Stopping server...")
                            kill_process_tree(process.pid)

                    except Exception as e:
                        error_message = str(e)
                        total_time = time.time() - model_start
                        print(f"\n❌ Error launching server: {error_message}")
                        status = "❌ ERROR"
                        all_results.append(
                            {
                                "model": display_name,
                                "tp_size": config.tp_size,
                                "accuracy": None,
                                "threshold": config.accuracy_threshold,
                                "passed": False,
                                "skipped": False,
                                "error": error_message,
                            }
                        )

                # Add to summary (use display name to show variant)
                acc_str = f"{acc:.3f}" if acc is not None else "N/A"
                startup_str = (
                    f"{startup_time:.0f}s" if startup_time is not None else "N/A"
                )
                bench_str = f"{bench_time:.0f}s" if bench_time is not None else "N/A"
                total_str = f"{total_time:.0f}s" if total_time is not None else "N/A"
                summary += f"| {display_name} | {config.tp_size} | {acc_str} | {config.accuracy_threshold} | {startup_str} | {bench_str} | {total_str} | {status} |\n"

        # Final summary
        total_test_time = time.time() - total_test_start
        failed_models = [
            r for r in all_results if not r["passed"] and not r.get("skipped", False)
        ]
        skipped_models = [r for r in all_results if r.get("skipped", False)]
        passed_models = [
            r for r in all_results if r["passed"] and not r.get("skipped", False)
        ]

        print(f"\n{'='*60}")
        print(f"SUMMARY - MI35x Model Group: {self.model_group}")
        print(f"{'='*60}")
        print(summary)
        print(
            f"\n📊 Passed: {len(passed_models)} | Failed: {len(failed_models)} | Skipped: {len(skipped_models)}"
        )
        print(f"⏱️  Total: {total_test_time:.1f}s ({total_test_time/60:.1f} min)")

        if is_in_ci():
            write_github_step_summary(summary)

        if failed_models:
            failure_msg = "\n".join(
                [
                    f"- {r['model']}: {r.get('error', 'below threshold')}"
                    for r in failed_models
                ]
            )
            raise AssertionError(f"The following models failed:\n{failure_msg}")


if __name__ == "__main__":
    unittest.main()
