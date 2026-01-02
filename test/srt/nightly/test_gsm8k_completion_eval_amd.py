"""
AMD GSM8K Completion Evaluation Test

This test uses the completion-based gsm8k benchmark (few-shot prompting)
which works with base models that don't have chat templates.

This complements test_gsm8k_eval_amd.py which uses mgsm_en (chat completions)
for instruction-tuned models.

Base models tested here:
- GPT-OSS series (lmsys/gpt-oss-20b-bf16, lmsys/gpt-oss-120b-bf16)
- GROK series (lmzheng/grok-1, amd/grok-1-W4A8KV8, xai-org/grok-2)
- DeepSeek series (deepseek-ai/DeepSeek-V3-0324, deepseek-ai/DeepSeek-R1-0528)

Model groups are selected via AMD_TEST_MODEL_GROUP environment variable:
- "gpt-oss" (default): GPT-OSS models only (nightly-amd-8-gpu-gpt-oss)
- "grok": All GROK models (nightly-amd-8-gpu-grok)
- "deepseek-v3-dp": DeepSeek-V3 with DP attention (nightly-amd-8-gpu-deepseek-v3-dp)
- "deepseek-v3-tc": DeepSeek-V3 with torch compile (nightly-amd-8-gpu-deepseek-v3-tc)
- "deepseek-v3-mtp": DeepSeek-V3 with MTP/EAGLE (nightly-amd-8-gpu-deepseek-v3-mtp)
- "deepseek-r1": DeepSeek-R1 reasoning model (nightly-amd-8-gpu-deepseek-r1)
- "all": All models
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
        env_vars={"SGLANG_USE_AITER": "0"},
    ),
    # GPT-OSS-120B - large model, needs longer timeout
    BaseModelConfig(
        model_path="lmsys/gpt-oss-120b-bf16",
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
        env_vars={"SGLANG_USE_AITER": "0"},
    ),
]

# Group 2: All GROK models
# Runner: nightly-amd-8-gpu-grok
# Order: GROK1-FP8 -> GROK1-IN4 -> GROK2.5
AMD_GROK_MODELS = [
    # GROK1-FP8 - verified accuracy: 0.860, runtime: ~12.5min
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
    # GROK1-IN4 - verified accuracy: 0.820, runtime: ~12.5min
    BaseModelConfig(
        model_path="amd/grok-1-W4A8KV8",
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
            "SGLANG_INT4_WEIGHT": "1",
        },
    ),
    # GROK2.5 - verified accuracy: 0.945, runtime: ~14.5min
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

# Group 3: DeepSeek-V3 with DP Attention
# Runner: nightly-amd-8-gpu-deepseek-v3-dp
# Note: Uses DP attention (dp-size=8) for better performance, requires ROCm 7.0+
AMD_DEEPSEEK_V3_DP_MODELS = [
    # DeepSeek-V3-0324 with DP attention
    BaseModelConfig(
        model_path="deepseek-ai/DeepSeek-V3-0324",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=3600,  # 1 hour for large model
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
]

# Group 3b: DeepSeek-V3 with Torch Compile
# Runner: nightly-amd-8-gpu-deepseek-v3-tc
# Note: Uses torch compile for performance optimization, requires ROCm 7.0+
AMD_DEEPSEEK_V3_TC_MODELS = [
    # DeepSeek-V3-0324 with torch compile
    BaseModelConfig(
        model_path="deepseek-ai/DeepSeek-V3-0324",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=7200,  # 2 hours for compilation + large model
        other_args=[
            "--chunked-prefill-size",
            "131072",
            "--mem-fraction-static",
            "0.70",  # Reduced further for torch compile
            "--cuda-graph-max-bs",
            "8",  # Reduced from 16 to reduce memory
            "--enable-torch-compile",
            "--disable-cuda-graph",  # Disable cuda graph to avoid memory issues
            "--trust-remote-code",
        ],
        env_vars={
            "SGLANG_USE_ROCM700A": "1",
            "SGLANG_USE_AITER": "1",
        },
    ),
]

# Group 3c: DeepSeek-V3 with MTP (EAGLE speculative decoding)
# Runner: nightly-amd-8-gpu-deepseek-v3-mtp
# Note: Uses MTP for improved throughput, requires ROCm 7.0+
AMD_DEEPSEEK_V3_MTP_MODELS = [
    # DeepSeek-V3-0324 with MTP (EAGLE speculative decoding)
    BaseModelConfig(
        model_path="deepseek-ai/DeepSeek-V3-0324",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=3600,  # 1 hour for large model
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
            "SGLANG_USE_ROCM700A": "1",
            "SGLANG_USE_AITER": "1",
        },
    ),
]

# Group 4: DeepSeek-R1 (reasoning model)
# Runner: nightly-amd-8-gpu-deepseek-r1
AMD_DEEPSEEK_R1_MODELS = [
    # DeepSeek-R1-0528 - reasoning model, ~80GB per GPU
    BaseModelConfig(
        model_path="deepseek-ai/DeepSeek-R1-0528",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=3600,  # 1 hour for large model
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
]


def get_model_group() -> str:
    """Get the model group to test from environment variable."""
    return os.environ.get("AMD_TEST_MODEL_GROUP", "gpt-oss")


def get_models_for_group(group: str) -> List[BaseModelConfig]:
    """Get the list of models for a given group."""
    if group == "gpt-oss":
        return AMD_GPT_OSS_MODELS
    elif group == "grok":
        return AMD_GROK_MODELS
    elif group == "deepseek-v3-dp":
        return AMD_DEEPSEEK_V3_DP_MODELS
    elif group == "deepseek-v3-tc":
        return AMD_DEEPSEEK_V3_TC_MODELS
    elif group == "deepseek-v3-mtp":
        return AMD_DEEPSEEK_V3_MTP_MODELS
    elif group == "deepseek-r1":
        return AMD_DEEPSEEK_R1_MODELS
    elif group == "all":
        return (
            AMD_GPT_OSS_MODELS
            + AMD_GROK_MODELS
            + AMD_DEEPSEEK_V3_DP_MODELS
            + AMD_DEEPSEEK_V3_TC_MODELS
            + AMD_DEEPSEEK_V3_MTP_MODELS
            + AMD_DEEPSEEK_R1_MODELS
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
    - "gpt-oss": GPT-OSS models only (default, nightly-amd-8-gpu)
    - "grok": All GROK models (nightly-amd-8-gpu-grok)
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
        total_test_start = time.time()

        # Summary table with runtime columns
        summary = f"### Model Group: {self.model_group}\n\n"
        summary += (
            "| Model | TP | Accuracy | Threshold | Startup | Bench | Total | Status |\n"
        )
        summary += (
            "| ----- | -- | -------- | --------- | ------- | ----- | ----- | ------ |\n"
        )

        for config in self.models:
            with self.subTest(model=config.model_path):
                print(f"\n{'='*60}")
                print(f"Testing: {config.model_path} (TP={config.tp_size})")
                print(f"{'='*60}")

                error_message = None
                acc, invalid, latency = None, None, None
                startup_time, bench_time, total_time = None, None, None
                skipped = False
                model_start = time.time()

                # Check model availability with detailed logging
                is_available, status_msg = log_model_status(config)

                if not is_available:
                    print(f"\n❌ MODEL NOT AVAILABLE: {status_msg}")
                    print(f"⏭️ SKIPPING: {config.model_path}")
                    status = f"⏭️ SKIP"
                    skipped = True
                    all_results.append(
                        {
                            "model": config.model_path,
                            "tp_size": config.tp_size,
                            "accuracy": None,
                            "threshold": config.accuracy_threshold,
                            "invalid": None,
                            "latency": None,
                            "startup_time": None,
                            "bench_time": None,
                            "total_time": None,
                            "passed": True,  # Don't count as failure
                            "skipped": True,
                            "error": status_msg,
                        }
                    )
                else:
                    try:
                        # Launch server with timing
                        print(f"\n🚀 Launching server for {config.model_path}...")
                        server_start = time.time()
                        process = popen_launch_server_for_base_model(
                            self.base_url, config
                        )
                        startup_time = time.time() - server_start
                        print(f"⏱️  Server startup: {startup_time:.1f}s")

                        try:
                            # Run benchmark with timing and retries
                            print(
                                f"📊 Running GSM8K benchmark ({self.num_questions} questions)..."
                            )
                            bench_start = time.time()
                            acc, invalid, latency = None, None, None
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
                                    print(
                                        f"   Attempt {attempt + 1} failed with error: {e}"
                                    )
                                    if attempt == 2:
                                        raise
                            bench_time = time.time() - bench_start

                            total_time = time.time() - model_start

                            print(f"\n📈 Results for {config.model_path}:")
                            print(
                                f"   Accuracy: {acc:.3f} (threshold: {config.accuracy_threshold})"
                            )
                            print(f"   Invalid: {invalid:.3f}")
                            print(f"   Benchmark latency: {latency:.1f}s")
                            print(f"\n⏱️  Runtime breakdown:")
                            print(f"   Server startup: {startup_time:.1f}s")
                            print(f"   Benchmark: {bench_time:.1f}s")
                            print(f"   Total: {total_time:.1f}s")

                            passed = acc >= config.accuracy_threshold
                            status = "✅ PASS" if passed else "❌ FAIL"

                            if passed:
                                print(f"\n   Status: ✅ PASSED")
                            else:
                                print(f"\n   Status: ❌ FAILED (below threshold)")

                            all_results.append(
                                {
                                    "model": config.model_path,
                                    "tp_size": config.tp_size,
                                    "accuracy": acc,
                                    "threshold": config.accuracy_threshold,
                                    "invalid": invalid,
                                    "latency": latency,
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
                                    "startup_time": startup_time,
                                    "bench_time": None,
                                    "total_time": total_time,
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
                        total_time = time.time() - model_start
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
                                "startup_time": None,
                                "bench_time": None,
                                "total_time": total_time,
                                "passed": False,
                                "skipped": False,
                                "error": error_message,
                            }
                        )

                # Add to summary with runtime
                acc_str = f"{acc:.3f}" if acc is not None else "N/A"
                startup_str = (
                    f"{startup_time:.0f}s" if startup_time is not None else "N/A"
                )
                bench_str = f"{bench_time:.0f}s" if bench_time is not None else "N/A"
                total_str = f"{total_time:.0f}s" if total_time is not None else "N/A"
                summary += f"| {config.model_path} | {config.tp_size} | {acc_str} | {config.accuracy_threshold} | {startup_str} | {bench_str} | {total_str} | {status} |\n"

        # Calculate total test runtime
        total_test_time = time.time() - total_test_start

        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY - Model Group: {self.model_group}")
        print(f"{'='*60}")
        print(summary)
        print(
            f"\n⏱️  Total test runtime: {total_test_time:.1f}s ({total_test_time/60:.1f} min)"
        )

        # Check for failures (exclude skipped models)
        failed_models = [
            r for r in all_results if not r["passed"] and not r.get("skipped", False)
        ]
        skipped_models = [r for r in all_results if r.get("skipped", False)]
        passed_models = [
            r for r in all_results if r["passed"] and not r.get("skipped", False)
        ]

        # Build GitHub summary with results and failure details
        # Note: summary already includes the "### Model Group:" header
        github_summary = f"{summary}\n"
        github_summary += f"\n**Statistics:** ✅ Passed: {len(passed_models)} | ❌ Failed: {len(failed_models)} | ⏭️ Skipped: {len(skipped_models)}\n"
        github_summary += f"\n**Total Runtime:** {total_test_time:.1f}s ({total_test_time/60:.1f} min)\n"

        if failed_models:
            github_summary += "\n#### ❌ Failed Models\n"
            for r in failed_models:
                acc_str = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "N/A"
                github_summary += f"- **{r['model']}**: accuracy={acc_str}, threshold={r['threshold']}"
                if r.get("error"):
                    # Truncate long errors for display
                    error_short = (
                        r["error"][:200] + "..."
                        if len(r["error"]) > 200
                        else r["error"]
                    )
                    github_summary += f"\n  - Error: `{error_short}`"
                github_summary += "\n"

        if skipped_models:
            github_summary += "\n#### ⏭️ Skipped Models\n"
            for r in skipped_models:
                github_summary += (
                    f"- **{r['model']}**: {r.get('error', 'Not available')}\n"
                )

        # Write GitHub step summary
        if is_in_ci():
            write_github_step_summary(github_summary)

        print(f"\n📊 Final Statistics:")
        print(f"   Passed: {len(passed_models)}")
        print(f"   Failed: {len(failed_models)}")
        print(f"   Skipped: {len(skipped_models)}")

        if skipped_models:
            print(f"\n⏭️ Skipped models (not available):")
            for r in skipped_models:
                print(f"   - {r['model']}: {r['error']}")

        if failed_models:
            print(f"\n❌ Failed models:")
            for r in failed_models:
                acc_str = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "N/A"
                print(
                    f"   - {r['model']}: accuracy={acc_str}, threshold={r['threshold']}"
                )
                if r.get("error"):
                    print(f"     Error: {r['error'][:200]}")

            failure_msg = "\n".join(
                [
                    f"- {r['model']}: accuracy={r['accuracy']}, threshold={r['threshold']}, error={r['error']}"
                    for r in failed_models
                ]
            )
            raise AssertionError(f"The following models failed:\n{failure_msg}")


if __name__ == "__main__":
    unittest.main()
