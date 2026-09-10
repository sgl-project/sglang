"""MI455x (gfx1250) cookbook configurations, as verified on real hardware.

Every entry here is transcribed from the MI455x bring-up cookbook: the exact
docker image, env vars, server flags and measured GSM8K accuracy from a run on
MI455 A0 (ctheliosr-rck-g02-j19-4). The point of keeping them in one table is
that the cookbook is the only record of *which* configuration produced a given
number -- and those configurations are not interchangeable. The cookbook's
GPT-OSS entry, for instance, runs a single Triton attention backend with CUDA
graphs off and no AITER env at all, which is a different code path from the
prefill-triton/decode-aiter split that `test_gpt_oss_w4a8_mxfp4_eval_mi45x.py`
uses. Reproducing the cookbook's 0.845 means reproducing its flags.

`accuracy` on each config is the measured value, not a threshold; the threshold
is derived from it via `ACCURACY_TOLERANCE` so a config cannot silently pass
with a number far below what the hardware already demonstrated.

The eval harness below is the 5-shot *completion* GSM8K over the full 1319
question test set, matching what the cookbook ran. It is deliberately not
`sglang.test.run_eval(eval_name="gsm8k")`, which prompts through the Chat API
and scores differently -- its numbers are not comparable to the cookbook's.
"""

from __future__ import annotations

import ast
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

INVALID = -9999999

# The cookbook ran the whole GSM8K test split. Anything less is a different
# measurement, so the count is part of the config rather than a free parameter;
# SGLANG_MI45X_NUM_QUESTIONS exists only to make a smoke run cheap.
COOKBOOK_NUM_QUESTIONS = 1319
COOKBOOK_NUM_SHOTS = 5
# `python3 -m sglang.test.few_shot_gsm8k` defaults to --parallel 128, which is
# what the cookbook's latency and throughput figures were produced at. Accuracy
# is unaffected (temperature 0), but those two are not comparable at any other
# value, so this is pinned rather than left to the caller.
COOKBOOK_PARALLEL = 128

# How far below the cookbook's measured accuracy a run may land before it fails.
# 0.02 absolute is wide enough to absorb GSM8K's own sampling noise at n=1319
# (one standard error is roughly 0.01 at these accuracies) but still catches the
# kind of regression that a broken kernel produces.
ACCURACY_TOLERANCE = float(os.environ.get("SGLANG_MI45X_ACCURACY_TOLERANCE", "0.02"))

# Docker image the cookbook numbers were taken under. Recorded for the report
# only -- the tests do not and cannot enforce it, since they run inside whatever
# container the caller started.
COOKBOOK_DOCKER_IMAGE = "lmsysorg/sglang-rocm:v0.5.19-rocm10-mi45x-20260906"
COOKBOOK_MACHINE = "MI455 A0: ctheliosr-rck-g02-j19-4"

# Where the weights live. The cookbook paths are container-side
# (/dockerx/data/models/...); a box that bind-mounts the host root at /dockerx
# sees the same tree twice, so both spellings of each root are listed and one
# config table works from inside or outside a container. SGLANG_MI45X_MODEL_ROOT
# wins if set. Roots are tried in order, so the bulk store comes before the
# smaller /data/models copy.
_MODEL_ROOTS: Tuple[str, ...] = (
    "/dockerx/mnt/raid/models",
    "/mnt/raid/models",
    "/dockerx/data/models",
    "/data/models",
)


def resolve_model_path(local_dirname: str, hf_repo_id: str) -> str:
    """First existing local copy of the weights, else the HF repo id.

    Preferring a local directory keeps a bring-up run from re-downloading
    hundreds of GB, and keeps it working on a box with no HF credentials. The
    HF id is the fallback so the same config still runs on a fresh CI runner.
    """
    roots: List[str] = []
    override = os.environ.get("SGLANG_MI45X_MODEL_ROOT")
    if override:
        roots.append(override)
    roots.extend(_MODEL_ROOTS)

    for root in roots:
        candidate = os.path.join(root, local_dirname)
        if os.path.isdir(candidate):
            return candidate
    return hf_repo_id


@dataclass
class CookbookConfig:
    """One cookbook entry: how to launch it, and what it scored."""

    name: str
    # Directory name under a model root, and the HF id to fall back to.
    local_dirname: str
    hf_repo_id: str
    tp_size: int
    server_args: List[str]
    env_vars: Dict[str, str] = field(default_factory=dict)
    # Env vars to drop before launching. The cookbook runs some entries only
    # after unsetting variables the image exports, so inheriting them silently
    # changes what is being measured.
    unset_env: Tuple[str, ...] = ()
    # Measured on MI455 A0, from the cookbook.
    accuracy: Optional[float] = None
    invalid: Optional[float] = None
    latency_s: Optional[float] = None
    output_throughput: Optional[float] = None
    # Server-launch budget. Large MXFP4 checkpoints take a long time to load on
    # a cold page cache, well past the 600s default.
    launch_timeout: int = 1800
    notes: str = ""

    @property
    def model_path(self) -> str:
        return resolve_model_path(self.local_dirname, self.hf_repo_id)

    @property
    def accuracy_threshold(self) -> Optional[float]:
        if self.accuracy is None:
            return None
        return round(self.accuracy - ACCURACY_TOLERANCE, 4)

    def full_args(self) -> List[str]:
        """Server args with TP appended, mirroring the cookbook command line."""
        return list(self.server_args) + ["--tp", str(self.tp_size)]

    def full_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        for key in self.unset_env:
            env.pop(key, None)
        env.update(self.env_vars)
        return env


# Shared by every cookbook command line. HIP_FORCE_DEV_KERNARG=0 appears on all
# six of them, and ENABLE_CK=0 on all but one -- gfx1250 has no CK kernels yet.
_COMMON_ENV = {
    "HIP_FORCE_DEV_KERNARG": "0",
    "ENABLE_CK": "0",
    # Not from the cookbook: coredumps on these images are multi-GB and filled
    # the disk on this box. The DSR1 entry already disables them explicitly.
    "HSA_COREDUMP_PATTERN": "/dev/null",
}


# --- 1. GPT-OSS-120B ---------------------------------------------------------
GPT_OSS_120B = CookbookConfig(
    name="gpt-oss-120b-w-mxfp4-a-fp8",
    local_dirname="gpt-oss-120b-w-mxfp4-a-fp8",
    hf_repo_id="amd/gpt-oss-120b-w-mxfp4-a-fp8",
    tp_size=1,
    server_args=[
        "--trust-remote-code",
        "--max-running-requests",
        "128",
        "--mem-fraction-static",
        "0.9",
        # One backend for both phases, and CUDA graphs off. This is the whole
        # difference from the other registered gpt-oss mi45x test; do not
        # "modernise" it to the prefill/decode split without re-measuring.
        "--attention-backend",
        "triton",
        "--disable-radix-cache",
        "--disable-cuda-graph",
    ],
    env_vars=dict(_COMMON_ENV),
    accuracy=0.845,
    invalid=0.010,
    latency_s=332.599,
    output_throughput=1316.405,
    notes="No AITER env; single triton attention backend, CUDA graphs disabled.",
)


# --- 2. DeepSeek-R1-0528-MXFP4 ----------------------------------------------
DEEPSEEK_R1_0528_MXFP4 = CookbookConfig(
    name="DeepSeek-R1-0528-MXFP4",
    local_dirname="DeepSeek-R1-0528-MXFP4",
    hf_repo_id="amd/DeepSeek-R1-0528-MXFP4",
    tp_size=1,
    server_args=[
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.90",
        "--chunked-prefill-size",
        "16384",
        "--attention-backend",
        "triton",
        "--max-running-requests",
        "32",
        "--kv-cache-dtype",
        "auto",
        "--page-size",
        "64",
    ],
    env_vars={
        **_COMMON_ENV,
        "HSA_ENABLE_COREDUMP": "0",
        "AMD_COREDUMP": "0",
        "SGLANG_USE_AITER": "1",
        "AITER_FORCE_A8W4": "1",
        "AITER_GROUPED_FORCE_SPLIT_K1": "1",
        "SGLANG_MOE_SHUFFLE_GFX1250": "1",
        "ROCM_QUICK_REDUCE_QUANTIZATION": "NONE",
        "SGLANG_AITER_FP8_PREFILL_ATTN": "0",
        "SGLANG_AITER_MLA_PERSIST": "0",
        "SGLANG_INT4_WEIGHT": "0",
        "SGLANG_MOE_PADDING": "1",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "SGLANG_ROCM_FUSED_DECODE_MLA": "0",
        "SGLANG_USE_ROCM700A": "1",
        "AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD": "16",
    },
    accuracy=0.948,
    invalid=0.000,
    latency_s=471.640,
    output_throughput=279.599,
    notes="Cookbook used --tensor-parallel-size 1 (alias of --tp).",
)


# --- 3. DeepSeek-V4-Flash ----------------------------------------------------
_DSV4_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "dsv4",
    "--page-size",
    "256",
    "--mem-fraction-static",
    "0.8",
    "--swa-full-tokens-ratio",
    "0.15",
    "--disable-shared-experts-fusion",
    "--tool-call-parser",
    "deepseekv4",
    "--reasoning-parser",
    "deepseek-v4",
    "--chunked-prefill-size",
    "2048",
    "--max-prefill-tokens",
    "2048",
    # Cookbook wrote --cuda-graph-max-bs; that alias was dropped after the args
    # were split into decode/prefill, and argparse now rejects it as ambiguous.
    "--cuda-graph-max-bs-decode",
    "256",
    "--max-running-requests",
    "256",
    "--disable-radix-cache",
    "--kv-cache-dtype",
    "fp8_e4m3",
]

_DSV4_ENV = {
    **_COMMON_ENV,
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "AITER_FORCE_A8W4": "1",
    "SGLANG_USE_AITER_MOE_GU_ITLV": "1",
    "AITER_GROUPED_FORCE_SPLIT_K1": "1",
    "SGLANG_OPT_FUSE_MHC_POST_PRE": "0",
    "SGLANG_USE_AITER": "1",
}

DEEPSEEK_V4_FLASH_TP1 = CookbookConfig(
    name="DeepSeek-V4-Flash (TP=1)",
    local_dirname="DeepSeek-V4-Flash",
    hf_repo_id="deepseek-ai/DeepSeek-V4-Flash",
    tp_size=1,
    server_args=list(_DSV4_ARGS),
    env_vars=dict(_DSV4_ENV),
    accuracy=0.932,
    invalid=0.000,
    latency_s=124.409,
    output_throughput=950.984,
)

DEEPSEEK_V4_FLASH_TP2 = CookbookConfig(
    name="DeepSeek-V4-Flash (TP=2)",
    local_dirname="DeepSeek-V4-Flash",
    hf_repo_id="deepseek-ai/DeepSeek-V4-Flash",
    tp_size=2,
    server_args=list(_DSV4_ARGS) + ["--disable-custom-all-reduce"],
    env_vars={
        **_DSV4_ENV,
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_MNNVL_ENABLE": "1",
        "HSA_HOTSWAP_DISABLE": "1",
    },
    # These ROCm images export NCCL_MIN_NCHANNELS=112, which is far past what a
    # 2-GPU all-reduce wants and hangs CUDA-graph capture.
    unset_env=("NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"),
    accuracy=0.926,
    invalid=0.000,
    latency_s=177.039,
    output_throughput=668.197,
    notes=(
        "Cookbook measured this on heliosr-1b114-a04-2 rather than the A0 box, "
        "so a delta here is not automatically a regression."
    ),
)


# --- 6. Qwen3.5-397B-A17B-MXFP4 ---------------------------------------------
QWEN35_397B_MXFP4 = CookbookConfig(
    name="Qwen3.5-397B-A17B-MXFP4",
    local_dirname="Qwen3.5-397B-A17B-MXFP4",
    hf_repo_id="amd/Qwen3.5-397B-A17B-MXFP4",
    tp_size=1,
    server_args=[
        "--trust-remote-code",
        "--max-running-requests",
        "128",
        "--mem-fraction-static",
        "0.9",
        "--disable-radix-cache",
        "--prefill-attention-backend",
        "triton",
        "--decode-attention-backend",
        "aiter",
        "--page-size",
        "16",
        "--prefill-max-requests",
        "1",
    ],
    env_vars={
        **_COMMON_ENV,
        "TRITON_HIP_USE_ASYNC_COPY": "0",
        "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    },
    accuracy=0.970,
    invalid=0.005,
    latency_s=216.519,
    output_throughput=258.800,
    notes=(
        "Measured by Marvin on rocm/sgl-dev:v0.5.18-rocm10-mi45x-dev-20260828, "
        "not the image the other entries used."
    ),
)


# Diffusion entries 4 (Wan2.2-T2V-A14B) and 5 (FLUX.2-dev) are in the cookbook
# but carry no accuracy metric -- they are `sglang generate` smoke runs judged
# by eye. They are intentionally absent here rather than represented by a
# config that nothing can gate on.


ALL_CONFIGS: Tuple[CookbookConfig, ...] = (
    GPT_OSS_120B,
    DEEPSEEK_R1_0528_MXFP4,
    DEEPSEEK_V4_FLASH_TP1,
    DEEPSEEK_V4_FLASH_TP2,
    QWEN35_397B_MXFP4,
)


# ------------------------------- eval harness -------------------------------


def _get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def _get_few_shot_examples(lines, k):
    return "".join(_get_one_example(lines, i, True) + "\n\n" for i in range(k))


def _get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def num_questions() -> int:
    """Question count for this run; the cookbook's 1319 unless overridden."""
    return int(
        os.environ.get("SGLANG_MI45X_NUM_QUESTIONS", str(COOKBOOK_NUM_QUESTIONS))
    )


@dataclass
class EvalOutcome:
    accuracy: float
    invalid: float
    latency: float
    output_throughput: Optional[float] = None


def run_gsm8k_completion(
    base_url: str,
    num_questions_: Optional[int] = None,
    num_shots: int = COOKBOOK_NUM_SHOTS,
    parallel: int = COOKBOOK_PARALLEL,
) -> EvalOutcome:
    """5-shot completion GSM8K, the same harness the cookbook numbers came from."""
    import numpy as np

    import sglang as sgl
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
    from sglang.utils import download_and_cache_file, read_jsonl

    count = num_questions_ if num_questions_ is not None else num_questions()

    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    lines = list(read_jsonl(download_and_cache_file(url)))

    few_shot_examples = _get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:count])):
        questions.append(_get_one_example(lines, i, False))
        labels.append(_get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"]
        )

    sgl.set_default_backend(RuntimeEndpoint(base_url))

    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments, temperature=0, num_threads=parallel, progress_bar=True
    )
    latency = time.perf_counter() - tic

    preds = [_get_answer_value(states[i]["answer"]) for i in range(len(states))]
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    invalid = float(np.mean(np.array(preds) == INVALID))

    # Same definition the cookbook reports: generated tokens over wall clock.
    output_tokens = sum(
        state.get_meta_info("answer")["completion_tokens"] for state in states
    )
    throughput = output_tokens / latency if latency > 0 else None

    return EvalOutcome(
        accuracy=acc,
        invalid=invalid,
        latency=latency,
        output_throughput=throughput,
    )


# ------------------------------- test driver --------------------------------


def run_cookbook_case(test_case, config: CookbookConfig) -> None:
    """Launch `config`, run the GSM8K eval, and gate on the cookbook accuracy.

    Shared by the per-model registered tests so that all of them report
    identically: one `Testing:` line, one `accuracy=... threshold=...` line, and
    one step-summary table row. The local collector
    (scripts/ci/amd/local/collect_results.py) parses exactly those.
    """
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import (
        DEFAULT_URL_FOR_TEST,
        is_in_ci,
        popen_launch_server,
        write_github_step_summary,
    )

    base_url = DEFAULT_URL_FOR_TEST
    model_path = config.model_path
    count = num_questions()

    print(f"\n{'=' * 60}")
    print(f"Testing: {config.name}")
    print(f"{'=' * 60}")
    print(f"  model path     : {model_path}")
    print(f"  tp             : {config.tp_size}")
    print(f"  questions      : {count}", flush=True)
    if model_path == config.hf_repo_id:
        print(
            f"  note           : no local copy found under {_MODEL_ROOTS}; "
            f"will download from HF",
            flush=True,
        )
    if count != COOKBOOK_NUM_QUESTIONS:
        # A short run is a smoke test, not a reproduction: at n=32 the accuracy
        # estimate is worth roughly +/-0.08, so the gate is meaningless.
        print(
            f"  WARNING        : {count} questions instead of the cookbook's "
            f"{COOKBOOK_NUM_QUESTIONS}; accuracy is not comparable",
            flush=True,
        )

    summary = f"### {config.name} (MI455x cookbook)\n\n"
    summary += "| Model | TP | Accuracy | Cookbook | Threshold | Delta | Status |\n"
    summary += "| ----- | -- | -------- | -------- | --------- | ----- | ------ |\n"

    threshold = config.accuracy_threshold
    error: Optional[str] = None
    outcome: Optional[EvalOutcome] = None
    process = None

    try:
        process = popen_launch_server(
            model=model_path,
            base_url=base_url,
            timeout=config.launch_timeout,
            other_args=config.full_args(),
            env=config.full_env(),
        )
        outcome = run_gsm8k_completion(base_url, num_questions_=count)
    except Exception as exc:  # noqa: BLE001 - reported, then re-raised below
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            kill_process_tree(process.pid)

    if outcome is None:
        status = "❌ ERROR"
        cookbook = "N/A" if config.accuracy is None else f"{config.accuracy:.3f}"
        summary += (
            f"| {config.name} | {config.tp_size} | N/A | {cookbook} "
            f"| {threshold} | N/A | {status} |\n"
        )
        if is_in_ci():
            write_github_step_summary(summary)
        raise AssertionError(f"{config.name} did not produce a result: {error}")

    passed = threshold is None or outcome.accuracy >= threshold
    status = "✅ PASS" if passed else "❌ FAIL"
    delta = (
        "N/A"
        if config.accuracy is None
        else f"{outcome.accuracy - config.accuracy:+.3f}"
    )

    print(f"  accuracy={outcome.accuracy:.3f} threshold={threshold} {status}")
    print(f"  invalid={outcome.invalid:.3f}")
    print(f"  latency={outcome.latency:.3f}s")
    if outcome.output_throughput is not None:
        print(f"  output throughput={outcome.output_throughput:.3f} token/s")
    if config.accuracy is not None:
        print(f"  cookbook accuracy={config.accuracy:.3f} (delta {delta})")
    print(
        "  cookbook reference: "
        f"latency={config.latency_s}s throughput={config.output_throughput} token/s",
        flush=True,
    )

    summary += (
        f"| {config.name} | {config.tp_size} | {outcome.accuracy:.3f} "
        f"| {'N/A' if config.accuracy is None else f'{config.accuracy:.3f}'} "
        f"| {threshold} | {delta} | {status} |\n"
    )
    summary += (
        f"\nInvalid {outcome.invalid:.3f} | latency {outcome.latency:.1f}s "
        f"(cookbook {config.latency_s}s)"
    )
    if outcome.output_throughput is not None:
        summary += (
            f" | output {outcome.output_throughput:.1f} tok/s "
            f"(cookbook {config.output_throughput} tok/s)"
        )
    summary += f" | {count} questions\n\n"

    if is_in_ci():
        write_github_step_summary(summary)

    if threshold is None:
        # No cookbook baseline to gate on; the run is a report, not a gate.
        return

    test_case.assertGreaterEqual(
        outcome.accuracy,
        threshold,
        f"{config.name}: accuracy {outcome.accuracy:.3f} below threshold "
        f"{threshold} (cookbook measured {config.accuracy})",
    )
