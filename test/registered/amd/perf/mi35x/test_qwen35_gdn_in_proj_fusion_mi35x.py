"""MI35x PR-CI accuracy gate for the Qwen3.5 GDN in_proj_qkvzba merge.

Two parallel TP2 servers on the MXFP4-AttnFP8 checkpoint: in_proj_ba folded into
in_proj_qkvz as one wider GEMM (GPUs 4,5, SGLANG_GDN_FUSE_QKVZBA=1) vs the two
separate projections (GPUs 6,7, the default).

Only the V2 line quantizes in_proj_ba to FP8, so only there do all four shards
resolve to one scheme. A mismatch falls back to separate projections, which would
turn this into baseline vs baseline, so the fused arm's log is checked for it.
"""

import ast
import os
import re
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file, read_jsonl

register_amd_ci(est_time=4800, suite="stage-c-test-large-8-gpu-amd-mi35x")

INVALID = -9999999

# Only AttnFP8 checkpoints quantize linear_attn, which makes the merge eligible.
QWEN35_ATTNFP8_MODEL_PATH = os.environ.get(
    "QWEN35_ATTNFP8_MODEL_PATH",
    "amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2",
)
SERVER_LAUNCH_TIMEOUT = 4800
GSM8K_NUM_QUESTIONS = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
GSM8K_NUM_SHOTS = 5
# Submit all at once; the mamba state pool caps concurrency server-side.
GSM8K_PARALLEL = int(os.environ.get("GSM8K_PARALLEL", str(GSM8K_NUM_QUESTIONS)))
# Reasoning model: leave room for the <think> block.
GSM8K_MAX_NEW_TOKENS = int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "8192"))
ACCURACY_THRESHOLD = 0.92
# GSM8K stderr at 1319 questions is ~0.006.
ACCURACY_DELTA_TOLERANCE = 0.02

# create_qkvzba_proj logs this when the shards disagree.
MERGE_FALLBACK_MARKER = "in_proj_qkvz and in_proj_ba kept separate"

GSM8K_DATA_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)

TP_SIZE = int(os.environ.get("QWEN35_TP_SIZE", "2"))
# First TP slice runs merged, second runs separate.
DEVICE_POOL: List[str] = os.environ.get("QWEN35_DEVICE_POOL", "4,5,6,7").split(",")

COMMON_ARGS: List[str] = [
    "--attention-backend",
    "aiter",
    "--tp",
    str(TP_SIZE),
    "--trust-remote-code",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.85",
    "--cuda-graph-max-bs-decode",
    "128",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--watchdog-timeout",
    "1200",
]

# Per-token a8w8 supplies the pre-shuffled weights the GEMM needs.
COMMON_ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
}

FUSED_VARIANT = "merged-in-proj-qkvzba"
BASELINE_VARIANT = "separate-baseline"


@dataclass
class InProjFusionVariant:
    """A Qwen3.5 GDN in_proj configuration to validate."""

    variant: str
    hip_visible_devices: str
    port_offset: int
    env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class ArmMetrics:
    """One arm's GSM8K result, plus whether the merge engaged."""

    accuracy: float
    invalid: float
    latency: float
    merge_fell_back: bool


def _base_url_with_port_offset(offset: int) -> str:
    host, port = DEFAULT_URL_FOR_TEST.rsplit(":", 1)
    return f"{host}:{int(port) + offset}"


def _device_slice(arm_index: int) -> str:
    start = arm_index * TP_SIZE
    devices = DEVICE_POOL[start : start + TP_SIZE]
    if len(devices) < TP_SIZE:
        raise ValueError(
            f"QWEN35_DEVICE_POOL={','.join(DEVICE_POOL)} needs {2 * TP_SIZE} "
            f"devices for two TP{TP_SIZE} arms"
        )
    return ",".join(devices)


def get_in_proj_fusion_variants() -> List[InProjFusionVariant]:
    return [
        InProjFusionVariant(
            variant=FUSED_VARIANT,
            hip_visible_devices=_device_slice(0),
            port_offset=0,
            env_vars={"SGLANG_GDN_FUSE_QKVZBA": "1"},
        ),
        InProjFusionVariant(
            variant=BASELINE_VARIANT,
            hip_visible_devices=_device_slice(1),
            port_offset=1,
        ),
    ]


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
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
    data_path: str,
    num_questions: int,
) -> Tuple[float, float, float]:
    import sglang as sgl
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

    lines = list(read_jsonl(data_path))
    few_shot_examples = get_few_shot_examples(lines, GSM8K_NUM_SHOTS)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        # Never stop on "Assistant:": the model opens its turn with it, which
        # empties 32% of the answers. EOS ends generation.
        s += sgl.gen(
            "answer",
            max_tokens=GSM8K_MAX_NEW_TOKENS,
            stop=["\n\nQuestion"],
        )

    # Both arms share this process, so pass the backend per call.
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        [{"question": q} for q in questions],
        temperature=0,
        num_threads=GSM8K_PARALLEL,
        backend=RuntimeEndpoint(base_url),
    )
    latency = time.perf_counter() - tic

    preds = [get_answer_value(states[i]["answer"]) for i in range(len(states))]
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    return float(acc), float(invalid), float(latency)


class TestQwen35GdnInProjFusionMI35x(CustomTestCase):
    """Validate the Qwen3.5 GDN merged in_proj accuracy on MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_ATTNFP8_MODEL_PATH
        cls.variants = get_in_proj_fusion_variants()
        # Pre-fetch so the two eval threads don't race the cache write.
        cls.gsm8k_data_path = download_and_cache_file(GSM8K_DATA_URL)

    def _run_variant(self, variant: InProjFusionVariant) -> ArmMetrics:
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = variant.hip_visible_devices
        env.update(COMMON_ENV)
        env.update(variant.env_vars)
        base_url = _base_url_with_port_offset(variant.port_offset)

        log_path = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"qwen35-in-proj-{variant.variant}.log"
        )
        # Grepped below for the fallback notice; stdout and stderr share one file
        # so it is found wherever logging is pointed.
        log_file = open(log_path, "w")
        try:
            process = popen_launch_server(
                self.model,
                base_url,
                timeout=SERVER_LAUNCH_TIMEOUT,
                other_args=list(COMMON_ARGS),
                env=env,
                return_stdout_stderr=(log_file, log_file),
            )
            try:
                requests.get(base_url + "/flush_cache", timeout=10)
                accuracy, invalid, latency = run_gsm8k_benchmark(
                    base_url, self.gsm8k_data_path, GSM8K_NUM_QUESTIONS
                )
            finally:
                kill_process_tree(process.pid)
                # Let the forwarding threads drain before closing the sink.
                time.sleep(2)
        finally:
            log_file.close()

        with open(log_path, "r", errors="replace") as f:
            fell_back = MERGE_FALLBACK_MARKER in f.read()

        metrics = ArmMetrics(
            accuracy=accuracy,
            invalid=invalid,
            latency=latency,
            merge_fell_back=fell_back,
        )
        print(f"[{variant.variant}] {metrics=}")
        return metrics

    def test_qwen35_gdn_in_proj_fusion_accuracy(self):
        results: Dict[str, ArmMetrics] = {}
        with ThreadPoolExecutor(max_workers=len(self.variants)) as executor:
            future_to_variant = {
                executor.submit(self._run_variant, variant): variant
                for variant in self.variants
            }
            for future in as_completed(future_to_variant):
                variant = future_to_variant[future]
                results[variant.variant] = future.result()

        summary = (
            "### Qwen3.5 MXFP4-AttnFP8 GDN in_proj_qkvzba merge GSM8K "
            f"(MI35x, parallel TP{TP_SIZE})\n\n"
        )
        summary += (
            "| Variant | GPUs | Accuracy | Invalid | Latency (s) | Threshold | "
            "Status |\n"
        )
        summary += (
            "| ------- | ---- | -------- | ------- | ----------- | --------- | "
            "------ |\n"
        )
        for variant in self.variants:
            metrics = results[variant.variant]
            passed = metrics.accuracy >= ACCURACY_THRESHOLD
            summary += (
                f"| {variant.variant} | {variant.hip_visible_devices} | "
                f"{metrics.accuracy:.3f} | {metrics.invalid:.3f} | "
                f"{metrics.latency:.2f} | {ACCURACY_THRESHOLD} | "
                f"{'PASS' if passed else 'FAIL'} |\n"
            )

        fused_accuracy = results[FUSED_VARIANT].accuracy
        baseline_accuracy = results[BASELINE_VARIANT].accuracy
        delta = fused_accuracy - baseline_accuracy
        summary += (
            f"\nmerged - separate = {delta:+.4f} "
            f"(tolerance -{ACCURACY_DELTA_TOLERANCE})\n"
        )

        if is_in_ci():
            write_github_step_summary(summary)
        print(summary)

        # Otherwise both arms are the same code path and agree trivially.
        self.assertFalse(
            results[FUSED_VARIANT].merge_fell_back,
            f"{self.model} left in_proj_qkvz and in_proj_ba separate, so the "
            "merged arm never ran; the accuracy comparison is meaningless",
        )

        below_threshold = [
            (name, metrics.accuracy)
            for name, metrics in sorted(results.items())
            if metrics.accuracy < ACCURACY_THRESHOLD
        ]
        self.assertEqual(
            below_threshold,
            [],
            f"Qwen3.5 GDN in_proj merge accuracy below {ACCURACY_THRESHOLD}: "
            f"{below_threshold}",
        )

        self.assertGreaterEqual(
            delta,
            -ACCURACY_DELTA_TOLERANCE,
            f"merged in_proj regressed vs separate baseline: {fused_accuracy:.4f} "
            f"vs {baseline_accuracy:.4f} (delta {delta:+.4f}, tolerance "
            f"-{ACCURACY_DELTA_TOLERANCE})",
        )


if __name__ == "__main__":
    unittest.main()
