"""MI35x PR-CI accuracy gate for the Qwen3.5 dense-FP8 fused down_proj path.

Two parallel TP4 servers on the MXFP4 checkpoint (8-GPU stage-c): --enable-dense-fp8
(GPUs 0-3, shared_expert.down_proj -> online w8a8 FP8 with a fused SiluAndMul+quant
kernel) vs bf16 baseline (GPUs 4-7). The fused path must hold GSM8K accuracy against
the baseline. The feature is default-off, so this is the only CI coverage that
actually executes it.
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

# Promotion only reaches layers the checkpoint excludes: this one leaves
# shared_expert.down_proj bf16, while AttnFP8-V2 quantizes it (nothing to promote).
QWEN35_MXFP4_MODEL_PATH = os.environ.get(
    "QWEN35_MXFP4_MODEL_PATH",
    "amd/Qwen3.5-397B-A17B-MXFP4",
)
SERVER_LAUNCH_TIMEOUT = 4800
GSM8K_NUM_QUESTIONS = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
GSM8K_NUM_SHOTS = 5
# Submit everything at once; concurrency is capped server-side anyway.
GSM8K_PARALLEL = int(os.environ.get("GSM8K_PARALLEL", str(GSM8K_NUM_QUESTIONS)))
# Reasoning model: the <think> block needs room before the answer line.
GSM8K_MAX_NEW_TOKENS = int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "8192"))
# bf16 baseline measures 0.973 here, so 0.92 leaves headroom.
ACCURACY_THRESHOLD = 0.92
# Promoting down_proj to FP8 must be accuracy-neutral; GSM8K stderr is ~0.006.
ACCURACY_DELTA_TOLERANCE = 0.02

GSM8K_DATA_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)

# TP=4 matches the AMD model-card recipe. Avoid TP=8: shared_expert.down_proj is
# row-parallel with a 1024 contraction dim, so TP=8 shards it to K=128. aiter's fp8
# bpreshuffle GEMM has no tuned config for that shape at decode-graph batch sizes,
# and its untuned CK fallback rejects K<=192, so cuda-graph capture crashes. That is
# an aiter FP8-GEMM tuning gap, not a dense-FP8 issue.
TP_SIZE = int(os.environ.get("QWEN35_TP_SIZE", "4"))
# Two arms run concurrently: first slice is fused, second is baseline.
DEVICE_POOL: List[str] = os.environ.get("QWEN35_DEVICE_POOL", "0,1,2,3,4,5,6,7").split(
    ","
)

COMMON_ARGS: List[str] = [
    "--attention-backend",
    "aiter",
    "--tp",
    str(TP_SIZE),
    "--trust-remote-code",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.85",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--watchdog-timeout",
    "1200",
]

# dense-FP8 is AITER-only and requires the per-token FP8 activation path.
COMMON_ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
}

FUSED_VARIANT = "dense-fp8-fused"
BASELINE_VARIANT = "bf16-baseline"


@dataclass
class DenseFp8Variant:
    """A Qwen3.5 dense-FP8 configuration to validate."""

    variant: str
    hip_visible_devices: str
    port_offset: int
    extra_args: List[str] = field(default_factory=list)


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


def get_dense_fp8_variants() -> List[DenseFp8Variant]:
    return [
        DenseFp8Variant(
            variant=FUSED_VARIANT,
            hip_visible_devices=_device_slice(0),
            port_offset=0,
            extra_args=["--enable-dense-fp8"],
        ),
        DenseFp8Variant(
            variant=BASELINE_VARIANT,
            hip_visible_devices=_device_slice(1),
            port_offset=1,
            extra_args=[],
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
        # Stop only at the next few-shot boundary. Never "Assistant:": the model
        # opens its turn with it, emptying 32% of answers. EOS ends generation.
        s += sgl.gen(
            "answer",
            max_tokens=GSM8K_MAX_NEW_TOKENS,
            stop=["\n\nQuestion"],
        )

    # Both arms share this process, so pass the backend per call instead of
    # through the global set_default_backend().
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


class TestQwen35DenseFp8MI35x(CustomTestCase):
    """Validate Qwen3.5 dense-FP8 fused down_proj accuracy on MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_MXFP4_MODEL_PATH
        cls.variants = get_dense_fp8_variants()
        # Pre-fetch once so the two parallel eval threads don't race the cache write.
        cls.gsm8k_data_path = download_and_cache_file(GSM8K_DATA_URL)

    def _run_variant(self, variant: DenseFp8Variant) -> Dict[str, float]:
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = variant.hip_visible_devices
        env.update(COMMON_ENV)
        base_url = _base_url_with_port_offset(variant.port_offset)

        process = popen_launch_server(
            self.model,
            base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=list(COMMON_ARGS) + variant.extra_args,
            env=env,
        )
        try:
            requests.get(base_url + "/flush_cache", timeout=10)
            accuracy, invalid, latency = run_gsm8k_benchmark(
                base_url, self.gsm8k_data_path, GSM8K_NUM_QUESTIONS
            )
            metrics = {
                "accuracy": accuracy,
                "invalid": invalid,
                "latency": latency,
            }
            print(f"[{variant.variant}] {metrics=}")
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_qwen35_dense_fp8_accuracy(self):
        results: Dict[str, Dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=len(self.variants)) as executor:
            future_to_variant = {
                executor.submit(self._run_variant, variant): variant
                for variant in self.variants
            }
            for future in as_completed(future_to_variant):
                variant = future_to_variant[future]
                results[variant.variant] = future.result()

        summary = "### Qwen3.5 MXFP4 --enable-dense-fp8 GSM8K (MI35x, parallel TP4)\n\n"
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
            passed = metrics["accuracy"] >= ACCURACY_THRESHOLD
            summary += (
                f"| {variant.variant} | {variant.hip_visible_devices} | "
                f"{metrics['accuracy']:.3f} | {metrics['invalid']:.3f} | "
                f"{metrics['latency']:.2f} | {ACCURACY_THRESHOLD} | "
                f"{'PASS' if passed else 'FAIL'} |\n"
            )

        fused_accuracy = results[FUSED_VARIANT]["accuracy"]
        baseline_accuracy = results[BASELINE_VARIANT]["accuracy"]
        delta = fused_accuracy - baseline_accuracy
        summary += (
            f"\nfused - baseline = {delta:+.4f} "
            f"(tolerance -{ACCURACY_DELTA_TOLERANCE})\n"
        )

        if is_in_ci():
            write_github_step_summary(summary)
        print(summary)

        below_threshold = [
            (name, metrics["accuracy"])
            for name, metrics in sorted(results.items())
            if metrics["accuracy"] < ACCURACY_THRESHOLD
        ]
        self.assertEqual(
            below_threshold,
            [],
            f"Qwen3.5 GSM8K accuracy below {ACCURACY_THRESHOLD}: {below_threshold}",
        )

        self.assertGreaterEqual(
            delta,
            -ACCURACY_DELTA_TOLERANCE,
            f"dense-FP8 regressed vs bf16 baseline: {fused_accuracy:.4f} vs "
            f"{baseline_accuracy:.4f} (delta {delta:+.4f}, tolerance "
            f"-{ACCURACY_DELTA_TOLERANCE})",
        )


if __name__ == "__main__":
    unittest.main()
