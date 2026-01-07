import ast
import json
import os
import re
import statistics
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    is_in_ci,
    popen_launch_server,
)
from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999


def _get_one_example(lines, i: int, include_answer: bool) -> str:
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def _get_few_shot_examples(lines, k: int) -> str:
    ret = ""
    for i in range(k):
        ret += _get_one_example(lines, i, True) + "\n\n"
    return ret


def _get_answer_value(answer_str: str) -> int:
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def _send_generate(base_url: str, prompt: str, *, max_new_tokens: int) -> dict:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_new_tokens": max_new_tokens,
                # Avoid extra decoding after an answer.
                "stop": ["Question", "Assistant:", "<|separator|>"],
            },
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def _run_generate_batch(
    base_url: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    parallel: int,
) -> tuple[float, list[dict]]:
    start = time.perf_counter()
    outputs: list[dict] = [None for _ in range(len(prompts))]  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(_send_generate, base_url, prompt, max_new_tokens=max_new_tokens): i
            for i, prompt in enumerate(prompts)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            outputs[idx] = fut.result()
    latency = time.perf_counter() - start
    return latency, outputs


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p90": None}
    values_sorted = sorted(values)
    p50 = values_sorted[int(0.50 * (len(values_sorted) - 1))]
    p90 = values_sorted[int(0.90 * (len(values_sorted) - 1))]
    return {
        "mean": float(statistics.mean(values_sorted)),
        "p50": float(p50),
        "p90": float(p90),
    }


class TestQwen3DFlashGSM8KBench(CustomTestCase):
    def test_qwen3_dflash_gsm8k_speedup_and_acceptance(self):
        if is_in_ci():
            self.skipTest("Manual benchmark; skipped in CI.")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this manual DFlash benchmark.")

        target_model = os.getenv("SGLANG_DFLASH_TARGET_MODEL", "Qwen/Qwen3-8B")
        draft_model_path = os.getenv(
            "SGLANG_DFLASH_DRAFT_MODEL_PATH", "/tmp/Qwen3-8B-DFlash-bf16"
        )
        if not os.path.isdir(draft_model_path):
            self.skipTest(
                f"Draft model folder not found: {draft_model_path}. "
                "Set SGLANG_DFLASH_DRAFT_MODEL_PATH to run this benchmark."
            )

        attention_backend = os.getenv("SGLANG_DFLASH_ATTENTION_BACKEND", "flashinfer")
        max_new_tokens = int(os.getenv("SGLANG_DFLASH_MAX_NEW_TOKENS", "2048"))
        parallel = int(os.getenv("SGLANG_DFLASH_PARALLEL", "1"))
        num_questions = int(os.getenv("SGLANG_DFLASH_NUM_QUESTIONS", "100"))
        num_shots = int(os.getenv("SGLANG_DFLASH_NUM_SHOTS", "1"))
        disable_radix_cache = os.getenv("SGLANG_DFLASH_DISABLE_RADIX_CACHE", "1") != "0"

        # Read GSM8K data (download if absent).
        data_path = os.getenv("SGLANG_DFLASH_GSM8K_PATH", "test.jsonl")
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        if not os.path.isfile(data_path):
            data_path = download_and_cache_file(url)
        lines = list(read_jsonl(data_path))

        few_shot = _get_few_shot_examples(lines, num_shots)
        prompts: list[str] = []
        labels: list[int] = []
        for i in range(len(lines[:num_questions])):
            prompts.append(few_shot + _get_one_example(lines, i, False))
            labels.append(_get_answer_value(lines[i]["answer"]))
        self.assertTrue(all(l != INVALID for l in labels), "Invalid labels in GSM8K data")

        common_server_args = ["--attention-backend", attention_backend]
        if disable_radix_cache:
            common_server_args.append("--disable-radix-cache")

        baseline_port = find_available_port(20000)
        dflash_port = find_available_port(baseline_port + 1)
        baseline_url = f"http://127.0.0.1:{baseline_port}"
        dflash_url = f"http://127.0.0.1:{dflash_port}"

        # 1) Target-only baseline.
        baseline_proc = popen_launch_server(
            target_model,
            baseline_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=common_server_args,
        )
        try:
            _send_generate(baseline_url, "Hello", max_new_tokens=8)  # warmup
            baseline_latency, baseline_outputs = _run_generate_batch(
                baseline_url,
                prompts,
                max_new_tokens=max_new_tokens,
                parallel=parallel,
            )
        finally:
            kill_process_tree(baseline_proc.pid)
            try:
                baseline_proc.wait(timeout=30)
            except Exception:
                pass

        # 2) DFLASH speculative decoding.
        dflash_proc = popen_launch_server(
            target_model,
            dflash_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *common_server_args,
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                draft_model_path,
            ],
        )
        try:
            _send_generate(dflash_url, "Hello", max_new_tokens=8)  # warmup
            dflash_latency, dflash_outputs = _run_generate_batch(
                dflash_url,
                prompts,
                max_new_tokens=max_new_tokens,
                parallel=parallel,
            )
        finally:
            kill_process_tree(dflash_proc.pid)
            try:
                dflash_proc.wait(timeout=30)
            except Exception:
                pass

        def _collect_common_metrics(outputs: list[dict]) -> tuple[int, list[int]]:
            completion_tokens = []
            preds = []
            total_completion_tokens = 0
            for out in outputs:
                meta = out.get("meta_info", {})
                total_completion_tokens += int(meta.get("completion_tokens", 0))
                completion_tokens.append(int(meta.get("completion_tokens", 0)))
                preds.append(_get_answer_value(out.get("text", "")))
            return total_completion_tokens, preds

        baseline_total_tokens, baseline_preds = _collect_common_metrics(baseline_outputs)
        dflash_total_tokens, dflash_preds = _collect_common_metrics(dflash_outputs)

        baseline_throughput = baseline_total_tokens / max(baseline_latency, 1e-6)
        dflash_throughput = dflash_total_tokens / max(dflash_latency, 1e-6)
        speedup = dflash_throughput / max(baseline_throughput, 1e-6)

        baseline_acc = sum(
            int(p == l) for p, l in zip(baseline_preds, labels, strict=True)
        ) / len(labels)
        dflash_acc = sum(
            int(p == l) for p, l in zip(dflash_preds, labels, strict=True)
        ) / len(labels)

        spec_accept_lengths: list[float] = []
        spec_accept_rates: list[float] = []
        spec_verify_cts: list[int] = []
        for out in dflash_outputs:
            meta = out.get("meta_info", {})
            if "spec_verify_ct" in meta:
                spec_verify_cts.append(int(meta["spec_verify_ct"]))
            if "spec_accept_length" in meta:
                spec_accept_lengths.append(float(meta["spec_accept_length"]))
            if "spec_accept_rate" in meta:
                spec_accept_rates.append(float(meta["spec_accept_rate"]))

        # Basic sanity checks that DFLASH actually ran.
        self.assertTrue(spec_verify_cts, "Missing spec_verify_ct in DFLASH responses.")
        self.assertGreater(sum(spec_verify_cts), 0, "DFLASH did not run verify steps.")

        report = {
            "settings": {
                "target_model": target_model,
                "draft_model_path": draft_model_path,
                "attention_backend": attention_backend,
                "max_new_tokens": max_new_tokens,
                "parallel": parallel,
                "num_questions": num_questions,
                "num_shots": num_shots,
                "disable_radix_cache": disable_radix_cache,
            },
            "baseline": {
                "latency_s": round(baseline_latency, 3),
                "completion_tokens": baseline_total_tokens,
                "throughput_tok_s": round(baseline_throughput, 3),
                "accuracy": round(baseline_acc, 3),
            },
            "dflash": {
                "latency_s": round(dflash_latency, 3),
                "completion_tokens": dflash_total_tokens,
                "throughput_tok_s": round(dflash_throughput, 3),
                "accuracy": round(dflash_acc, 3),
                "spec_accept_length": _summarize(spec_accept_lengths),
                "spec_accept_rate": _summarize(spec_accept_rates),
                "spec_verify_ct_mean": float(statistics.mean(spec_verify_cts)),
            },
            "speedup": round(speedup, 3),
        }
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    unittest.main()
