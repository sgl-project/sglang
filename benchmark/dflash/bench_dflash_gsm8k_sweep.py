"""DFLASH vs baseline GSM8K sweep.

This is a *benchmark script* (not a CI test): it can take a long time because it
launches servers for multiple (attention_backend, tp_size) configs and runs a
GSM8K workload for each (concurrency, num_questions) setting.

Example usage:
  ./venv/bin/python benchmark/gsm8k/bench_dflash_gsm8k_sweep.py --output-md dflash_gsm8k_sweep.md
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import requests
import torch
from transformers import AutoTokenizer

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999


def _is_blackwell() -> bool:
    # Prefer explicit env var, but also infer from compute capability (SM100+).
    if envs.IS_BLACKWELL.get():
        return True
    return get_device_sm() >= 100


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


def _maybe_download_gsm8k(data_path: str) -> str:
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    if os.path.isfile(data_path):
        return data_path
    return download_and_cache_file(url)


def _flush_cache(base_url: str) -> None:
    resp = requests.get(base_url + "/flush_cache", timeout=60)
    resp.raise_for_status()


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> dict:
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    return resp.json()


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    accuracy: Optional[float]
    invalid_rate: Optional[float]
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int


def _run_gsm8k_requests(
    base_url: str,
    *,
    prompts: list[str],
    labels: Optional[list[int]],
    max_new_tokens: int,
    concurrency: int,
    stop: list[str],
    timeout_s: int,
    expect_dflash: bool,
) -> BenchMetrics:
    if labels is not None and len(labels) != len(prompts):
        raise ValueError("labels length must match prompts length")

    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []
    correct = 0
    invalid = 0

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        futures = {
            pool.submit(
                _send_generate,
                base_url,
                prompt,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
            ): i
            for i, prompt in enumerate(prompts)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            out = fut.result()
            meta = out.get("meta_info", {}) or {}
            total_tokens += int(meta.get("completion_tokens", 0))
            spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
            if "spec_accept_length" in meta:
                try:
                    spec_accept_lengths.append(float(meta["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass

            if labels is not None:
                pred = _get_answer_value(out.get("text", ""))
                if pred == INVALID:
                    invalid += 1
                if pred == labels[i]:
                    correct += 1

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_dflash and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "DFLASH sanity check failed: did not observe any `spec_verify_ct` in responses "
            "(DFLASH may not have been enabled)."
        )

    spec_accept_length = (
        float(statistics.mean(spec_accept_lengths)) if spec_accept_lengths else None
    )

    if labels is None:
        acc = None
        invalid_rate = None
    else:
        acc = correct / max(len(prompts), 1)
        invalid_rate = invalid / max(len(prompts), 1)

    return BenchMetrics(
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        accuracy=acc,
        invalid_rate=invalid_rate,
        spec_accept_length=spec_accept_length,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
    )


def _format_table(
    *,
    tp_sizes: list[int],
    concurrencies: list[int],
    values: dict[tuple[int, int], Optional[float]],
    float_fmt: str,
) -> str:
    header = ["tp\\conc"] + [str(c) for c in concurrencies]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for tp in tp_sizes:
        row = [str(tp)]
        for c in concurrencies:
            v = values.get((tp, c), None)
            row.append("N/A" if v is None else format(v, float_fmt))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-md", type=str, default="dflash_gsm8k_sweep.md")
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["fewshot_qa", "chat"],
        default="chat",
        help="Prompting style: 'chat' matches the DFlash HF demo prompt.",
    )
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument(
        "--tp-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list, filtered by visible CUDA devices.",
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of client concurrency levels.",
    )
    parser.add_argument(
        "--questions-per-concurrency-base",
        type=int,
        default=128,
        help="num_questions = base * concurrency (default matches the sweep plan).",
    )
    parser.add_argument(
        "--max-questions-per-config",
        type=int,
        default=1024,
        help="Cap num_questions per (tp, concurrency) run (default: 1024).",
    )
    parser.add_argument(
        "--attention-backends",
        type=str,
        default="flashinfer,fa3",
        help="Comma-separated list. Will auto-skip fa3 on Blackwell/SM<90.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")

    visible_gpus = int(torch.cuda.device_count())
    tp_sizes = [int(x) for x in args.tp_sizes.split(",") if x.strip()]
    tp_sizes = [tp for tp in tp_sizes if tp >= 1 and tp <= visible_gpus]
    if not tp_sizes:
        raise RuntimeError(
            f"No tp sizes are runnable with visible_gpus={visible_gpus}. "
            "Set CUDA_VISIBLE_DEVICES accordingly."
        )

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    num_questions_by_conc = {
        c: min(int(args.questions_per_concurrency_base) * int(c), int(args.max_questions_per_config))
        for c in concurrencies
    }
    max_questions = max(num_questions_by_conc.values())

    attention_backends = [s.strip() for s in args.attention_backends.split(",") if s.strip()]
    is_blackwell = _is_blackwell()
    device_sm = get_device_sm()
    if is_blackwell:
        attention_backends = [b for b in attention_backends if b == "flashinfer"]
    if device_sm < 90:
        attention_backends = [b for b in attention_backends if b != "fa3"]
    attention_backends = attention_backends or ["flashinfer"]

    data_path = _maybe_download_gsm8k(args.data_path)
    lines = list(read_jsonl(data_path))
    if len(lines) < max_questions:
        raise RuntimeError(
            f"GSM8K file only has {len(lines)} lines, but need {max_questions}."
        )

    tokenizer = None
    if args.prompt_style == "chat":
        tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    few_shot = (
        _get_few_shot_examples(lines, int(args.num_shots))
        if args.prompt_style == "fewshot_qa"
        else ""
    )

    prompts: list[str] = []
    labels: list[int] = []
    for i in range(max_questions):
        if args.prompt_style == "fewshot_qa":
            prompts.append(few_shot + _get_one_example(lines, i, False))
        else:
            assert tokenizer is not None
            user_content = (
                lines[i]["question"]
                + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            )
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        labels.append(_get_answer_value(lines[i]["answer"]))
    if not all(l != INVALID for l in labels):
        raise RuntimeError("Invalid labels in GSM8K data.")

    default_stop = (
        ["Question", "Assistant:", "<|separator|>"] if args.prompt_style == "fewshot_qa" else []
    )

    # Results indexed by (backend, tp, concurrency) for baseline + dflash.
    baseline_toks: dict[tuple[str, int, int], Optional[float]] = {}
    dflash_toks: dict[tuple[str, int, int], Optional[float]] = {}
    dflash_accept_len: dict[tuple[str, int, int], Optional[float]] = {}
    baseline_acc: dict[tuple[str, int, int], Optional[float]] = {}
    dflash_acc: dict[tuple[str, int, int], Optional[float]] = {}

    for backend in attention_backends:
        for tp in tp_sizes:
            print(f"\n=== backend={backend} tp={tp} (baseline) ===")
            baseline_port = find_available_port(20000)
            baseline_url = f"http://127.0.0.1:{baseline_port}"

            common_server_args: list[str] = [
                "--trust-remote-code",
                "--attention-backend",
                backend,
                "--tp-size",
                str(tp),
                "--dtype",
                str(args.dtype),
                "--mem-fraction-static",
                str(args.mem_fraction_static),
                "--max-running-requests",
                str(args.max_running_requests),
            ]
            common_server_args.extend(
                ["--cuda-graph-bs", *[str(i) for i in range(1, 33)], "--cuda-graph-max-bs", "32"]
            )
            if args.disable_radix_cache:
                common_server_args.append("--disable-radix-cache")

            baseline_proc = popen_launch_server(
                args.target_model,
                baseline_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=common_server_args,
            )
            try:
                # Warm up.
                _send_generate(
                    baseline_url,
                    "Hello",
                    max_new_tokens=8,
                    stop=[],
                    timeout_s=min(int(args.timeout_s), 300),
                )

                for conc in concurrencies:
                    n = num_questions_by_conc[conc]
                    _flush_cache(baseline_url)
                    metrics = _run_gsm8k_requests(
                        baseline_url,
                        prompts=prompts[:n],
                        labels=labels[:n],
                        max_new_tokens=int(args.max_new_tokens),
                        concurrency=int(conc),
                        stop=default_stop,
                        timeout_s=int(args.timeout_s),
                        expect_dflash=False,
                    )
                    baseline_toks[(backend, tp, conc)] = metrics.output_toks_per_s
                    baseline_acc[(backend, tp, conc)] = metrics.accuracy
                    print(
                        f"[baseline] conc={conc:>2} n={n:<4} "
                        f"toks/s={metrics.output_toks_per_s:,.2f} "
                        f"latency={metrics.latency_s:.1f}s "
                        f"acc={metrics.accuracy:.3f} invalid={metrics.invalid_rate:.3f}"
                    )
            finally:
                kill_process_tree(baseline_proc.pid)
                try:
                    baseline_proc.wait(timeout=30)
                except Exception:
                    pass

            print(f"\n=== backend={backend} tp={tp} (DFLASH) ===")
            dflash_port = find_available_port(baseline_port + 1)
            dflash_url = f"http://127.0.0.1:{dflash_port}"
            dflash_proc = popen_launch_server(
                args.target_model,
                dflash_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    *common_server_args,
                    "--speculative-algorithm",
                    "DFLASH",
                    "--speculative-draft-model-path",
                    args.draft_model,
                ],
            )
            try:
                _send_generate(
                    dflash_url,
                    "Hello",
                    max_new_tokens=8,
                    stop=[],
                    timeout_s=min(int(args.timeout_s), 300),
                )
                for conc in concurrencies:
                    n = num_questions_by_conc[conc]
                    _flush_cache(dflash_url)
                    metrics = _run_gsm8k_requests(
                        dflash_url,
                        prompts=prompts[:n],
                        labels=labels[:n],
                        max_new_tokens=int(args.max_new_tokens),
                        concurrency=int(conc),
                        stop=default_stop,
                        timeout_s=int(args.timeout_s),
                        expect_dflash=True,
                    )
                    dflash_toks[(backend, tp, conc)] = metrics.output_toks_per_s
                    dflash_accept_len[(backend, tp, conc)] = metrics.spec_accept_length
                    dflash_acc[(backend, tp, conc)] = metrics.accuracy
                    print(
                        f"[DFLASH]   conc={conc:>2} n={n:<4} "
                        f"toks/s={metrics.output_toks_per_s:,.2f} "
                        f"latency={metrics.latency_s:.1f}s "
                        f"acc={metrics.accuracy:.3f} invalid={metrics.invalid_rate:.3f} "
                        f"accept_len={metrics.spec_accept_length:.3f} "
                        f"spec_verify_ct_sum={metrics.spec_verify_ct_sum}"
                    )
            finally:
                kill_process_tree(dflash_proc.pid)
                try:
                    dflash_proc.wait(timeout=30)
                except Exception:
                    pass

    # Render markdown.
    md_lines: list[str] = []
    md_lines.append("# DFLASH GSM8K Sweep")
    md_lines.append("")
    md_lines.append("## Settings")
    md_lines.append(f"- target_model: `{args.target_model}`")
    md_lines.append(f"- draft_model: `{args.draft_model}`")
    md_lines.append(f"- prompt_style: `{args.prompt_style}`")
    if args.prompt_style == "fewshot_qa":
        md_lines.append(f"- num_shots: `{args.num_shots}`")
    md_lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    md_lines.append(f"- attention_backends: `{', '.join(attention_backends)}`")
    md_lines.append(f"- tp_sizes: `{', '.join(str(x) for x in tp_sizes)}`")
    md_lines.append(f"- concurrencies: `{', '.join(str(x) for x in concurrencies)}`")
    md_lines.append(f"- questions_per_concurrency: `base={args.questions_per_concurrency_base}`")
    md_lines.append(f"- device_sm: `{device_sm}`")
    md_lines.append(f"- is_blackwell: `{is_blackwell}`")
    md_lines.append("")
    md_lines.append(
        "Note: DFLASH and baseline greedy outputs may diverge on some prompts due to numerical differences "
        "(e.g. verify path vs decode path). This sweep focuses on throughput."
    )
    md_lines.append("")

    for backend in attention_backends:
        md_lines.append(f"## Backend: `{backend}`")
        md_lines.append("")

        baseline_values = {
            (tp, conc): baseline_toks.get((backend, tp, conc), None)
            for tp in tp_sizes
            for conc in concurrencies
        }
        dflash_values = {
            (tp, conc): dflash_toks.get((backend, tp, conc), None)
            for tp in tp_sizes
            for conc in concurrencies
        }
        speedup_values: dict[tuple[int, int], Optional[float]] = {}
        for tp in tp_sizes:
            for conc in concurrencies:
                b = baseline_values.get((tp, conc), None)
                d = dflash_values.get((tp, conc), None)
                speedup_values[(tp, conc)] = None if (b is None or d is None or b <= 0) else (d / b)

        md_lines.append("### Baseline output tok/s")
        md_lines.append(
            _format_table(
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                values=baseline_values,
                float_fmt=",.2f",
            )
        )
        md_lines.append("")
        md_lines.append("### Baseline accuracy")
        md_lines.append(
            _format_table(
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                values={
                    (tp, conc): baseline_acc.get((backend, tp, conc), None)
                    for tp in tp_sizes
                    for conc in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")
        md_lines.append("### DFLASH output tok/s")
        md_lines.append(
            _format_table(
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                values=dflash_values,
                float_fmt=",.2f",
            )
        )
        md_lines.append("")
        md_lines.append("### DFLASH accuracy")
        md_lines.append(
            _format_table(
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                values={
                    (tp, conc): dflash_acc.get((backend, tp, conc), None)
                    for tp in tp_sizes
                    for conc in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")
        md_lines.append("### Speedup (DFLASH / baseline)")
        md_lines.append(
            _format_table(
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                values=speedup_values,
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH acceptance length (mean per-request spec_accept_length)")
        md_lines.append(
            _format_table(
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                values={
                    (tp, conc): dflash_accept_len.get((backend, tp, conc), None)
                    for tp in tp_sizes
                    for conc in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        f.write("\n")

    print(f"\nWrote markdown report to: {args.output_md}")


if __name__ == "__main__":
    main()
