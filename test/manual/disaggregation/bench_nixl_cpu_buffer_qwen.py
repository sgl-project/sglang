"""Real-model unified/CPU-buffer PD comparison, run manually on a shared CC host.

Download Qwen/Qwen3.5-0.8B with ModelScope first. Example:
  PYTHONPATH=python python test/manual/disaggregation/bench_nixl_cpu_buffer_qwen.py \
    --model /path/to/Qwen3.5-0.8B --output /path/to/results --tp 1
Repeat with --tp 2 (four GPUs required for PD). This is a small client-observed
smoke benchmark, not a dedicated-hardware capacity or model-quality evaluation.
For Qwen3.5-35B-A3B on free H20 GPUs, use --mem-fraction-static 0.9,
--cpu-buffer-size-gb 1, and --startup-timeout 1200.
"""

import argparse
import itertools
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

from sglang.test.test_utils import terminate_and_kill_process_tree
from sglang.utils import wait_for_http_ready


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stream_generate(url, body, start):
    first = None
    result = None
    with requests.post(
        url + "/generate", json={**body, "stream": True}, stream=True, timeout=300
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            result = json.loads(line[6:])
            if result.get("error"):
                raise RuntimeError(result)
            if result.get("output_ids") and first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    if first is None or result is None:
        raise RuntimeError(f"Missing generated tokens: {result}")
    ids = result["output_ids"]
    return {
        "ids": ids,
        "ttft_s": first - start,
        "latency_s": end - start,
        "tpot_s": (end - first) / max(1, len(ids) - 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--linear-attn-prefill-backend", choices=("triton", "flashinfer")
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.6)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--cpu-buffer-size-gb", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.mem_fraction_static < 1:
        parser.error("--mem-fraction-static must be between 0 and 1")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    if not 0 < args.cpu_buffer_size_gb < float("inf"):
        parser.error("--cpu-buffer-size-gb must be finite and positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    questions = [
        "Explain why the sky looks blue in two sentences.",
        "Write a Python function that returns the greatest common divisor of two positive integers.",
        "用中文简要解释什么是大语言模型的 KV cache，以及它为什么能加速生成。",
        "Summarize the difference between prefill and decode in language model inference.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        for q in questions
    ]
    context = tokenizer.encode(
        "A deployment separates prompt processing from autoregressive token generation. "
        "Key and value tensors are transferred between workers. " * 100,
        add_special_tokens=False,
    )
    for length in (512, 1024):
        suffix = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Summarize the deployment described above."}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        prompts.append(context[: length - len(suffix)] + suffix)
    common = [
        "--tp",
        str(args.tp),
        "--load-format",
        "safetensors",
        "--skip-tokenizer-init",
        "--random-seed",
        "42",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "triton",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--disable-overlap-schedule",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-total-tokens",
        "8192",
        "--max-running-requests",
        "4",
        "--context-length",
        "2048",
        "--chunked-prefill-size",
        "128",
        "--page-size",
        "1",
        "--disable-custom-all-reduce",
        "--disable-radix-cache",
    ]
    if args.deterministic:
        common += [
            "--enable-deterministic-inference",
            "--linear-attn-prefill-backend",
            "triton",
        ]
    if args.linear_attn_prefill_backend:
        common += ["--linear-attn-prefill-backend", args.linear_attn_prefill_backend]
    report = {
        "model": args.model,
        "tp": args.tp,
        "common_args": common,
        "cpu_buffer_size_gb": args.cpu_buffer_size_gb,
        "input_lengths": list(map(len, prompts)),
        "output_tokens": 64,
        "note": "Shared H20 CC host; unified uses TP GPUs, PD uses 2*TP GPUs; "
        "client timings include local HTTP and PD coordination; no router.",
        "runs": {},
    }
    report_path = out / f"tp{args.tp}.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    for mode in ("unified", "pd"):
        processes = []
        handles = []
        urls = []
        bootstrap = free_port()
        rooms = itertools.count(10000)
        env = {
            **os.environ,
            "UCX_TLS": "tcp,cuda_copy,self",
            "UCX_ACC_DEVICES": "",
            "SGLANG_DISAGGREGATION_NIXL_BACKEND": "UCX",
            "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS": '{"num_threads":"1"}',
        }
        try:
            for role, gpu in (
                [("unified", 0)]
                if mode == "unified"
                else [("prefill", 0), ("decode", args.tp)]
            ):
                port = free_port()
                url = f"http://127.0.0.1:{port}"
                extra = (
                    []
                    if role == "unified"
                    else [
                        "--disaggregation-mode",
                        role,
                        "--disaggregation-transfer-backend",
                        "nixl",
                        "--nixl-use-cpu-buffer",
                        "--nixl-cpu-buffer-size-gb",
                        str(args.cpu_buffer_size_gb),
                        "--disaggregation-enable-kv-checksum",
                        "--disaggregation-bootstrap-port",
                        str(bootstrap),
                    ]
                )
                command = [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    args.model,
                    *common,
                    *extra,
                    "--base-gpu-id",
                    str(gpu),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ]
                log = open(out / f"tp{args.tp}-{role}.log", "w")
                handles.append(log)
                print("Starting", role, "TP", args.tp, flush=True)
                processes.append(
                    subprocess.Popen(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                )
                urls.append(url)
            for url, process in zip(urls, processes, strict=True):
                wait_for_http_ready(
                    url=url + "/health", timeout=args.startup_timeout, process=process
                )

            def request(ids):
                body = {
                    "input_ids": ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 64,
                        "ignore_eos": True,
                    },
                }
                start = time.perf_counter()
                if mode == "unified":
                    result = stream_generate(urls[0], body, start)
                else:
                    body.update(
                        bootstrap_host="127.0.0.1",
                        bootstrap_port=bootstrap,
                        bootstrap_room=next(rooms),
                    )
                    with ThreadPoolExecutor(max_workers=1) as executor:

                        def prefill():
                            response = requests.post(
                                urls[0] + "/generate", json=body, timeout=300
                            )
                            response.raise_for_status()
                            return response.json()

                        future = executor.submit(prefill)
                        result = stream_generate(urls[1], body, start)
                        future.result(timeout=300)
                assert len(result["ids"]) == 64, result
                return result

            # Exclude startup and kernel warmup from recorded measurements.
            request(prompts[0])
            request(prompts[-1])
            correctness = [request(prompt) for prompt in prompts]
            for item in correctness:
                item["text"] = tokenizer.decode(item["ids"], skip_special_tokens=True)
            runs = []
            workload = [prompts[4], prompts[5]] * 4
            for concurrency in (1, 4):
                for repeat in range(2):
                    start = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        results = list(executor.map(request, workload))
                    elapsed = time.perf_counter() - start
                    runs.append(
                        {
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "requests": len(results),
                            "elapsed_s": elapsed,
                            "output_tokens_per_s": sum(len(r["ids"]) for r in results)
                            / elapsed,
                            "ttft_median_s": statistics.median(
                                r["ttft_s"] for r in results
                            ),
                            "latency_median_s": statistics.median(
                                r["latency_s"] for r in results
                            ),
                            "tpot_median_s": statistics.median(
                                r["tpot_s"] for r in results
                            ),
                            "samples": results,
                        }
                    )
                    print(
                        mode,
                        "TP",
                        args.tp,
                        {k: v for k, v in runs[-1].items() if k != "samples"},
                        flush=True,
                    )
            report["runs"][mode] = {"correctness": correctness, "performance": runs}
            save()
        finally:
            for process in reversed(processes):
                terminate_and_kill_process_tree(process)
            for handle in handles:
                handle.close()
    baseline = report["runs"]["unified"]["correctness"]
    actual = report["runs"]["pd"]["correctness"]
    report["correctness_matches"] = [
        a["ids"] == b["ids"] for a, b in zip(baseline, actual, strict=True)
    ]
    report["performance_token_matches"] = [
        [a["ids"] == b["ids"] for a, b in zip(x["samples"], y["samples"], strict=True)]
        for x, y in zip(
            report["runs"]["unified"]["performance"],
            report["runs"]["pd"]["performance"],
            strict=True,
        )
    ]
    save()
    print(
        "COMPARISON_COMPLETE",
        report_path,
        "correctness",
        report["correctness_matches"],
        "performance matches",
        [sum(x) for x in report["performance_token_matches"]],
        flush=True,
    )


if __name__ == "__main__":
    main()
