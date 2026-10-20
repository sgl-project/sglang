"""Diagnose generation differences without PD by varying companion lifetimes.

Run manually with the same model and common_args as a saved PD benchmark JSON.
This is a diagnostic, not a model-quality or throughput benchmark.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

from sglang.test.test_utils import terminate_and_kill_process_tree
from sglang.utils import wait_for_http_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    ref = json.loads(Path(args.reference).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ref["model"], local_files_only=True)
    context = tokenizer.encode(
        "A deployment separates prompt processing from autoregressive token generation. "
        "Key and value tensors are transferred between workers. " * 100,
        add_special_tokens=False,
    )
    suffix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Summarize the deployment described above."}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    prompts = {size: context[: size - len(suffix)] + suffix for size in (512, 1024)}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    common = list(ref["common_args"])
    if args.deterministic:
        common += [
            "--enable-deterministic-inference",
            "--linear-attn-prefill-backend",
            "triton",
        ]
    report = {"model": ref["model"], "common_args": common, "runs": []}
    log = open(out / "unified.log", "w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            ref["model"],
            *common,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=os.environ.copy(),
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    def generate(length, tokens, delay=0):
        time.sleep(delay)
        response = requests.post(
            url + "/generate",
            json={
                "input_ids": prompts[length],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
                "logprob_start_len": -1,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        assert len(result["output_ids"]) == tokens, result
        return result

    def save():
        (out / "results.json").write_text(json.dumps(report, indent=2))

    try:
        wait_for_http_ready(url=url + "/health", timeout=1200, process=process)
        generate(1024, 8)
        baseline = generate(512, 64)
        report["baseline"] = baseline
        save()
        for delay in (0, 0.5, 1.0, 2.0):
            for repeat in range(2):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    target = executor.submit(generate, 512, 64)
                    companions = [
                        executor.submit(generate, 1024, n, delay) for n in (8, 24, 40)
                    ]
                    actual = target.result()
                    for future in companions:
                        future.result()
                matches = actual["output_ids"] == baseline["output_ids"]
                first = next(
                    (
                        i + 1
                        for i, (a, b) in enumerate(
                            zip(
                                actual["output_ids"],
                                baseline["output_ids"],
                                strict=True,
                            )
                        )
                        if a != b
                    ),
                    None,
                )
                report["runs"].append(
                    {
                        "delay_s": delay,
                        "repeat": repeat,
                        "matches": matches,
                        "first_difference": first,
                        "result": actual,
                    }
                )
                print(delay, repeat, matches, first, flush=True)
                save()
    finally:
        terminate_and_kill_process_tree(process)
        log.close()


if __name__ == "__main__":
    main()
