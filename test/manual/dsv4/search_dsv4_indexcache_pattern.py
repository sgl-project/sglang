"""Greedy training-free IndexCache pattern search for DSV4 C4 layers."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional

import requests


def load_calibration_texts(path: Path, limit: int) -> list[str]:
    texts = []
    with path.open() as f:
        for line in f:
            if limit > 0 and len(texts) >= limit:
                break
            row = json.loads(line)
            text = row.get("text") or row.get("prompt")
            if text:
                texts.append(text)
    if not texts:
        raise ValueError(f"no calibration texts found in {path}")
    return texts


def protected_c4_indices(num_c4_layers: int, block_size: int) -> set[int]:
    protected = {0}
    if block_size > 0:
        protected.update(range(0, num_c4_layers, block_size))
    return protected


def target_f_layers(num_c4_layers: int, retention: str) -> int:
    if retention == "1/2":
        return math.ceil(num_c4_layers / 2)
    if retention == "1/4":
        return math.ceil(num_c4_layers / 4)
    raise ValueError(f"unsupported retention {retention}")


def pattern_to_str(pattern: list[str]) -> str:
    return "".join(pattern)


def finite_logprobs(meta_info: dict) -> Iterable[float]:
    for item in meta_info.get("input_token_logprobs", []):
        if item is None:
            continue
        value = item[0] if isinstance(item, list) else item
        if isinstance(value, (float, int)) and math.isfinite(value):
            yield float(value)


def score_endpoint(endpoint: str, texts: list[str], timeout: int) -> float:
    losses = []
    for text in texts:
        res = requests.post(
            endpoint.rstrip("/") + "/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=timeout,
        )
        res.raise_for_status()
        logprobs = list(finite_logprobs(res.json().get("meta_info", {})))
        if not logprobs:
            raise RuntimeError("endpoint returned no input_token_logprobs")
        losses.append(-sum(logprobs) / len(logprobs))
    return sum(losses) / len(losses)


def wait_for_endpoint(endpoint: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(endpoint.rstrip("/") + "/health", timeout=5).raise_for_status()
            return
        except requests.RequestException:
            time.sleep(2)
    raise TimeoutError(f"endpoint did not become healthy: {endpoint}")


def run_candidate(
    command_template: Optional[str],
    endpoint: str,
    pattern: str,
    startup_timeout: int,
    texts: list[str],
    request_timeout: int,
) -> float:
    proc = None
    try:
        if command_template:
            cmd = command_template.format(pattern=pattern)
            proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
            wait_for_endpoint(endpoint, startup_timeout)
        return score_endpoint(endpoint, texts, request_timeout)
    finally:
        if proc is not None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)


def search(args) -> list[dict]:
    texts = load_calibration_texts(args.calibration_jsonl, args.limit)
    pattern = ["F"] * args.num_c4_layers
    protected = protected_c4_indices(args.num_c4_layers, args.pp_block_c4_layers)
    target_f = target_f_layers(args.num_c4_layers, args.retention)
    history = []

    while pattern.count("F") > target_f:
        candidates = [i for i, v in enumerate(pattern) if v == "F" and i not in protected]
        if not candidates:
            raise RuntimeError("no candidate F layers left to flip")

        scored = []
        for idx in candidates:
            candidate = pattern.copy()
            candidate[idx] = "S"
            candidate_str = pattern_to_str(candidate)
            loss = run_candidate(
                args.command_template,
                args.endpoint,
                candidate_str,
                args.startup_timeout,
                texts,
                args.request_timeout,
            )
            scored.append((loss, idx, candidate_str))
            print(json.dumps({"candidate": candidate_str, "flip": idx, "loss": loss}))

        loss, idx, candidate_str = min(scored)
        pattern[idx] = "S"
        step = {"pattern": candidate_str, "flip": idx, "loss": loss}
        history.append(step)
        print(json.dumps({"selected": step}))

    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--command-template")
    parser.add_argument("--num-c4-layers", type=int, required=True)
    parser.add_argument("--retention", choices=["1/2", "1/4"], required=True)
    parser.add_argument("--pp-block-c4-layers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    history = search(args)
    result = {"retention": args.retention, "history": history}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
