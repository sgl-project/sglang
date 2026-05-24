"""Greedy training-free IndexCache pattern search for DSV4 C4 layers."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests

sys.path.insert(0, str(Path(__file__).parent))
from indexcache_base_path import (
    fetch_server_info,
    validate_server_info_for_base_path,
)


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


def pattern_counts(pattern: str) -> dict[str, int]:
    return {"F": pattern.count("F"), "S": pattern.count("S")}


def finite_logprobs(meta_info: dict) -> Iterable[float]:
    for item in meta_info.get("input_token_logprobs", []):
        if item is None:
            continue
        value = item[0] if isinstance(item, list) else item
        if isinstance(value, (float, int)) and math.isfinite(value):
            yield float(value)


def score_endpoint(
    endpoint: str, texts: list[str], timeout: int, min_prompt_tokens: int
) -> float:
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
        if min_prompt_tokens > 0 and len(logprobs) < min_prompt_tokens:
            raise RuntimeError(
                f"calibration prompt has {len(logprobs)} tokens, below "
                f"IndexCache floor {min_prompt_tokens}"
            )
        losses.append(-sum(logprobs) / len(logprobs))
    return sum(losses) / len(losses)


def wait_for_endpoint(endpoint: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(endpoint.rstrip("/") + "/health", timeout=5).raise_for_status()
            server_info = fetch_server_info(endpoint, timeout=5)
            validate_server_info_for_base_path(server_info, "candidate endpoint")
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
    min_prompt_tokens: int,
) -> float:
    proc = None
    try:
        if command_template:
            cmd = command_template.format(pattern=pattern)
            proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
            wait_for_endpoint(endpoint, startup_timeout)
        return score_endpoint(endpoint, texts, request_timeout, min_prompt_tokens)
    finally:
        if proc is not None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=60)


def greedy_search_pattern(
    num_c4_layers: int,
    retention: str,
    pp_block_c4_layers: int,
    score_pattern: Callable[[str], float],
) -> dict:
    pattern = ["F"] * num_c4_layers
    protected = protected_c4_indices(num_c4_layers, pp_block_c4_layers)
    target_f = target_f_layers(num_c4_layers, retention)
    if len(protected) > target_f:
        raise ValueError(
            f"retention {retention} requires {target_f} F layers but "
            f"{len(protected)} C4 anchors are protected; reduce "
            "--pp-block-c4-layers or use a higher retention"
        )
    initial_pattern = pattern_to_str(pattern)
    baseline_loss = score_pattern(initial_pattern)
    print(json.dumps({"baseline": {"pattern": initial_pattern, "loss": baseline_loss}}))
    history = []

    while pattern.count("F") > target_f:
        candidates = [
            i for i, v in enumerate(pattern) if v == "F" and i not in protected
        ]
        if not candidates:
            raise RuntimeError("no candidate F layers left to flip")

        scored = []
        for idx in candidates:
            candidate = pattern.copy()
            candidate[idx] = "S"
            candidate_str = pattern_to_str(candidate)
            loss = score_pattern(candidate_str)
            loss_delta = loss - baseline_loss
            item = {
                "pattern": candidate_str,
                "flip": idx,
                "loss": loss,
                "loss_delta": loss_delta,
            }
            scored.append(item)
            print(json.dumps({"candidate": item}))

        selected = min(scored, key=lambda item: (item["loss_delta"], item["flip"]))
        idx = selected["flip"]
        candidate_str = selected["pattern"]
        pattern[idx] = "S"
        step = {
            "pattern": candidate_str,
            "flip": idx,
            "loss": selected["loss"],
            "loss_delta": selected["loss_delta"],
            "candidates": sorted(
                scored, key=lambda item: (item["loss_delta"], item["flip"])
            ),
        }
        history.append(step)
        print(json.dumps({"selected": step}))

    final_pattern = pattern_to_str(pattern)
    return {
        "retention": retention,
        "search_method": "greedy_training_free",
        "uniform_candidate": False,
        "initial_pattern": initial_pattern,
        "baseline_loss": baseline_loss,
        "final_pattern": final_pattern,
        "final_pattern_counts": pattern_counts(final_pattern),
        "target_f_layers": target_f,
        "protected_c4_indices": sorted(protected),
        "history": history,
    }


def search(args, retention: str) -> dict:
    texts = load_calibration_texts(args.calibration_jsonl, args.limit)

    def score_pattern(pattern: str) -> float:
        return run_candidate(
            args.command_template,
            args.endpoint,
            pattern,
            args.startup_timeout,
            texts,
            args.request_timeout,
            args.min_indexcache_prompt_tokens,
        )

    return greedy_search_pattern(
        args.num_c4_layers,
        retention,
        args.pp_block_c4_layers,
        score_pattern,
    )


def validate_args(args) -> None:
    if not args.command_template:
        raise SystemExit(
            "--command-template is required so each candidate pattern is actually "
            "deployed/scored; otherwise every candidate would hit the same endpoint"
        )
    if "{pattern}" not in args.command_template:
        raise SystemExit("--command-template must contain {pattern}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--command-template")
    parser.add_argument("--num-c4-layers", type=int, required=True)
    parser.add_argument(
        "--retention",
        action="append",
        choices=["1/2", "1/4"],
        required=True,
        help="Target retention. Repeat to search both 1/2 and 1/4.",
    )
    parser.add_argument("--pp-block-c4-layers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--min-indexcache-prompt-tokens", type=int, default=75000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_args(args)

    result = {
        "search_method": "greedy_training_free",
        "uniform_1_4_candidate": "not generated; only searched 1/4 is produced",
        "min_indexcache_prompt_tokens": args.min_indexcache_prompt_tokens,
        "retentions": {
            retention: search(args, retention) for retention in args.retention
        },
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
