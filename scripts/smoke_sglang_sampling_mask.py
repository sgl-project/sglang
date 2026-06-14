#!/usr/bin/env python3
"""Smoke test SGLang sparse sampling-mask responses.

This starts an SGLang server from the current checkout by default, sends a
single /generate request with return_sampling_mask=true, and verifies that the
returned sparse support is aligned with generated tokens and consistent with
the requested top-p/top-k truncation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


DEFAULT_MODEL_PATH = "/mnt/lustre/slime/models/Qwen3.5-35B-A3B-FP8"
DEFAULT_PROMPT = (
    "Complete one concise mathematical sentence: The sum of two even integers is"
)


def post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc


def tail_file(path: Path, max_lines: int = 100) -> str:
    if not path.exists():
        return f"{path} does not exist"
    return "\n".join(path.read_text(errors="replace").splitlines()[-max_lines:])


def wait_for_health(
    base_url: str,
    proc: subprocess.Popen | None,
    log_path: Path | None,
    timeout: int,
) -> None:
    deadline = time.time() + timeout
    next_progress = 0.0
    health_url = f"{base_url.rstrip('/')}/health"

    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = tail_file(log_path) if log_path is not None else ""
            raise RuntimeError(
                f"SGLang server exited with code {proc.returncode} before /health was ready.\n"
                f"Last server log lines:\n{tail}"
            )

        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    print(f"SGLang health check passed: {health_url}", flush=True)
                    return
        except Exception:
            pass

        now = time.time()
        if now >= next_progress:
            print(f"Waiting for SGLang server on {health_url}", flush=True)
            next_progress = now + 30
        time.sleep(2)

    raise TimeoutError(f"SGLang did not become healthy within {timeout} seconds")


def parse_top_logprob_item(item: object, position: int) -> tuple[float, int]:
    if isinstance(item, dict):
        if "logprob" not in item:
            raise AssertionError(
                f"Malformed top logprob at position {position}: {item}"
            )
        token_id = item.get("token_id", item.get("id"))
        if token_id is None:
            raise AssertionError(
                f"Malformed top logprob at position {position}: {item}"
            )
        return float(item["logprob"]), int(token_id)

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return float(item[0]), int(item[1])

    raise AssertionError(f"Malformed top logprob at position {position}: {item}")


def validate_sampling_mask(
    response: dict | list,
    top_p: float,
    top_k: int,
    min_p: float,
) -> dict:
    if isinstance(response, list):
        if len(response) != 1:
            raise AssertionError(f"Expected one response item, got {len(response)}")
        response = response[0]

    output_ids = response.get("output_ids")
    meta = response.get("meta_info") or {}
    masks = meta.get("output_token_sampling_mask")
    sampling_logprobs = meta.get("output_token_sampling_logprobs")
    sampling_mask_length = meta.get("output_token_sampling_mask_length")
    top_logprobs = meta.get("output_top_logprobs")

    if not isinstance(output_ids, list) or not output_ids:
        raise AssertionError(
            f"Response did not include non-empty output_ids: {response}"
        )
    if not isinstance(masks, list) or len(masks) != len(output_ids):
        raise AssertionError(
            f"Sampling masks are missing or misaligned: masks={masks}, output_ids={output_ids}"
        )
    if not isinstance(sampling_logprobs, list) or len(sampling_logprobs) != len(
        output_ids
    ):
        raise AssertionError(
            "Sampling logprobs are missing or misaligned: "
            f"sampling_logprobs={sampling_logprobs}, output_ids={output_ids}"
        )
    if sampling_mask_length is not None and int(sampling_mask_length) != len(
        output_ids
    ):
        raise AssertionError(
            "output_token_sampling_mask_length does not match output length: "
            f"{sampling_mask_length} != {len(output_ids)}"
        )
    if not isinstance(top_logprobs, list) or len(top_logprobs) != len(output_ids):
        raise AssertionError(
            "Response did not include output_top_logprobs aligned with output_ids; "
            f"got top_logprobs={top_logprobs}"
        )

    mask_lengths: list[int] = []
    mask_prob_masses: list[float] = []
    mask_prob_masses_without_last: list[float] = []
    sampled_top1_like_count = 0
    probability_tie_tolerance = 1e-6

    for i, (sampled_token_id, mask, sampling_logprob) in enumerate(
        zip(output_ids, masks, sampling_logprobs)
    ):
        if not isinstance(mask, list) or not mask:
            raise AssertionError(
                f"Mask at output position {i} is not a non-empty list: {mask}"
            )
        if sampled_token_id not in mask:
            raise AssertionError(
                f"Sampled token id {sampled_token_id} at output position {i} is not in its mask"
            )
        if top_k > 0 and len(mask) > top_k:
            raise AssertionError(
                f"Mask length {len(mask)} exceeds top_k={top_k} at output position {i}"
            )

        position_top_logprobs = top_logprobs[i]
        if not isinstance(position_top_logprobs, list) or not position_top_logprobs:
            raise AssertionError(f"Missing top logprobs at output position {i}")

        top_pairs = [parse_top_logprob_item(item, i) for item in position_top_logprobs]
        logprob_by_token_id = {token_id: logprob for logprob, token_id in top_pairs}
        missing = [token_id for token_id in mask if token_id not in logprob_by_token_id]
        if missing:
            raise AssertionError(
                f"Top logprobs did not cover all mask tokens at output position {i}; "
                f"missing first ids={missing[:10]}. Increase --top-logprobs-num."
            )

        mask_logprobs = [logprob_by_token_id[token_id] for token_id in mask]
        for left, right in zip(mask_logprobs, mask_logprobs[1:]):
            if left + probability_tie_tolerance < right:
                raise AssertionError(
                    "Mask token ids are not sorted by descending probability at "
                    f"output position {i}"
                )

        top1_logprob = max(logprob for logprob, _ in top_pairs)
        if abs(mask_logprobs[0] - top1_logprob) > 1e-5:
            raise AssertionError(
                f"Mask first token is not top-probability at output position {i}: "
                f"{mask_logprobs[0]} vs top {top1_logprob}"
            )
        if abs(logprob_by_token_id[sampled_token_id] - top1_logprob) <= 1e-5:
            sampled_top1_like_count += 1

        mass = math.fsum(math.exp(logprob) for logprob in mask_logprobs)
        mass_without_last = mass - math.exp(mask_logprobs[-1])
        top_k_is_binding = top_k > 0 and len(mask) >= top_k
        min_p_is_active = min_p > 0.0
        if not top_k_is_binding and not min_p_is_active:
            if mass + 1e-4 < top_p:
                raise AssertionError(
                    f"Returned mask mass {mass:.6f} is below top_p={top_p} "
                    f"at output position {i}"
                )
            if len(mask_logprobs) > 1 and mass_without_last > top_p + 1e-3:
                raise AssertionError(
                    f"Mask is not the minimal top-p prefix at output position {i}: "
                    f"mass_without_last={mass_without_last:.6f}, top_p={top_p}"
                )

        expected_sampling_logprob = logprob_by_token_id[sampled_token_id] - math.log(
            mass
        )
        if not math.isfinite(float(sampling_logprob)):
            raise AssertionError(
                f"Sampling logprob at output position {i} is not finite: {sampling_logprob}"
            )
        if abs(float(sampling_logprob) - expected_sampling_logprob) > 3e-3:
            raise AssertionError(
                f"Sampling logprob mismatch at output position {i}: "
                f"got {sampling_logprob}, expected {expected_sampling_logprob}"
            )

        mask_lengths.append(len(mask))
        mask_prob_masses.append(mass)
        mask_prob_masses_without_last.append(mass_without_last)

    return {
        "text": response.get("text"),
        "output_ids": output_ids,
        "mask_lengths": mask_lengths,
        "mask_prob_masses": mask_prob_masses,
        "mask_prob_masses_without_last": mask_prob_masses_without_last,
        "sampling_logprobs": sampling_logprobs,
        "sampling_mask_length_field": sampling_mask_length,
        "sampled_top1_like_count": sampled_top1_like_count,
        "checked_positions": len(output_ids),
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_server_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--tp-size",
        str(args.tp_size),
        "--trust-remote-code",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--watchdog-timeout",
        str(args.watchdog_timeout),
    ]

    optional_args = [
        ("--dp-size", args.dp_size),
        ("--context-length", args.context_length),
        ("--mem-fraction-static", args.mem_fraction_static),
        ("--page-size", args.page_size),
        ("--max-running-requests", args.max_running_requests),
        ("--cuda-graph-max-bs", args.cuda_graph_max_bs),
        ("--tool-call-parser", args.tool_call_parser),
        ("--reasoning-parser", args.reasoning_parser),
        ("--sampling-backend", args.sampling_backend),
    ]
    for name, value in optional_args:
        if value is not None:
            cmd.extend([name, str(value)])

    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if args.disable_piecewise_cuda_graph:
        cmd.append("--disable-piecewise-cuda-graph")

    cmd.extend(args.extra_server_arg)
    return cmd


def start_server(args: argparse.Namespace) -> subprocess.Popen:
    root = repo_root()
    model_path = Path(args.model_path)
    if model_path.is_absolute() and not model_path.exists():
        raise RuntimeError(f"Model path does not exist: {model_path}")

    env = os.environ.copy()
    python_dir = str(root / "python")
    env["PYTHONPATH"] = (
        python_dir
        if not env.get("PYTHONPATH")
        else python_dir + os.pathsep + env["PYTHONPATH"]
    )

    log_path = Path(args.server_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_server_command(args)

    print("Starting SGLang server", flush=True)
    print("+ " + " ".join(cmd), flush=True)
    with log_path.open("w") as log_file:
        return subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )


def build_payload(args: argparse.Namespace) -> dict:
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k
    if args.min_p:
        sampling_params["min_p"] = args.min_p

    return {
        "text": args.prompt,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "top_logprobs_num": args.top_logprobs_num,
        "return_sampling_mask": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Model path used when launching a server.",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "30000"))
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL"),
        help="Existing server URL. Defaults to http://127.0.0.1:<port>.",
    )
    parser.add_argument(
        "--no-launch-server",
        action="store_true",
        help="Run the request and validation against an already-running server.",
    )
    parser.add_argument(
        "--tp-size", type=int, default=int(os.environ.get("TP_SIZE", "1"))
    )
    parser.add_argument("--dp-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--mem-fraction-static", default=os.environ.get("MEM_FRACTION_STATIC")
    )
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-piecewise-cuda-graph", action="store_true")
    parser.add_argument(
        "--tool-call-parser", default=os.environ.get("TOOL_CALL_PARSER")
    )
    parser.add_argument(
        "--reasoning-parser", default=os.environ.get("REASONING_PARSER")
    )
    parser.add_argument(
        "--sampling-backend", default=os.environ.get("SGLANG_SAMPLING_BACKEND")
    )
    parser.add_argument("--extra-server-arg", action="append", default=[])
    parser.add_argument(
        "--server-log",
        default=os.environ.get("SERVER_LOG", "/tmp/sglang_sampling_mask_server.log"),
    )
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    parser.add_argument(
        "--top-p", type=float, default=float(os.environ.get("TOP_P", "0.9"))
    )
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "16")))
    parser.add_argument(
        "--min-p", type=float, default=float(os.environ.get("MIN_P", "0.0"))
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", "0.7")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("MAX_NEW_TOKENS", "4")),
    )
    parser.add_argument(
        "--top-logprobs-num",
        type=int,
        default=int(os.environ.get("TOP_LOGPROBS_NUM", "32")),
        help="Must cover the full expected support for probability-mass checks.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.environ.get("TIMEOUT_SECONDS", "1200")),
    )
    parser.add_argument("--watchdog-timeout", type=int, default=1000000)
    return parser.parse_args()


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    print("Stopping SGLang server", flush=True)
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def main() -> None:
    args = parse_args()
    base_url = (args.base_url or f"http://127.0.0.1:{args.port}").rstrip("/")
    proc = None

    try:
        if not args.no_launch_server:
            proc = start_server(args)
        wait_for_health(
            base_url=base_url,
            proc=proc,
            log_path=Path(args.server_log) if proc is not None else None,
            timeout=args.timeout_seconds,
        )

        payload = build_payload(args)
        print(f"Querying /generate with payload: {json.dumps(payload)}", flush=True)
        response = post_json(
            f"{base_url}/generate",
            payload=payload,
            timeout=args.timeout_seconds,
        )
        try:
            summary = validate_sampling_mask(
                response=response,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
            )
        except Exception:
            print("Raw /generate response:", json.dumps(response)[:12000], flush=True)
            raise

        print("Sampling mask smoke test passed.", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if proc is not None:
            stop_server(proc)
            print(f"Server log: {args.server_log}", flush=True)


if __name__ == "__main__":
    main()
