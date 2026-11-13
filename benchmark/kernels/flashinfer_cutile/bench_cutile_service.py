"""Reproducible serving and memory-saver checks against an existing server.

This driver never starts a model server or downloads weights/tokenizers. Run it
on a dedicated benchmark instance: serving runs flush the prefix cache, and the
explicit lifecycle mode releases and restores weights and KV cache. Random-ID
throughput and deterministic lifecycle checks are not a model-quality evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS = (
    "completed",
    "duration",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
    "output_throughput",
    "request_throughput",
)
DEFAULT_PROMPTS = [
    "Explain why database indexes can speed up a query in three sentences.",
    "What is 17 multiplied by 23? Show the calculation.",
    "Write a Python function that returns the larger of two integers.",
]


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _capture(argv):
    """Diagnostic failures remain visible without hiding benchmark failures."""
    try:
        result = subprocess.run(
            argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=False
        )
        return {
            "argv": argv,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc)}


def _metadata(args):
    diff = _capture(["git", "--no-optional-locks", "diff", "HEAD", "--binary"])
    packages = {}
    for name in (
        "sglang",
        "torch",
        "flashinfer-python",
        "cuda-tile",
        "nvidia-cuda-tileiras",
        "nvidia-cuda-nvcc",
        "nvidia-nvvm",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "settings": vars(args),
        "client_git_head": _capture(
            ["git", "--no-optional-locks", "rev-parse", "HEAD"]
        ),
        "client_git_status": _capture(
            ["git", "--no-optional-locks", "status", "--porcelain"]
        ),
        "client_tracked_diff_sha256": hashlib.sha256(
            diff.get("stdout", "").encode()
        ).hexdigest(),
        "client_packages": packages,
        "client_python": sys.version,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_module_sha256": hashlib.sha256(
            (REPO_ROOT / "python/sglang/benchmark/serving.py").read_bytes()
        ).hexdigest(),
        "client_gpu_inventory": _capture(["nvidia-smi", "-q"]),
        "client_gpu_topology": _capture(["nvidia-smi", "topo", "-m"]),
        "provenance_note": (
            "Client metadata is not server metadata. The server revision and "
            "FlashInfer variant are operator-supplied labels; archive the server "
            "environment, source diff and launch log alongside this directory."
        ),
    }


class ServerResponseError(RuntimeError):
    def __init__(self, status, body):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


class Server:
    def __init__(self, base_url, timeout):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def call(self, path, payload=None):
        # Match the benchmark client's existing authentication environment.
        headers = {"Content-Type": "application/json"}
        if os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
        elif os.environ.get("API_KEY"):
            headers["Authorization"] = os.environ["API_KEY"]
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as exc:
            # Preserve status AND body: a busy preflight rejection and a partial
            # scheduler failure can both use 400, but need different recovery.
            raise ServerResponseError(exc.code, exc.read().decode()) from exc
        try:
            value = json.loads(body) if body else None
        except json.JSONDecodeError:
            value = body
        if isinstance(value, dict) and (
            value.get("success") is False or "error" in value
        ):
            raise ServerResponseError(200, value)
        if path.split("?", 1)[0] == "/flush_cache" and (
            not isinstance(value, str) or not value.startswith("Cache flushed.")
        ):
            raise RuntimeError(f"{path} did not confirm a cache flush: {value}")
        return value


def _validate_server(info, args):
    if not isinstance(info, dict):
        raise ValueError("/server_info did not return an object")
    if info.get("moe_runner_backend") != args.expected_backend:
        raise ValueError(
            "The running server's MoE backend does not match --expected-backend"
        )
    if args.mode in ("lifecycle", "all"):
        # Without CPU backup, resume can reallocate uninitialized weight memory.
        # Refuse before even generating the baseline or releasing any resource.
        for flag in ("enable_memory_saver", "enable_weights_cpu_backup"):
            if info.get(flag) is not True:
                raise ValueError(f"Lifecycle mode requires server {flag}=true")

        if info.get("dp_size") != 1:
            raise ValueError("Lifecycle mode requires an explicitly reported dp_size=1")


def _validate_model(info, args):
    if not isinstance(info, dict) or args.model not in (
        info.get("served_model_name"),
        info.get("model_path"),
    ):
        raise ValueError("--model does not match the running server's model identity")
    return tuple(
        info.get(key) for key in ("model_path", "served_model_name", "weight_version")
    )


def _recovery_request(tags):
    return {
        "method": "POST",
        "path": "/resume_memory_occupation",
        "body": tags,
        "instruction": "Inspect server logs and pending operations before retrying "
        "this request; the server may require a restart.",
    }


def _confirmed_release_rejection(error, server_info):
    if not isinstance(error, ServerResponseError):
        return False
    if error.status in (401, 403, 404, 405, 422):
        return True
    # This exact idle-gate message precedes offload_tags mutation. Restrict
    # this interpretation to DP1; multiple DP workers may have mixed outcomes.
    return (
        server_info is not None
        and server_info.get("dp_size") == 1
        and error.status in (200, 400)
        and "Memory was not released because requests are still running or queued."
        in str(error.body)
    )


def _signature(response, *, expected_tokens=None):
    if not isinstance(response, dict):
        raise ValueError("Expected a single non-streaming generation response")
    ids = response.get("output_ids")
    meta = response.get("meta_info", {})
    if not isinstance(meta, dict):
        raise ValueError("Generation response must contain a meta_info object")
    # Native /generate can return HTTP 200 for an abort without a status code.
    # Matching partial outputs before and after resume do not prove a complete
    # decode worked. These requests ignore EOS and must finish at their limit.
    finish_reason = meta.get("finish_reason")
    if not isinstance(finish_reason, dict) or finish_reason.get("type") != "length":
        raise ValueError(
            f"Lifecycle generation must finish by length, got {finish_reason!r}"
        )
    logprobs = meta.get("output_token_logprobs")
    if (
        not isinstance(ids, list)
        or not ids
        or not isinstance(logprobs, list)
        or len(logprobs) != len(ids)
    ):
        raise ValueError(
            "Response must contain nonempty output_ids and one logprob per token"
        )
    if expected_tokens is not None and len(ids) != expected_tokens:
        raise ValueError(
            f"Lifecycle generation returned {len(ids)} tokens; "
            f"expected {expected_tokens}"
        )
    if meta.get("completion_tokens") != len(ids):
        raise ValueError("completion_tokens does not match output_ids length")
    values = []
    for token_id, row in zip(ids, logprobs):
        if (
            not isinstance(row, (list, tuple))
            or len(row) < 2
            or row[1] != token_id
            or not isinstance(row[0], (int, float))
            or not math.isfinite(row[0])
        ):
            raise ValueError("Malformed or non-finite output logprob")
        values.append(float(row[0]))
    return ids, values


def _compare_generations(before, after, atol):
    if len(before) != len(after):
        raise AssertionError("Generation count changed")
    maximum = 0.0
    for first, second in zip(before, after):
        first_ids, first_logprobs = _signature(first)
        second_ids, second_logprobs = _signature(second)
        if first_ids != second_ids:
            raise AssertionError("Output token IDs changed after weight release/resume")
        if first.get("text") != second.get("text"):
            raise AssertionError("Decoded text changed after weight release/resume")
        error = max(abs(x - y) for x, y in zip(first_logprobs, second_logprobs))
        maximum = max(maximum, error)
        if error > atol:
            raise AssertionError(f"Output logprob error {error} exceeds {atol}")
    return {"output_ids_equal": True, "max_abs_logprob_error": maximum}


def _lifecycle(server, args, output_dir, *, server_info=None):
    payloads = [
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 64,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
            "stream": False,
        }
        for prompt in DEFAULT_PROMPTS
    ]

    def generate():
        responses = []
        for payload in payloads:
            # Force prefill after restoring weights, not a hit in the old KV
            # cache. The baseline uses the same cold-cache request sequence.
            server.call("/flush_cache?timeout=60", {})
            response = server.call("/generate", payload)
            _signature(
                response, expected_tokens=payload["sampling_params"]["max_new_tokens"]
            )
            responses.append(response)
        return responses

    before = generate()
    _write_json(output_dir / "lifecycle-before.json", before)
    # Check deterministic service behavior before blaming release/resume.
    repeated = generate()
    _compare_generations(before, repeated, args.logprob_atol)
    server.call("/flush_cache?timeout=60", {})
    tags = {"tags": ["weights", "kv_cache"]}
    cycle = {"release": "attempted", "resume": "not_attempted"}
    release_error = None
    should_resume = True
    try:
        server.call("/release_memory_occupation", tags)
        cycle["release"] = "confirmed"
        _write_json(
            output_dir / "lifecycle-released-server-info.json",
            server.call("/server_info"),
        )
    except Exception as exc:
        release_error = exc
        cycle["release_or_diagnostic_error"] = repr(exc)
        if cycle["release"] != "confirmed" and isinstance(exc, ServerResponseError):
            should_resume = False
            if _confirmed_release_rejection(exc, server_info):
                cycle["release"] = "rejected"
            else:
                # Generic HTTP 400 can represent a partial scheduler failure,
                # not merely a preflight refusal. Do not guess which tags exist.
                cycle["release"] = "failed_state_unknown"
                cycle["recovery"] = _recovery_request(tags)
    finally:
        # Resume once after confirmed release, or an uncertain transport error.
        # A definite rejection must not resume: the scheduler removes offload
        # tags unconditionally, so resuming absent tags can itself fail.
        if should_resume:
            cycle["resume"] = "attempted"
            try:
                server.call("/resume_memory_occupation", tags)
                cycle["resume"] = "confirmed"
            except Exception as exc:
                cycle["resume_error"] = repr(exc)
                cycle["recovery"] = _recovery_request(tags)
                _write_json(output_dir / "lifecycle-resource-cycle.json", cycle)
                raise RuntimeError(
                    "Memory-saver resume failed; inspect lifecycle-resource-cycle.json. "
                    f"Release/diagnostic error: {release_error!r}; resume error: {exc!r}"
                ) from exc
        _write_json(output_dir / "lifecycle-resource-cycle.json", cycle)
    if release_error is not None:
        raise release_error
    after = generate()
    _write_json(output_dir / "lifecycle-after.json", after)
    result = _compare_generations(before, after, args.logprob_atol)
    result.update(
        {
            "prompts": DEFAULT_PROMPTS,
            "tags": tags["tags"],
            "logprob_atol": args.logprob_atol,
            "scope": "release/resume API and cold-cache generation consistency only; "
            "not a quality evaluation or a measurement of released GPU bytes",
            "memory_note": "/server_info memory fields may describe startup allocation "
            "rather than live usage. Collect independent server-side NVML snapshots "
            "before release, while released and after resume to quantify reclamation.",
        }
    )
    _write_json(output_dir / "lifecycle-result.json", result)


def _commands(args, output_dir):
    commands = []
    for input_len in args.input_lens:
        for output_len in args.output_lens:
            for concurrency in args.concurrencies:
                name = f"in{input_len}-out{output_len}-c{concurrency}"
                argv = [
                    sys.executable,
                    "-m",
                    "sglang.benchmark.serving",
                    "--backend",
                    "sglang",
                    "--base-url",
                    args.base_url,
                    "--model",
                    args.model,
                    "--tokenizer",
                    args.tokenizer,
                    "--dataset-name",
                    "random-ids",
                    "--tokenize-prompt",
                    "--random-input-len",
                    str(input_len),
                    "--random-output-len",
                    str(output_len),
                    # range_ratio=1 makes both lengths fixed in this API.
                    "--random-range-ratio",
                    "1",
                    "--num-prompts",
                    str(args.num_prompts),
                    "--max-concurrency",
                    str(concurrency),
                    "--seed",
                    str(args.seed),
                    "--temperature",
                    "0",
                    "--warmup-requests",
                    str(max(args.warmup_requests, concurrency)),
                    "--flush-cache",
                    "--flush-cache-timeout",
                    "60",
                    "--output-details",
                    "--disable-tqdm",
                    "--cache-report",
                    "--output-file",
                    str(output_dir / f"{name}.jsonl"),
                ]
                pilot_argv = argv.copy()
                pilot_argv[-1] = str(output_dir / f"{name}-pilot.jsonl")
                commands.append(
                    {
                        "name": name,
                        "input_len": input_len,
                        "output_len": output_len,
                        "concurrency": concurrency,
                        "argv": argv,
                        "pilot_argv": pilot_argv,
                    }
                )
    return commands


def _validate_result(result, case, num_prompts, args=None):
    if args is not None:
        _validate_server(result.get("server_info"), args)
    if result.get("completed") != num_prompts:
        raise AssertionError(
            "Some benchmark requests failed; do not report partial throughput"
        )
    for key, expected in (
        ("input_lens", case["input_len"]),
        ("output_lens", case["output_len"]),
    ):
        values = result.get(key)
        if (
            not isinstance(values, list)
            or len(values) != num_prompts
            or any(value != expected for value in values)
        ):
            raise AssertionError(
                f"{key} does not match the requested fixed-length workload"
            )
    if any(result.get("errors", [])):
        raise AssertionError("Benchmark reported request errors")
    for name in METRICS:
        value = result.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise AssertionError(f"Invalid benchmark metric: {name}")


def _serving(args, output_dir, commands, server, model_info):
    rows = []
    environment = os.environ.copy()
    # A missing local tokenizer should fail instead of starting an unplanned
    # download. random-ids also avoids the random dataset's ShareGPT download.
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    environment["PYTHONPATH"] = (
        str(REPO_ROOT / "python") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    model_identity = _validate_model(model_info, args)
    for case in commands:
        # Run an entire unscored case before the measured case, including its
        # concurrency and decode workload. Keep pilot evidence separate. This
        # warms more service shapes than repeating one short warmup prompt, but
        # does not claim to cover every possible scheduler shape or future JIT.
        for stage, argv in (("pilot", case["pilot_argv"]), ("measured", case["argv"])):
            if _validate_model(server.call("/model_info"), args) != model_identity:
                raise AssertionError(
                    "The model identity/version changed during the run"
                )
            _validate_server(server.call("/server_info"), args)
            name = case["name"] + ("-pilot" if stage == "pilot" else "")
            print(shlex.join(argv), flush=True)
            log_path = output_dir / f"{name}.log"
            try:
                with log_path.open("w") as log:
                    # This bounds the whole benchmark client, including its
                    # internal HTTP calls. --http-timeout only bounds this
                    # driver's own control requests, not those of the child.
                    subprocess.run(
                        argv,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=args.case_timeout,
                    )
            except subprocess.TimeoutExpired as exc:
                # subprocess.run kills and waits for its child on timeout.
                # The separately launched model server is left running; stop
                # the matrix so its outstanding work can be inspected first.
                _write_json(
                    output_dir / f"{name}-timeout.json",
                    {
                        "status": "failed",
                        "case": case["name"],
                        "stage": stage,
                        "timeout_seconds": args.case_timeout,
                        "argv": argv,
                        "log_path": str(log_path),
                        "error": repr(exc),
                        "note": "Benchmark client timed out; inspect its log and "
                        "server requests before retrying. The server was not stopped.",
                    },
                )
                raise TimeoutError(
                    f"Serving case {case['name']} ({stage}) exceeded "
                    f"--case-timeout={args.case_timeout}s; see {log_path}. "
                    "The benchmark client was stopped; the server was not stopped."
                ) from exc
            lines = (output_dir / f"{name}.jsonl").read_text().splitlines()
            if len(lines) != 1:
                raise AssertionError(
                    "Expected exactly one result per benchmark process"
                )
            result = json.loads(lines[0])
            _validate_result(result, case, args.num_prompts, args)
            if _validate_model(server.call("/model_info"), args) != model_identity:
                raise AssertionError(
                    "The model identity/version changed during the run"
                )
        rows.append(
            {key: case[key] for key in ("input_len", "output_len", "concurrency")}
            | {key: result[key] for key in METRICS}
        )
        # Save completed cases incrementally; failures retain all raw evidence.
        with (output_dir / "serving-summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer", required=True, help="Existing local tokenizer directory"
    )
    parser.add_argument("--output-dir", required=True, help="Must not exist")
    parser.add_argument("--implementation-label", required=True)
    parser.add_argument(
        "--server-revision", required=True, help="Exact server Git commit"
    )
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument(
        "--flashinfer-variant",
        required=True,
        help="For example stock-0.7.3 or 0.7.3+9129@PATCH_SHA",
    )
    parser.add_argument("--expected-backend", default="flashinfer_cutile")
    parser.add_argument(
        "--mode", choices=("serving", "lifecycle", "all"), default="serving"
    )
    parser.add_argument("--input-lens", type=int, nargs="+", default=[128, 2048, 8192])
    parser.add_argument("--output-lens", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--http-timeout", type=float, default=600)
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=1800,
        help="Seconds allowed for each pilot or measured benchmark subprocess "
        "separately (default: 1800); does not stop the model server",
    )
    parser.add_argument("--logprob-atol", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; no file, network or subprocess operations",
    )
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    args.tokenizer = str(Path(args.tokenizer).expanduser().resolve())
    url = urlsplit(args.base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.username
        or url.password
    ):
        parser.error("--base-url must be an HTTP(S) URL without credentials")
    if url.query or url.fragment:
        parser.error("--base-url cannot contain a query or fragment")
    if any(
        value <= 0 for value in args.input_lens + args.output_lens + args.concurrencies
    ):
        parser.error("Lengths and concurrency must be positive")
    if args.num_prompts < max(args.concurrencies) or args.warmup_requests < 1:
        parser.error("Use at least one warmup and num-prompts >= max concurrency")
    if not math.isfinite(args.logprob_atol) or args.logprob_atol < 0:
        parser.error("--logprob-atol must be finite and nonnegative")
    if not math.isfinite(args.http_timeout) or args.http_timeout <= 0:
        parser.error("--http-timeout must be finite and positive")
    if not math.isfinite(args.case_timeout) or args.case_timeout <= 0:
        parser.error("--case-timeout must be finite and positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    commands = _commands(args, output_dir) if args.mode != "lifecycle" else []
    if args.dry_run:
        print(json.dumps({"settings": vars(args), "commands": commands}, indent=2))
        return
    if not Path(args.tokenizer).expanduser().is_dir():
        parser.error("--tokenizer must be an existing local directory")
    # Refuse an existing destination so reruns cannot append to an old result.
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "commands.json", commands)
    _write_json(output_dir / "metadata.json", _metadata(args))
    server = Server(args.base_url, args.http_timeout)
    started = time.monotonic()
    try:
        server.call("/health")
        info = server.call("/server_info")
        _write_json(output_dir / "server-info-before.json", info)
        model_info = server.call("/model_info")
        _write_json(output_dir / "model-info.json", model_info)
        _validate_model(model_info, args)
        _validate_server(info, args)
        if args.mode in ("lifecycle", "all"):
            _lifecycle(server, args, output_dir, server_info=info)
        if args.mode in ("serving", "all"):
            _serving(args, output_dir, commands, server, model_info)
        _write_json(output_dir / "server-info-after.json", server.call("/server_info"))
    except Exception as exc:
        _write_json(
            output_dir / "status.json",
            {
                "status": "failed",
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    _write_json(
        output_dir / "status.json",
        {
            "status": "passed",
            "elapsed_seconds": time.monotonic() - started,
            "scope": "requested checks only; no model-quality score",
        },
    )


if __name__ == "__main__":
    main()
