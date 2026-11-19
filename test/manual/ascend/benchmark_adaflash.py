#!/usr/bin/env python3
"""Portable single-NPU AdaFlash benchmark supporting AIME2025, MATH-500, and GSM8K."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

METHODS = ("ar", "dflash", "adaflash-dynamic", "adaflash-fixed")
DATASETS = ("aime", "math500", "gsm8k")
DATASET_ALIASES = {
    "aime": "aime",
    "aime25": "aime",
    "aime2025": "aime",
    "math500": "math500",
    "math-500": "math500",
    "gsm8k": "gsm8k",
}

TARGET_MODEL_DEFAULT = "Qwen/Qwen3-8B"
DFLASH_MODEL_NON_THINKING = "z-lab/Qwen3-8B-DFlash-b16"
DFLASH_MODEL_THINKING = "z-lab/Qwen3-8B-DFlash-b16"
ADAFLASH_MODEL_NON_THINKING = "AdaFlash/Qwen3-8B-AdaFlash"
ADAFLASH_MODEL_THINKING = "AdaFlash/Qwen3-8B-AdaFlash-Thinking"

REVISIONS = {
    DFLASH_MODEL_NON_THINKING: "9b41424b7109f9c5413454f481b09a82b85333f4",
    ADAFLASH_MODEL_THINKING: "d2fd7eb77c2a9da7b2b04acecbc87a55a6aaecc4",
    ADAFLASH_MODEL_NON_THINKING: "7fa2e0719c89b73d8ae25c63e8cc5cdfd7f75fab",
}

DATASET_PRESETS = {
    "aime": {
        "hf_repo": "opencompass/AIME2025",
        "hf_revision": "a6ad95f611d72cf628a80b58bd0432ef6638f958",
        "local_candidates": (
            Path("/data/adaflash/datasets/aime2025-questions.jsonl"),
        ),
        "thinking": True,
        "limit": 30,
        "concurrency": 8,
        "max_new_tokens": 32768,
        "context_length": 40960,
        "mem_fraction_static": 0.90,
    },
    "math500": {
        "hf_repo": "HuggingFaceH4/MATH-500",
        "hf_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "local_candidates": (
            Path("/data/adaflash/datasets/math500-test.jsonl"),
            Path("/data/adaflash/datasets/math500-test-first128.jsonl"),
        ),
        "thinking": True,
        "limit": 500,
        "concurrency": 32,
        "max_new_tokens": 32768,
        "context_length": 40960,
        "mem_fraction_static": 0.90,
    },
    "gsm8k": {
        "hf_repo": "openai/gsm8k",
        "hf_revision": "e53f048856ff4f594e959d75785d2c2d37b678ee",
        "local_candidates": (
            Path("/data/adaflash/datasets/gsm8k-test.jsonl"),
        ),
        "thinking": False,
        "limit": 1319,
        "concurrency": 64,
        "max_new_tokens": 2048,
        "context_length": 4096,
        "mem_fraction_static": 0.80,
    },
}

LOCAL_MODEL_DEFAULTS = {
    "target": Path("/data/adaflash/models/Qwen3-8B"),
    "dflash_thinking": Path("/data/adaflash/models/Qwen3-8B-DFlash-Thinking"),
    "dflash_non_thinking": Path("/data/adaflash/models/Qwen3-8B-DFlash-b16"),
    "adaflash_thinking": Path("/data/adaflash/models/Qwen3-8B-AdaFlash-Thinking"),
    "adaflash_non_thinking": Path("/data/adaflash/models/Qwen3-8B-AdaFlash"),
}

INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def http(url, payload=None, *, timeout=30, json_response=True):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error
    return json.loads(body) if json_response else body


def default_model_path(kind: str, thinking: bool) -> str:
    if kind == "target":
        local = LOCAL_MODEL_DEFAULTS["target"]
        return str(local) if local.is_dir() else TARGET_MODEL_DEFAULT
    suffix = "thinking" if thinking else "non_thinking"
    local = LOCAL_MODEL_DEFAULTS[f"{kind}_{suffix}"]
    if local.is_dir():
        return str(local)
    if kind == "dflash":
        fallback_local = LOCAL_MODEL_DEFAULTS["dflash_non_thinking"]
        if fallback_local.is_dir():
            return str(fallback_local)
        return DFLASH_MODEL_THINKING if thinking else DFLASH_MODEL_NON_THINKING
    return ADAFLASH_MODEL_THINKING if thinking else ADAFLASH_MODEL_NON_THINKING


def resolve_model(model, revision=None):
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision or REVISIONS.get(model))


def _normalize_row(row: dict, index: int, prefix: str) -> dict:
    question = row.get("question")
    if question is None:
        question = row.get("problem")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Row {index} is missing a non-empty 'question' or 'problem'")
    row_id = row.get("id") or row.get("unique_id") or f"{prefix}-{index:04d}"
    return {
        **row,
        "id": str(row_id),
        "question": question,
    }


def _load_jsonl(path: Path, prefix: str) -> list[dict]:
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [_normalize_row(row, i, prefix) for i, row in enumerate(raw, 1)]


def _download_aime(preset: dict) -> list[dict]:
    from huggingface_hub import hf_hub_download

    rows = []
    for split in ("I", "II"):
        path = hf_hub_download(
            preset["hf_repo"],
            f"aime2025-{split}.jsonl",
            repo_type="dataset",
            revision=preset["hf_revision"],
        )
        part = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
        if len(part) != 15:
            raise ValueError(f"Expected 15 questions in AIME2025-{split}")
        for i, row in enumerate(part, 1):
            rows.append(
                _normalize_row(
                    {**row, "id": f"AIME2025-{split}-{i:02d}", "split": split},
                    i,
                    f"AIME2025-{split}",
                )
            )
    return rows


def _download_math500(preset: dict) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        preset["hf_repo"],
        "test.jsonl",
        repo_type="dataset",
        revision=preset["hf_revision"],
    )
    part = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    return [
        _normalize_row(
            {**row, "id": f"MATH500-{i:04d}", "split": "test"},
            i,
            "MATH500",
        )
        for i, row in enumerate(part)
    ]


def _download_gsm8k(preset: dict) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        preset["hf_repo"],
        "main/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=preset["hf_revision"],
    )
    part = pq.read_table(path).to_pylist()
    return [
        _normalize_row(
            {**row, "id": f"GSM8K-{i:04d}", "split": "test"},
            i,
            "GSM8K",
        )
        for i, row in enumerate(part, 1)
    ]


def load_questions(
    dataset: str = "aime",
    dataset_path: Path | None = None,
    *,
    limit: int | None = None,
) -> tuple[list[dict], str, str | None]:
    preset = DATASET_PRESETS[dataset]
    prefix = dataset.upper()

    if dataset_path is not None:
        resolved = Path(dataset_path).expanduser().resolve()
        rows = _load_jsonl(resolved, prefix)
        source = str(resolved)
        revision = None
    else:
        rows = None
        source = preset["hf_repo"]
        revision = preset["hf_revision"]
        for candidate in preset["local_candidates"]:
            if candidate.is_file():
                loaded = _load_jsonl(candidate, prefix)
                if limit is None or len(loaded) >= limit:
                    rows = loaded
                    source = str(candidate.resolve())
                    revision = None
                    break
        if rows is None:
            if dataset == "aime":
                rows = _download_aime(preset)
            elif dataset == "math500":
                rows = _download_math500(preset)
            elif dataset == "gsm8k":
                rows = _download_gsm8k(preset)
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset must contain unique question IDs")
    return rows, source, revision


def prepare_prompts(tokenizer, rows, max_new_tokens, context_length, *, thinking=True):
    prepared = []
    for row in rows:
        messages = [{"role": "user", "content": row["question"] + INSTRUCTION}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        alternate = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not thinking,
        )
        if prompt == alternate or ("</think>" in prompt[-100:]) == thinking:
            raise ValueError(
                "The tokenizer did not apply the requested Qwen3 thinking mode"
            )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) + max_new_tokens >= context_length:
            raise ValueError(f"Context length is too small for {row['id']}")
        prepared.append(
            {
                **row,
                "input_ids": ids,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    return prepared


def build_server_command(args, models, method, probe=False):
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        models["target"],
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--dtype",
        "bfloat16",
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--context-length",
        str(args.context_length),
        "--max-running-requests",
        str(args.concurrency),
        "--chunked-prefill-size",
        "2048",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-bs-decode",
        *map(str, range(1, args.concurrency + 1)),
        "--cuda-graph-backend-prefill",
        "disabled",
        "--disable-overlap-schedule",
        "--return-output-ids",
        "--random-seed",
        "42",
    ]
    if method != "ar":
        draft = models["dflash"] if method == "dflash" else models["adaflash"]
        command += [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            draft,
            "--speculative-num-draft-tokens",
            "16",
            "--speculative-draft-attention-backend",
            "ascend",
        ]
    if probe:
        command += ["--decode-log-interval", "1"]
    return command


class Server:
    def __init__(self, args, models):
        self.args, self.models = args, models
        self.process = None
        self.log = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.args.port}"

    def start(self, method, prefix, probe=False):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.args.port))
        env = os.environ.copy()
        resolved = Path(__file__).resolve()
        repo_python = (
            resolved.parents[3] / "python" if len(resolved.parents) > 3 else None
        )
        if repo_python is not None and repo_python.is_dir():
            env["PYTHONPATH"] = (
                str(repo_python) + os.pathsep + env.get("PYTHONPATH", "")
            )
        env["SGLANG_RAGGED_VERIFY_MODE"] = (
            "compact" if method == "adaflash-dynamic" else "static"
        )
        if self.args.gamma is not None:
            env["ADAFLASH_LENGTH_SCALE"] = str(self.args.gamma)
        if self.args.min_dynamic_bs is not None:
            env["ADAFLASH_MIN_DYNAMIC_BS"] = str(self.args.min_dynamic_bs)
        env["ASCEND_USE_PAGED_MTP"] = "0"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env.setdefault("OMP_NUM_THREADS", "8")
        env.pop("SGLANG_LOG_DECODE_GRAPH_KEY", None)
        if probe:
            env["SGLANG_LOG_DECODE_GRAPH_KEY"] = "1"
        command = build_server_command(self.args, self.models, method, probe)
        write_json(self.args.output_dir / f"{prefix}-server-command.json", command)
        self.log = (self.args.output_dir / f"{prefix}-server.log").open("x")
        self.process = subprocess.Popen(
            command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Server exited; inspect {prefix}-server.log")
            try:
                http(self.url + "/health", timeout=3, json_response=False)
                info = http(self.url + "/server_info")
                write_json(self.args.output_dir / f"{prefix}-server-info.json", info)
                return
            except (OSError, RuntimeError):
                time.sleep(5)
        raise TimeoutError(f"Server startup exceeded {self.args.startup_timeout}s")

    def stop(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=15)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()
                self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


def evaluate(args, url, tokenizer, rows, method, prefix, probe=False):
    limit = 64 if probe else args.max_new_tokens
    selected = rows[: args.concurrency] if probe else rows
    root = args.output_dir
    raw = root / f"{prefix}-raw"
    raw.mkdir()

    def generate(ids, count):
        start = time.perf_counter()
        response = http(
            url + "/generate",
            {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "max_new_tokens": count,
                    "skip_special_tokens": False,
                    "ignore_eos": args.ignore_eos,
                },
            },
            timeout=args.request_timeout,
        )
        elapsed = time.perf_counter() - start
        if "error" in response:
            raise RuntimeError(response["error"])
        return response, elapsed

    warmups = []
    for i in range(args.concurrency):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Compute {i + 1} plus 2."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )
        warmups.append(tokenizer.encode(prompt, add_special_tokens=False))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(lambda ids: generate(ids, 32), warmups))
    http(url + "/flush_cache", {}, timeout=60, json_response=False)

    def one(row):
        try:
            response, elapsed = generate(row["input_ids"], limit)
            safe_id = row["id"].replace("/", "_")
            write_json(
                raw / f"{safe_id}.json", {"response": response, "seconds": elapsed}
            )
            ids = response.get("output_ids")
            if not isinstance(ids, list) or not ids:
                raise RuntimeError("Response did not contain output_ids")
            return {
                "id": row["id"],
                "split": row.get("split"),
                "prompt_sha256": row["prompt_sha256"],
                "request_seconds": elapsed,
                "response": response,
            }
        except Exception as error:
            return {"id": row["id"], "error": repr(error)}

    results = []
    start = time.perf_counter()
    with (root / f"{prefix}-responses.jsonl").open("x") as stream:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = [pool.submit(one, row) for row in selected]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    f"{prefix}: {len(results)}/{len(selected)} {result['id']} "
                    f"{result.get('error', 'done')}",
                    flush=True,
                )
    wall = time.perf_counter() - start
    valid = [row for row in results if "error" not in row]
    metadata = [row["response"]["meta_info"] for row in valid]
    tokens = sum(row["completion_tokens"] for row in metadata)
    verifies = sum(row.get("spec_verify_ct", 0) for row in metadata)
    accepted = sum(row.get("spec_num_correct_drafts", 0) for row in metadata)
    summary = {
        "dataset": args.dataset,
        "method": method,
        "gamma": args.gamma
        if method == "adaflash-dynamic"
        else None,
        "min_dynamic_bs": args.min_dynamic_bs
        if method == "adaflash-dynamic"
        else None,
        "questions": len(selected),
        "valid_requests": len(valid),
        "errors": len(results) - len(valid),
        "concurrency": args.concurrency,
        "enable_thinking": args.thinking,
        "ignore_eos": args.ignore_eos,
        "temperature": 0,
        "max_new_tokens": limit,
        "wall_seconds": wall,
        "output_tokens": tokens,
        "output_tokens_per_second": tokens / wall,
        "mean_request_seconds": statistics.mean(row["request_seconds"] for row in valid)
        if valid
        else None,
        "spec_verify_ct": verifies,
        "mean_accepted_tokens_with_bonus": accepted / verifies + 1
        if verifies
        else None,
        "length_limit_count": sum(
            row.get("finish_reason", {}).get("type") == "length" for row in metadata
        ),
        "retractions": sum(row.get("num_retractions", 0) for row in metadata),
    }
    write_json(root / f"{prefix}-summary.json", summary)
    if summary["errors"]:
        raise RuntimeError(
            f"{prefix}: {summary['errors']} failed requests; raw results preserved"
        )
    return summary


def graph_stats(path):
    lines = path.read_text(errors="replace").splitlines()
    flushes = [
        i for i, line in enumerate(lines) if "/flush_cache" in line and "200" in line
    ]
    if not flushes:
        raise RuntimeError(f"Missing benchmark start marker in {path}")
    decode = [line for line in lines[flushes[-1] + 1 :] if "Decode batch," in line]
    result = {
        "decode_graph_hits_logged": sum("npu graph: True" in line for line in decode),
        "decode_graph_misses_logged": sum(
            "npu graph: False" in line for line in decode
        ),
    }
    if result["decode_graph_misses_logged"]:
        raise RuntimeError(f"Eager decode fallback detected in {path}")
    return result


def write_comparison(root, summaries):
    baseline = next(
        (s["output_tokens_per_second"] for s in summaries if s["method"] == "ar"), None
    )
    fixed = next(
        (
            s["output_tokens_per_second"]
            for s in summaries
            if s["method"] == "adaflash-fixed"
        ),
        None,
    )
    lines = [
        "| 方法 | 吞吐（token/s） | 相对 AR 加速 | 相对 Fixed 加速 | 平均接受长度 |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        vs_fixed = (
            summary["output_tokens_per_second"] / fixed if fixed is not None else None
        )
        summary["speedup_vs_adaflash_fixed"] = vs_fixed
        speed = summary["output_tokens_per_second"] / baseline if baseline else None
        summary["speedup_vs_ar"] = speed
        speed_text = f"{speed:.2f}×" if speed is not None else "—"
        fixed_text = f"{vs_fixed:.2f}×" if vs_fixed is not None else "—"
        accept = summary["mean_accepted_tokens_with_bonus"]
        accept_text = f"{accept:.2f}" if accept is not None else "—"
        lines.append(
            f"| {summary['method']} | {summary['output_tokens_per_second']:.2f} "
            f"| {speed_text} | {fixed_text} | {accept_text} |"
        )
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    write_json(root / "comparison.json", summaries)


def _normalize_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        raise argparse.ArgumentTypeError(
            f"Unsupported dataset {name!r}; expected one of {DATASETS}"
        )
    return DATASET_ALIASES[key]


def parse_args(argv=None, *, default_dataset: str = "aime"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=_normalize_dataset_name,
        default=default_dataset,
        help="Benchmark dataset preset: aime, math500, or gsm8k",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Optional local JSONL with question/problem fields (overrides dataset source)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Target HF ID or local directory (auto-detects local Qwen3-8B if present)",
    )
    parser.add_argument(
        "--dflash-model",
        default=None,
        help="DFlash HF ID or local directory (auto-selected by thinking mode)",
    )
    parser.add_argument(
        "--adaflash-model",
        default=None,
        help="AdaFlash HF ID or local directory (auto-selected by thinking mode)",
    )
    parser.add_argument("--model-revision")
    parser.add_argument("--dflash-revision")
    parser.add_argument("--adaflash-revision")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-dir", type=Path, default=None)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode",
    )
    thinking_group.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode",
    )
    parser.set_defaults(thinking=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Override ADAFLASH_LENGTH_SCALE for adaflash-dynamic (default: env or 1.2)",
    )
    parser.add_argument(
        "--min-dynamic-bs",
        type=int,
        default=None,
        help="Override ADAFLASH_MIN_DYNAMIC_BS for adaflash-dynamic (default: env or 12)",
    )
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=21600)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    preset = DATASET_PRESETS[args.dataset]
    if args.thinking is None:
        args.thinking = preset["thinking"]
    if args.concurrency is None:
        args.concurrency = preset["concurrency"]
    if args.max_new_tokens is None:
        args.max_new_tokens = preset["max_new_tokens"]
    if args.context_length is None:
        args.context_length = preset["context_length"]
    if args.mem_fraction_static is None:
        args.mem_fraction_static = preset["mem_fraction_static"]
    if args.limit is None:
        args.limit = preset["limit"]
    if args.model is None:
        args.model = default_model_path("target", args.thinking)
    if args.dflash_model is None:
        args.dflash_model = default_model_path("dflash", args.thinking)
    if args.adaflash_model is None:
        args.adaflash_model = default_model_path("adaflash", args.thinking)
    if args.output_dir is None:
        args.output_dir = Path("results") / time.strftime(
            f"adaflash-{args.dataset}-%Y%m%d-%H%M%S"
        )

    if (
        min(
            args.concurrency,
            args.max_new_tokens,
            args.limit,
            args.startup_timeout,
            args.request_timeout,
        )
        < 1
    ):
        parser.error("Counts and timeouts must be positive")
    if args.context_length <= args.max_new_tokens:
        parser.error("--context-length must leave room for the prompt")
    if not 0 < args.mem_fraction_static < 1:
        parser.error("--mem-fraction-static must be between 0 and 1")
    if args.gamma is not None and args.gamma <= 0:
        parser.error("--gamma must be positive")
    if args.min_dynamic_bs is not None and args.min_dynamic_bs < 0:
        parser.error("--min-dynamic-bs must be non-negative")
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods cannot contain duplicates")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv=None, *, default_dataset: str = "aime"):
    args = parse_args(argv, default_dataset=default_dataset)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            "Use an empty --output-dir; existing results are never overwritten"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions, dataset_source, dataset_revision = load_questions(
        args.dataset,
        args.dataset_path,
        limit=args.limit,
    )
    if args.limit > len(questions):
        raise ValueError(
            f"Requested {args.limit} questions, dataset contains {len(questions)}"
        )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        dataset_preset=args.dataset,
        dataset=dataset_source,
        dataset_revision=dataset_revision,
        enable_thinking=args.thinking,
        temperature=0,
    )
    write_json(args.output_dir / "config.json", config)
    server = Server(args, {})
    summaries = []

    def status(stage, **extra):
        value = {"stage": stage, "updated_unix": time.time(), **extra}
        write_json(args.output_dir / "status.json", value)
        print(json.dumps(value), flush=True)

    def interrupted(signum, _frame):
        server.stop()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        status("prepare")
        models = {"target": resolve_model(args.model, args.model_revision)}
        if "dflash" in args.methods:
            models["dflash"] = resolve_model(args.dflash_model, args.dflash_revision)
        if any(method.startswith("adaflash") for method in args.methods):
            models["adaflash"] = resolve_model(
                args.adaflash_model, args.adaflash_revision
            )
        server.models = models
        write_json(args.output_dir / "resolved-models.json", models)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(models["target"])
        rows = prepare_prompts(
            tokenizer,
            questions[: args.limit],
            args.max_new_tokens,
            args.context_length,
            thinking=args.thinking,
        )
        write_json(args.output_dir / "prompts.json", rows)
        for probe in [True, False] if not args.skip_preflight else [False]:
            for method in args.methods:
                prefix = f"preflight-{method}" if probe else method
                status("preflight" if probe else "startup", method=method)
                server.start(method, prefix, probe)
                status("preflight" if probe else "benchmark", method=method)
                summary = evaluate(
                    args, server.url, tokenizer, rows, method, prefix, probe
                )
                proof = graph_stats(args.output_dir / f"{prefix}-server.log")
                if probe and not proof["decode_graph_hits_logged"]:
                    raise RuntimeError(f"No Graph decode proof in {prefix}")
                write_json(args.output_dir / f"{prefix}-graph-proof.json", proof)
                server.stop()
                if not probe:
                    summaries.append(summary)
                    write_comparison(args.output_dir, summaries)
        status("complete", report=str(args.output_dir / "comparison.md"))
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        server.stop()


if __name__ == "__main__":
    main()
