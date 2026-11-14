#!/usr/bin/env python3
"""Portable single-NPU, thinking-enabled AIME2025 performance comparison."""

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
DFLASH_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
ADAFLASH_MODEL = "AdaFlash/Qwen3-8B-AdaFlash-Thinking"
REVISIONS = {
    DFLASH_MODEL: "9b41424b7109f9c5413454f481b09a82b85333f4",
    ADAFLASH_MODEL: "d2fd7eb77c2a9da7b2b04acecbc87a55a6aaecc4",
}
DATASET = "opencompass/AIME2025"
DATASET_REVISION = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
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


def resolve_model(model, revision=None):
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision or REVISIONS.get(model))


def load_questions(dataset_path=None):
    if dataset_path:
        rows = [json.loads(s) for s in Path(dataset_path).read_text().splitlines() if s]
    else:
        from huggingface_hub import hf_hub_download

        rows = []
        for split in ("I", "II"):
            path = hf_hub_download(
                DATASET,
                f"aime2025-{split}.jsonl",
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
            part = [json.loads(s) for s in Path(path).read_text().splitlines() if s]
            if len(part) != 15:
                raise ValueError(f"Expected 15 questions in AIME2025-{split}")
            for i, row in enumerate(part, 1):
                rows.append({**row, "id": f"AIME2025-{split}-{i:02d}", "split": split})
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset must contain unique question IDs")
    return rows


def prepare_prompts(tokenizer, rows, max_new_tokens, context_length):
    prepared = []
    for row in rows:
        messages = [{"role": "user", "content": row["question"] + INSTRUCTION}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        disabled = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if prompt == disabled or "</think>" in prompt[-100:]:
            raise ValueError("The tokenizer did not enable the Qwen3 thinking template")
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
        repo_python = Path(__file__).resolve().parents[3] / "python"
        env["PYTHONPATH"] = str(repo_python) + os.pathsep + env.get("PYTHONPATH", "")
        env["SGLANG_RAGGED_VERIFY_MODE"] = (
            "compact" if method == "adaflash-dynamic" else "static"
        )
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
            enable_thinking=True,
        )
        warmups.append(tokenizer.encode(prompt, add_special_tokens=False))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(lambda ids: generate(ids, 32), warmups))
    http(url + "/flush_cache", {}, timeout=60, json_response=False)

    def one(row):
        try:
            response, elapsed = generate(row["input_ids"], limit)
            write_json(
                raw / f"{row['id']}.json", {"response": response, "seconds": elapsed}
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
        "method": method,
        "questions": len(selected),
        "valid_requests": len(valid),
        "errors": len(results) - len(valid),
        "concurrency": args.concurrency,
        "enable_thinking": True,
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
    lines = [
        "| 方法 | 吞吐（token/s） | 相对 AR 加速 | 平均接受长度 |",
        "|---|---:|---:|---:|",
    ]
    for summary in summaries:
        speed = summary["output_tokens_per_second"] / baseline if baseline else None
        summary["speedup_vs_ar"] = speed
        speed_text = f"{speed:.2f}×" if speed is not None else "—"
        accept = summary["mean_accepted_tokens_with_bonus"]
        accept_text = f"{accept:.2f}" if accept is not None else "—"
        lines.append(
            f"| {summary['method']} | {summary['output_tokens_per_second']:.2f} | {speed_text} | {accept_text} |"
        )
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    write_json(root / "comparison.json", summaries)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B", help="Target HF ID or local directory"
    )
    parser.add_argument(
        "--dflash-model", default=DFLASH_MODEL, help="DFlash HF ID or local directory"
    )
    parser.add_argument(
        "--adaflash-model",
        default=ADAFLASH_MODEL,
        help="Thinking draft HF ID or local directory",
    )
    parser.add_argument("--model-revision")
    parser.add_argument("--dflash-revision")
    parser.add_argument("--adaflash-revision")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / time.strftime("adaflash-aime25-%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Optional local AIME JSONL with id and question fields",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--context-length", type=int, default=40960)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=21600)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)
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
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods cannot contain duplicates")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            "Use an empty --output-dir; existing results are never overwritten"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        dataset=DATASET,
        dataset_revision=DATASET_REVISION,
        enable_thinking=True,
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
        questions = load_questions(args.dataset_path)
        if args.limit > len(questions):
            raise ValueError(
                f"Requested {args.limit} questions, dataset contains {len(questions)}"
            )
        rows = prepare_prompts(
            tokenizer, questions[: args.limit], args.max_new_tokens, args.context_length
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
