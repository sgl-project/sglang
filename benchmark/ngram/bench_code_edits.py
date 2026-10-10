"""Serial, isolated SGLang code-edit benchmark using the prepared TRT workload.

See README.md for token export, smoke/full runs, and measurement definitions.
This module deliberately imports neither SGLang nor CUDA in the client process.
"""

import argparse
import hashlib
import http.client
import importlib.metadata
import inspect
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from code_edit_report import build_summary, summarize_trial, trt_reference

REPO = Path(__file__).resolve().parents[2]
CORPUS = "current-prompt"
COUNTERS = ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts")
# (global allocation, legacy ranking, external prompt SAM)
MODES = {
    "baseline": (None, None, False),
    "legacy_bfs": ("disabled", "BFS", True),
    "legacy_prob": ("disabled", "PROB", True),
    "path_probability": ("path_probability", "PROB", True),
    "specificity_path_probability": ("specificity_path_probability", "PROB", True),
    "trie_only": ("disabled", "BFS", False),
}
DEFAULT_MODES = list(MODES)[:-1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def model_assets(model):
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "model.safetensors.index.json",
    )
    return {name: sha256(model / name) for name in names if (model / name).is_file()}


def load_workload(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    dataset = directory / "dataset.jsonl"
    for key in (
        "model",
        "model_revision",
        "dataset",
        "dataset_revision",
        "row_indices",
    ):
        if not manifest.get(key):
            raise ValueError(f"Manifest is missing {key}")
    if sha256(dataset) != manifest["dataset_sha256"]:
        raise ValueError("dataset.jsonl does not match the prepared manifest SHA256")
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if [r["task_id"] for r in rows] != manifest["row_indices"]:
        raise ValueError("Dataset order/IDs differ from manifest row_indices")
    if len({str(r["task_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate task IDs")
    if any(not isinstance(r["prompt"], str) or r["output_tokens"] != 256 for r in rows):
        raise ValueError("Expected raw prompts and the shared 256-token output budget")
    return manifest, rows


def export_trt_inputs(args):
    """Run in the same TRT container as bench.py; no inference is performed."""
    from tensorrt_llm.bench.utils.data import (
        create_dataset_from_stream,
        initialize_tokenizer,
    )

    manifest, rows = load_workload(args.inputs_dir)
    model = args.model_path or args.inputs_dir / "model"
    tokenizer = initialize_tokenizer(str(model.resolve()))
    with (args.inputs_dir / "dataset.jsonl").open() as stream:
        _, requests = create_dataset_from_stream(
            tokenizer, stream, num_requests=len(rows)
        )
    exported = []
    if len(requests) != len(rows):
        raise ValueError("TRT loader changed the request count")
    for row, request in zip(rows, requests):
        if (
            request.task_id != row["task_id"]
            or request.output_tokens != row["output_tokens"]
        ):
            raise ValueError("TRT loader changed request order or output length")
        exported.append({**row, "input_ids": list(request.input_ids)})
    artifact = {
        "schema_version": 1,
        "producer": "tensorrt_llm.bench.utils.data.create_dataset_from_stream",
        "packages": package_versions(["tensorrt_llm", "transformers", "tokenizers"]),
        "loader_sha256": sha256(inspect.getfile(create_dataset_from_stream)),
        "manifest": manifest,
        "model_assets": model_assets(model),
        "requests": exported,
    }
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2)
        stream.write("\n")
    print(f"Exported {len(exported)} exact TRT inputs to {args.output}")


def verify_inputs(args):
    from transformers import AutoTokenizer

    manifest, rows = load_workload(args.inputs_dir)
    artifact = json.loads(args.tokens.read_text())
    if (
        artifact.get("producer")
        != "tensorrt_llm.bench.utils.data.create_dataset_from_stream"
    ):
        raise ValueError("Use export-trt-inputs in the TRT benchmark environment first")
    if artifact["manifest"] != manifest:
        raise ValueError("Token export belongs to a different manifest")
    if artifact["model_assets"] != model_assets(args.model_path):
        raise ValueError("Local model/tokenizer metadata differs from the TRT export")
    token_rows = artifact["requests"]
    if len(token_rows) != len(rows):
        raise ValueError("Token export request count differs from dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    started = time.perf_counter()
    for row, exported in zip(rows, token_rows):
        if any(
            exported[key] != row[key] for key in ("task_id", "prompt", "output_tokens")
        ):
            raise ValueError("Token export order/content differs from dataset")
        ids = exported["input_ids"]
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Invalid input token IDs")
        if tokenizer(row["prompt"])["input_ids"] != ids:
            raise ValueError(f"Tokenizer/BOS mismatch for task {row['task_id']}")
    return artifact, time.perf_counter() - started


class Client:
    """One persistent local HTTP connection; exactly one request in flight."""

    def __init__(self, port, timeout=600):
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)

    def close(self):
        self.connection.close()

    def request(self, method, path, body=None):
        payload = json.dumps(body) if body is not None else None
        self.connection.request(
            method, path, payload, {"Content-Type": "application/json"}
        )
        response = self.connection.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"{path}: HTTP {response.status}: {response.read().decode()}"
            )
        return response

    def control(self, path, body=None, method="POST"):
        raw = self.request(method, path, body).read()
        # /flush_cache returns text and reports failure through its HTTP status.
        if path.startswith("/flush_cache"):
            return raw.decode()
        result = json.loads(raw)
        if result.get("success") is False:
            raise RuntimeError(f"{path}: {result}")
        return result

    def check_corpora(self, expected):
        actual = self.control("/list_external_corpora", method="GET")[
            "corpus_token_counts"
        ]
        if actual != expected:
            raise RuntimeError(
                f"Corpus isolation failed: expected {expected}, got {actual}"
            )

    def generate(self, row):
        payload = {
            "input_ids": row["input_ids"],
            "sampling_params": {
                "temperature": 0,
                "top_k": 1,
                "max_new_tokens": row["output_tokens"],
                "ignore_eos": True,
                "skip_special_tokens": False,
            },
            "stream": True,
            "return_prompt_token_ids": True,
        }
        start = time.perf_counter()
        try:
            response = self.request("POST", "/generate", payload)
            result = collect_stream(response, start)
        except BaseException:
            # Discard an unfinished HTTP response so cleanup can reconnect.
            self.close()
            raise
        result["generation_s"] = time.perf_counter() - start
        return result


def collect_stream(response, start):
    """Parse cumulative /generate SSE, preserving IDs and final metadata.

    TTFT starts at the first token-bearing event, including empty/special text.
    Speculation can deliver multiple tokens per event; TPOT is an average,
    not a claim that individual tokens were observed at separate times.
    """
    final, events, data = None, [], []
    previous_ids = []
    first, last = None, None
    done = False
    while line := response.readline():
        line = line.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            event = "\n".join(data)
            data = []
            if event == "[DONE]":
                done = True
                break
            current = json.loads(event)
            if "error" in current:
                raise RuntimeError(f"Generation stream failed: {current}")
            ids = current.get("output_ids", [])
            if ids[: len(previous_ids)] != previous_ids:
                raise RuntimeError("Expected cumulative streaming token IDs")
            now = time.perf_counter() - start
            if len(ids) > len(previous_ids):
                first = now if first is None else first
                last = now
                events.append({"elapsed_s": now, "completion_tokens": len(ids)})
            previous_ids = ids
            final = current
    # Consume the HTTP terminator before reusing the persistent connection.
    response.read()
    if not done or final is None or first is None:
        raise RuntimeError("Incomplete generation stream (missing tokens or DONE)")
    if final.get("meta_info", {}).get("finish_reason") is None:
        raise RuntimeError("Stream ended without final request metadata")
    return {
        "output_ids": final["output_ids"],
        "text": final.get("text", ""),
        "prompt_token_ids": final.get("prompt_token_ids"),
        "meta_info": final["meta_info"],
        "ttft_s": first,
        "last_token_s": last,
        "tpot_s": (last - first) / (len(previous_ids) - 1)
        if len(previous_ids) > 1
        else None,
        "stream_events": events,
    }


def validate_result(result, row, speculative):
    meta = result["meta_info"]
    issues = []
    if result["prompt_token_ids"] != row["input_ids"]:
        issues.append("server prompt IDs differ from TRT input IDs")
    if meta.get("prompt_tokens") != len(row["input_ids"]):
        issues.append("server prompt token count differs from input")
    if (
        len(result["output_ids"]) != row["output_tokens"]
        or meta.get("completion_tokens") != row["output_tokens"]
    ):
        issues.append("output length is not the fixed budget")
    if meta.get("finish_reason", {}).get("type") != "length":
        issues.append("generation did not finish at the length limit")
    if speculative:
        if any(type(meta.get(k)) is not int or meta[k] < 0 for k in COUNTERS):
            issues.append("missing/invalid final speculative counters")
        elif meta[COUNTERS[0]] <= 0 or meta[COUNTERS[2]] != 4 * meta[COUNTERS[0]]:
            issues.append("verification/draft capacity counters differ from width five")
        elif meta[COUNTERS[1]] > meta[COUNTERS[2]]:
            issues.append("accepted drafts exceed proposed capacity")
    return issues


def measure_request(client, row, mode):
    speculative = mode != "baseline"
    use_sam = MODES[mode][2]
    start = time.perf_counter()
    if speculative:
        client.check_corpora({})
    flush_start = time.perf_counter()
    client.control("/flush_cache?timeout=10")
    flush_s = time.perf_counter() - flush_start
    load_s = 0.0
    try:
        if use_sam:
            load_start = time.perf_counter()
            loaded = client.control(
                "/add_external_corpus",
                {
                    "corpus_id": CORPUS,
                    "token_chunks": [row["input_ids"]],
                },
            )
            load_s = time.perf_counter() - load_start
            if loaded.get("loaded_token_count") != len(row["input_ids"]):
                raise RuntimeError(
                    "Corpus load did not preserve the exact prompt length"
                )
            client.check_corpora({CORPUS: len(row["input_ids"])})
        setup_s = time.perf_counter() - start
        result = client.generate(row)
    finally:
        cleanup_start = time.perf_counter()
        if use_sam:
            client.control("/remove_external_corpus", {"corpus_id": CORPUS})
        if speculative:
            client.check_corpora({})
        cleanup_s = time.perf_counter() - cleanup_start
    result.update(
        {
            "task_id": row["task_id"],
            "setup_s": setup_s,
            "flush_s": flush_s,
            "corpus_load_s": load_s,
            "cleanup_s": cleanup_s,
            "workflow_s": time.perf_counter() - start,
        }
    )
    result["validation_errors"] = validate_result(result, row, speculative)
    return result


def server_command(args, mode):
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--tp-size",
        "1",
        "--attention-backend",
        args.attention_backend,
        "--page-size",
        "64",
        "--max-running-requests",
        "1",
        "--chunked-prefill-size",
        "-1",
        "--disable-radix-cache",
        "--context-length",
        "8192",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "disabled" if args.disable_cuda_graph else "full",
        "--cuda-graph-bs-decode",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--stream-interval",
        "1",
        "--random-seed",
        str(args.seed),
    ]
    if args.disable_overlap_schedule:
        command.append("--disable-overlap-schedule")
    allocation, ranking, use_sam = MODES[mode]
    if mode != "baseline":
        command.extend(
            [
                "--speculative-algorithm",
                "NGRAM",
                "--speculative-num-draft-tokens",
                "5",
                "--speculative-ngram-global-tree-mode",
                allocation,
                "--speculative-ngram-match-type",
                ranking,
                "--speculative-ngram-external-sam-budget",
                "2" if use_sam else "0",
                "--speculative-ngram-min-bfs-breadth",
                "1",
                "--speculative-ngram-max-bfs-breadth",
                "4",
                "--speculative-ngram-max-trie-depth",
                str(args.max_trie_depth),
                "--speculative-ngram-capacity",
                str(args.ngram_capacity),
            ]
        )
    return command


def validate_server(info, args, mode):
    expected = {
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "tp_size": 1,
        "max_running_requests": 1,
        "disable_radix_cache": True,
        "chunked_prefill_size": -1,
        "disable_overlap_schedule": args.disable_overlap_schedule,
        "incremental_streaming_output": False,
        "stream_interval": 1,
        "attention_backend": args.attention_backend,
        "page_size": 64,
        "speculative_algorithm": None if mode == "baseline" else "NGRAM",
    }
    if mode != "baseline":
        expected.update(
            {
                "speculative_num_draft_tokens": 5,
                "speculative_ngram_global_tree_mode": MODES[mode][0],
                "speculative_ngram_match_type": MODES[mode][1],
                "speculative_ngram_external_sam_budget": 2 if MODES[mode][2] else 0,
                "speculative_ngram_max_trie_depth": args.max_trie_depth,
            }
        )
    errors = {
        key: {"expected": value, "resolved": info.get(key)}
        for key, value in expected.items()
        if info.get(key) != value
    }
    graph = info.get("cuda_graph_config", {})
    decode = graph.get("decode", {})
    if decode.get("backend") != ("disabled" if args.disable_cuda_graph else "full"):
        errors["decode_graph"] = decode
    if not args.disable_cuda_graph and decode.get("bs") != [1]:
        errors["decode_graph_batch_sizes"] = decode
    if graph.get("prefill", {}).get("backend") != "disabled":
        errors["prefill_graph"] = graph.get("prefill")
    if errors:
        raise RuntimeError(
            f"Resolved server settings differ from the experiment: {errors}"
        )


@contextmanager
def launch_server(args, command, log_path):
    # Own the server lifecycle; never flush an unrelated server on an occupied port.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    with log_path.open("w") as log:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = (
            str(REPO / "python") + os.pathsep + environment.get("PYTHONPATH", "")
        )
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.perf_counter()
        try:
            while time.perf_counter() - started < args.startup_timeout:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited with {process.returncode}; see {log_path}"
                    )
                client = Client(args.port, timeout=2)
                try:
                    client.request("GET", "/health").read()
                    break
                except (OSError, http.client.HTTPException, RuntimeError):
                    time.sleep(1)
                finally:
                    client.close()
            else:
                raise TimeoutError(f"Server startup timed out; see {log_path}")
            yield time.perf_counter() - started
        finally:
            # Also reap children if the parent exited while a worker survived.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def capture(command, cwd=REPO):
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {type(error).__name__}"


def redact(value):
    if isinstance(value, dict):
        return {
            k: "<redacted>"
            if "api_key" in k or k in {"hf_token", "access_token", "password", "secret"}
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def run(args):
    args.model_path = (args.model_path or args.inputs_dir / "model").resolve()
    if args.requests < 1 or args.repeats < 1 or args.warmup < 1:
        raise ValueError("Requests, repetitions, and warmup must all be positive")
    if len(set(args.modes)) != len(args.modes) or "baseline" not in args.modes:
        raise ValueError("Select distinct modes including baseline")
    if args.max_trie_depth < 2 or args.ngram_capacity < 1:
        raise ValueError("Trie depth must be at least two and capacity positive")
    artifact, tokenization_s = verify_inputs(args)
    rows = artifact["requests"][: args.requests]
    if len(rows) != args.requests or any(
        len(row["input_ids"]) + 256 > 8192 for row in rows
    ):
        raise ValueError(
            "Insufficient prepared requests or sequence exceeds 8192 tokens"
        )
    # Reject an incompatible reference before spending time loading GPU models.
    trt = trt_reference(args.trt_results, args.requests) if args.trt_results else None
    args.directory.mkdir()  # Never overwrite or mix previous experiment results.
    write_json(args.directory / "inputs.json", artifact)
    provenance = {
        "schema_version": 1,
        "framework": "sglang",
        "source_revision": capture(["git", "rev-parse", "HEAD"]),
        "source_status": capture(["git", "status", "--short"]),
        "tracked_diff_sha256": hashlib.sha256(
            capture(["git", "diff", "HEAD"]).encode()
        ).hexdigest(),
        "runner_sha256": sha256(__file__),
        "reporter_sha256": sha256(Path(__file__).with_name("code_edit_report.py")),
        "packages": package_versions(
            ["sglang", "torch", "transformers", "tokenizers", "flashinfer-python"]
        ),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": capture(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,power.limit,clocks.sm,clocks.mem",
                "--format=csv",
            ]
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "token_export_sha256": sha256(args.tokens),
        "token_verification_s": tokenization_s,
        "timing_boundary": "persistent local HTTP client; streaming /generate",
        "source_policy": "static current-prompt SAM plus online Trie; reset every request",
        "warmup_policy": "first selected requests, cycling deterministically; same isolated lifecycle; excluded",
    }
    write_json(args.directory / "experiment.json", provenance)
    trials = []
    try:
        for repetition in range(1, args.repeats + 1):
            offset = (repetition - 1) % len(args.modes)
            order = args.modes[offset:] + args.modes[:offset]
            for mode in order:
                prefix = args.directory / f"{mode}-{repetition}"
                command = server_command(args, mode)
                write_json(prefix.with_suffix(".command.json"), command)
                print(f"Running {mode}, repetition {repetition}", flush=True)
                with launch_server(
                    args, command, prefix.with_suffix(".server.log")
                ) as startup_s:
                    client = Client(args.port, args.request_timeout)
                    try:
                        info = client.control("/server_info", method="GET")
                        write_json(prefix.with_suffix(".server.json"), redact(info))
                        validate_server(info, args, mode)
                        warm_start = time.perf_counter()
                        for index in range(args.warmup):
                            warm = measure_request(
                                client, rows[index % len(rows)], mode
                            )
                            if warm["validation_errors"]:
                                write_json(
                                    prefix.with_suffix(".failed-warmup.json"), warm
                                )
                                raise RuntimeError(
                                    f"Warmup validation failed: {warm['validation_errors']}"
                                )
                        warmup_s = time.perf_counter() - warm_start
                        records = []
                        window_start = time.perf_counter()
                        with prefix.with_suffix(".requests.jsonl").open("w") as stream:
                            for row in rows:
                                result = measure_request(client, row, mode)
                                records.append(result)
                                stream.write(json.dumps(result, allow_nan=False) + "\n")
                                stream.flush()
                                if result["validation_errors"]:
                                    raise RuntimeError(
                                        f"Task {row['task_id']}: {result['validation_errors']}"
                                    )
                        window_s = time.perf_counter() - window_start
                        trial = summarize_trial(mode, repetition, records, window_s)
                        trial.update({"startup_s": startup_s, "warmup_s": warmup_s})
                        write_json(prefix.with_suffix(".report.json"), trial)
                        trials.append({**trial, "requests": records})
                    finally:
                        client.close()
        summary = build_summary(trials)
        if trt is not None:
            summary["trt_reference"] = trt
        write_json(args.directory / "summary.json", summary)
        print(
            "\nMode                              HTTP tok/s    Workflow tok/s    vs baseline    mismatches"
        )
        for mode, report in summary["modes"].items():
            print(
                f"{mode:33} {report['generation_tok_s']['median']:10.2f} {report['workflow_tok_s']['median']:17.2f} {report['generation_speedup']['median']:14.3f} {report['mismatch_count']:13}"
            )
        if args.fail_on_mismatch and any(
            m["mismatch_count"] for m in summary["modes"].values()
        ):
            raise RuntimeError(
                "Output mismatches recorded; see summary.json and request artifacts"
            )
    except BaseException as error:
        write_json(
            args.directory / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "completed_trials": len(trials),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export-trt-inputs",
        help="Export exact tokens with the installed TRT benchmark loader",
    )
    export.add_argument("--inputs-dir", type=Path, default=Path("."))
    export.add_argument("--model-path", type=Path)
    export.add_argument("--output", type=Path, required=True)
    bench = commands.add_parser(
        "run", help="Launch and measure a serial SGLang mode matrix"
    )
    bench.add_argument("requests", type=int)
    bench.add_argument("repeats", type=int)
    bench.add_argument("directory", type=Path)
    bench.add_argument("--inputs-dir", type=Path, default=Path("."))
    bench.add_argument("--tokens", type=Path, required=True)
    bench.add_argument("--model-path", type=Path)
    bench.add_argument("--modes", nargs="+", choices=MODES, default=DEFAULT_MODES)
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--port", type=int, default=30000)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--attention-backend", default="flashinfer")
    bench.add_argument("--mem-fraction-static", type=float, default=0.8)
    bench.add_argument("--max-trie-depth", type=int, default=18)
    bench.add_argument("--ngram-capacity", type=int, default=10000000)
    bench.add_argument("--disable-overlap-schedule", action="store_true")
    bench.add_argument("--disable-cuda-graph", action="store_true")
    bench.add_argument("--startup-timeout", type=float, default=900)
    bench.add_argument("--request-timeout", type=float, default=600)
    bench.add_argument("--trt-results", type=Path)
    bench.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()
    if args.command == "export-trt-inputs":
        export_trt_inputs(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
