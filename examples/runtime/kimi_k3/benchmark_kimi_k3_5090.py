#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a one-shot Kimi K3 expert-pack benchmark on one RTX 5090."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = "请介绍深圳"
DEFAULT_SERVER_LOG = SCRIPT_DIR / "logs/kimi-k3-5090-benchmark-server.log"
DEFAULT_LOCK = "/tmp/sglang-kimi-k3-5090-benchmark.lock"
METADATA_FORMAT_VERSION = 2
ACTIVE_MOE_LAYERS = tuple(range(1, 93))
IMMUTABLE_TOP_K = 16


def cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()


def artifact_dir_for_source(gguf: Path, expert_pack: Path) -> Path:
    stat = gguf.stat()
    pack_stat = expert_pack.stat()
    fingerprint = hashlib.sha256(
        f"{gguf.parent.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"{expert_pack.resolve()}:{pack_stat.st_size}:{pack_stat.st_mtime_ns}:{METADATA_FORMAT_VERSION}".encode()
    ).hexdigest()[:20]
    return cache_root() / "sglang-expert-pack" / "kimi-k3" / fingerprint


def _tokenizer_candidate(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any(
            (path / name).is_file()
            for name in ("tokenizer.json", "tiktoken.model", "tokenizer_config.json")
        )
    )


def resolve_kimi_assets(gguf: Path) -> tuple[Path, Path, Path]:
    gguf_dir = gguf.parent
    packs = sorted(gguf_dir.glob("*.expert-major.pack"))
    if len(packs) != 1:
        raise RuntimeError(
            f"expected exactly one Kimi Expert Pack beside --gguf in {gguf_dir}; found {len(packs)}"
        )
    candidates = [gguf_dir / "tokenizer", gguf_dir.parent / "kimi-k3-tokenizer"]
    candidates.extend(
        sorted(path for path in gguf_dir.parent.glob("*tokenizer*") if path.is_dir())
    )
    tokenizers = []
    for path in candidates:
        path = path.resolve()
        if path not in tokenizers and _tokenizer_candidate(path):
            tokenizers.append(path)
    if len(tokenizers) != 1:
        names = ", ".join(str(path) for path in tokenizers) or "none"
        raise RuntimeError(
            f"could not uniquely derive Kimi tokenizer beside {gguf_dir}; candidates: {names}"
        )
    return gguf_dir, packs[0].resolve(), tokenizers[0]


def find_sglang_repo() -> Path:
    configured = os.environ.get("SGLANG_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (candidate / "python" / "sglang").is_dir() and (
            candidate / "tools" / "expert_pack"
        ).is_dir():
            return candidate
    raise RuntimeError("could not locate the SGLang repository")


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def detect_rtx_5090() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    gpu_zero = next(
        (line.split(",", 1)[1].strip() for line in rows if line.startswith("0,")),
        None,
    )
    if gpu_zero is None or "5090" not in gpu_zero:
        raise RuntimeError(
            f"CUDA device 0 must be an RTX 5090; detected: {', '.join(rows)}"
        )
    return gpu_zero


def prepare_model_metadata(tokenizer_dir: Path, artifact_dir: Path) -> Path:
    tokenizer_dir = tokenizer_dir.resolve(strict=True)
    source_config = json.loads(
        (tokenizer_dir / "config.json").read_text(encoding="utf-8")
    )
    if "text_config" not in source_config:
        raise ValueError("Kimi tokenizer config does not contain text_config")
    config = dict(source_config["text_config"])
    config["architectures"] = ["KimiK3LinearForCausalLM"]
    config["model_type"] = "kimi_linear"
    config.pop("auto_map", None)
    config.pop("quantization_config", None)

    output_dir = artifact_dir / "model-meta"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in tokenizer_dir.iterdir():
        if source.is_file() and source.name != "config.json":
            destination = output_dir / source.name
            if (
                not destination.is_file()
                or destination.stat().st_size != source.stat().st_size
                or destination.stat().st_mtime_ns != source.stat().st_mtime_ns
            ):
                shutil.copy2(source, destination)
    write_json_atomic(output_dir / "config.json", config)
    return output_dir


def prepare_manifest(args: argparse.Namespace, model_dir: Path) -> Path:
    manifest = args.artifact_dir / "kimi-k3-expert-pack.manifest.json"
    command = [
        sys.executable,
        str(args.sglang_repo / "tools" / "expert_pack" / "prepare_kimi_manifest.py"),
        "--gguf-dir",
        str(args.gguf_dir),
        "--expert-pack",
        str(args.expert_pack),
        "--model-config",
        str(model_dir / "config.json"),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
        "--output",
        str(manifest),
        "--payload-samples",
        str(args.payload_samples),
    ]
    if args.full_source_hashes:
        command.append("--full-source-hashes")
    if args.full_pack_hash:
        command.append("--full-pack-hash")
    subprocess.run(command, cwd=args.sglang_repo, check=True)
    return manifest


def validate_manifest(
    manifest_path: Path,
    expert_pack: Path,
    gguf_dir: Path | None = None,
    tokenizer_dir: Path | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_constraints = {
        "all_selected_experts_must_execute": True,
        "expert_pruning_allowed": False,
        "requantization_allowed": False,
        "top_k": IMMUTABLE_TOP_K,
        "top_k_is_immutable": True,
    }
    if manifest.get("hard_constraints") != expected_constraints:
        raise ValueError("manifest does not enforce immutable Top-K=16 execution")
    model = manifest["model"]
    if tuple(model["active_moe_layer_ids"]) != ACTIVE_MOE_LAYERS:
        raise ValueError("manifest active MoE layers must be exactly 1..92")
    if int(model["num_experts_per_token"]) != IMMUTABLE_TOP_K:
        raise ValueError("manifest changed Kimi K3 Top-K away from 16")
    pack = manifest["expert_pack"]
    if Path(pack["path"]).resolve() != expert_pack.resolve(strict=True):
        raise ValueError("manifest points to a different expert-pack")
    if int(pack["size"]) != expert_pack.stat().st_size:
        raise ValueError("expert-pack size changed after preparation")
    if gguf_dir is not None:
        shard_paths = [
            Path(item["path"]).resolve() for item in manifest["source"]["shards"]
        ]
        if not shard_paths or any(
            path.parent != gguf_dir.resolve() for path in shard_paths
        ):
            raise ValueError("manifest source shards do not match --gguf directory")
    if (
        tokenizer_dir is not None
        and Path(manifest["tokenizer"]["path"]).resolve() != tokenizer_dir.resolve()
    ):
        raise ValueError(
            "manifest tokenizer does not match derived tokenizer directory"
        )
    return manifest


def make_prompt(tokenizer_dir: Path, prompt: str) -> tuple[list[int], str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True, local_files_only=True
    )
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    if not input_ids:
        raise ValueError("Kimi chat template produced an empty prompt")
    return [int(value) for value in input_ids], prompt_text


def server_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}"


def port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_server(
    args: argparse.Namespace,
    model_dir: Path,
    manifest_path: Path,
) -> subprocess.Popen:
    if port_in_use(args.host, args.port):
        raise RuntimeError(f"server address is already in use: {server_url(args)}")
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    log = args.server_log.open("wb", buffering=0)
    extra_config = {
        "pack_path": str(args.expert_pack),
        "manifest_path": str(manifest_path),
        "cache_vram_mib": args.expert_cache_mib,
        "cache_vram_reserve_mib": args.expert_cache_reserve_mib,
        "stage_slots": args.stage_slots,
        "read_splits": args.read_splits,
        "direct_io": args.direct_io,
        "stats_flush_interval": len(ACTIVE_MOE_LAYERS),
        "stats_path": str(args.stats_path),
        "verify_pack_sha256": args.verify_pack_sha256,
    }
    env = os.environ.copy()
    python_path = [str(args.sglang_repo), str(args.sglang_repo / "python")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    conda_lib = str(Path(sys.prefix) / "lib")
    cuda_root = Path("/usr/local/cuda")
    if (cuda_root / "bin" / "nvcc").is_file():
        env["CUDA_HOME"] = str(cuda_root)
        env["CUDA_PATH"] = str(cuda_root)
        env["PATH"] = os.pathsep.join((str(cuda_root / "bin"), env.get("PATH", "")))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in (
            conda_lib,
            str(cuda_root / "lib64") if (cuda_root / "lib64").is_dir() else None,
            env.get("LD_LIBRARY_PATH"),
        )
        if value
    )
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_dir),
        "--tokenizer-path",
        str(model_dir),
        "--trust-remote-code",
        "--load-format",
        "expert_pack",
        "--model-loader-extra-config",
        json.dumps(extra_config, separators=(",", ":")),
        "--tp-size",
        "1",
        "--ep-size",
        "1",
        "--disable-cuda-graph",
        "--disable-shared-experts-fusion",
        "--disable-radix-cache",
        "--mamba-radix-cache-strategy",
        "no_buffer",
        "--disable-overlap-schedule",
        "--skip-server-warmup",
        "--context-length",
        str(args.context_length),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--watchdog-timeout",
        str(args.watchdog_timeout),
        "--max-running-requests",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(
        f"SERVICE_STARTING url={server_url(args)} timeout={args.startup_timeout:.0f}s "
        f"log={args.server_log}",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    process._benchmark_log = log  # type: ignore[attr-defined]
    try:
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"SGLang exited during startup with code {process.returncode}; "
                    f"see {args.server_log}"
                )
            if port_in_use(args.host, args.port):
                print(
                    f"SERVICE_READY pid={process.pid} url={server_url(args)}",
                    flush=True,
                )
                return process
            time.sleep(2)
        raise TimeoutError(
            f"SGLang did not become ready within {args.startup_timeout:.0f}s; "
            f"see {args.server_log}"
        )
    except BaseException:
        stop_server(process, server_url(args))
        raise


def stop_server(process: subprocess.Popen | None, url: str) -> None:
    if process is None:
        return
    log = getattr(process, "_benchmark_log", None)
    try:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)
        deadline = time.monotonic() + 10
        while (
            port_in_use(*server_address(url), timeout=0.2)
            and time.monotonic() < deadline
        ):
            time.sleep(0.2)
        print(f"SERVICE_STOPPED pid={process.pid} url={url}", flush=True)
    finally:
        if log is not None:
            log.close()


def server_address(url: str) -> tuple[str, int]:
    without_scheme = url.removeprefix("http://")
    host, port = without_scheme.rsplit(":", 1)
    return host, int(port)


def generate(
    url: str,
    input_ids: list[int],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    *,
    stream_output: bool,
) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": temperature,
            "top_p": top_p,
            "sampling_seed": seed,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "stream": True,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter_ns()
    first_token = None
    last_token = None
    completion_tokens = 0
    prompt_tokens = None
    output = ""
    finish_reason = None
    output_token_ids: dict[int, int] = {}

    if stream_output:
        print(f"prompt: {prompt}", flush=True)
        print("output: ", end="", flush=True)
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw_line in response:
            now = time.perf_counter_ns()
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                continue
            event = json.loads(line)
            meta = event.get("meta_info") or {}
            current_tokens = int(meta.get("completion_tokens", 0))
            if current_tokens > completion_tokens:
                first_token = first_token or now
                last_token = now
                completion_tokens = current_tokens
            if meta.get("prompt_tokens") is not None:
                prompt_tokens = int(meta["prompt_tokens"])
            logprobs = meta.get("output_token_logprobs") or []
            logprob_length = int(
                meta.get("output_token_logprobs_length", current_tokens)
            )
            offset = logprob_length - len(logprobs)
            for index, item in enumerate(logprobs):
                output_token_ids[offset + index] = int(item[1])
            event_output = event.get("text")
            if event_output is not None:
                if stream_output and event_output != output:
                    if event_output.startswith(output):
                        print(event_output[len(output) :], end="", flush=True)
                    else:
                        print(f"\n[output revised]\n{event_output}", end="", flush=True)
                output = event_output
            finish_reason = meta.get("finish_reason", finish_reason)
    if stream_output:
        print(flush=True)

    if first_token is None or last_token is None or prompt_tokens is None:
        raise RuntimeError("SGLang response omitted token timing metadata")
    if prompt_tokens != len(input_ids):
        raise RuntimeError(
            f"server prompt token count {prompt_tokens} != tokenizer count {len(input_ids)}"
        )
    ordered_token_ids = [output_token_ids[index] for index in sorted(output_token_ids)]
    if len(ordered_token_ids) != completion_tokens:
        raise RuntimeError(
            "SGLang response omitted output token IDs: "
            f"{len(ordered_token_ids)} != {completion_tokens}"
        )
    ttft_s = (first_token - started) / 1e9
    decode_span_s = (last_token - first_token) / 1e9
    total_s = (time.perf_counter_ns() - started) / 1e9
    return {
        "output": output,
        "output_token_ids": ordered_token_ids,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_ms": ttft_s * 1000,
        "prefill_token_rate": prompt_tokens / ttft_s if ttft_s > 0 else None,
        "decode_token_rate": (
            (completion_tokens - 1) / decode_span_s
            if completion_tokens > 1 and decode_span_s > 0
            else None
        ),
        "tpot_ms": (
            decode_span_s * 1000 / (completion_tokens - 1)
            if completion_tokens > 1
            else None
        ),
        "total_elapsed_s": total_s,
        "end_to_end_token_rate": completion_tokens / total_s if total_s > 0 else None,
    }


def read_stats(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"expert-pack stats were not written: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_routes(stats: dict, expected_tokens: int) -> None:
    token_counts = stats["route_tokens_by_layer"]
    call_counts = stats["route_calls_by_layer"]
    for layer in ACTIVE_MOE_LAYERS:
        if (
            token_counts[layer] != expected_tokens
            or call_counts[layer] != expected_tokens
        ):
            raise RuntimeError(
                f"layer {layer} routed {token_counts[layer]} tokens in "
                f"{call_counts[layer]} calls; expected {expected_tokens} exact Top-16 calls"
            )
    if any(
        token_counts[layer]
        for layer in set(range(len(token_counts))) - set(ACTIVE_MOE_LAYERS)
    ):
        raise RuntimeError("routed experts outside model layers 1..92")
    if int(stats.get("fallback_count", 0)) != 0:
        raise RuntimeError("the request used an expert fallback")


def git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gguf",
        type=Path,
        required=True,
        help="one Kimi-K3 GGUF shard; sibling shards, Pack and tokenizer are derived",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.set_defaults(
        host="127.0.0.1",
        port=30001,
        prompt=DEFAULT_PROMPT,
        temperature=0.0,
        top_p=0.95,
        seed=20260813,
        startup_timeout=1200,
        context_length=384,
        max_total_tokens=512,
        chunked_prefill_size=64,
        watchdog_timeout=1800,
        mem_fraction_static=0.98,
        expert_cache_mib=5120,
        expert_cache_reserve_mib=1536,
        stage_slots=16,
        read_splits=4,
        direct_io=True,
        payload_samples=6,
        full_source_hashes=False,
        full_pack_hash=False,
        verify_pack_sha256=False,
        prepare_only=False,
    )
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    for name in (
        "expert_cache_mib",
        "expert_cache_reserve_mib",
        "stage_slots",
        "read_splits",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.gguf = args.gguf.expanduser().resolve(strict=True)
    args.gguf_dir, args.expert_pack, args.tokenizer_dir = resolve_kimi_assets(args.gguf)
    args.sglang_repo = find_sglang_repo()
    args.artifact_dir = artifact_dir_for_source(args.gguf, args.expert_pack).resolve()
    args.server_log = args.artifact_dir / DEFAULT_SERVER_LOG.name
    args.stats_path = args.artifact_dir / "kimi-k3-expert-pack.stats.json"
    args.report_path = args.artifact_dir / "kimi-k3-5090-benchmark.json"
    return args


def print_result(result: dict, stats: dict) -> None:
    def value(name: str) -> str:
        item = result[name]
        return "n/a" if item is None else f"{item:.3f}"

    print(f"prompt_tokens: {result['prompt_tokens']}")
    print(f"completion_tokens: {result['completion_tokens']}")
    print(f"ttft_ms: {value('ttft_ms')}")
    print(f"prefill_token_rate: {value('prefill_token_rate')} tok/s")
    print(f"decode_token_rate: {value('decode_token_rate')} tok/s")
    print(f"tpot_ms: {value('tpot_ms')} ms/token")
    print(f"end_to_end_token_rate: {value('end_to_end_token_rate')} tok/s")
    print(
        f"expert_cache: hits={stats['cache_hits']} misses={stats['cache_misses']} "
        f"evictions={stats['cache_evictions']} reads={stats['pack_reads']} "
        f"read_bytes={stats['pack_read_bytes']} h2d_bytes={stats['h2d_bytes']}"
    )


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGHUP, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    lock = Path(DEFAULT_LOCK).open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"error: another benchmark is running (lock: {DEFAULT_LOCK})",
            file=sys.stderr,
        )
        return 2
    lock.write(f"{os.getpid()}\n")
    lock.flush()

    process = None
    try:
        gpu = detect_rtx_5090()
        model_dir = prepare_model_metadata(args.tokenizer_dir, args.artifact_dir)
        manifest_path = prepare_manifest(args, model_dir)
        manifest = validate_manifest(
            manifest_path, args.expert_pack, args.gguf_dir, args.tokenizer_dir
        )
        prompt_ids, prompt_text = make_prompt(args.tokenizer_dir, args.prompt)
        prepared = {
            "status": "prepared",
            "gpu": gpu,
            "model_metadata": str(model_dir),
            "manifest": str(manifest_path),
            "source_inventory_sha256": manifest["source"]["inventory_sha256"],
            "expert_pack_index_sha256": manifest["expert_pack"]["index_sha256"],
            "expert_pack_reused": True,
            "expert_pack_modified": False,
            "top_k": IMMUTABLE_TOP_K,
            "prompt_token_ids": prompt_ids,
        }
        print(
            f"MODEL_READY gpu={gpu} top_k={IMMUTABLE_TOP_K} "
            f"expert_pack={args.expert_pack}",
            flush=True,
        )
        if args.prepare_only:
            return 0

        if args.stats_path.exists():
            args.stats_path.unlink()
        process = start_server(args, model_dir, manifest_path)
        result = generate(
            server_url(args),
            prompt_ids,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.seed,
            stream_output=True,
        )
        model_tokens = result["prompt_tokens"] + result["completion_tokens"] - 1
        stop_server(process, server_url(args))
        process = None
        stats = read_stats(args.stats_path)
        audit_routes(stats, model_tokens)

        report = {
            **prepared,
            "status": "passed",
            "sglang_commit": git_commit(args.sglang_repo),
            "python": sys.version,
            "command": sys.argv,
            "server_url": server_url(args),
            "server_log": str(args.server_log),
            "stats_path": str(args.stats_path),
            "prompt": args.prompt,
            "formatted_prompt": prompt_text,
            "result": result,
            "expert_pack_stats": stats,
            "route_audit": {
                "active_moe_layers": list(ACTIVE_MOE_LAYERS),
                "immutable_top_k": IMMUTABLE_TOP_K,
                "model_tokens_per_layer": model_tokens,
                "fallback_count": int(stats.get("fallback_count", 0)),
            },
        }
        write_json_atomic(args.report_path, report)
        print_result(result, stats)
        print(f"report: {args.report_path}")
        print(f"server_log: {args.server_log}")
        return 0
    except KeyboardInterrupt:
        print("error: benchmark interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_server(process, server_url(args))


if __name__ == "__main__":
    raise SystemExit(main())
