#!/usr/bin/env python3
"""Benchmark an existing DeepSeek V4 Flash Expert Pack on one RTX 5090."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = "Please introduce Shenzhen"
DEFAULT_WARMUP_PROMPT = "Briefly explain why the sky appears blue."
CHAT_PREFIX = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>"
CHAT_SUFFIX = "<｜Assistant｜><think>"
DEFAULT_LOCK = Path("/tmp/sglang-deepseek-v4-5090-benchmark.lock")
METADATA_FORMAT_VERSION = 4
ACTIVE_MOE_LAYERS = tuple(range(43))


def cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()


def source_fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(
        f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{METADATA_FORMAT_VERSION}".encode()
    ).hexdigest()[:20]


def artifact_dir_for_source(path: Path) -> Path:
    return (
        cache_root()
        / "sglang-expert-pack"
        / "deepseek-v4-flash"
        / source_fingerprint(path)
    )


def find_sglang_repo() -> Path:
    configured = os.getenv("SGLANG_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (candidate / "python" / "sglang").is_dir():
            return candidate
    raise RuntimeError("cannot locate the SGLang checkout; set SGLANG_REPO")


def format_prompt(prompt: str) -> str:
    return f"{CHAT_PREFIX}{prompt}{CHAT_SUFFIX}"


def server_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}"


def port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


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
        detected = ", ".join(rows) if rows else "none"
        raise RuntimeError(f"CUDA device 0 must be an RTX 5090; detected: {detected}")
    return gpu_zero


def _digest(value: Any, field: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"manifest {field} is not a SHA-256 digest")
    return digest


def _sha256(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _gguf_value(reader: object, name: str) -> object:
    fields = getattr(reader, "fields")
    if name not in fields:
        raise ValueError(f"GGUF metadata is missing required field: {name}")
    return fields[name].contents()


def _model_config_from_gguf(reader: object) -> dict[str, Any]:
    value = lambda name: _gguf_value(reader, f"deepseek4.{name}")
    architecture = _gguf_value(reader, "general.architecture")
    if architecture != "deepseek4":
        raise ValueError(f"expected GGUF architecture deepseek4, got {architecture!r}")
    if int(value("expert_gating_func")) != 4:
        raise ValueError("unsupported deepseek4.expert_gating_func; expected 4")
    swiglu_limits = [float(item) for item in value("swiglu_clamp_exp")]
    if not swiglu_limits or any(item != swiglu_limits[0] for item in swiglu_limits):
        raise ValueError("deepseek4.swiglu_clamp_exp must be constant")
    tokens = _gguf_value(reader, "tokenizer.ggml.tokens")
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": int(_gguf_value(reader, "tokenizer.ggml.bos_token_id")),
        "eos_token_id": int(_gguf_value(reader, "tokenizer.ggml.eos_token_id")),
        "expert_dtype": "fp4",
        "hc_eps": float(value("hyper_connection.epsilon")),
        "hc_mult": int(value("hyper_connection.count")),
        "hc_sinkhorn_iters": int(value("hyper_connection.sinkhorn_iterations")),
        "head_dim": int(value("attention.key_length")),
        "hidden_act": "silu",
        "hidden_size": int(value("embedding_length")),
        "index_head_dim": int(value("attention.indexer.key_length")),
        "index_n_heads": int(value("attention.indexer.head_count")),
        "index_topk": int(value("attention.indexer.top_k")),
        "initializer_range": 0.02,
        "max_position_embeddings": int(value("context_length")),
        "model_type": "deepseek_v4",
        "moe_intermediate_size": int(value("expert_feed_forward_length")),
        "n_routed_experts": int(value("expert_count")),
        "n_shared_experts": int(value("expert_shared_count")),
        "norm_topk_prob": bool(value("expert_weights_norm")),
        "num_attention_heads": int(value("attention.head_count")),
        "num_experts_per_tok": int(value("expert_used_count")),
        "num_hidden_layers": int(value("block_count")),
        "num_hash_layers": int(value("hash_layer_count")),
        "num_key_value_heads": int(value("attention.head_count_kv")),
        "num_nextn_predict_layers": 1,
        "o_groups": int(value("attention.output_group_count")),
        "o_lora_rank": int(value("attention.output_lora_rank")),
        "q_lora_rank": int(value("attention.q_lora_rank")),
        "qk_rope_head_dim": int(value("rope.dimension_count")),
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
        "rms_norm_eps": float(value("attention.layer_norm_rms_epsilon")),
        "rope_scaling": {
            "beta_fast": float(value("rope.scaling.yarn_beta_fast")),
            "beta_slow": float(value("rope.scaling.yarn_beta_slow")),
            "factor": float(value("rope.scaling.factor")),
            "original_max_position_embeddings": int(
                value("rope.scaling.original_context_length")
            ),
            "type": str(value("rope.scaling.type")),
        },
        "rope_theta": float(value("rope.freq_base")),
        "routed_scaling_factor": float(value("expert_weights_scale")),
        "scoring_func": "sqrtsoftplus",
        "sliding_window": int(value("attention.sliding_window")),
        "swiglu_limit": swiglu_limits[0],
        "tie_word_embeddings": False,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "transformers_version": importlib.metadata.version("transformers"),
        "use_cache": True,
        "vocab_size": len(tokens),
        "compress_rope_theta": float(value("attention.compress_rope_freq_base")),
        "compress_ratios": [int(item) for item in value("attention.compress_ratios")],
    }


def _write_tokenizer_from_gguf(
    reader: object, output_dir: Path, config: dict[str, Any]
) -> None:
    from tokenizers import AddedToken, Regex, normalizers, pre_tokenizers
    from transformers.integrations.ggml import convert_gguf_tokenizer

    tokenizer_type = _gguf_value(reader, "tokenizer.ggml.model")
    pre_tokenizer_type = _gguf_value(reader, "tokenizer.ggml.pre")
    if tokenizer_type != "gpt2" or pre_tokenizer_type != "joyai-llm":
        raise ValueError(
            f"unsupported GGUF tokenizer: model={tokenizer_type!r} pre={pre_tokenizer_type!r}"
        )
    tokens = list(_gguf_value(reader, "tokenizer.ggml.tokens"))
    token_types = list(_gguf_value(reader, "tokenizer.ggml.token_type"))
    tokenizer_data = {
        "tokenizer_type": tokenizer_type,
        "tokens": tokens,
        "token_type": token_types,
        "merges": list(_gguf_value(reader, "tokenizer.ggml.merges")),
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
        "pad_token_id": int(_gguf_value(reader, "tokenizer.ggml.padding_token_id")),
    }
    tokenizer, _ = convert_gguf_tokenizer("gpt2", tokenizer_data)
    tokenizer.add_special_tokens(
        [
            AddedToken(token, normalized=False, special=True)
            for token, token_type in zip(tokens, token_types)
            if token_type in (3, 4)
        ]
    )
    tokenizer.normalizer = normalizers.Sequence([])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\p{N}{1,3}"), behavior="isolated"),
            pre_tokenizers.Split(
                Regex(r"[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+"),
                behavior="isolated",
            ),
            pre_tokenizers.Split(
                Regex(
                    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|"
                    r"[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|"
                    r" ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                ),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False, trim_offsets=True, use_regex=False
            ),
        ]
    )
    tokenizer.save(str(output_dir / "tokenizer.json"))
    token = lambda token_id: {
        "__type": "AddedToken",
        "content": tokens[token_id],
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    }
    tokenizer_config = {
        "add_bos_token": bool(_gguf_value(reader, "tokenizer.ggml.add_bos_token")),
        "add_eos_token": bool(_gguf_value(reader, "tokenizer.ggml.add_eos_token")),
        "bos_token": token(config["bos_token_id"]),
        "chat_template": _gguf_value(reader, "tokenizer.chat_template"),
        "clean_up_tokenization_spaces": False,
        "eos_token": token(config["eos_token_id"]),
        "model_max_length": config["max_position_embeddings"],
        "pad_token": token(int(_gguf_value(reader, "tokenizer.ggml.padding_token_id"))),
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": None,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_model_metadata(args: argparse.Namespace) -> Path:
    source = args.gguf
    output_dir = artifact_dir_for_source(source) / "model-meta"
    marker = output_dir / "metadata.json"
    stat = source.stat()
    if (
        marker.is_file()
        and (output_dir / "config.json").is_file()
        and (output_dir / "tokenizer.json").is_file()
    ):
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            if (
                metadata.get("size") == stat.st_size
                and metadata.get("mtime_ns") == stat.st_mtime_ns
            ):
                return output_dir / "config.json"
        except (OSError, ValueError):
            pass
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError(
            "the gguf Python package is required to prepare DeepSeek metadata"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = gguf.GGUFReader(str(source), "r")
    config = _model_config_from_gguf(reader)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    generation_config = {
        "_from_model_config": True,
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
        "do_sample": True,
        "temperature": float(_gguf_value(reader, "general.sampling.temp")),
        "top_p": float(_gguf_value(reader, "general.sampling.top_p")),
    }
    (output_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2) + "\n", encoding="utf-8"
    )
    _write_tokenizer_from_gguf(reader, output_dir, config)
    marker.write_text(
        json.dumps(
            {
                "format_version": METADATA_FORMAT_VERSION,
                "gguf": str(source),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir / "config.json"


def prepare_expert_pack(
    args: argparse.Namespace, model_config: Path
) -> tuple[Path, Path]:
    tool = args.sglang_repo / "tools" / "expert_pack" / "prepare_deepseek_pack.py"
    if not tool.is_file():
        raise FileNotFoundError(f"missing DeepSeek Expert Pack preparer: {tool}")
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--gguf",
            str(args.gguf),
            "--model-config",
            str(model_config),
        ],
        cwd=args.sglang_repo,
        check=True,
    )
    return (
        args.gguf.parent / "DeepSeek-V4-Flash.expert-pack",
        args.gguf.parent / "DeepSeek-V4-Flash.expert-pack.manifest.json",
    )


def validate_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("model_path", "tokenizer_path"):
        path = getattr(args, name)
        if not path.is_dir():
            raise FileNotFoundError(
                f"--{name.replace('_', '-')} is not a directory: {path}"
            )
    for name in ("source_path", "pack_path", "manifest_path"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} is not a file: {path}")

    try:
        args.artifact_dir.relative_to(args.sglang_repo)
    except ValueError:
        pass
    else:
        raise ValueError("--artifact-dir must be outside the SGLang checkout")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Expert Pack manifest is not complete")
    if manifest.get("format") != "SGLANG-EXPERTPACK-v1":
        raise ValueError(f"unsupported Expert Pack format: {manifest.get('format')!r}")
    if int(manifest.get("pack_size", -1)) != args.pack_path.stat().st_size:
        raise ValueError("Expert Pack size does not match its manifest")

    source = manifest.get("source") or {}
    model = manifest.get("model") or {}
    if int(source.get("size", -1)) != args.source_path.stat().st_size:
        raise ValueError("GGUF source size does not match its manifest")
    source_sha256 = _digest(source.get("sha256"), "source.sha256")
    model_identity_sha256 = _digest(
        model.get("model_identity_sha256") or model.get("ollama_manifest_sha256"),
        "model.ollama_manifest_sha256",
    )
    config_sha256 = _digest(model.get("config_sha256"), "model.config_sha256")
    if args.validate_only and args.verify_source_sha256:
        if _sha256(args.source_path) != source_sha256:
            raise ValueError("GGUF source SHA-256 does not match its manifest")
    if args.validate_only and args.verify_pack_sha256:
        pack_sha256 = _digest(manifest.get("pack_sha256"), "pack_sha256")
        if _sha256(args.pack_path) != pack_sha256:
            raise ValueError("Expert Pack SHA-256 does not match its manifest")

    return {
        "manifest": manifest,
        "loader_extra_config": {
            "pack_path": str(args.pack_path),
            "manifest_path": str(args.manifest_path),
            "source_path": str(args.source_path),
            "source_sha256": source_sha256,
            "ollama_manifest_sha256": model_identity_sha256,
            "config_sha256": config_sha256,
            "cache_vram_mib": args.expert_cache_mib,
            "cache_vram_reserve_mib": args.expert_cache_reserve_mib,
            "stage_slots": args.stage_slots,
            "read_splits": args.read_splits,
            "direct_io": args.direct_io,
            "stats_flush_interval": args.stats_flush_interval,
            "stats_path": str(args.stats_path),
        },
    }


def build_server_command(
    args: argparse.Namespace, loader_extra_config: dict[str, Any]
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model_path),
        "--tokenizer-path",
        str(args.tokenizer_path),
        "--load-format",
        "expert_pack",
        "--model-loader-extra-config",
        json.dumps(loader_extra_config, separators=(",", ":")),
        "--attention-backend",
        "dsv4",
        "--tp-size",
        "1",
        "--ep-size",
        "1",
        "--disable-flashinfer-autotune",
        "--skip-server-warmup",
        "--context-length",
        str(args.context_length),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--max-running-requests",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--watchdog-timeout",
        str(args.watchdog_timeout),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]


def start_server(
    args: argparse.Namespace, loader_extra_config: dict[str, Any]
) -> subprocess.Popen:
    if port_in_use(args.host, args.port):
        raise RuntimeError(f"server address is already in use: {server_url(args)}")

    log = args.server_log.open("wb", buffering=0)
    env = os.environ.copy()
    python_path = [str(args.sglang_repo), str(args.sglang_repo / "python")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env.setdefault("SGLANG_OPT_USE_TILELANG_INDEXER", "1")
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
    command = build_server_command(args, loader_extra_config)
    print(
        f"SERVICE_STARTING url={server_url(args)} "
        f"timeout={args.startup_timeout:.0f}s log={args.server_log}",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=args.sglang_repo,
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
        stop_server(process, args)
        raise


def stop_server(process: subprocess.Popen | None, args: argparse.Namespace) -> None:
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
            port_in_use(args.host, args.port, timeout=0.2)
            and time.monotonic() < deadline
        ):
            time.sleep(0.2)
        print(f"SERVICE_STOPPED pid={process.pid} url={server_url(args)}", flush=True)
    finally:
        if log is not None:
            log.close()


def generate(
    args: argparse.Namespace,
    prompt: str,
    max_new_tokens: int,
    *,
    stream_output: bool,
) -> dict[str, Any]:
    payload = {
        "text": format_prompt(prompt),
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sampling_seed": args.seed,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    request = urllib.request.Request(
        server_url(args) + "/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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
    if stream_output:
        print(f"prompt: {prompt}", flush=True)
        print("output: ", end="", flush=True)

    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
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
        raise RuntimeError(
            "SGLang response did not contain complete token timing metadata"
        )
    ttft_s = (first_token - started) / 1e9
    decode_span_s = (last_token - first_token) / 1e9
    total_s = (time.perf_counter_ns() - started) / 1e9
    decode_intervals = max(0, completion_tokens - 1)
    return {
        "prompt": prompt,
        "output": output,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_ms": ttft_s * 1000,
        "prefill_token_rate": prompt_tokens / ttft_s if ttft_s > 0 else None,
        "decode_token_rate": (
            decode_intervals / decode_span_s if decode_span_s > 0 else None
        ),
        "tpot_ms": (
            decode_span_s * 1000 / decode_intervals if decode_intervals else None
        ),
        "total_elapsed_s": total_s,
        "end_to_end_token_rate": completion_tokens / total_s if total_s > 0 else None,
    }


def run_benchmark(
    args: argparse.Namespace, loader_extra_config: dict[str, Any]
) -> dict[str, Any]:
    process = None
    try:
        stats_path = getattr(args, "stats_path", None)
        if stats_path is not None and stats_path.exists():
            stats_path.unlink()
        process = start_server(args, loader_extra_config)
        if args.warmup:
            generate(
                args,
                DEFAULT_WARMUP_PROMPT,
                args.warmup_tokens,
                stream_output=False,
            )
        return generate(
            args,
            args.prompt,
            args.max_new_tokens,
            stream_output=True,
        )
    finally:
        stop_server(process, args)


def read_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"expert-pack stats were not written: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_routes(stats: dict[str, Any], expected_tokens: int) -> None:
    token_counts = stats.get("route_tokens_by_layer") or []
    call_counts = stats.get("route_calls_by_layer") or []
    if len(token_counts) != len(ACTIVE_MOE_LAYERS) or len(call_counts) != len(
        ACTIVE_MOE_LAYERS
    ):
        raise RuntimeError("DeepSeek Expert Pack stats have an unexpected layer count")
    for layer in ACTIVE_MOE_LAYERS:
        if call_counts[layer] <= 0 or token_counts[layer] != expected_tokens:
            raise RuntimeError(
                f"layer {layer} routed {token_counts[layer]} tokens in "
                f"{call_counts[layer]} calls; expected {expected_tokens} tokens"
            )
    if int(stats.get("fallback_count", 0)) != 0:
        raise RuntimeError("the request used an expert fallback")
    if int(stats.get("io_errors", 0)) != 0:
        raise RuntimeError("the request encountered Expert Pack I/O errors")


def _git_sha(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_report(
    args: argparse.Namespace,
    gpu: str,
    artifacts: dict[str, Any],
    result: dict[str, Any],
) -> None:
    stats = None
    if args.stats_path.is_file():
        stats = json.loads(args.stats_path.read_text(encoding="utf-8"))
    report = {
        "format": "SGLANG-DEEPSEEK-V4-FLASH-EXPERT-PACK-BENCHMARK-v1",
        "git_sha": _git_sha(args.sglang_repo),
        "gpu": gpu,
        "model_path": str(args.model_path),
        "source_path": str(args.source_path),
        "pack_path": str(args.pack_path),
        "manifest_path": str(args.manifest_path),
        "loader_extra_config": artifacts["loader_extra_config"],
        "result": result,
        "expert_pack_stats": stats,
        "server_log": str(args.server_log),
    }
    temporary = args.report_path.with_suffix(args.report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.report_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gguf",
        type=Path,
        required=True,
        help="DeepSeek-V4-Flash source GGUF; all other assets are derived",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.set_defaults(
        prompt=DEFAULT_PROMPT,
        temperature=0.0,
        top_p=0.95,
        seed=20260810,
        warmup=False,
        warmup_tokens=20,
        host="127.0.0.1",
        port=30000,
        startup_timeout=1200,
        request_timeout=3600,
        watchdog_timeout=1800,
        context_length=32768,
        max_total_tokens=32768,
        mem_fraction_static=0.96,
        expert_cache_mib=21 * 1024,
        expert_cache_reserve_mib=2 * 1024,
        stage_slots=12,
        read_splits=4,
        stats_flush_interval=len(ACTIVE_MOE_LAYERS),
        direct_io=True,
        verify_source_sha256=False,
        verify_pack_sha256=False,
        validate_only=False,
    )
    args = parser.parse_args(argv)

    args.sglang_repo = find_sglang_repo()
    args.gguf = args.gguf.expanduser().resolve(strict=True)
    args.source_path = args.gguf
    args.artifact_dir = artifact_dir_for_source(args.gguf).resolve()
    args.model_path = args.artifact_dir / "model-meta"
    args.tokenizer_path = args.model_path
    args.pack_path = args.gguf.parent / "DeepSeek-V4-Flash.expert-pack"
    args.manifest_path = (
        args.gguf.parent / "DeepSeek-V4-Flash.expert-pack.manifest.json"
    )
    args.server_log = args.artifact_dir / "deepseek-v4-5090-server.log"
    args.stats_path = args.artifact_dir / "deepseek-v4-expert-pack.stats.json"
    args.report_path = args.artifact_dir / "deepseek-v4-5090-benchmark.json"

    positive = (
        "max_new_tokens",
        "warmup_tokens",
        "port",
        "startup_timeout",
        "request_timeout",
        "watchdog_timeout",
        "context_length",
        "max_total_tokens",
        "expert_cache_mib",
        "expert_cache_reserve_mib",
        "stage_slots",
        "read_splits",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.stats_flush_interval < 0:
        parser.error("--stats-flush-interval cannot be negative")
    if args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0 < args.mem_fraction_static <= 1:
        parser.error("--mem-fraction-static must be in (0, 1]")
    return args


def handle_termination(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def print_result(result: dict[str, Any], gpu: str) -> None:
    print(f"gpu: {gpu}")
    print(f"prompt_tokens: {result['prompt_tokens']}")
    print(f"completion_tokens: {result['completion_tokens']}")
    print(f"ttft_ms: {result['ttft_ms']:.3f}")
    print(f"prefill_token_rate: {result['prefill_token_rate']:.3f} tok/s")
    print(f"decode_token_rate: {result['decode_token_rate']:.3f} tok/s")
    print(f"tpot_ms: {result['tpot_ms']:.3f} ms/token")
    print(f"end_to_end_token_rate: {result['end_to_end_token_rate']:.3f} tok/s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGHUP, handle_termination)
    lock = DEFAULT_LOCK.open("w")
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

    try:
        gpu = detect_rtx_5090()
        config_path = prepare_model_metadata(args)
        args.model_path = config_path.parent
        args.tokenizer_path = config_path.parent
        args.pack_path, args.manifest_path = prepare_expert_pack(args, config_path)
        artifacts = validate_artifacts(args)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "manifest": str(args.manifest_path),
                        "pack": str(args.pack_path),
                        "source": str(args.source_path),
                        "metadata": str(config_path.parent),
                    },
                    sort_keys=True,
                )
            )
            return 0
        print(
            f"HARDWARE_READY gpu={gpu} metadata={config_path.parent} pack={args.pack_path}",
            flush=True,
        )
        result = run_benchmark(args, artifacts["loader_extra_config"])
        stats = read_stats(args.stats_path)
        audit_routes(stats, result["prompt_tokens"] + result["completion_tokens"])
        write_report(args, gpu, artifacts, result)
        print_result(result, gpu)
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
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
