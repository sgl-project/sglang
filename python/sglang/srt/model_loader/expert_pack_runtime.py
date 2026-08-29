# SPDX-License-Identifier: Apache-2.0
"""Internal preparation of source assets for the expert-pack load format."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

METADATA_FORMAT_VERSION = 3
GGUF_SHARD_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
DEEPSEEK_METADATA_FORMAT_VERSION = 4


def cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()


def artifact_dir_for_source(gguf: Path) -> Path:
    stat = gguf.stat()
    fingerprint = hashlib.sha256(
        f"{gguf.parent.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"{METADATA_FORMAT_VERSION}".encode()
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


def resolve_kimi_tokenizer(gguf: Path, explicit: str | None = None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not _tokenizer_candidate(candidate):
            raise ValueError(f"Kimi tokenizer directory is invalid: {candidate}")
        return candidate

    candidates = [gguf.parent / "tokenizer", gguf.parent.parent / "kimi-k3-tokenizer"]
    candidates.extend(
        sorted(path for path in gguf.parent.parent.glob("*tokenizer*") if path.is_dir())
    )
    tokenizers = []
    for path in candidates:
        path = path.resolve()
        if path not in tokenizers and _tokenizer_candidate(path):
            tokenizers.append(path)
    if len(tokenizers) != 1:
        names = ", ".join(str(path) for path in tokenizers) or "none"
        raise RuntimeError(
            f"could not uniquely derive Kimi tokenizer beside {gguf.parent}; "
            f"candidates: {names}"
        )
    return tokenizers[0]


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_kimi_model_metadata(tokenizer_dir: Path, artifact_dir: Path) -> Path:
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
    _write_json_atomic(output_dir / "config.json", config)
    return output_dir


def _expert_pack_path(gguf: Path) -> Path:
    match = GGUF_SHARD_SUFFIX_RE.search(gguf.name)
    if match is None:
        raise ValueError(f"Kimi GGUF is not a numbered shard: {gguf}")
    return gguf.parent / f"{gguf.name[: match.start()]}.expert-major.pack"


def _expert_pack_tools_dir() -> Path:
    tools_dir = Path(__file__).with_name("expert_pack")
    required_tools = (
        "prepare_deepseek_pack.py",
        "prepare_kimi_manifest.py",
        "prepare_kimi_pack.py",
    )
    missing = [name for name in required_tools if not (tools_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "expert_pack preparation tools are missing: " + ", ".join(missing)
        )
    return tools_dir


def ensure_kimi_assets(
    gguf: Path,
    *,
    tokenizer_dir: str | None = None,
) -> dict[str, Path]:
    """Build or reuse Kimi artifacts and return the internal serving paths."""
    gguf = gguf.expanduser().resolve(strict=True)
    if not gguf.is_file() or gguf.suffix != ".gguf":
        raise ValueError(f"expert_pack expects a local Kimi GGUF shard, got {gguf}")
    if "KIMI-K3" not in gguf.name.upper():
        raise ValueError(
            "raw GGUF auto-preparation currently supports only Kimi-K3; "
            "provide the model metadata and loader artifacts for other models"
        )
    gguf_dir = gguf.parent
    artifact_dir = artifact_dir_for_source(gguf).resolve()
    pack = _expert_pack_path(gguf).resolve()
    manifest = artifact_dir / "kimi-k3-expert-pack.manifest.json"
    tokenizer = resolve_kimi_tokenizer(gguf, tokenizer_dir)
    lock_path = pack.with_name(pack.name + ".startup.lock")
    tools_dir = _expert_pack_tools_dir()
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        model_dir = prepare_kimi_model_metadata(tokenizer, artifact_dir)
        subprocess.run(
            [
                sys.executable,
                str(tools_dir / "prepare_kimi_pack.py"),
                "--gguf",
                str(gguf),
                "--model-config",
                str(model_dir / "config.json"),
            ],
            cwd=tools_dir,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(tools_dir / "prepare_kimi_manifest.py"),
                "--gguf-dir",
                str(gguf_dir),
                "--expert-pack",
                str(pack),
                "--model-config",
                str(model_dir / "config.json"),
                "--tokenizer-dir",
                str(tokenizer),
                "--output",
                str(manifest),
                "--payload-samples",
                "6",
            ],
            cwd=tools_dir,
            check=True,
        )
    return {
        "gguf": gguf,
        "gguf_dir": gguf_dir,
        "tokenizer_dir": tokenizer,
        "model_dir": model_dir,
        "pack_path": pack,
        "manifest_path": manifest,
        "stats_path": artifact_dir / "kimi-k3-expert-pack.stats.json",
        "artifact_dir": artifact_dir,
    }


def prepare_raw_kimi_server_args(
    server_args: Any, loader_config: dict[str, Any]
) -> None:
    """Resolve a raw GGUF model path into the normal loader inputs."""

    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    model_path = Path(cfg.model_path).expanduser()
    if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
        return
    tokenizer_path = cfg.tokenizer_path
    if tokenizer_path and Path(tokenizer_path).expanduser() == model_path:
        tokenizer_path = None
    assets = ensure_kimi_assets(
        model_path,
        tokenizer_dir=tokenizer_path,
    )
    server_args._declare(
        "prepare_raw_kimi_server_args",
        model_path=str(assets["model_dir"]),
        tokenizer_path=str(assets["model_dir"]),
    )
    for key in ("pack_path", "manifest_path", "stats_path"):
        loader_config.setdefault(key, str(assets[key]))
    loader_config.setdefault("source_path", str(assets["gguf"]))


def _deepseek_cache_root() -> Path:
    return cache_root() / "sglang-expert-pack" / "deepseek-v4-flash"


def _deepseek_artifact_dir_for_source(source: Path) -> Path:
    stat = source.stat()
    fingerprint = hashlib.sha256(
        f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"{DEEPSEEK_METADATA_FORMAT_VERSION}".encode()
    ).hexdigest()[:20]
    return _deepseek_cache_root() / fingerprint


def _deepseek_gguf_value(reader: object, name: str) -> object:
    fields = getattr(reader, "fields")
    if name not in fields:
        raise ValueError(f"GGUF metadata is missing required field: {name}")
    return fields[name].contents()


def _deepseek_model_config_from_gguf(reader: object) -> dict[str, Any]:
    def value(name: str) -> object:
        return _deepseek_gguf_value(reader, f"deepseek4.{name}")

    architecture = _deepseek_gguf_value(reader, "general.architecture")
    if architecture != "deepseek4":
        raise ValueError(f"expected GGUF architecture deepseek4, got {architecture!r}")
    if int(value("expert_gating_func")) != 4:
        raise ValueError("unsupported deepseek4.expert_gating_func; expected 4")
    swiglu_limits = [float(item) for item in value("swiglu_clamp_exp")]
    if not swiglu_limits or any(item != swiglu_limits[0] for item in swiglu_limits):
        raise ValueError("deepseek4.swiglu_clamp_exp must be constant")
    tokens = _deepseek_gguf_value(reader, "tokenizer.ggml.tokens")
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": int(
            _deepseek_gguf_value(reader, "tokenizer.ggml.bos_token_id")
        ),
        "eos_token_id": int(
            _deepseek_gguf_value(reader, "tokenizer.ggml.eos_token_id")
        ),
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


def _write_deepseek_tokenizer_from_gguf(
    reader: object, output_dir: Path, config: dict[str, Any]
) -> None:
    from tokenizers import AddedToken, Regex, normalizers, pre_tokenizers
    from transformers.integrations.ggml import convert_gguf_tokenizer

    tokenizer_type = _deepseek_gguf_value(reader, "tokenizer.ggml.model")
    pre_tokenizer_type = _deepseek_gguf_value(reader, "tokenizer.ggml.pre")
    if tokenizer_type != "gpt2" or pre_tokenizer_type != "joyai-llm":
        raise ValueError(
            f"unsupported GGUF tokenizer: model={tokenizer_type!r} "
            f"pre={pre_tokenizer_type!r}"
        )
    tokens = list(_deepseek_gguf_value(reader, "tokenizer.ggml.tokens"))
    token_types = list(_deepseek_gguf_value(reader, "tokenizer.ggml.token_type"))
    tokenizer_data = {
        "tokenizer_type": tokenizer_type,
        "tokens": tokens,
        "token_type": token_types,
        "merges": list(_deepseek_gguf_value(reader, "tokenizer.ggml.merges")),
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
        "pad_token_id": int(
            _deepseek_gguf_value(reader, "tokenizer.ggml.padding_token_id")
        ),
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

    def token(token_id: int) -> dict[str, object]:
        return {
            "__type": "AddedToken",
            "content": tokens[token_id],
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        }

    tokenizer_config = {
        "add_bos_token": bool(
            _deepseek_gguf_value(reader, "tokenizer.ggml.add_bos_token")
        ),
        "add_eos_token": bool(
            _deepseek_gguf_value(reader, "tokenizer.ggml.add_eos_token")
        ),
        "bos_token": token(config["bos_token_id"]),
        "chat_template": _deepseek_gguf_value(reader, "tokenizer.chat_template"),
        "clean_up_tokenization_spaces": False,
        "eos_token": token(config["eos_token_id"]),
        "model_max_length": config["max_position_embeddings"],
        "pad_token": token(
            int(_deepseek_gguf_value(reader, "tokenizer.ggml.padding_token_id"))
        ),
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": None,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_deepseek_model_metadata(source: Path, artifact_dir: Path) -> Path:
    output_dir = artifact_dir / "model-meta"
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
    config = _deepseek_model_config_from_gguf(reader)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    generation_config = {
        "_from_model_config": True,
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
        "do_sample": True,
        "temperature": float(_deepseek_gguf_value(reader, "general.sampling.temp")),
        "top_p": float(_deepseek_gguf_value(reader, "general.sampling.top_p")),
    }
    (output_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2) + "\n", encoding="utf-8"
    )
    _write_deepseek_tokenizer_from_gguf(reader, output_dir, config)
    marker.write_text(
        json.dumps(
            {
                "format_version": DEEPSEEK_METADATA_FORMAT_VERSION,
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


def _deepseek_digest(value: object, field: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"manifest {field} is not a SHA-256 digest")
    return digest


def _prepare_deepseek_pack(
    source: Path, model_config: Path, tools_dir: Path
) -> tuple[Path, Path]:
    tool = tools_dir / "prepare_deepseek_pack.py"
    if not tool.is_file():
        raise FileNotFoundError(f"missing DeepSeek Expert Pack preparer: {tool}")
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--gguf",
            str(source),
            "--model-config",
            str(model_config),
        ],
        cwd=tools_dir,
        check=True,
    )
    return (
        source.parent / "DeepSeek-V4-Flash.expert-pack",
        source.parent / "DeepSeek-V4-Flash.expert-pack.manifest.json",
    )


def prepare_raw_deepseek_server_args(
    server_args: Any, loader_config: dict[str, Any]
) -> None:
    """Resolve a raw DeepSeek V4 GGUF into metadata and Expert Pack inputs."""

    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    source = Path(cfg.model_path).expanduser().resolve(strict=True)
    if not source.is_file():
        return
    tools_dir = _expert_pack_tools_dir()
    artifact_dir = _deepseek_artifact_dir_for_source(source).resolve()
    lock_path = artifact_dir / "deepseek-v4-startup.lock"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        model_config = _prepare_deepseek_model_metadata(source, artifact_dir)
        pack, manifest = _prepare_deepseek_pack(source, model_config, tools_dir)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    source_value = manifest_value.get("source") or {}
    model_value = manifest_value.get("model") or {}
    source_sha256 = _deepseek_digest(source_value.get("sha256"), "source.sha256")
    model_identity_sha256 = _deepseek_digest(
        model_value.get("model_identity_sha256"),
        "model.model_identity_sha256",
    )
    config_sha256 = _deepseek_digest(
        model_value.get("config_sha256"), "model.config_sha256"
    )
    server_args._declare(
        "prepare_raw_deepseek_server_args",
        model_path=str(model_config.parent),
        tokenizer_path=str(model_config.parent),
    )
    for key, value in {
        "pack_path": pack,
        "manifest_path": manifest,
        "source_path": source,
        "source_sha256": source_sha256,
        "model_identity_sha256": model_identity_sha256,
        "config_sha256": config_sha256,
        "stats_path": artifact_dir / "deepseek-v4-expert-pack.stats.json",
    }.items():
        loader_config.setdefault(key, str(value) if isinstance(value, Path) else value)


def prepare_raw_expert_pack_server_args(
    server_args: Any, loader_config: dict[str, Any]
) -> None:
    """Dispatch a raw GGUF to the model-specific expert-pack preparation path."""

    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    source = Path(cfg.model_path).expanduser()
    if not source.is_file():
        return
    name = source.name.upper()
    if "KIMI" in name:
        prepare_raw_kimi_server_args(server_args, loader_config)
        return
    if "DEEPSEEK" in name:
        prepare_raw_deepseek_server_args(server_args, loader_config)
        return
    try:
        import gguf

        reader = gguf.GGUFReader(str(source), "r")
        architecture = _deepseek_gguf_value(reader, "general.architecture")
    except Exception as exc:
        raise ValueError(
            "raw GGUF auto-preparation currently supports DeepSeek-V4 and Kimi-K3; "
            f"could not identify {source}: {exc}"
        ) from exc
    if architecture == "deepseek4":
        prepare_raw_deepseek_server_args(server_args, loader_config)
        return
    raise ValueError(
        "raw GGUF auto-preparation currently supports DeepSeek-V4 and Kimi-K3; "
        f"detected architecture {architecture!r}"
    )
