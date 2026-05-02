#!/usr/bin/env python3
"""Validate cross-framework cookbook benchmark configs.

The validator is intentionally shallow: it proves that every config can be
loaded, translated into bounded candidate commands, and checked against the
known server flag surface. It does not launch model servers.
"""

from __future__ import annotations

import argparse
import itertools
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

FRAMEWORKS = ("sglang", "vllm", "tensorrt_llm")
ALLOWED_SOURCE_KINDS = {"llm_serving_cookbook"}

SEQUENCE_LIMIT_KEY = {
    "sglang": "context_length",
    "vllm": "max_model_len",
    "tensorrt_llm": "max_seq_len",
}

ALLOWED_SLA_KEYS = {
    "max_p99_ttft_ms",
    "max_p99_tpot_ms",
    "min_success_rate",
    "max_p99_e2e_ms",
}

DEPRECATED_SLA_KEYS = {
    "max_ttft_ms": "max_p99_ttft_ms",
    "max_tpot_ms": "max_p99_tpot_ms",
    "max_e2e_ms": "max_p99_e2e_ms",
}

STATIC_SERVER_FLAGS = {
    "sglang": {
        "attention_backend",
        "chunked_prefill_size",
        "context_length",
        "decode_attention_backend",
        "dllm_algorithm",
        "dtype",
        "enable_multimodal",
        "enable_symm_mem",
        "ep_size",
        "host",
        "kv_cache_dtype",
        "max_running_requests",
        "mem_fraction_static",
        "model_loader_extra_config",
        "model_path",
        "moe_runner_backend",
        "nnodes",
        "port",
        "pp_size",
        "prefill_attention_backend",
        "reasoning_parser",
        "schedule_policy",
        "tool_call_parser",
        "tp_size",
        "trust_remote_code",
    },
    "vllm": {
        "block_size",
        "dtype",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "gpu_memory_utilization",
        "host",
        "kv_cache_dtype",
        "long_prefill_token_threshold",
        "max_long_partial_prefills",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_partial_prefills",
        "max_num_seqs",
        "pipeline_parallel_size",
        "port",
        "tensor_parallel_size",
        "trust_remote_code",
    },
    "tensorrt_llm": {
        "backend",
        "ep_size",
        "extra_llm_api_options",
        "host",
        "kv_cache_free_gpu_memory_fraction",
        "max_batch_size",
        "max_num_tokens",
        "max_seq_len",
        "port",
        "pp_size",
        "tp_size",
        "trust_remote_code",
    },
}

HELP_FILE_HINTS = {
    "sglang": ("sglang", "launch"),
    "vllm": ("vllm", "serve"),
    "tensorrt_llm": ("trtllm", "serve"),
}


def flag_name(framework: str, key: str) -> str:
    if framework in {"sglang", "vllm"}:
        return "--" + key.replace("_", "-")
    return "--" + key


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _enabled(config: dict[str, Any], framework: str) -> bool:
    return bool(config.get("frameworks", {}).get(framework, {}).get("enabled", False))


def _max_required_sequence(dataset: dict[str, Any]) -> int:
    input_len = dataset.get("input_len")
    output_len = dataset.get("output_len")
    if not isinstance(input_len, list) or not isinstance(output_len, list):
        raise ValueError("dataset.input_len and dataset.output_len must be lists")
    if len(input_len) != len(output_len):
        raise ValueError("dataset.input_len and dataset.output_len must be aligned")
    if not input_len:
        raise ValueError("dataset.input_len and dataset.output_len must not be empty")
    return max(int(i) + int(o) for i, o in zip(input_len, output_len, strict=True))


def _candidate_dicts(
    base_flags: dict[str, Any],
    search_space: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    candidates = [dict(base_flags)]
    keys = list(search_space)
    values = [_as_list(search_space[key]) for key in keys]
    for combo in itertools.product(*values):
        candidate = dict(base_flags)
        candidate.update(dict(zip(keys, combo, strict=True)))
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def _command_tokens(
    framework: str,
    config: dict[str, Any],
    flags: dict[str, Any],
) -> list[str]:
    server = config["frameworks"][framework]
    command = shlex.split(server["server_command"])
    model = config["model"]["name"]

    if framework in {"vllm", "tensorrt_llm"}:
        command.append(model)

    for key, value in flags.items():
        if value is None or value is False:
            continue
        command.append(flag_name(framework, key))
        if value is not True:
            command.append(str(value))

    return command


def render_command(
    framework: str, config: dict[str, Any], flags: dict[str, Any]
) -> str:
    return shlex.join(_command_tokens(framework, config, flags))


def _extract_help_flags(text: str) -> set[str]:
    return {
        item.lstrip("-") for item in re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", text)
    }


def load_help_flags(help_dir: Path) -> dict[str, set[str]]:
    help_flags: dict[str, set[str]] = {}
    for framework, hints in HELP_FILE_HINTS.items():
        matches = []
        for path in help_dir.rglob("*.txt"):
            name = path.name.lower()
            if all(hint in name for hint in hints):
                matches.append(path)
        if matches:
            text = "\n".join(
                path.read_text(encoding="utf-8", errors="replace") for path in matches
            )
            help_flags[framework] = _extract_help_flags(text)
    return help_flags


def _known_flag(
    framework: str,
    key: str,
    help_flags: dict[str, set[str]] | None,
) -> bool:
    static_keys = STATIC_SERVER_FLAGS[framework]
    if key not in static_keys:
        return False
    if not help_flags or framework not in help_flags:
        return True

    concrete = flag_name(framework, key).lstrip("-")
    aliases = {concrete, concrete.replace("-", "_"), concrete.replace("_", "-")}
    return bool(aliases & help_flags[framework])


def _validate_framework(
    config: dict[str, Any],
    framework: str,
    help_flags: dict[str, set[str]] | None,
    max_candidates: int,
) -> list[str]:
    errors: list[str] = []
    server = config["frameworks"].get(framework)
    if not isinstance(server, dict):
        return [f"missing frameworks.{framework}"]
    if not server.get("enabled", False):
        return []

    base_flags = server.get("base_server_flags")
    search_space = server.get("search_space")
    if not isinstance(base_flags, dict):
        errors.append(f"{framework}: base_server_flags must be a mapping")
        base_flags = {}
    if not isinstance(search_space, dict):
        errors.append(f"{framework}: search_space must be a mapping")
        search_space = {}
    server_command_is_valid = isinstance(server.get("server_command"), str)
    if not server_command_is_valid:
        errors.append(f"{framework}: server_command must be a string")

    for key in set(base_flags) | set(search_space):
        if not _known_flag(framework, key, help_flags):
            errors.append(f"{framework}: unknown or unsupported server flag {key!r}")

    if framework == "tensorrt_llm":
        if server.get("backend_policy") != "fixed_pytorch":
            errors.append("tensorrt_llm: backend_policy must be fixed_pytorch")
        if base_flags.get("backend") != "pytorch":
            errors.append("tensorrt_llm: base backend must be pytorch")
        if "backend" in search_space:
            errors.append("tensorrt_llm: backend must not appear in search_space")

    candidates = _candidate_dicts(base_flags, search_space, max_candidates)
    if not candidates:
        errors.append(f"{framework}: no candidates generated")
    can_render = server_command_is_valid and isinstance(
        config.get("model", {}).get("name"), str
    )
    if can_render:
        for candidate in candidates:
            command = render_command(framework, config, candidate)
            if not command:
                errors.append(f"{framework}: rendered an empty command")

    return errors


def validate_config(
    path: Path,
    help_flags: dict[str, set[str]] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        config = load_yaml(path)
    except Exception as exc:  # noqa: BLE001
        return [str(exc)]

    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(config.get("model", {}).get("name"), str):
        errors.append("model.name must be set")
    if config.get("source", {}).get("kind") not in ALLOWED_SOURCE_KINDS:
        errors.append(f"source.kind must be one of {sorted(ALLOWED_SOURCE_KINDS)}")

    try:
        required_sequence = _max_required_sequence(config["dataset"])
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        required_sequence = 0

    search = config.get("search")
    if not isinstance(search, dict):
        errors.append("search must be a mapping")
        max_candidates = 1
    else:
        try:
            max_candidates = int(search.get("max_candidates_per_framework", 0))
        except (TypeError, ValueError):
            errors.append("search.max_candidates_per_framework must be an integer")
            max_candidates = 1
        if max_candidates < 1:
            errors.append("search.max_candidates_per_framework must be positive")
            max_candidates = 1

    frameworks = config.get("frameworks")
    if not isinstance(frameworks, dict):
        return errors + ["frameworks must be a mapping"]

    for framework in FRAMEWORKS:
        errors.extend(
            _validate_framework(config, framework, help_flags, max_candidates)
        )

    for framework in FRAMEWORKS:
        if not _enabled(config, framework):
            continue
        key = SEQUENCE_LIMIT_KEY[framework]
        fw = frameworks[framework]
        base_flags = fw.get("base_server_flags", {}) or {}
        search_space = fw.get("search_space", {}) or {}
        if not isinstance(base_flags, dict) or not isinstance(search_space, dict):
            continue

        try:
            if framework == "sglang":
                base_value = int(base_flags.get(key, required_sequence))
            else:
                base_value = int(base_flags.get(key, 0))
        except (TypeError, ValueError):
            errors.append(f"{framework}: base {key} is not an integer")
            continue
        if base_value < required_sequence:
            errors.append(
                f"{framework}: base {key} ({base_value}) is smaller than the largest dataset scenario ({required_sequence})"
            )

        if key in search_space:
            for value in _as_list(search_space[key]):
                try:
                    if int(value) < required_sequence:
                        errors.append(
                            f"{framework}: search_space {key} candidate {value} is smaller than the largest dataset scenario ({required_sequence})"
                        )
                except (TypeError, ValueError):
                    errors.append(
                        f"{framework}: search_space {key} candidate {value!r} is not an integer"
                    )

    sla_block = (
        config.get("benchmark", {}).get("sla")
        if isinstance(config.get("benchmark"), dict)
        else None
    )
    if sla_block is None:
        sla_block = config.get("sla")
    if isinstance(sla_block, dict):
        for key in sla_block:
            if key in DEPRECATED_SLA_KEYS:
                errors.append(
                    f"sla: {key!r} is deprecated; use {DEPRECATED_SLA_KEYS[key]!r} (see references/result-schema.md)"
                )
            elif key not in ALLOWED_SLA_KEYS:
                errors.append(
                    f"sla: unknown key {key!r}; allowed keys are {sorted(ALLOWED_SLA_KEYS)}"
                )

    return errors


def iter_config_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.yaml")))
            files.extend(sorted(path.rglob("*.yml")))
        else:
            files.append(path)
    return sorted(dict.fromkeys(files))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--help-dir", type=Path)
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()

    help_flags = load_help_flags(args.help_dir) if args.help_dir else None
    failed = False
    for path in iter_config_files(args.paths):
        errors = validate_config(path, help_flags)
        if errors:
            failed = True
            for error in errors:
                print(f"{path}: {error}")
            continue

        if args.print_commands:
            config = load_yaml(path)
            limit = int(config["search"].get("max_candidates_per_framework", 1))
            for framework in FRAMEWORKS:
                if not _enabled(config, framework):
                    continue
                server = config["frameworks"][framework]
                candidates = _candidate_dicts(
                    server["base_server_flags"],
                    server["search_space"],
                    limit,
                )
                print(f"# {path.name} {framework}")
                print(render_command(framework, config, candidates[0]))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
