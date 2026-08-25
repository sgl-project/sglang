# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the public expert-pack load format."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
)
from sglang.srt.model_loader.expert_pack_config import (
    DEEPSEEK_V4_MODEL_TYPE,
    KIMI_K3_MODEL_TYPE,
    validate_expert_pack_model_config,
)

logger = logging.getLogger(__name__)


def handle_expert_pack(server_args: Any) -> None:
    """Normalize expert-pack settings and report all startup errors together."""
    if server_args.load_format != "expert_pack":
        return

    errors = []
    parallelism = (
        ("tensor", "--tp-size", server_args.tp_size),
        ("data", "--dp-size", server_args.dp_size),
        ("expert", "--ep-size", server_args.ep_size),
    )
    for label, option, size in parallelism:
        if size != 1:
            errors.append(f"{label} parallelism ({option}) must be 1, got {size}")

    if server_args.enforce_shared_experts_fusion:
        errors.append(
            "--enforce-shared-experts-fusion is incompatible with expert_pack"
        )
    if server_args.enable_waterfill:
        errors.append("--enable-waterfill is incompatible with expert_pack")

    explicit_cuda_graph_backends = {
        Phase.DECODE: server_args.cuda_graph_backend_decode,
        Phase.PREFILL: server_args.cuda_graph_backend_prefill,
    }
    raw_cuda_graph_config = server_args.cuda_graph_config
    if isinstance(raw_cuda_graph_config, CudaGraphConfig):
        raw_cuda_graph_config = raw_cuda_graph_config.to_dict()
    for phase in Phase.ALL:
        phase_config = (
            raw_cuda_graph_config.get(phase, {})
            if isinstance(raw_cuda_graph_config, dict)
            else {}
        )
        explicit_backend = phase_config.get(
            "backend", explicit_cuda_graph_backends[phase]
        )
        if explicit_backend not in (None, Backend.DISABLED):
            errors.append(
                f"expert_pack requires the {phase} CUDA graph backend to be "
                f"disabled, got {explicit_backend!r}"
            )

    loader_config = server_args.model_loader_extra_config or {}
    if isinstance(loader_config, str):
        try:
            loader_config = json.loads(loader_config)
        except (TypeError, json.JSONDecodeError) as exc:
            errors.append(f"--model-loader-extra-config must be valid JSON: {exc}")
            loader_config = {}
    if not isinstance(loader_config, dict):
        errors.append("--model-loader-extra-config must be a JSON object")
        loader_config = {}

    # A raw GGUF path is the public input form.  Preparation is performed once
    # here, before model-config parsing and before the loader is constructed.
    raw_model_path = Path(server_args.model_path).expanduser()
    raw_preparation_failed = False
    if not errors and raw_model_path.is_file():
        try:
            from sglang.srt.model_loader import expert_pack_runtime

            model_name = raw_model_path.name.upper()
            if "KIMI" in model_name:
                expert_pack_runtime.prepare_raw_kimi_server_args(
                    server_args, loader_config
                )
            elif "DEEPSEEK" in model_name:
                expert_pack_runtime.prepare_raw_deepseek_server_args(
                    server_args, loader_config
                )
            else:
                expert_pack_runtime.prepare_raw_expert_pack_server_args(
                    server_args, loader_config
                )
            declare_resolution(
                server_args,
                "handle_expert_pack",
                model_loader_extra_config=loader_config,
            )
        except Exception as exc:
            errors.append(f"failed to prepare raw expert_pack GGUF input: {exc}")
            raw_preparation_failed = True

    def parse_path(label: str, value: Any) -> Path | None:
        if not value:
            return None
        try:
            return Path(value).expanduser()
        except TypeError:
            errors.append(
                f"{label} must be a filesystem path, got {type(value).__name__}"
            )
            return None

    pack_path = None
    if not raw_preparation_failed:
        pack_path_value = loader_config.get("pack_path") or os.getenv(
            "SGLANG_EXPERT_PACK_PATH"
        )
        pack_path = parse_path("pack_path", pack_path_value)
        if pack_path is None:
            errors.append(
                "pack_path is required in --model-loader-extra-config or "
                "SGLANG_EXPERT_PACK_PATH"
            )
        elif not pack_path.is_file():
            errors.append(f"expert-pack file does not exist: {pack_path}")

    model_kind = None
    model_path = parse_path("--model-path", server_args.model_path)
    if not raw_preparation_failed:
        if model_path is None or not model_path.is_dir():
            errors.append(
                "--model-path must be a local GGUF shard or tokenizer/config "
                f"directory for expert_pack, got {server_args.model_path!r}"
            )
        else:
            try:
                hf_config = server_args.get_model_config().hf_config
            except Exception as exc:
                errors.append(f"failed to load expert_pack model config: {exc}")
            else:
                model_kind, model_errors = validate_expert_pack_model_config(hf_config)
                errors.extend(model_errors)

    manifest_path_value = loader_config.get("manifest_path")
    if model_kind == KIMI_K3_MODEL_TYPE and not manifest_path_value:
        errors.append("Kimi-K3 requires manifest_path in loader config")
    if manifest_path_value:
        manifest_path = parse_path("manifest_path", manifest_path_value)
    elif model_kind == DEEPSEEK_V4_MODEL_TYPE and pack_path is not None:
        manifest_path = Path(str(pack_path) + ".manifest.json")
    else:
        manifest_path = None
    if manifest_path is not None and not manifest_path.is_file():
        errors.append(f"expert-pack manifest does not exist: {manifest_path}")

    if model_kind == DEEPSEEK_V4_MODEL_TYPE:
        required = (
            "source_path",
            "source_sha256",
            "model_identity_sha256",
            "config_sha256",
        )
        for name in required:
            if not loader_config.get(name):
                errors.append(f"deepseek-v4-flash loader config requires {name}")
        source_path = parse_path("source_path", loader_config.get("source_path"))
        if source_path is not None and not source_path.is_file():
            errors.append(
                f"deepseek-v4-flash source GGUF does not exist: {source_path}"
            )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid expert_pack configuration:\n{details}")

    declare_resolution(
        server_args,
        "handle_expert_pack",
        disable_cuda_graph=True,
        disable_shared_experts_fusion=True,
    )
    if model_kind == DEEPSEEK_V4_MODEL_TYPE:
        envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)
    logger.info(
        "expert_pack selected: CUDA graph and shared-experts fusion are "
        "disabled for correctness."
    )
