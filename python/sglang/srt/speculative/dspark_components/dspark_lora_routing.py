# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for immutable, per-request DSpark draft adapters."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from types import MappingProxyType


@lru_cache(maxsize=16)
def parse_draft_adapters(raw: str | None):
    if raw is None:
        return MappingProxyType({})

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate draft adapter name: {key}")
            result[key] = value
        return result

    entries = json.loads(raw, object_pairs_hook=unique_pairs)
    if not isinstance(entries, dict) or not entries:
        raise ValueError(
            "--speculative-dspark-lora-paths must be a nonempty JSON object."
        )
    for name, path in entries.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError(f"Invalid draft adapter name: {name!r}")
        if not isinstance(path, str) or not path:
            raise ValueError(f"Draft adapter {name!r} needs a local directory path.")
    return MappingProxyType(entries)


def validate_draft_adapter_server_config(cfg):
    parse_draft_adapters(cfg.speculative_dspark_lora_paths)
    if (cfg.speculative_algorithm or "").upper() != "DSPARK":
        raise ValueError("Per-request draft adapters require DSPARK.")
    if cfg.speculative_dspark_lora_path:
        raise ValueError(
            "Choose either startup draft LoRA or per-request draft adapters."
        )
    if (cfg.speculative_draft_load_format or cfg.load_format) not in {
        "auto",
        "pt",
        "safetensors",
    }:
        raise ValueError("Per-request draft adapters require an ordinary dense loader.")
    required = {
        "tp_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "attn_cp_size": 1,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "schedule_policy": "fcfs",
        "enable_priority_scheduling": False,
        "enable_mixed_chunk": False,
        "enable_dp_attention": False,
        "enable_hierarchical_cache": False,
        "enable_hisparse": False,
        "enable_flexkv": False,
        "enable_unified_cache_external_linker": False,
        "enable_unified_memory": False,
        "enable_torch_compile": False,
        "enable_memory_saver": False,
        "checkpoint_engine_wait_weights_before_ready": False,
        "cpu_offload_gb": 0,
        "disaggregation_mode": "null",
    }
    for field, expected in required.items():
        if getattr(cfg, field) != expected:
            raise ValueError(
                f"Per-request draft adapters require {field}={expected!r}."
            )
    if not cfg.device.startswith("cuda"):
        raise ValueError("Per-request draft adapters currently require CUDA.")
    # The graph resolution hook has already run. Inspect resolved backends,
    # not legacy booleans, so explicit JSON graph settings cannot bypass this.
    for phase in (cfg.cuda_graph_config.prefill, cfg.cuda_graph_config.decode):
        if phase.backend != "disabled":
            raise ValueError(
                "Per-request draft adapters require both prefill and decode "
                "CUDA graphs disabled."
            )


def normalize_draft_adapter(value, *, batch_size, parallel_samples, single):
    def validate_name(name):
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("draft_adapter entries must be nonempty names or null.")

    if single:
        validate_name(value)
    elif isinstance(value, list):
        if len(value) != batch_size:
            raise ValueError("draft_adapter list length must match the input batch.")
        for name in value:
            validate_name(name)
    else:
        validate_name(value)
    if value is None or (single and parallel_samples == 1):
        return value
    values = value if isinstance(value, list) else [value] * batch_size
    # Matches GenerateReqInput._expand_inputs: repeat the entire input batch.
    return values * parallel_samples


def validate_draft_adapter_request(request, registry):
    names = request.draft_adapter
    names = names if isinstance(names, list) else [names]
    for name in names:
        if name is not None and name not in registry:
            raise ValueError(f"Unknown draft_adapter {name!r}; use a configured name.")
    # Session-held KV may outlive a cohort. Reject for the whole bank deployment,
    # including base-draft requests, until adapter affinity is stored in sessions.
    if registry:
        params = request.sampling_params
        params = params if isinstance(params, list) else [params or {}]
        if any((item.get("beam_width") or 1) > 1 for item in params):
            raise ValueError(
                "Beam search is not supported with per-request draft adapters."
            )
    if registry and (
        request.session_id is not None or request.session_params is not None
    ):
        raise ValueError("Sessions are not supported with per-request draft adapters.")


def homogeneous_draft_adapter(names):
    if not names or any(name != names[0] for name in names[1:]):
        raise ValueError(
            "DSpark draft adapter batches must be nonempty and homogeneous."
        )
    return names[0]


class DraftAdapterCohort:
    """FCFS admission gate. Stop scanning at the first foreign adapter.

    Existing requests (including chunked prefills) keep their draft until they
    finish or retract. A retracted request receives a full prefill because radix
    caching is disabled. Keeping Req.draft_adapter immutable preserves affinity.
    """

    def __init__(self, active_names):
        self.selected = bool(active_names)
        self.name = homogeneous_draft_adapter(active_names) if active_names else None

    def admit(self, name):
        if not self.selected:
            self.name = name
            self.selected = True
        return name == self.name
