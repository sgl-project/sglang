"""Capability fingerprint a PD decode checks against its prefill peer before
registering any destination buffer."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, List, Optional

_SNAPSHOT_REVISION = re.compile(r"/snapshots/([0-9a-f]{40})(?:/|$)")
# Written into hf_config from per-role server args; they do not change KV bytes.
_RUNTIME_KEYS = frozenset(
    {
        "transformers_version",
        "context_len",
        "max_position_embeddings",
        "encoder_only",
        "language_only",
        "language_model_only",
    }
)
_SPECULATION_KEYS = (
    "algorithm",
    "num_draft_tokens",
    "num_steps",
    "eagle_topk",
    "dspark_block_size",
    "dflash_block_size",
)


def _config_dict(config: Any) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    return config.to_dict()


def _without_local_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_local_keys(item)
            for key, item in value.items()
            if not str(key).startswith("_") and key not in _RUNTIME_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_local_keys(item) for item in value]
    return value


def config_digest(config: Any) -> str:
    canonical = json.dumps(
        _without_local_keys(_config_dict(config)), sort_keys=True, default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def model_revision(model_path: Optional[str], revision: Optional[str]) -> Optional[str]:
    if revision:
        return revision
    match = _SNAPSHOT_REVISION.search(model_path or "")
    return match.group(1) if match else None


def build_peer_fingerprint(
    *,
    kv_args: Any,
    target_config: Any,
    target_model_path: Optional[str],
    target_revision: Optional[str],
    speculation: Optional[dict],
    draft_config: Any = None,
    draft_model_path: Optional[str] = None,
    draft_revision: Optional[str] = None,
    attn_tp_size: int,
    attn_cp_size: int,
    pp_size: int,
    dcp_size: int,
    draft_attn_tp_size: Optional[int],
    one_to_one_capable: bool,
) -> dict:
    """``speculation`` is None without spec decoding; otherwise it carries the
    _SPECULATION_KEYS plus ``requires_prefill_draft_kv``. A bundled draft
    (no separate draft checkpoint) is identified by the target model."""
    bundled = speculation is None or draft_model_path in (None, target_model_path)
    return {
        "target": {
            "config_digest": config_digest(target_config),
            "revision": model_revision(target_model_path, target_revision),
        },
        "draft": (
            None
            if bundled
            else {
                "config_digest": config_digest(draft_config),
                "revision": model_revision(draft_model_path, draft_revision),
            }
        ),
        "speculation": speculation,
        "layout": {
            "page_size": kv_args.page_size,
            "kv_cache_dtype": kv_args.kv_cache_dtype_str,
            "kv_layout": kv_args.kv_layout,
            "kv_item_lens": list(kv_args.kv_item_lens),
            "kv_layer_ids": list(kv_args.kv_layer_ids),
            "num_draft_entries": kv_args.num_draft_entries,
            "state_types": [state_type.value for state_type in kv_args.state_types],
            "state_item_lens": [list(lens) for lens in kv_args.state_item_lens],
            "attn_tp_size": attn_tp_size,
            "attn_cp_size": attn_cp_size,
            "pp_size": pp_size,
            "dcp_size": dcp_size,
            "draft_attn_tp_size": draft_attn_tp_size,
            "one_to_one_capable": one_to_one_capable,
        },
    }


def _strides_differ(local: List[int], remote: List[int]) -> List[int]:
    # A zero-length entry is a buffer the owning role elided; transfers skip it.
    return [i for i, (a, b) in enumerate(zip(local, remote)) if a and b and a != b]


def _requires_matched_draft(fingerprint: dict) -> bool:
    speculation = fingerprint["speculation"]
    return bool(speculation and speculation["requires_prefill_draft_kv"])


def peer_fingerprint_mismatches(decode: dict, prefill: Optional[dict]) -> List[str]:
    """Reasons the decode must not register with this prefill; empty if compatible."""
    if prefill is None:
        if _requires_matched_draft(decode):
            return [
                "the prefill publishes no peer fingerprint, so the draft KV it "
                "injects during prefill cannot be verified"
            ]
        return []

    problems = []
    for key in ("config_digest", "revision"):
        local, remote = decode["target"][key], prefill["target"][key]
        if local is not None and remote is not None and local != remote:
            problems.append(f"target model {key}: decode={local}, prefill={remote}")

    local_layout, remote_layout = decode["layout"], prefill["layout"]
    local_draft = local_layout["num_draft_entries"]
    remote_draft = remote_layout["num_draft_entries"]
    if (
        _requires_matched_draft(decode)
        or _requires_matched_draft(prefill)
        or (local_draft and remote_draft)
    ):
        local_spec = decode["speculation"] or {}
        remote_spec = prefill["speculation"] or {}
        for key in _SPECULATION_KEYS:
            if local_spec.get(key) != remote_spec.get(key):
                problems.append(
                    f"speculation {key}: decode={local_spec.get(key)}, "
                    f"prefill={remote_spec.get(key)}"
                )
        if decode["draft"] != prefill["draft"]:
            problems.append(
                f"draft model: decode={decode['draft']}, prefill={prefill['draft']}"
            )
        for key in ("attn_tp_size", "draft_attn_tp_size"):
            if local_layout[key] != remote_layout[key]:
                problems.append(
                    f"{key} with draft KV on the wire: decode={local_layout[key]}, "
                    f"prefill={remote_layout[key]} (no cross-TP draft relayout)"
                )

    one_to_one = (
        local_layout["one_to_one_capable"]
        and remote_layout["one_to_one_capable"]
        and local_layout["kv_layout"] is None
        and remote_layout["kv_layout"] is None
        and local_layout["pp_size"] == remote_layout["pp_size"] == 1
        and local_layout["dcp_size"] == remote_layout["dcp_size"] == 1
        and local_layout["attn_tp_size"] == remote_layout["attn_tp_size"]
        and local_layout["attn_cp_size"] == remote_layout["attn_cp_size"]
    )
    if one_to_one:
        problems += _layout_mismatches(local_layout, remote_layout)
    return problems


def _layout_mismatches(decode: dict, prefill: dict) -> List[str]:
    problems = []
    local_target = len(decode["kv_item_lens"]) - decode["num_draft_entries"]
    remote_target = len(prefill["kv_item_lens"]) - prefill["num_draft_entries"]
    if local_target != remote_target:
        return [f"target KV entries: decode={local_target}, prefill={remote_target}"]

    checks = [("target", slice(0, local_target))]
    same_draft = decode["num_draft_entries"] == prefill["num_draft_entries"]
    if same_draft and decode["num_draft_entries"]:
        checks.append(("draft", slice(local_target, None)))
    for name, entries in checks:
        if decode["kv_layer_ids"][entries] != prefill["kv_layer_ids"][entries]:
            problems.append(f"{name} KV entry order differs")
        differ = _strides_differ(
            decode["kv_item_lens"][entries], prefill["kv_item_lens"][entries]
        )
        if differ:
            i = differ[0] + entries.start
            problems.append(
                f"{name} KV byte stride differs at entry {i}: "
                f"decode={decode['kv_item_lens'][i]}, prefill={prefill['kv_item_lens'][i]}"
            )
    if (
        decode["num_draft_entries"] != prefill["num_draft_entries"]
        and prefill["num_draft_entries"]
    ):
        problems.append(
            f"draft KV entries: decode={decode['num_draft_entries']}, "
            f"prefill={prefill['num_draft_entries']}"
        )
    if same_draft:
        if decode["state_types"] != prefill["state_types"]:
            problems.append(
                f"state components: decode={decode['state_types']}, "
                f"prefill={prefill['state_types']}"
            )
        else:
            for state, (a, b) in enumerate(
                zip(decode["state_item_lens"], prefill["state_item_lens"])
            ):
                if len(a) != len(b) or _strides_differ(a, b):
                    problems.append(f"state component {state} byte strides differ")
    return problems


def collect_peer_fingerprint(
    scheduler: Any, kv_args: Any, token_to_kv_pool: Any
) -> dict:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.runtime_context import get_parallel, get_spec

    parallel = get_parallel()
    spec = get_spec()
    algorithm = scheduler.spec_algorithm
    model_config = scheduler.model_config
    speculation = None
    draft_config = None
    draft_attn_tp_size = None
    if not algorithm.is_none():
        speculation = {
            "algorithm": str(spec.speculative_algorithm).upper(),
            "num_draft_tokens": spec.speculative_num_draft_tokens,
            "num_steps": spec.speculative_num_steps,
            "eagle_topk": spec.speculative_eagle_topk,
            "dspark_block_size": spec.speculative_dspark_block_size,
            "dflash_block_size": spec.speculative_dflash_block_size,
            "requires_prefill_draft_kv": algorithm.injects_prefill_draft_kv(),
        }
        if spec.speculative_draft_model_path not in (None, model_config.model_path):
            draft_config = ModelConfig.from_server_args(
                scheduler.server_args,
                model_path=spec.speculative_draft_model_path,
                model_revision=spec.speculative_draft_model_revision,
                is_draft_model=True,
            ).hf_config
        draft_worker = scheduler.draft_worker
        layout = getattr(draft_worker, "draft_parallel_layout", None) or getattr(
            draft_worker, "parallel_layout", None
        )
        draft_attn_tp_size = (
            layout.attn_tp_size if layout is not None else parallel.attn_tp_size
        )
    return build_peer_fingerprint(
        kv_args=kv_args,
        target_config=model_config.hf_config,
        target_model_path=model_config.model_path,
        target_revision=model_config.revision,
        speculation=speculation,
        draft_config=draft_config,
        draft_model_path=spec.speculative_draft_model_path if speculation else None,
        draft_revision=spec.speculative_draft_model_revision,
        attn_tp_size=parallel.attn_tp_size,
        attn_cp_size=parallel.attn_cp_size,
        pp_size=parallel.pp_size,
        dcp_size=parallel.attn_dcp_size,
        draft_attn_tp_size=draft_attn_tp_size,
        one_to_one_capable=not (
            getattr(token_to_kv_pool, "layer_shard_enabled", False)
            or parallel.enable_kv_cache_sharding
        ),
    )
