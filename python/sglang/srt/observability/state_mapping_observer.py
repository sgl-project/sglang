"""Opt-in request-bound state mapping evidence for inference validation.

This observer intentionally records metadata only. It is disabled unless both
``SGLANG_STATE_OBSERVER_DIR`` and ``SGLANG_STATE_OBSERVER_RID_PREFIX`` are set.
The intended consumer is a short-lived validation lane, never a performance
benchmark lane.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from sglang.srt.environ import envs

_CONTEXT_ENV = {
    "variant": "SGLANG_STATE_OBSERVER_VARIANT",
    "role": "SGLANG_STATE_OBSERVER_ROLE",
    "source_commit": "SGLANG_SOURCE_COMMIT",
    "source_tree": "SGLANG_SOURCE_TREE",
    "source_archive_sha256": "SGLANG_SOURCE_ARCHIVE_SHA256",
    "pod_name": "POD_NAME",
    "pod_uid": "POD_UID",
    "node_name": "NODE_NAME",
}


def _class_name(value: Any) -> Optional[str]:
    return None if value is None else type(value).__name__


def _shape(value: Any) -> Optional[list[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _small_ints(value: Any, *, limit: int = 8) -> Optional[list[int]]:
    """Copy only explicitly bounded index metadata, never tensor payloads."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    if len(value) > limit:
        return None
    return [int(item) for item in value]


def _enum_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(getattr(value, "name", value))


def _enum_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _bounded_int_list(value: Any, *, limit: int = 512) -> Optional[list[int]]:
    if value is None:
        return []
    value = list(value)
    if len(value) > limit:
        return None
    return [int(item) for item in value]


def _bounded_optional_int_list(
    value: Any, *, limit: int = 512
) -> Optional[list[Optional[int]]]:
    if value is None:
        return []
    value = list(value)
    if len(value) > limit:
        return None
    return [None if item is None else int(item) for item in value]


def _bounded_conv_groups(
    value: Any, *, outer_limit: int = 512, inner_limit: int = 8
) -> Optional[list[Optional[list[int]]]]:
    if value is None:
        return []
    value = list(value)
    if len(value) > outer_limit:
        return None
    result = []
    for group in value:
        if group is None:
            result.append(None)
            continue
        group = list(group)
        if len(group) > inner_limit:
            return None
        result.append([int(item) for item in group])
    return result


def _input_ids_digest(req: Any) -> dict[str, Any]:
    token_ids = [int(token_id) for token_id in req.origin_input_ids]
    canonical = json.dumps(token_ids, separators=(",", ":")).encode()
    return {
        "count": len(token_ids),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _pool_details(pool: Any) -> dict[str, Any]:
    full_pool = getattr(pool, "full_kv_pool", pool)
    return {
        "pool_class": _class_name(pool),
        "full_pool_class": _class_name(full_pool),
        "page_size": getattr(full_pool, "page_size", getattr(pool, "page_size", None)),
        "use_mla": getattr(pool, "use_mla", None),
        "use_dsa": getattr(pool, "use_dsa", None),
        "full_attention_layer_ids": sorted(
            int(layer_id)
            for layer_id in getattr(pool, "full_attention_layer_id_mapping", {})
        ),
        "index_cache_class": _class_name(getattr(full_pool, "index_key_cache", None)),
        "index_head_dim": getattr(full_pool, "index_head_dim", None),
        "index_kpool": getattr(full_pool, "index_kpool", None),
        "index_kpool_compress": getattr(full_pool, "index_kpool_compress", None),
        "kpool_use_compress": getattr(full_pool, "kpool_use_compress", None),
        "tail_extra_slots": getattr(full_pool, "tail_extra_slots", None),
    }


def _dsa_tail_state_indices(pool: Any, req_pool_idx: int, seq_len: int) -> list[int]:
    """Metadata-only mirror of the PD transfer's DSA-tail addressing."""
    full_pool = getattr(pool, "full_kv_pool", pool)
    if not getattr(full_pool, "kpool_use_compress", False):
        return []
    pool_size = int(full_pool.index_kpool)
    tail_size = pool_size + int(getattr(full_pool, "tail_extra_slots", 0))
    if pool_size <= 1 or tail_size < pool_size:
        raise ValueError(
            "DSA kpool-compress requires pool_size > 1 and "
            f"tail_size >= pool_size, got {pool_size=} and {tail_size=}"
        )
    n_valid = int(seq_len) % pool_size
    if n_valid == 0:
        return []
    start_phys = (int(seq_len) - n_valid) % tail_size
    first_n = min(n_valid, tail_size - start_phys)
    return [
        int(req_pool_idx),
        start_phys,
        first_n,
        0,
        n_valid - first_n,
        tail_size,
    ]


def _spec_shapes(value: Any) -> dict[str, Optional[list[int]]]:
    return {
        name: _shape(getattr(value, name, None))
        for name in (
            "topk_p",
            "topk_index",
            "hidden_states",
            "bonus_tokens",
            "dsa_topk_indices",
            "draft_probs",
        )
    }


def _runner_details(runner: Any) -> dict[str, Any]:
    if runner is None:
        return {"runner_class": None, "model_class": None, "layers": None}
    layer_info = getattr(runner, "layer_info", None)
    start = getattr(layer_info, "start_layer", None)
    end = getattr(layer_info, "end_layer", None)
    model = getattr(runner, "model", None)
    if start is None:
        start = getattr(model, "start_layer", None)
    if end is None:
        end = getattr(model, "end_layer", None)
    return {
        "runner_class": _class_name(runner),
        "model_class": _class_name(model),
        "layers": None if start is None or end is None else [int(start), int(end)],
        "req_pool_class": _class_name(getattr(runner, "req_to_token_pool", None)),
        "kv_pool": _pool_details(getattr(runner, "token_to_kv_pool", None)),
    }


def _draft_runners(scheduler: Any) -> list[Any]:
    worker = getattr(scheduler, "model_worker", None)
    method = getattr(worker, "_draft_model_runners", None)
    if callable(method):
        return list(method())
    return []


def _transfer_manager(scheduler: Any) -> Any:
    mode = _enum_name(getattr(scheduler, "disaggregation_mode", None))
    queue_name = {
        "PREFILL": "disagg_prefill_bootstrap_queue",
        "DECODE": "disagg_decode_prealloc_queue",
    }.get(mode)
    queue = getattr(scheduler, queue_name, None) if queue_name else None
    return getattr(queue, "kv_manager", None)


def _transfer_contract(scheduler: Any) -> dict[str, Any]:
    """Return the registered PD transfer schema without buffer addresses."""
    manager = _transfer_manager(scheduler)
    kv_args = getattr(manager, "kv_args", None)
    if kv_args is None:
        return {"present": False}

    state_types = list(getattr(kv_args, "state_types", []) or [])
    state_ptrs = list(getattr(kv_args, "state_data_ptrs", []) or [])
    state_item_lens = list(getattr(kv_args, "state_item_lens", []) or [])
    state_dims = list(getattr(kv_args, "state_dim_per_tensor", []) or [])
    state_outer_counts = list(getattr(kv_args, "state_slice_outer_counts", []) or [])
    state_conv_groups = list(getattr(kv_args, "state_conv_shard_groups", []) or [])
    state_layer_ids = list(getattr(kv_args, "state_layer_ids", []) or [])

    components = []
    for index, state_type in enumerate(state_types):
        ptrs = state_ptrs[index] if index < len(state_ptrs) else []
        components.append(
            {
                "type": _enum_value(state_type),
                "entry_count": len(ptrs),
                "item_lens": _bounded_int_list(
                    state_item_lens[index] if index < len(state_item_lens) else []
                ),
                "dim_per_tensor": _bounded_optional_int_list(
                    state_dims[index] if index < len(state_dims) else []
                ),
                "slice_outer_counts": _bounded_int_list(
                    state_outer_counts[index] if index < len(state_outer_counts) else []
                ),
                "conv_shard_groups": _bounded_conv_groups(
                    state_conv_groups[index] if index < len(state_conv_groups) else []
                ),
                "layer_ids": _bounded_int_list(
                    state_layer_ids[index] if index < len(state_layer_ids) else []
                ),
            }
        )

    kv_ptrs = list(getattr(kv_args, "kv_data_ptrs", []) or [])
    return {
        "present": True,
        "backend": _enum_value(getattr(scheduler, "transfer_backend", None)),
        "manager_class": _class_name(manager),
        "engine_rank": int(getattr(kv_args, "engine_rank")),
        "pp_rank": int(getattr(kv_args, "pp_rank")),
        "system_dp_rank": int(getattr(kv_args, "system_dp_rank")),
        "page_size": int(getattr(kv_args, "page_size")),
        "kv_cache_dtype": str(getattr(kv_args, "kv_cache_dtype_str")),
        "kv_entry_count": len(kv_ptrs),
        "kv_layer_ids": _bounded_int_list(getattr(kv_args, "kv_layer_ids", [])),
        "state_components": components,
    }


def _token_slots(req_pool: Any, req_pool_idx: Optional[int], kv: Any) -> dict[str, Any]:
    if req_pool is None or req_pool_idx is None:
        return {"positions": [], "slot_ids": []}
    positions = sorted(
        {
            pos
            for pos in (0, int(kv.kv_committed_len) - 1, int(kv.kv_allocated_len) - 1)
            if pos >= 0
        }
    )
    if not positions:
        return {"positions": [], "slot_ids": []}
    mapping = req_pool.req_to_token[int(req_pool_idx), positions]
    slots = _small_ints(mapping, limit=3)
    return {"positions": positions, "slot_ids": slots}


@dataclass(frozen=True)
class StateMappingObserver:
    output_dir: Path
    rid_prefix: str
    rank: int
    local_rank: int
    scheduler: Any

    @classmethod
    def from_scheduler(cls, scheduler: Any) -> Optional[StateMappingObserver]:
        output_dir = envs.SGLANG_STATE_OBSERVER_DIR.get()
        rid_prefix = envs.SGLANG_STATE_OBSERVER_RID_PREFIX.get()
        if not output_dir or not rid_prefix:
            return None
        world = getattr(scheduler, "world_group", None)
        rank = int(getattr(world, "rank", getattr(world, "rank_in_group", 0)))
        local_rank = int(getattr(world, "local_rank", scheduler.ps.gpu_id))
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return cls(path, str(rid_prefix), rank, local_rank, scheduler)

    def observe(self, batch: Any, phase: str, batch_result: Any = None) -> None:
        for req in batch.reqs:
            if str(req.rid).startswith(self.rid_prefix):
                try:
                    self._append(self._build_record(req, batch, phase, batch_result))
                except Exception as exc:
                    raise RuntimeError(
                        "state-mapping observer failed for "
                        f"rid={req.rid!r}, phase={phase}, rank={self.rank}"
                    ) from exc

    def _build_record(
        self, req: Any, batch: Any, phase: str, batch_result: Any
    ) -> dict[str, Any]:
        scheduler = self.scheduler
        ps = scheduler.ps
        kv = req.kv
        req_pool = scheduler.req_to_token_pool
        target_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        target_runner = scheduler.tp_worker.model_runner
        draft_runners = _draft_runners(scheduler)
        dsa_tail = []
        if kv.req_pool_idx is not None:
            try:
                dsa_tail = _dsa_tail_state_indices(
                    target_pool, int(kv.req_pool_idx), int(req.seqlen)
                )
            except (AttributeError, ValueError):
                dsa_tail = []
        context = {name: os.getenv(env_name) for name, env_name in _CONTEXT_ENV.items()}
        context["role"] = context["role"] or _enum_name(
            getattr(scheduler, "disaggregation_mode", None)
        )
        return {
            "schema_version": 2,
            "event": {
                "phase": phase,
                "timestamp_ns": time.time_ns(),
                "forward_iter": int(batch.forward_iter),
            },
            "context": context,
            "process": {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "rank": self.rank,
                "local_rank": self.local_rank,
                "gpu_id": int(ps.gpu_id),
            },
            "parallel": asdict(ps),
            "request": {
                "rid": str(req.rid),
                "input_ids": _input_ids_digest(req),
                "sequence_length": int(req.seqlen),
                "req_pool_idx": (
                    None if kv.req_pool_idx is None else int(kv.req_pool_idx)
                ),
                "kv_committed_len": int(kv.kv_committed_len),
                "kv_allocated_len": int(kv.kv_allocated_len),
                "cache_protected_len": int(kv.cache_protected_len),
                "token_slots": _token_slots(req_pool, kv.req_pool_idx, kv),
            },
            "target": {
                **_runner_details(target_runner),
                "req_pool_class": _class_name(req_pool),
                "kv_pool": _pool_details(target_pool),
            },
            "draft": {
                "present_on_rank": bool(draft_runners),
                "runners": [_runner_details(runner) for runner in draft_runners],
                "shares_target_req_pool": (
                    all(
                        getattr(runner, "req_to_token_pool", None) is req_pool
                        for runner in draft_runners
                    )
                    if draft_runners
                    else None
                ),
            },
            "transfer": _transfer_contract(scheduler),
            "dsa": {
                **_pool_details(target_pool),
                "tail_state_indices": [int(item) for item in dsa_tail],
            },
            "mamba": {
                "layer_ids": sorted(
                    int(layer) for layer in getattr(req_pool, "mamba_map", {})
                ),
                "layer_mapping": {
                    str(layer): int(local)
                    for layer, local in getattr(req_pool, "mamba_map", {}).items()
                },
                "pool_idx": _small_ints(kv.mamba_pool_idx, limit=1),
                "mapping_value": _small_ints(
                    (
                        getattr(req_pool, "req_index_to_mamba_index_mapping", None)[
                            int(kv.req_pool_idx)
                        ]
                        if kv.req_pool_idx is not None
                        and hasattr(req_pool, "req_index_to_mamba_index_mapping")
                        else None
                    ),
                    limit=1,
                ),
                "ping_pong_slots": _small_ints(
                    kv.mamba_ping_pong_track_buffer, limit=2
                ),
                "next_track_idx": kv.mamba_next_track_idx,
                "last_track_idx": kv.mamba_last_track_idx,
                "last_track_seqlen": kv.mamba_last_track_seqlen,
                "batch_track_indices_shape": _shape(batch.mamba_track_indices),
                "batch_track_buffer_indices": batch.mamba_track_buffer_indices,
            },
            "speculative": {
                "algorithm": _enum_name(batch.spec_algorithm),
                "forward_mode": _enum_name(batch.forward_mode),
                "spec_info_class": _class_name(batch.spec_info),
                "spec_info_shapes": _spec_shapes(batch.spec_info),
                "next_draft_input_class": _class_name(
                    getattr(batch_result, "next_draft_input", None)
                ),
                "next_draft_input_shapes": _spec_shapes(
                    getattr(batch_result, "next_draft_input", None)
                ),
            },
        }

    def _append(self, record: dict[str, Any]) -> None:
        role = record["context"].get("role") or "unknown"
        path = self.output_dir / f"state-mapping-{role}-rank-{self.rank}.jsonl"
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
        try:
            os.write(fd, payload.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
