from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


def _readonly_sequence(value: Any, *, field_name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{field_name} must be a list or tuple for read-only observation payloads"
        )
    return value


def build_runtime_observation_payloads(
    *,
    batch: Any,
    layer_ids: Sequence[int],
    batch_id: str,
    phase: str,
    runtime_policy_state: str,
) -> list[dict[str, Any]]:
    """Build read-only RelayKV runtime observation payloads from a batch-like object.

    This helper only reads plain Python batch metadata. It intentionally does not
    touch KV pools, create snapshots, copy host backup tensors, mutate scheduler
    state, or call tensor conversion helpers that may synchronize GPU work.
    """

    rids = _readonly_sequence(getattr(batch, "rids"), field_name="batch.rids")
    req_pool_indices = _readonly_sequence(
        getattr(batch, "req_pool_indices"),
        field_name="batch.req_pool_indices",
    )
    seq_lens = _readonly_sequence(getattr(batch, "seq_lens"), field_name="batch.seq_lens")
    layer_id_values = _readonly_sequence(layer_ids, field_name="layer_ids")

    batch_size = getattr(batch, "batch_size", len(rids))
    if batch_size != len(rids):
        raise ValueError("batch.batch_size must match len(batch.rids)")
    if len(req_pool_indices) != len(rids):
        raise ValueError("batch.req_pool_indices must match len(batch.rids)")
    if len(seq_lens) != len(rids):
        raise ValueError("batch.seq_lens must match len(batch.rids)")

    payloads: list[dict[str, Any]] = []
    for request_index, request_id in enumerate(rids):
        for layer_id in layer_id_values:
            payloads.append(
                {
                    "event_type": "runtime_observation",
                    "batch_id": batch_id,
                    "request_id": request_id,
                    "request_index": request_index,
                    "req_pool_index": req_pool_indices[request_index],
                    "seq_len": seq_lens[request_index],
                    "layer_id": layer_id,
                    "phase": phase,
                    "runtime_policy_state": runtime_policy_state,
                    "source_mutated": False,
                    "attention_override": False,
                    "kv_cache_mutation": False,
                    "runtime_writeback": False,
                    "scheduler_policy_noop": True,
                }
            )
    return payloads


def summarize_runtime_observation_payloads(
    payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize read-only runtime observation payloads.

    This reads payload dictionaries only. It does not touch KV pools, create
    snapshots, copy host backup tensors, mutate scheduler state, or write back
    runtime state.
    """

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_batch: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "source_mutated_true_count": 0,
            "attention_override_true_count": 0,
            "kv_cache_mutation_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
        }
    )

    for payload in payloads:
        per_request[str(payload.get("request_id"))] += 1
        layer_value = payload.get("layer_id")
        if layer_value is None:
            layer_value = payload.get("layer_idx")
        per_layer[str(layer_value)] += 1
        per_batch[str(payload.get("batch_id"))] += 1

        if payload.get("source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1
        if payload.get("attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if payload.get("kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if payload.get("runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if payload.get("scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1

    return {
        "total_payloads": len(payloads),
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_batch_counts": dict(sorted(per_batch.items())),
        **dict(safety_counts),
    }


def log_runtime_observation_summary(
    summary: dict[str, Any],
    *,
    logger_: logging.Logger | None = None,
    prefix: str = "relaykv_runtime_observation_summary",
) -> None:
    """Log a precomputed read-only runtime observation summary."""

    target_logger = logger_ if logger_ is not None else logger
    target_logger.info("%s=%s", prefix, json.dumps(summary, sort_keys=True))
