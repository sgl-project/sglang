from __future__ import annotations

from collections.abc import Sequence
from typing import Any


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
