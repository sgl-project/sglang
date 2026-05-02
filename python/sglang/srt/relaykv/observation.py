from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)
RUNTIME_OBSERVATION_ENV = "SGLANG_RELAYKV_RUNTIME_OBSERVATION"


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


def log_runtime_observation_skip(
    skip_event: dict[str, Any],
    *,
    logger_: logging.Logger | None = None,
    prefix: str = "relaykv_runtime_observation_skip",
) -> None:
    """Log that the read-only runtime observation hook was reached and skipped."""

    target_logger = logger_ if logger_ is not None else logger
    target_logger.warning("%s=%s", prefix, json.dumps(skip_event, sort_keys=True))


def _safe_metadata_attr_description(value: Any, attr_name: str) -> dict[str, Any]:
    description: dict[str, Any] = {
        f"has_{attr_name}": False,
        f"{attr_name}_repr": "",
    }
    try:
        attr_value = getattr(value, attr_name)
    except Exception as exc:
        description[f"{attr_name}_error"] = type(exc).__name__
        return description

    description[f"has_{attr_name}"] = True
    description[f"{attr_name}_repr"] = str(attr_value)
    return description


def _describe_runtime_metadata_value(value: Any) -> dict[str, Any]:
    """Describe metadata without reading tensor values or synchronizing devices."""

    value_type = type(value)
    description: dict[str, Any] = {
        "type_name": value_type.__name__,
        "type_module": value_type.__module__,
        "type_qualname": value_type.__qualname__,
        "is_list_or_tuple": isinstance(value, (list, tuple)),
        "list_or_tuple_len": None,
    }
    if isinstance(value, _MetadataAccessError):
        description["metadata_access_error_field"] = value.field_name
        description["metadata_access_error_reason"] = value.reason
    if description["is_list_or_tuple"]:
        description["list_or_tuple_len"] = len(value)

    description.update(_safe_metadata_attr_description(value, "shape"))
    description.update(_safe_metadata_attr_description(value, "device"))
    description.update(_safe_metadata_attr_description(value, "dtype"))
    return description


def _safe_forward_batch_metadata_value(batch: Any, field_name: str) -> Any:
    try:
        return getattr(batch, field_name)
    except Exception as exc:
        return _MetadataAccessError(field_name=field_name, reason=type(exc).__name__)


class _MetadataAccessError:
    def __init__(self, *, field_name: str, reason: str) -> None:
        self.field_name = field_name
        self.reason = reason


def _describe_runtime_forward_batch_metadata(
    *,
    forward_batch: Any,
    layer_ids: Sequence[int],
) -> dict[str, Any]:
    """Describe ForwardBatch-like metadata without values, iteration, or tensor sync."""

    return {
        "rids": _describe_runtime_metadata_value(
            _safe_forward_batch_metadata_value(forward_batch, "rids")
        ),
        "req_pool_indices": _describe_runtime_metadata_value(
            _safe_forward_batch_metadata_value(forward_batch, "req_pool_indices")
        ),
        "seq_lens": _describe_runtime_metadata_value(
            _safe_forward_batch_metadata_value(forward_batch, "seq_lens")
        ),
        "layer_ids": _describe_runtime_metadata_value(layer_ids),
    }


def run_model_runner_forward_observation_hook(
    *,
    forward_batch: Any,
    forward_pass_id: int,
    env_value: str | None = None,
    logger_: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the default-off ModelRunner.forward read-only observation hook.

    The hook is payload-only. It never touches KV pools, snapshots, host backup
    copy, attention state, scheduler decisions, or runtime writeback.
    """

    target_logger = logger_ if logger_ is not None else logger
    if env_value is None:
        env_value = os.getenv(RUNTIME_OBSERVATION_ENV)
    if env_value != "1":
        return {
            "enabled": False,
            "skipped": True,
            "skip_reason": "env_disabled",
            "summary": None,
        }

    try:
        layer_ids = [0]
        payloads = build_runtime_observation_payloads(
            batch=forward_batch,
            layer_ids=layer_ids,
            batch_id=f"forward-{forward_pass_id}",
            phase="forward",
            runtime_policy_state="runtime_observation",
        )
        summary = summarize_runtime_observation_payloads(payloads)
        log_runtime_observation_summary(summary, logger_=target_logger)
        return {
            "enabled": True,
            "skipped": False,
            "skip_reason": "",
            "summary": summary,
        }
    except (TypeError, ValueError) as exc:
        metadata_description = _describe_runtime_forward_batch_metadata(
            forward_batch=forward_batch,
            layer_ids=[0],
        )
        log_runtime_observation_skip(
            {
                "forward_pass_id": forward_pass_id,
                "reason": type(exc).__name__,
                "metadata_description": metadata_description,
            },
            logger_=target_logger,
        )
        return {
            "enabled": True,
            "skipped": True,
            "skip_reason": type(exc).__name__,
            "metadata_description": metadata_description,
            "summary": None,
        }
    except Exception as exc:
        target_logger.debug(
            "relaykv_runtime_observation_error=%s",
            json.dumps(
                {
                    "forward_pass_id": forward_pass_id,
                    "reason": type(exc).__name__,
                },
                sort_keys=True,
            ),
            exc_info=True,
        )
        return {
            "enabled": True,
            "skipped": True,
            "skip_reason": type(exc).__name__,
            "summary": None,
        }
