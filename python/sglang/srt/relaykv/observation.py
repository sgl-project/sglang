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


def _readonly_optional_sequence(
    value: Any,
    *,
    field_name: str,
) -> Sequence[Any] | None:
    if value is None:
        return None
    return _readonly_sequence(value, field_name=field_name)


def build_runtime_observation_cpu_metadata_payloads(
    *,
    rids: Sequence[str],
    req_pool_indices_cpu: Sequence[int],
    seq_lens_cpu: Sequence[int],
    layer_ids: Sequence[int],
    batch_id: str,
    phase: str,
    runtime_policy_state: str,
    extend_seq_lens_cpu: Sequence[int] | None = None,
    extend_prefix_lens_cpu: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Build RelayKV runtime observation payload candidates from CPU metadata.

    This pure helper fixes the future CPU metadata schema only. It accepts plain
    Python list/tuple metadata and rejects tensor-like inputs without converting
    or reading tensor values. It does not touch ForwardBatch, KV pools, snapshots,
    host backup copy, attention state, scheduler decisions, or writeback.
    """

    rid_values = _readonly_sequence(rids, field_name="rids")
    req_pool_idx_values = _readonly_sequence(
        req_pool_indices_cpu,
        field_name="req_pool_indices_cpu",
    )
    seq_len_values = _readonly_sequence(seq_lens_cpu, field_name="seq_lens_cpu")
    layer_id_values = _readonly_sequence(layer_ids, field_name="layer_ids")
    extend_seq_len_values = _readonly_optional_sequence(
        extend_seq_lens_cpu,
        field_name="extend_seq_lens_cpu",
    )
    extend_prefix_len_values = _readonly_optional_sequence(
        extend_prefix_lens_cpu,
        field_name="extend_prefix_lens_cpu",
    )

    request_count = len(rid_values)
    if len(req_pool_idx_values) != request_count:
        raise ValueError("req_pool_indices_cpu must match len(rids)")
    if len(seq_len_values) != request_count:
        raise ValueError("seq_lens_cpu must match len(rids)")
    if (
        extend_seq_len_values is not None
        and len(extend_seq_len_values) != request_count
    ):
        raise ValueError("extend_seq_lens_cpu must match len(rids)")
    if (
        extend_prefix_len_values is not None
        and len(extend_prefix_len_values) != request_count
    ):
        raise ValueError("extend_prefix_lens_cpu must match len(rids)")

    payloads: list[dict[str, Any]] = []
    for request_index, request_id in enumerate(rid_values):
        for layer_id in layer_id_values:
            payload: dict[str, Any] = {
                "event_type": "runtime_observation_cpu_metadata_candidate",
                "batch_id": batch_id,
                "request_id": request_id,
                "request_index_in_batch": request_index,
                "request_index": request_index,
                "req_pool_idx": req_pool_idx_values[request_index],
                "req_pool_index": req_pool_idx_values[request_index],
                "seq_len": seq_len_values[request_index],
                "layer_id": layer_id,
                "phase": phase,
                "runtime_policy_state": runtime_policy_state,
                "source": "cpu_metadata",
                "source_mutated": False,
                "attention_override": False,
                "kv_cache_mutation": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
            }
            if extend_seq_len_values is not None:
                payload["extend_seq_len"] = extend_seq_len_values[request_index]
            if extend_prefix_len_values is not None:
                payload["extend_prefix_len"] = extend_prefix_len_values[request_index]
            payloads.append(payload)
    return payloads


def build_runtime_observation_payload_candidates_from_forward_batch_existing_metadata(
    *,
    forward_batch: Any,
    layer_ids: Sequence[int],
    batch_id: str,
    phase: str,
    runtime_policy_state: str,
) -> list[dict[str, Any]]:
    """Build payload candidates from existing ForwardBatch CPU-side attributes.

    This helper is intentionally disconnected from real ForwardBatch runtime
    hooks. It only accepts list/tuple metadata already present on a
    ForwardBatch-like object and fixes the schema for a future read-only path.
    req_pool_idx is not available from the existing attributes and is therefore
    emitted as None. Tensor-like values are rejected without conversion.
    """

    rid_values = _readonly_sequence(
        getattr(forward_batch, "rids"),
        field_name="forward_batch.rids",
    )
    seq_len_values = _readonly_sequence(
        getattr(forward_batch, "seq_lens_cpu"),
        field_name="forward_batch.seq_lens_cpu",
    )
    layer_id_values = _readonly_sequence(layer_ids, field_name="layer_ids")
    extend_seq_len_values = _readonly_optional_sequence(
        getattr(forward_batch, "extend_seq_lens_cpu", None),
        field_name="forward_batch.extend_seq_lens_cpu",
    )
    extend_prefix_len_values = _readonly_optional_sequence(
        getattr(forward_batch, "extend_prefix_lens_cpu", None),
        field_name="forward_batch.extend_prefix_lens_cpu",
    )

    request_count = len(rid_values)
    if len(seq_len_values) != request_count:
        raise ValueError("forward_batch.seq_lens_cpu must match len(forward_batch.rids)")
    if (
        extend_seq_len_values is not None
        and len(extend_seq_len_values) != request_count
    ):
        raise ValueError(
            "forward_batch.extend_seq_lens_cpu must match len(forward_batch.rids)"
        )
    if (
        extend_prefix_len_values is not None
        and len(extend_prefix_len_values) != request_count
    ):
        raise ValueError(
            "forward_batch.extend_prefix_lens_cpu must match len(forward_batch.rids)"
        )

    payloads: list[dict[str, Any]] = []
    for request_index, request_id in enumerate(rid_values):
        for layer_id in layer_id_values:
            payload: dict[str, Any] = {
                "event_type": (
                    "runtime_observation_forward_batch_existing_metadata_candidate"
                ),
                "batch_id": batch_id,
                "request_id": request_id,
                "request_index_in_batch": request_index,
                "request_index": request_index,
                "req_pool_idx": None,
                "req_pool_index": None,
                "seq_len": seq_len_values[request_index],
                "layer_id": layer_id,
                "phase": phase,
                "runtime_policy_state": runtime_policy_state,
                "source": "forward_batch_existing_cpu_metadata",
                "source_mutated": False,
                "attention_override": False,
                "kv_cache_mutation": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
            }
            if extend_seq_len_values is not None:
                payload["extend_seq_len"] = extend_seq_len_values[request_index]
            if extend_prefix_len_values is not None:
                payload["extend_prefix_len"] = extend_prefix_len_values[request_index]
            payloads.append(payload)
    return payloads


def _device_type_name(value: Any) -> str:
    device = getattr(value, "device", None)
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type
    if device is None:
        return ""
    return str(device).split(":", 1)[0]


def _dtype_is_int64_like(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    return "int64" in str(dtype) or str(dtype).endswith(".long")


def _cpu_tensor_1d_to_list_for_runtime_observation(
    value: Any,
    *,
    field_name: str,
    expected_len: int,
) -> list[int]:
    """Read a 1D CPU tensor-like value for observation-only metadata.

    This intentionally refuses GPU tensors before any value conversion. The only
    value read allowed here is CPU tensor metadata needed to form a read-only
    observation payload candidate.
    """

    if _device_type_name(value) != "cpu":
        raise TypeError(f"{field_name} must be a CPU tensor for runtime observation")
    if not _dtype_is_int64_like(value):
        raise TypeError(f"{field_name} must have int64-like dtype")

    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 1:
        raise ValueError(f"{field_name} must be a 1D CPU tensor")
    if int(shape[0]) != expected_len:
        raise ValueError(f"{field_name} must match len(forward_batch.rids)")

    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        raise TypeError(f"{field_name} must support CPU tensor tolist()")
    values = tolist()
    if not isinstance(values, list) or len(values) != expected_len:
        raise ValueError(f"{field_name} CPU tensor values must match len(rids)")
    return [int(item) for item in values]


def _runtime_existing_metadata_seq_lens(
    value: Any,
    *,
    expected_len: int,
) -> tuple[Sequence[Any], str]:
    if isinstance(value, (list, tuple)):
        if len(value) != expected_len:
            raise ValueError(
                "forward_batch.seq_lens_cpu must match len(forward_batch.rids)"
            )
        return value, "list_tuple_observation_only"
    return (
        _cpu_tensor_1d_to_list_for_runtime_observation(
            value,
            field_name="forward_batch.seq_lens_cpu",
            expected_len=expected_len,
        ),
        "cpu_tensor_observation_only",
    )


def build_runtime_observation_payload_candidates_from_forward_batch_runtime_existing_metadata(
    *,
    forward_batch: Any,
    layer_ids: Sequence[int],
    batch_id: str,
    phase: str,
    runtime_policy_state: str,
) -> list[dict[str, Any]]:
    """Build observation-only candidates from existing ForwardBatch metadata.

    This runtime helper is still payload-only. It reads only rids and
    seq_lens_cpu/extend CPU metadata that already exists on ForwardBatch-like
    objects. req_pool_idx remains None and no KV pool, attention, scheduler, or
    writeback state is touched.
    """

    rid_values = _readonly_sequence(
        getattr(forward_batch, "rids"),
        field_name="forward_batch.rids",
    )
    request_count = len(rid_values)
    seq_len_values, seq_lens_cpu_value_source = _runtime_existing_metadata_seq_lens(
        getattr(forward_batch, "seq_lens_cpu"),
        expected_len=request_count,
    )
    layer_id_values = _readonly_sequence(layer_ids, field_name="layer_ids")
    extend_seq_len_values = _readonly_optional_sequence(
        getattr(forward_batch, "extend_seq_lens_cpu", None),
        field_name="forward_batch.extend_seq_lens_cpu",
    )
    extend_prefix_len_values = _readonly_optional_sequence(
        getattr(forward_batch, "extend_prefix_lens_cpu", None),
        field_name="forward_batch.extend_prefix_lens_cpu",
    )

    if (
        extend_seq_len_values is not None
        and len(extend_seq_len_values) != request_count
    ):
        raise ValueError(
            "forward_batch.extend_seq_lens_cpu must match len(forward_batch.rids)"
        )
    if (
        extend_prefix_len_values is not None
        and len(extend_prefix_len_values) != request_count
    ):
        raise ValueError(
            "forward_batch.extend_prefix_lens_cpu must match len(forward_batch.rids)"
        )

    payloads: list[dict[str, Any]] = []
    for request_index, request_id in enumerate(rid_values):
        for layer_id in layer_id_values:
            payload: dict[str, Any] = {
                "event_type": (
                    "runtime_observation_forward_batch_existing_metadata_candidate"
                ),
                "batch_id": batch_id,
                "request_id": request_id,
                "request_index_in_batch": request_index,
                "request_index": request_index,
                "req_pool_idx": None,
                "req_pool_index": None,
                "seq_len": seq_len_values[request_index],
                "seq_lens_cpu_value_source": seq_lens_cpu_value_source,
                "layer_id": layer_id,
                "phase": phase,
                "runtime_policy_state": runtime_policy_state,
                "source": "forward_batch_existing_cpu_metadata_runtime_observation",
                "source_mutated": False,
                "attention_override": False,
                "kv_cache_mutation": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
            }
            if extend_seq_len_values is not None:
                payload["extend_seq_len"] = extend_seq_len_values[request_index]
            if extend_prefix_len_values is not None:
                payload["extend_prefix_len"] = extend_prefix_len_values[request_index]
            payloads.append(payload)
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


def _describe_forward_batch_cpu_metadata(forward_batch: Any) -> dict[str, Any]:
    """Inventory existing ForwardBatch CPU-side metadata without reading values."""

    return {
        "rids": _describe_runtime_metadata_value(
            _safe_forward_batch_metadata_value(forward_batch, "rids")
        ),
        "seq_lens_cpu": _describe_runtime_metadata_value(
            _safe_forward_batch_metadata_value(forward_batch, "seq_lens_cpu")
        ),
        "extend_seq_lens_cpu": _describe_runtime_metadata_value(
            _safe_forward_batch_metadata_value(forward_batch, "extend_seq_lens_cpu")
        ),
        "extend_prefix_lens_cpu": _describe_runtime_metadata_value(
            _safe_forward_batch_metadata_value(forward_batch, "extend_prefix_lens_cpu")
        ),
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
        cpu_metadata_description = _describe_forward_batch_cpu_metadata(forward_batch)
        try:
            layer_ids = [0]
            fallback_payloads = (
                build_runtime_observation_payload_candidates_from_forward_batch_runtime_existing_metadata(
                    forward_batch=forward_batch,
                    layer_ids=layer_ids,
                    batch_id=f"forward-{forward_pass_id}",
                    phase="forward",
                    runtime_policy_state="runtime_observation",
                )
            )
            fallback_summary = summarize_runtime_observation_payloads(
                fallback_payloads
            )
            fallback_summary.update(
                {
                    "forward_pass_id": forward_pass_id,
                    "source": (
                        "forward_batch_existing_cpu_metadata_runtime_observation"
                    ),
                    "seq_lens_cpu_value_source": fallback_payloads[0].get(
                        "seq_lens_cpu_value_source", ""
                    )
                    if fallback_payloads
                    else "",
                    "req_pool_idx_none": all(
                        payload.get("req_pool_idx") is None
                        and payload.get("req_pool_index") is None
                        for payload in fallback_payloads
                    ),
                    "initial_skip_reason": type(exc).__name__,
                }
            )
            log_runtime_observation_summary(
                fallback_summary,
                logger_=target_logger,
                prefix=(
                    "relaykv_runtime_observation_forward_batch_"
                    "existing_metadata_summary"
                ),
            )
            return {
                "enabled": True,
                "skipped": False,
                "skip_reason": "",
                "summary": fallback_summary,
                "metadata_description": metadata_description,
                "forward_batch_cpu_metadata_description": cpu_metadata_description,
            }
        except Exception as fallback_exc:
            fallback_skip_reason = type(fallback_exc).__name__
        log_runtime_observation_skip(
            {
                "forward_pass_id": forward_pass_id,
                "reason": type(exc).__name__,
                "runtime_existing_metadata_skip_reason": fallback_skip_reason,
                "metadata_description": metadata_description,
                "forward_batch_cpu_metadata_description": cpu_metadata_description,
            },
            logger_=target_logger,
        )
        return {
            "enabled": True,
            "skipped": True,
            "skip_reason": type(exc).__name__,
            "runtime_existing_metadata_skip_reason": fallback_skip_reason,
            "metadata_description": metadata_description,
            "forward_batch_cpu_metadata_description": cpu_metadata_description,
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
