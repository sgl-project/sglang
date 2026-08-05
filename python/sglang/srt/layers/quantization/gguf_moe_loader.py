"""Pure loading-contract helpers for GGUF MoE weights.

This module intentionally has no torch or platform imports so its shape and
type checks can be exercised in CPU-only unit tests.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence

GGUF_MOE_SHARDS = frozenset({"w1", "w2", "w3"})


def plan_k3_a_log_tp_shard(
    *,
    checkpoint_shape: Sequence[int],
    parameter_shape: Sequence[int],
    tp_rank: int,
    tp_size: int,
) -> tuple[int, int]:
    """Return the flat start/length for one validated K3 A_log TP shard."""

    checkpoint_shape = tuple(int(value) for value in checkpoint_shape)
    parameter_shape = tuple(int(value) for value in parameter_shape)
    if len(checkpoint_shape) == 4:
        if checkpoint_shape[0:2] != (1, 1) or checkpoint_shape[3] != 1:
            raise ValueError(
                f"K3 A_log checkpoint shape is invalid: {checkpoint_shape}"
            )
        checkpoint_elements = checkpoint_shape[2]
    elif len(checkpoint_shape) == 1:
        checkpoint_elements = checkpoint_shape[0]
    else:
        raise ValueError(
            f"K3 A_log checkpoint must be 1-D or legacy 4-D: {checkpoint_shape}"
        )
    if (
        len(parameter_shape) != 4
        or parameter_shape[0:2] != (1, 1)
        or parameter_shape[3] != 1
        or parameter_shape[2] <= 0
    ):
        raise ValueError(f"K3 A_log parameter shape is invalid: {parameter_shape}")
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(
            f"K3 A_log TP topology is invalid: rank={tp_rank}, size={tp_size}"
        )

    shard_size = parameter_shape[2]
    start_idx = tp_rank * shard_size
    expected_checkpoint_elements = tp_size * shard_size
    if checkpoint_elements != expected_checkpoint_elements:
        raise ValueError(
            "K3 A_log checkpoint extent differs from the exact TP topology: "
            f"elements={checkpoint_elements}, expected={expected_checkpoint_elements}, "
            f"rank={tp_rank}, size={tp_size}, shard_size={shard_size}"
        )
    return start_idx, shard_size


def record_gguf_moe_qtype(
    shard_weight_types: MutableMapping[str, int],
    shard_id: str,
    qtype: int,
) -> None:
    """Record one GGUF MoE shard type and reject ambiguous W13 fusion."""

    if shard_id not in GGUF_MOE_SHARDS:
        raise ValueError(f"unsupported GGUF MoE shard id: {shard_id!r}")
    qtype = int(qtype)
    previous = shard_weight_types.get(shard_id)
    if previous is not None and previous != qtype:
        raise ValueError(
            f"GGUF MoE shard {shard_id} repeats with different qtypes: "
            f"{previous} and {qtype}"
        )

    if shard_id in {"w1", "w3"}:
        other_id = "w3" if shard_id == "w1" else "w1"
        other = shard_weight_types.get(other_id)
        if other is not None and other != qtype:
            w1_qtype = qtype if shard_id == "w1" else other
            w3_qtype = qtype if shard_id == "w3" else other
            raise ValueError(
                "GGUF MoE cannot fuse W1 and W3 with different qtypes: "
                f"w1={w1_qtype}, w3={w3_qtype}"
            )

    # Mutate only after every invariant has passed so a caller that catches a
    # validation error cannot observe a partially accepted qtype inventory.
    shard_weight_types[shard_id] = qtype


def plan_gguf_moe_tp_shard(
    *,
    shard_id: str,
    shape: Sequence[int],
    tp_size: int,
    tp_rank: int,
    packed_type_size: int | None = None,
) -> tuple[int, int, int]:
    """Return ``(axis, start, length)`` for one rank's packed MoE shard.

    W1/W3 are column-parallel in logical weight space and therefore split
    their output rows (axis 0). W2 is row-parallel and must split its packed
    input columns (axis 1). For quantized byte tensors, ``packed_type_size``
    makes the W2 split fail closed unless every rank receives whole GGML
    blocks.
    """

    if shard_id not in GGUF_MOE_SHARDS:
        raise ValueError(f"unsupported GGUF MoE shard id: {shard_id!r}")
    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError(f"GGUF MoE expert weight must be a positive 2D shape: {shape}")
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid MoE TP topology: size={tp_size}, rank={tp_rank}")

    axis = 1 if shard_id == "w2" else 0
    extent = int(shape[axis])
    if extent % tp_size:
        raise ValueError(
            f"GGUF MoE {shard_id} dimension {extent} is not divisible by "
            f"TP size {tp_size}"
        )
    length = extent // tp_size
    if packed_type_size is not None:
        if packed_type_size <= 0:
            raise ValueError("GGUF packed type size must be positive")
        if axis != 1:
            raise ValueError("packed block validation only applies to W2 input shards")
        if extent % packed_type_size or length % packed_type_size:
            raise ValueError(
                "GGUF MoE W2 TP split cuts a packed quantization block: "
                f"packed_width={extent}, local_width={length}, "
                f"type_size={packed_type_size}, tp_size={tp_size}"
            )
    return axis, tp_rank * length, length


def plan_gguf_moe_stream_destination(
    *,
    shard_id: str,
    expert_id: int,
    num_experts: int,
    local_shape: Sequence[int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Plan one expert's final slot in a streaming packed-MoE parameter.

    Returns ``(parameter_shape, (expert, row_start, row_length))``.  W1 and W3
    share the fused W13 parameter, so their rows occupy its lower and upper
    halves.  W2 owns the complete row range of its separate parameter.

    Keeping this shape arithmetic pure makes the peak-memory invariant
    testable without CUDA: the runtime can materialize the final parameter
    once and copy each incoming expert directly into its permanent slot,
    rather than retaining every expert tensor and allocating a second fused
    copy after the whole checkpoint has been read.
    """

    if shard_id not in GGUF_MOE_SHARDS:
        raise ValueError(f"unsupported GGUF MoE shard id: {shard_id!r}")
    if num_experts <= 0 or not 0 <= expert_id < num_experts:
        raise ValueError(
            f"invalid GGUF MoE expert: id={expert_id}, count={num_experts}"
        )
    if len(local_shape) != 2 or any(int(value) <= 0 for value in local_shape):
        raise ValueError(
            f"GGUF MoE local expert weight must be a positive 2D shape: "
            f"{local_shape}"
        )

    rows, columns = (int(value) for value in local_shape)
    if shard_id == "w2":
        return (num_experts, rows, columns), (expert_id, 0, rows)

    row_start = 0 if shard_id == "w1" else rows
    return (num_experts, 2 * rows, columns), (expert_id, row_start, rows)
