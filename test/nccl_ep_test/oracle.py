"""CPU oracle, independent of the communication and Graph implementations."""

from collections import Counter
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutingBatch:
    tokens: tuple[torch.Tensor, ...]
    expert_ids: tuple[torch.Tensor, ...]
    weights: tuple[torch.Tensor, ...]
    num_experts: int


def make_fixture(
    bucket: int,
    *,
    case: str = "balanced",
    step: int = 0,
    hidden: int = 2048,
    seed: int = 32774,
    change: str = "all",
) -> RoutingBatch:
    """Exactly representable payloads, distinct rows and nonuniform route weights.

    Powers of two survive BF16 and block FP8 quantize/dequantize. Changing step
    changes the selected data component without changing any tensor shape.
    A masked rank retains its bucket size and participates in communication.
    """
    if bucket < 0 or hidden < 1:
        raise ValueError("Expected nonnegative bucket and positive hidden size")
    cases = {"balanced", "hotspot", "empty_rank", "all_masked", "padding", "duplicates"}
    if case not in cases:
        raise ValueError(f"Unknown routing case: {case}")
    if change not in ("all", "tokens", "routing", "weights"):
        raise ValueError(f"Unknown dynamic-data component: {change}")
    token_step = step if change in ("all", "tokens") else 0
    routing_step = step if change in ("all", "routing") else 0
    weight_step = step if change in ("all", "weights") else 0
    tokens, expert_ids, weights = [], [], []
    for rank in range(2):
        code = torch.arange(bucket, dtype=torch.int64) + rank * bucket + 1 + seed % 31
        digit = (torch.arange(hidden) % 8) * 2
        exponent = (code[:, None] >> digit[None, :]) & 3
        tokens.append((2.0**exponent * (1 + token_step % 2)).to(torch.bfloat16))
        first = (torch.arange(bucket) + rank + routing_step) % 4
        ids = torch.stack((first, (first + 2) % 4), dim=1)
        if case == "hotspot":
            ids[:] = torch.tensor([0, 1]) + 2 * (routing_step % 2)
        elif case == "duplicates":
            ids[:] = routing_step % 4
        route_weights = torch.tensor(
            [0.25, 0.75] if weight_step % 2 == 0 else [0.75, 0.25]
        )
        route_weights = route_weights.expand(bucket, 2).clone()
        if case == "all_masked" or (case == "empty_rank" and rank == routing_step % 2):
            ids.fill_(-1)
        elif case == "padding":
            ids[bucket // 2 :] = -1
        # CUDA SGLang TopK masks padded ids, but need not zero their weights.
        # Nonzero weights make stale combine routing observable in the oracle.
        expert_ids.append(ids)
        weights.append(route_weights)
    return RoutingBatch(tuple(tokens), tuple(expert_ids), tuple(weights), 4)


def expected_dispatch(batch: RoutingBatch):
    """Rows for each global expert, including repeated route contributions."""
    rows = [[] for _ in range(batch.num_experts)]
    for tokens, ids in zip(batch.tokens, batch.expert_ids):
        for token in range(tokens.shape[0]):
            for expert in ids[token].tolist():
                if expert >= 0:
                    rows[expert].append(tokens[token].float())
    hidden = batch.tokens[0].shape[1]
    return tuple(
        torch.stack(items) if items else torch.empty((0, hidden)) for items in rows
    )


def assert_dispatch_matches(batch: RoutingBatch, rank: int, received, counters):
    """Compare all valid elements and multiplicities, independent of receive order."""
    expected = expected_dispatch(batch)
    local_experts = batch.num_experts // len(batch.tokens)
    if not 0 <= rank < len(batch.tokens):
        raise AssertionError(f"Invalid rank: {rank}")
    if (
        received.ndim != 3
        or received.shape[0] != local_experts
        or received.shape[2] != batch.tokens[rank].shape[1]
    ):
        raise AssertionError(f"Unexpected receive shape: {tuple(received.shape)}")
    received = received.detach().cpu().float()
    counts = counters.detach().cpu().tolist()
    if len(counts) != local_experts:
        raise AssertionError("Unexpected expert counter shape")
    for local_expert, count in enumerate(counts):
        expert = rank * local_experts + local_expert
        wanted = expected[expert]
        if count != len(wanted) or not 0 <= count <= received.shape[1]:
            raise AssertionError(
                f"Expert {expert}: count={count}, expected={len(wanted)}"
            )
        actual = received[local_expert, :count]
        if not torch.isfinite(actual).all():
            raise AssertionError(f"Expert {expert}: nonfinite valid payload")
        actual_rows = Counter(tuple(row) for row in actual.tolist())
        wanted_rows = Counter(tuple(row) for row in wanted.tolist())
        if actual_rows != wanted_rows:
            raise AssertionError(
                f"Expert {expert}: received token contents/multiplicity differ"
            )


def expected_combine(batch: RoutingBatch, *, identity: bool = False):
    """Sum weighted synthetic experts in float64, independently per source token."""
    outputs = []
    for tokens, ids, weights in zip(batch.tokens, batch.expert_ids, batch.weights):
        output = torch.zeros_like(tokens, dtype=torch.float64, device="cpu")
        for token in range(tokens.shape[0]):
            for route in range(ids.shape[1]):
                expert = int(ids[token, route])
                if expert < 0:
                    continue
                factor = 1 if identity else expert + 1
                output[token] += (
                    tokens[token].double() * factor * float(weights[token, route])
                )
        outputs.append(output.float())
    return tuple(outputs)


def assert_combine_matches(batch: RoutingBatch, rank: int, actual, *, identity=False):
    """Strict comparison for the exactly representable lab fixtures."""
    wanted = expected_combine(batch, identity=identity)[rank]
    actual = actual.detach().cpu().float()
    if not torch.isfinite(actual).all():
        raise AssertionError("Combined output contains nonfinite values")
    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def validate_capacity(batch: RoutingBatch, capacity: int):
    """Reject unsafe synthetic routes before entering any collective operation."""
    world = len(batch.tokens)
    if not world or batch.num_experts <= 0 or batch.num_experts % world:
        raise ValueError("Experts must divide evenly across nonempty ranks")
    if len(batch.expert_ids) != world or len(batch.weights) != world:
        raise ValueError("Missing rank inputs")
    hidden, topk = batch.tokens[0].shape[1], batch.expert_ids[0].shape[1]
    for tokens, ids, weights in zip(batch.tokens, batch.expert_ids, batch.weights):
        if (
            tokens.ndim != 2
            or tokens.shape[1] != hidden
            or ids.shape != (len(tokens), topk)
            or weights.shape != ids.shape
        ):
            raise ValueError("Incompatible token/routing/weight shapes")
        if len(tokens) > capacity:
            raise ValueError("Token count exceeds dispatch capacity")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expert ids must be integers")
        if ((ids < -1) | (ids >= batch.num_experts)).any():
            raise ValueError("Invalid expert id")
        if not torch.isfinite(tokens).all() or not torch.isfinite(weights).all():
            raise ValueError("Nonfinite test input")
    for expert, rows in enumerate(expected_dispatch(batch)):
        if len(rows) > world * capacity:
            raise ValueError(
                f"Expert {expert} exceeds receive capacity {world * capacity}"
            )


def dequantize_fp8(payload: torch.Tensor, scales: torch.Tensor, group_size=128):
    """Apply every block scale before the synthetic expert's BF16 input cast."""
    if payload.shape[-1] % group_size:
        raise ValueError("Hidden size must be divisible by the FP8 group size")
    expected_shape = (*payload.shape[:-1], payload.shape[-1] // group_size)
    if tuple(scales.shape) != expected_shape:
        raise ValueError(
            f"Expected FP8 scales shaped {expected_shape}, got {scales.shape}"
        )
    blocks = payload.float().reshape(*expected_shape, group_size)
    return (
        (blocks * scales.float().unsqueeze(-1))
        .reshape(payload.shape)
        .to(torch.bfloat16)
    )
