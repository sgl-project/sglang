"""Redundant expert replicas for Muse Spark EP4 decode / verify (``SGLANG_MUSE_EXPERT_REPLICAS``).

Each MoE layer's verify step ends in an all-reduce that waits for the rank with the most distinct active experts
(compact A4 costs about 16 us + 4.2 us per distinct active local expert). With replicas, every rank also holds R
experts of the other ranks (R / (EP - 1) from each), named by a plan JSON (harness ``replica_plan.py``):

    {"num_experts": 256, "ep_size": 4, "replicas_per_rank": R, "layers": {"<layer>": [[slot, holder], ...]}}

listing per layer the replicated slot ids (in the placement's slot space) and their holder rank, in the fixed
order of the per-step greedy. Every rank holds the same top-k ids, so ``_scatter_a4_replica`` derives the same
assignment everywhere: each active replicated expert runs on the less loaded of its home and holder rank; every
route still runs on exactly one rank and the combine all-reduce is unchanged. The replica weights are copied from
their home ranks once after loading (one all-gather per expert tensor per layer).

A holder's replica slots follow its 64 home slots, ordered by (home rank, plan order); the prefill a4w4 path keeps
reading only the home slots.
"""

from __future__ import annotations

import json
import logging

import msgspec
import torch
import triton

logger = logging.getLogger(__name__)


class LayerReplicas(msgspec.Struct, frozen=True):
    """One MoE layer's replicas on this rank: the kernel table (``build_table``, on device), entry count, local
    experts incl. replicas, and the compact A4 operands (w13, w2, w13 scales, w2 scales) as uint8
    [num_local, N, K] views."""

    table: torch.Tensor
    num_entries: int
    num_local: int
    operands: tuple


def load_plan(path: str, num_experts: int, ep_size: int, replicas: int) -> dict:
    """{layer: [(slot, holder), ...]} from the plan JSON, validated for this EP split and replica count."""
    with open(path) as f:
        spec = json.load(f)
    if int(spec["num_experts"]) != num_experts or int(spec["ep_size"]) != ep_size:
        raise ValueError(
            f"Replica plan {path} is for {spec['num_experts']} experts / EP{spec['ep_size']}, "
            f"running {num_experts} / EP{ep_size}"
        )
    if not 0 < replicas * ep_size <= 31:
        # _scatter_a4_replica keeps one bit per replicated expert in a 32-bit word.
        raise ValueError(
            f"{replicas} replicas per rank x EP{ep_size} exceeds 31 entries"
        )
    if int(spec["replicas_per_rank"]) != replicas:
        raise ValueError(
            f"Replica plan {path} holds {spec['replicas_per_rank']} replicas per rank, "
            f"SGLANG_MUSE_EXPERT_REPLICAS={replicas}"
        )
    local = num_experts // ep_size
    plan = {}
    for layer, entries in spec["layers"].items():
        entries = [(int(g), int(h)) for g, h in entries]
        if not entries:
            continue
        slots = [g for g, _ in entries]
        if len(set(slots)) != len(slots) or any(
            not 0 <= g < num_experts or not 0 <= h < ep_size or g // local == h
            for g, h in entries
        ):
            raise ValueError(f"Replica plan {path} layer {layer}: bad entries")
        for h in range(ep_size):
            if sum(1 for _, o in entries if o == h) != replicas:
                raise ValueError(
                    f"Replica plan {path} layer {layer}: rank {h} does not hold {replicas} replicas"
                )
        plan[int(layer)] = entries
    logger.info(
        "Muse expert replicas %s: %d per rank on %d layers (EP%d)",
        path,
        replicas,
        len(plan),
        ep_size,
    )
    return plan


def replica_slots(entries, ep_size: int, home: int) -> dict:
    """{(slot, holder): local slot on the holder}: home + j, ordered by (home rank, plan order)."""
    out = {}
    for h in range(ep_size):
        mine = sorted((g // home, i, g) for i, (g, o) in enumerate(entries) if o == h)
        for j, (_, _, g) in enumerate(mine):
            out[(g, h)] = home + j
    return out


def build_table(
    entries, ep_size: int, home: int, rank: int, num_experts: int
) -> torch.Tensor:
    """Rank ``rank``'s int32 table for ``_scatter_a4_replica`` (CPU): per global slot id its entry index in plan
    order, its home slot on this rank and its replica slot on this rank (-1 when none); then each entry's home rank
    (KP = next power of two of the entry count); then the entries' home and holder ranks packed 2 bits per entry,
    entries 0-15 and 16-31 in separate words."""
    slots = replica_slots(entries, ep_size, home)
    kp = triton.next_power_of_2(len(entries))
    ne = num_experts
    table = torch.full((3 * ne + kp + 4,), -1, dtype=torch.int64)
    g = torch.arange(ne)
    table[ne : 2 * ne] = torch.where(g // home == rank, g - rank * home, -1)
    table[3 * ne : 3 * ne + kp] = 0
    pack = [0, 0, 0, 0]
    for i, (slot, holder) in enumerate(entries):
        table[slot] = i
        if holder == rank:
            table[2 * ne + slot] = slots[(slot, holder)]
        table[3 * ne + i] = slot // home
        pack[i // 16] |= (slot // home) << (2 * (i % 16))
        pack[2 + i // 16] |= holder << (2 * (i % 16))
    table[3 * ne + kp :] = torch.tensor(pack)
    # Packed words are two's-complement int32 (bit 31 may be set).
    return torch.where(table >= 2**31, table - 2**32, table).to(torch.int32)


@torch.no_grad()
def replicate_expert_blocks(
    operands, entries, *, rank: int, ep_size: int, home: int, group
) -> tuple:
    """New [home + R, ...] uint8 tensors per operand: this rank's home expert blocks, then its replicas copied
    from their home ranks. ``operands`` are [home, ...] tensors with each expert's block contiguous; ``group`` is the
    TP GroupCoordinator. One all-gather per tensor of the blocks every rank donates (plan order): the warmed-up
    pynccl communicator, where a torch.distributed all-to-all would bring up a second RCCL communicator (+2.4 GB)."""
    slots = replica_slots(entries, ep_size, home)
    donate = [[g for g, _ in entries if g // home == r] for r in range(ep_size)]
    width = max(len(d) for d in donate)
    mine = sorted((s, g) for (g, h), s in slots.items() if h == rank)
    dev = operands[0].device
    pick = torch.tensor(
        [(g // home) * width + donate[g // home].index(g) for _, g in mine],
        dtype=torch.long,
        device=dev,
    )
    send_idx = torch.tensor(
        [g - rank * home for g in donate[rank]], dtype=torch.long, device=dev
    )
    out = []
    for t in operands:
        flat = t.view(torch.uint8).reshape(home, -1)
        assert flat.shape[1] % 2 == 0
        full = torch.empty(
            (home + len(mine), flat.shape[1]), dtype=torch.uint8, device=dev
        )
        full[:home].copy_(flat)
        send = torch.zeros((width, flat.shape[1]), dtype=torch.uint8, device=dev)
        send[: len(donate[rank])] = flat.index_select(0, send_idx)
        gathered = torch.empty(
            (ep_size * width, flat.shape[1]), dtype=torch.uint8, device=dev
        )
        # Bytes travel as bf16 words (the dtypes every all-gather backend takes); a gather copies them exactly.
        group.all_gather_into_tensor(
            gathered.view(torch.bfloat16), send.view(torch.bfloat16)
        )
        full[home:] = gathered.index_select(0, pick)
        out.append(full)
    return tuple(out)
