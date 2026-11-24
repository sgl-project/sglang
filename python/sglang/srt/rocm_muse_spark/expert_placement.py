"""Online expert placement for the Muse Spark MoE under expert parallelism.

In a sequence-parallel prefill every rank runs its local experts over all tokens, and the MoE reduce-scatter waits
for the rank with the most routed rows. The routing load per expert is uneven and stable across requests (a rank's
share of a layer correlates strongly between different 8K prompts), so it is a property of the placement: rank r owns
experts [r * E_local, (r + 1) * E_local). Each MoE layer counts its routed rows per expert in the SP prefill
branch (``MuseSpark1xMoE.expert_load``); once ``MIN_TOKENS`` prefill tokens are counted, every layer swaps expert
slots between ranks until its heaviest rank is near the mean. A swap moves the expert weights between the two
ranks and renumbers the router's output columns (router weight, ``expert_bias``, the joint router copies) in place,
so the model computes the same function and captured graphs keep their addresses. Runs outside any graph.

Generated tokens load the experts differently from prompts: after the prefill placement, a wide verify step's MoE time
per (layer, rank) still deviates by a stable amount (correlated between steps), the wait of its reduce-scatter.
So the verify-SP gather also counts routed rows per expert (``expert_load_verify``), and once ``MIN_VERIFY_ROWS`` are
counted every layer is re-placed once more, against the prefill and verify loads together (``plan_swaps_joint``).
At verify widths a rank's MoE time follows the number of its experts a step touches, not its routed rows (gfx1250
A8W4: a fixed cost plus a per-touched-expert cost that dominates the per-route cost), so the
re-placement also balances each rank's expected touched experts per verify step, 1 - exp(-rows / step) per expert.

gfx950 (``verify_blocks``; MXFP4 experts at TP = EP, off by default): a rank's compact A4 verify time follows its
expert blocks, which the routing launch adds to ``expert_load_verify`` (each rank its own experts). The verify load
is re-placed after ``min_verify_rows`` rows whether or not an SP prefill ran, jointly with the prefill rows when there
are any; those are counted up to the first placement only and then follow their experts.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.rocm_muse_spark import router_pooler
from sglang.srt.rocm_muse_spark.sequence_parallel import (
    SP_MIN_TOKENS,
    VERIFY_SP_MIN_ROWS,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

MIN_TOKENS = 6144
# Verify-SP rows counted before the joint re-placement: about 256 verify steps of 128 rows.
MIN_VERIFY_ROWS = 32768
# Per layer: stop when a swap gains under MIN_GAIN of the mean rank load, or after MAX_SWAPS swaps (each moves
# two experts, ~25 MB apiece).
MAX_SWAPS = 12
MIN_GAIN = 0.005
# gfx950 block counts: one swap moves at most one block a step, so the same balance takes more swaps than rows do.
MAX_SWAPS_BLOCKS = 24


def _attention_dp() -> bool:
    from sglang.srt.rocm_muse_spark import dp_ep

    return dp_ep.enabled()


# Attention DP: forwards between two sums of the ranks' verify rows (an all-reduce on the CPU group).
DP_CHECK_EVERY = 64


def plan_swaps(
    load: np.ndarray, num_local: int, max_swaps: int = MAX_SWAPS
) -> list[tuple[int, int]]:
    """Greedy slot swaps lowering the heaviest rank's load; ``load`` is per physical slot (rank = slot // num_local).
    Deterministic: the same counts give the same swaps on every rank."""
    load = load.astype(np.int64).copy()
    ranks = load.size // num_local
    mean = load.sum() / ranks
    swaps = []
    for _ in range(max_swaps):
        rank_load = load.reshape(ranks, num_local).sum(1)
        a = int(np.argmax(rank_load))
        la = load[a * num_local : (a + 1) * num_local]
        best = None
        for b in np.argsort(rank_load, kind="stable"):
            b = int(b)
            if b == a:
                continue
            lb = load[b * num_local : (b + 1) * num_local]
            d = la[:, None] - lb[None, :]
            new_max = np.maximum(rank_load[a] - d, rank_load[b] + d)
            new_max = np.where(
                (d > 0) & (d < rank_load[a] - rank_load[b]),
                new_max,
                np.iinfo(np.int64).max,
            )
            k = int(np.argmin(new_max))
            if best is None or new_max.flat[k] < best[0]:
                best = (
                    int(new_max.flat[k]),
                    a * num_local + k // num_local,
                    b * num_local + k % num_local,
                )
        if best is None or rank_load[a] - best[0] < MIN_GAIN * mean:
            break
        _, i, j = best
        load[[i, j]] = load[[j, i]]
        swaps.append((i, j))
    return swaps


def plan_swaps_joint(
    loads: np.ndarray, num_local: int, max_swaps: int = MAX_SWAPS
) -> list[tuple[int, int]]:
    """``plan_swaps`` for several loads per slot (rows [k, slots]): a rank's cost is its largest load relative to that
    load's mean over ranks, and each swap lowers the heaviest rank's cost the most. Deterministic."""
    loads = loads.astype(np.float64)
    ranks = loads.shape[1] // num_local
    means = loads.sum(1, keepdims=True) / ranks
    norm = loads / np.where(means > 0, means, 1.0)
    swaps = []
    for _ in range(max_swaps):
        rank_load = norm.reshape(-1, ranks, num_local).sum(2)  # [k, ranks]
        cost = rank_load.max(0)
        a = int(np.argmax(cost))
        la = norm[:, a * num_local : (a + 1) * num_local]
        best = None
        for b in np.argsort(cost, kind="stable"):
            b = int(b)
            if b == a:
                continue
            lb = norm[:, b * num_local : (b + 1) * num_local]
            d = la[:, :, None] - lb[:, None, :]  # [k, slot of a, slot of b]
            new_a = (rank_load[:, a, None, None] - d).max(0)
            new_b = (rank_load[:, b, None, None] + d).max(0)
            new_max = np.maximum(new_a, new_b)
            k = int(np.argmin(new_max))
            if best is None or new_max.flat[k] < best[0]:
                best = (
                    float(new_max.flat[k]),
                    a * num_local + k // num_local,
                    b * num_local + k % num_local,
                )
        if best is None or cost[a] - best[0] < MIN_GAIN:
            break
        _, i, j = best
        norm[:, [i, j]] = norm[:, [j, i]]
        swaps.append((i, j))
    return swaps


def _source_slots(swaps: list[tuple[int, int]], num_experts: int) -> list[int]:
    """src[p] = the slot whose expert ends up in slot p after applying ``swaps`` in order."""
    src = list(range(num_experts))
    for i, j in swaps:
        src[i], src[j] = src[j], src[i]
    return src


def _expert_tensors(experts) -> list[torch.Tensor]:
    """Every tensor holding the experts' weights, each with one contiguous block per local expert."""
    a8w4 = getattr(experts, "gfx1250_a8w4", None)
    if a8w4 is not None:
        return list(a8w4[:4])
    tensors = []
    if getattr(experts, "a4w4_weights", None) is not None:
        # gfx950 MXFP4: the preshuffled weights and scales (the only copy unless the stored layout is kept)
        tensors += list(experts.a4w4_weights)
    if not getattr(experts, "native_dropped", False):
        tensors += [experts.w13_weight, experts.w2_weight]
        tensors += [
            scale
            for name in ("w13_weight_scale", "w2_weight_scale")
            if (scale := getattr(experts, name, None)) is not None
        ]
    return tensors


@torch.no_grad()
def _move_experts(moe, src: list[int], group) -> None:
    """Each rank fills its slots p from slot src[p]: local copies, or point-to-point from the owning rank. Every
    send reads the pre-move weights; writes land after all transfers complete."""
    n = moe.num_local_experts
    me = group.rank_in_group
    slots = [t.view(torch.uint8).view(n, -1) for t in _expert_tensors(moe.experts)]
    ops, fills = [], []
    for p, s in enumerate(src):
        if s == p or (p // n != me and s // n != me):
            continue
        if p // n == me:
            if s // n == me:
                fills.append((p % n, [x[s % n].clone() for x in slots]))
            else:
                bufs = [torch.empty_like(x[0]) for x in slots]
                ops += [
                    dist.P2POp(dist.irecv, b, group.ranks[s // n], group.device_group)
                    for b in bufs
                ]
                fills.append((p % n, bufs))
        else:
            ops += [
                dist.P2POp(
                    dist.isend, x[s % n], group.ranks[p // n], group.device_group
                )
                for x in slots
            ]
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    for k, bufs in fills:
        for x, b in zip(slots, bufs):
            x[k].copy_(b)


@torch.no_grad()
def _renumber_router(moe, src: list[int]) -> None:
    perm = torch.tensor(src, device=moe.expert_bias.device)
    router = moe.router.weight
    router.copy_(router[:, perm] if router.shape[1] == perm.numel() else router[perm])
    moe.expert_bias.copy_(moe.expert_bias[perm])
    if getattr(moe, "router_pooler_packed_weight", None) is not None:
        router_pooler.prepare_joint_router_pooler(moe)


class ExpertPlacement:
    """Counts SP prefill tokens across forwards and re-places the experts once, then once more from the prefill and
    verify-SP loads together after ``MIN_VERIFY_ROWS`` verify rows. ``verify_blocks``: the gfx950 form (module
    docstring), re-placing on the counted expert blocks after ``min_verify_rows`` target-verify rows."""

    def __init__(
        self,
        moe_layers,
        verify_blocks: bool = False,
        min_verify_rows: int = MIN_VERIFY_ROWS,
    ):
        self.moes = [m for m in moe_layers if m.expert_load is not None]
        self.verify_blocks = verify_blocks
        self.min_verify_rows = min_verify_rows
        self.tokens = 0
        self.verify_rows = 0
        self.verify_steps = 0
        self.started = False
        self.dp_forwards = 0
        self.placed = False
        self.verified = False
        self.done = not self.moes
        # verify_blocks: the prefill rows per slot, kept here once the layers stop counting them
        self._prefill = [m.expert_load for m in self.moes]

    def on_forward_end(self, forward_batch: ForwardBatch) -> None:
        if self.done:
            return
        if not self.started:
            # Graph capture and the server warmup ran the counters on dummy rows.
            self.started = True
            self._zero_counts()
            return
        num_tokens = forward_batch.input_ids.shape[0]
        if self.verify_blocks and _attention_dp():
            # The re-placement is collective, but a DP rank runs its own batch (or an idle one). The ranks run their
            # forwards in lockstep (MORI-EP is collective in every MoE layer), so each counts its own verify rows and
            # every DP_CHECK_EVERY forwards they add them up: all see the same sum and re-place at the same forward.
            if forward_batch.forward_mode.is_target_verify():
                self.verify_rows += num_tokens
                self.verify_steps += 1
            self.dp_forwards += 1
            if self.dp_forwards % DP_CHECK_EVERY == 0:
                total = torch.tensor([self.verify_rows], dtype=torch.int64)
                dist.all_reduce(total, group=get_parallel().tp_group.cpu_group)
                if int(total.item()) >= self.min_verify_rows:
                    self.verified = True
                    self.rebalance_blocks()
            self.done = self.verified
            return
        if self.verify_blocks:
            mode = forward_batch.forward_mode
            if mode.is_target_verify():
                if not self.verified:
                    self.verify_rows += num_tokens
                    self.verify_steps += 1
                    if self.verify_rows >= self.min_verify_rows:
                        self.verified = True
                        self.rebalance_blocks()
            elif (
                mode == ForwardMode.EXTEND
                and num_tokens >= SP_MIN_TOKENS
                and not self.placed
            ):
                self.tokens += num_tokens
                if self.tokens >= MIN_TOKENS:
                    self.placed = True
                    self.rebalance_blocks()
                    # The counted rows follow their experts from here on: no more counting in the prefill path.
                    for moe in self.moes:
                        moe.expert_load = None
            self.done = self.placed and self.verified
            return
        if self.placed:
            if (
                forward_batch.forward_mode.is_target_verify()
                and num_tokens >= VERIFY_SP_MIN_ROWS
            ):
                self.verify_rows += num_tokens
                self.verify_steps += 1
                if self.verify_rows >= MIN_VERIFY_ROWS:
                    self.done = True
                    self.rebalance(joint=True)
            return
        if (
            forward_batch.forward_mode != ForwardMode.EXTEND
            or num_tokens < SP_MIN_TOKENS
        ):
            return
        self.tokens += num_tokens
        if self.tokens >= MIN_TOKENS:
            self.placed = True
            self.rebalance()

    def _zero_counts(self) -> None:
        for moe, prefill in zip(self.moes, self._prefill):
            prefill.zero_()
            if moe.expert_load_verify is not None:
                moe.expert_load_verify.zero_()

    @torch.no_grad()
    def rebalance_blocks(self) -> None:
        """The gfx950 re-placement: per layer, swaps planned on the prefill rows and the verify blocks counted so
        far (whichever are non-zero). The prefill rows then follow their experts; the block counters restart."""
        start = time.perf_counter()
        group = get_parallel().tp_group
        blocks = torch.stack([m.expert_load_verify for m in self.moes]).to(torch.int64)
        # every rank counted the blocks of its own experts
        dist.all_reduce(blocks, group=group.device_group)
        loads = torch.stack([torch.stack(self._prefill).to(torch.int64), blocks], dim=1)
        # Rank 0 decides: every rank applies the swaps planned from its counts.
        dist.broadcast(loads, src=group.ranks[0], group=group.device_group)
        loads = loads.cpu().numpy()
        before, after, layers, moved = np.zeros(2), np.zeros(2), np.zeros(2), 0
        for moe, prefill, both in zip(self.moes, self._prefill, loads):
            n = moe.num_local_experts
            counted = both.sum(1) > 0
            if not counted.any():
                continue
            swaps = plan_swaps_joint(both[counted], n, MAX_SWAPS_BLOCKS)
            src = _source_slots(swaps, both.shape[1])
            for k in np.flatnonzero(counted):
                mean = both[k].sum() * n / both.shape[1]
                before[k] += both[k].reshape(-1, n).sum(1).max() / mean
                after[k] += both[k][src].reshape(-1, n).sum(1).max() / mean
                layers[k] += 1
            moved += sum(s != p for p, s in enumerate(src))
            if swaps:
                _move_experts(moe, src, group)
                _renumber_router(moe, src)
                prefill.copy_(prefill[torch.tensor(src, device=prefill.device)])
            moe.expert_load_verify.zero_()
        steps = self.verify_steps
        if not self.verified:
            self.verify_rows = self.verify_steps = 0
        torch.cuda.synchronize()
        layers = np.maximum(layers, 1)
        logger.info(
            "Muse expert placement (gfx950, %d verify steps / %d prefill tokens counted): heaviest rank %.3fx -> "
            "%.3fx the mean verify blocks (%d layers), %.3fx -> %.3fx the mean prefill rows (%d layers), "
            "%d experts moved, %.0f ms",
            steps, self.tokens, before[1] / layers[1], after[1] / layers[1], int(layers[1]),
            before[0] / layers[0], after[0] / layers[0], int(layers[0]), moved,
            (time.perf_counter() - start) * 1e3,
        )  # fmt: skip

    @torch.no_grad()
    def rebalance(self, joint: bool = False) -> None:
        start = time.perf_counter()
        group = get_parallel().tp_group
        verify = [m.expert_load_verify if m.expert_load_verify is not None else torch.zeros_like(p)
                  for m, p in zip(self.moes, self._prefill)]  # fmt: skip
        loads = torch.stack(
            [torch.stack([p, v]) for p, v in zip(self._prefill, verify)]
        ).to(torch.int64)
        # Rank 0 decides: every rank applies the swaps planned from its counts.
        dist.broadcast(loads, src=group.ranks[0], group=group.device_group)
        loads = loads.cpu().numpy()
        before = after = moved = 0
        for moe, both in zip(self.moes, loads):
            n = moe.num_local_experts
            if joint and both[1].sum() == 0:
                continue
            if joint:
                # expected experts touched per verify step (Poisson rows per step); identical on every rank
                both = np.vstack(
                    [both, 1.0 - np.exp(-both[1] / max(self.verify_steps, 1))]
                )
            load = both[2] if joint else both[0]
            swaps = plan_swaps_joint(both, n) if joint else plan_swaps(load, n)
            src = _source_slots(swaps, load.size)
            mean = max(load.sum() * n / load.size, 1)
            before += load.reshape(-1, n).sum(1).max() / mean
            after += load[src].reshape(-1, n).sum(1).max() / mean
            moved += sum(s != p for p, s in enumerate(src))
            if swaps:
                _move_experts(moe, src, group)
                _renumber_router(moe, src)
        self._zero_counts()
        torch.cuda.synchronize()
        logger.info(
            "Muse expert placement (%s): %d layers, heaviest rank %.2fx -> %.2fx the mean %s load, %d experts moved, %.0f ms",
            "prefill + verify" if joint else "prefill", len(self.moes), before / len(self.moes),
            after / len(self.moes), "verify touched-expert" if joint else "prefill", moved,
            (time.perf_counter() - start) * 1e3,
        )  # fmt: skip
