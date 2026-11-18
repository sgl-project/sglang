"""KDA TP state on the PD decode worker: full-width staging slots, imports, and state-slot ops."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.communication.kda_tp import (
    all_gather_padded,
    group,
    lazy_mamba_slot,
    lazy_stage_key,
    width,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


class KdaTpStaging:
    """PD-received full-width KDA state slots and their all-to-all into each rank's head shard."""

    def __init__(self, pool, cache_params, num_slots: int, device):
        from sglang.srt.configs.mamba_utils import KimiLinearStateShape

        shape = cache_params.shape
        if not isinstance(shape, KimiLinearStateShape):
            raise TypeError(
                f"KDA TP staging needs Kimi linear state, got {type(shape)}"
            )
        if num_slots < 1:
            raise ValueError(f"KDA TP staging needs at least one slot, got {num_slots}")
        full = KimiLinearStateShape.create(
            tp_world_size=1,
            num_heads=shape.num_heads,
            head_dim=shape.head_dim,
            num_k_heads=shape.num_k_heads,
            head_k_dim=shape.head_k_dim,
            conv_kernel_size=shape.conv_kernel,
        )
        W = width()
        if full.temporal[0] % W:
            raise ValueError(
                f"KDA TP: {full.temporal[0]} heads do not split over {W} ranks"
            )
        self.layer_ids = list(pool.mamba_layer_ids)
        L = len(self.layer_ids)
        with pool.state_allocation_region():
            self.conv = [
                torch.zeros(
                    (L, num_slots) + tuple(c),
                    dtype=cache_params.dtype.conv,
                    device=device,
                )
                for c in full.conv
            ]
            self.temporal = torch.zeros(
                (L, num_slots) + tuple(full.temporal),
                dtype=cache_params.dtype.temporal,
                device=device,
            )
        self.conv_slice_axis = full.conv_slice_axis
        # Conv channels reordered rank-major so rank r's block is its local layout.
        groups = full.conv_shard_groups
        cols, off = [[] for _ in range(W)], 0
        for g in groups:
            if g % W:
                raise ValueError(
                    f"KDA TP: conv group {g} does not split over {W} ranks"
                )
            for r in range(W):
                cols[r].extend(range(off + r * g // W, off + (r + 1) * g // W))
            off += g
        self._conv_rank_major = torch.tensor(
            [c for r in cols for c in r], dtype=torch.int64, device=device
        )
        self.num_slots = num_slots
        self._free: List[int] = list(range(num_slots))
        self._stage_of: Dict[int, int] = {}  # request main slot -> stage
        self._pending: List[int] = []  # slots whose request joined the running batch
        self._reserved = 0  # admitted requests whose assign is still to come
        self._draining: List[
            Tuple[Any, List[int]]
        ] = []  # (event, stages) being imported
        nbytes = sum(t.nbytes for t in self.conv) + self.temporal.nbytes
        logger.info(
            "KDA TP PD decode: %d staging slots for prefill state (%.2f GB)",
            num_slots,
            nbytes / 1e9,
        )

    # Transfer registration, in MambaPool._iter_transfer_state_entries order and format.
    def iter_transfer_state_entries(self):
        for conv in self.conv:
            for i, layer_id in enumerate(self.layer_ids):
                yield "conv", conv[i], self.conv_slice_axis, layer_id
        for i, layer_id in enumerate(self.layer_ids):
            yield "temporal", self.temporal[i], 0, layer_id

    def _rank_major(self, t: torch.Tensor, kind: str) -> torch.Tensor:
        # Sizes are explicit: a rank with nothing to send has c == 0.
        W = width()
        Lc, c = t.shape[:2]
        per = math.prod(t.shape[2:]) // W
        if kind == "temporal":
            t = t.reshape(Lc, c, W, per)
            return t.permute(2, 1, 0, 3).contiguous()
        # conv: [Lc, c, K-1, D] (channels last) -> [Lc, c, K-1, W, D/W]
        k1, d = t.shape[2], t.shape[3]
        t = t.index_select(3, self._conv_rank_major).reshape(Lc, c, k1, W, d // W)
        return t.permute(3, 1, 0, 2, 4).reshape(W, c, Lc, per).contiguous()

    def _state_pairs(self, mamba_pool):
        cache = mamba_pool.mamba_cache
        pairs = [(stage, shard, "conv") for stage, shard in zip(self.conv, cache.conv)]
        pairs.append((self.temporal, cache.temporal, "temporal"))
        return pairs

    # Bounds the all-to-all buffers of one chunk of layers.
    chunk_bytes = 256 << 20

    def _layer_chunk(self, L: int, nbytes_per_layer: int) -> int:
        # Every rank must derive it from the same global sizes.
        return max(1, min(L, self.chunk_bytes // max(1, nbytes_per_layer)))

    def scatter(
        self, mamba_pool, stages: List[int], counts: List[int], dst: torch.Tensor
    ) -> None:
        # Collective over the TP group; every rank passes the same ``counts``.
        g = group()
        W = g.world_size
        dev = self.temporal.device
        n_max, n_total = max(counts), sum(counts)
        c = len(stages)
        stage_idx = torch.tensor(stages, dtype=torch.int64, device=dev)
        L = len(self.layer_ids)
        for src, dst_t, kind in self._state_pairs(mamba_pool):
            Lc = self._layer_chunk(L, n_max * src[0, 0].nbytes)
            for l0 in range(0, L, Lc):
                l1 = min(L, l0 + Lc)
                send = self._rank_major(src[l0:l1].index_select(1, stage_idx), kind)
                per = send.shape[-1]
                recv = torch.empty((n_total, l1 - l0, per), dtype=src.dtype, device=dev)
                torch.distributed.all_to_all_single(
                    recv.view(-1),
                    send.view(-1),
                    output_split_sizes=[n * (l1 - l0) * per for n in counts],
                    input_split_sizes=[c * (l1 - l0) * per] * W,
                    group=g.device_group,
                )
                slot_shape = dst_t.shape[2:]
                dst_t[l0:l1, dst] = recv.view(n_total, l1 - l0, *slot_shape).transpose(
                    0, 1
                )

    def _reclaim(self) -> None:
        still = []
        for event, stages in self._draining:
            if event is None or event.query():
                self._free.extend(stages)
            else:
                still.append((event, stages))
        self._draining = still

    def has_room(self, allocator, req_pool=None) -> bool:
        self._reclaim()
        if len(self._free) <= self._reserved and self._stage_of:
            # Reclaim stages leaked by requests freed outside free_mamba_cache (one sync).
            free_slots = set(allocator.free_slots.tolist())
            free_rows = set(req_pool.free_slots) if req_pool is not None else set()
            stale = [
                s
                for s in self._stage_of
                if s not in self._pending
                and ((s > 0 and s in free_slots) or (s < 0 and -s in free_rows))
            ]
            for slot in stale:
                self.release_slot(slot)
        return len(self._free) > self._reserved

    def reserve(self) -> None:
        if len(self._free) <= self._reserved:
            raise RuntimeError(
                "KDA TP PD staging: no stage to reserve (check has_room first)"
            )
        self._reserved += 1

    def unreserve(self) -> None:
        if self._reserved <= 0:
            raise RuntimeError("KDA TP PD staging: unreserve without a reservation")
        self._reserved -= 1

    def rekey(self, key: int, slot: int) -> None:
        if key not in self._stage_of:
            raise RuntimeError(f"KDA TP PD staging: key {key} has no staged state")
        if slot in self._stage_of:
            # The slot was freed and reallocated without release_slot (as in assign).
            self.release_slot(slot)
        if key in self._pending:
            raise RuntimeError(
                f"KDA TP PD staging: key {key} was queued before admission"
            )
        self._stage_of[slot] = self._stage_of.pop(key)

    def assign(self, slot: int, reserved: bool = False) -> int:
        if slot in self._stage_of:
            # The slot was freed and reallocated without release_slot.
            self.release_slot(slot)
        if reserved:
            self.unreserve()
        if len(self._free) <= self._reserved:
            raise RuntimeError(
                "KDA TP PD staging: no free stage (check has_room first)"
            )
        stage = self._free.pop()
        self._stage_of[slot] = stage
        return stage

    def queue_import(self, slot: int) -> None:
        if slot not in self._stage_of:
            raise RuntimeError(f"KDA TP PD staging: slot {slot} has no staged state")
        if slot not in self._pending:
            self._pending.append(slot)

    def take_pending(self) -> List[Tuple[int, int]]:
        out = [(self._stage_of[s], s) for s in self._pending]
        self._pending = []
        return out

    def finish_import(self, pairs: List[Tuple[int, int]], event) -> None:
        stages = []
        for stage, slot in pairs:
            if self._stage_of.get(slot) == stage:
                del self._stage_of[slot]
                stages.append(stage)
        if stages:
            self._draining.append((event, stages))

    def release_request(self, req) -> None:
        # Both keys are host ints recorded at assign / admission, so this does not sync.
        for attr in ("kda_tp_staged_slot", "kda_tp_state_slot"):
            key = getattr(req, attr, None)
            if key is not None:
                self.release_slot(key)
                setattr(req, attr, None)

    def release_slot(self, slot: int) -> None:
        stage = self._stage_of.pop(slot, None)
        if stage is None:
            return
        if slot in self._pending:
            self._pending.remove(slot)
        self._free.append(stage)

    def import_states(
        self, mamba_pool, local: List[Tuple[int, int]], counts: List[int]
    ) -> None:
        # Collective over the TP group (every rank, counts from the MLP sync).
        dev = self.temporal.device
        slots = torch.tensor([s for _, s in local], dtype=torch.int64, device=dev)
        gathered = all_gather_padded(slots, max(counts), -1).view(
            group().world_size, max(counts)
        )
        dst = torch.cat(
            [gathered[j, : counts[j]] for j in range(len(counts)) if counts[j]]
        )
        self.scatter(mamba_pool, [st for st, _ in local], counts, dst)
        event = None
        if dev.type == "cuda":
            event = torch.cuda.Event()
            event.record()
        self.finish_import(local, event)

    def run_imports(self, mamba_pool, forward_batch) -> None:
        # Every rank, before the KDA layers or a graph replay read the states.
        counts = forward_batch.dp_kda_tp_state_imports
        local = forward_batch.kda_tp_state_imports or []
        forward_batch.kda_tp_state_imports = None
        if not counts or not any(counts):
            return
        rank = get_parallel().attn_dp_rank
        if len(local) != counts[rank]:
            raise RuntimeError(
                f"KDA TP: {len(local)} local state imports, MLP sync advertised {counts[rank]}"
            )
        self.import_states(mamba_pool, local, counts)

    def commit_after_verify(
        self,
        model_runner,
        batch,
        last_correct_step_indices,
        mamba_steps_to_track,
        state_slots=None,
    ):
        # A collective: idle DP ranks call it with no requests.
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            scatter_mamba_states_after_mtp_verify,
        )
        from sglang.srt.mem_cache.kv.decl import STATE_PART

        req_pool = model_runner.req_to_token_pool
        W = width()
        n = req_pool.size  # per-DP-rank request capacity
        dev = torch.device("cuda", torch.cuda.current_device())
        bs = (
            0
            if last_correct_step_indices is None
            else last_correct_step_indices.shape[0]
        )
        if bs > n:
            raise RuntimeError(
                f"KDA TP commit: {bs} requests exceed the per-rank capacity {n}"
            )
        pack = torch.zeros((5, n), dtype=torch.int64, device=dev)
        # Padding rows commit nothing (step -1); a step 0 would copy into slot 0 every step.
        pack[1].fill_(-1)
        pack[3].fill_(-1)  # no track step
        if bs:
            if state_slots is None:
                # Not the request table, which the scheduler may have rewritten since the forward.
                state_slots = model_runner.attn_backend.forward_state_slots(
                    bs, batch.mamba_track_indices
                )
            slots = state_slots.working[:bs]
            track = state_slots.track
            pack[0, :bs] = slots.to(torch.int64)
            pack[1, :bs] = last_correct_step_indices.to(torch.int64)
            if (
                batch.mamba_track_indices is not None
                and track is not None
                and mamba_steps_to_track is not None
            ):
                pack[2, :bs] = track[:bs].to(torch.int64)
                pack[3, :bs] = mamba_steps_to_track.to(torch.int64)
        rank = get_parallel().attn_dp_rank
        pack[4] = rank * n + torch.arange(
            n, device=dev
        )  # see _fast_context's scratch rows
        mamba_pool = model_runner.token_to_kv_pool.layer_router.part_view(STATE_PART)
        caches = mamba_pool.get_speculative_mamba2_params_all_layers()
        out = pack.new_empty((W, 5, n))
        group().all_gather_into_tensor(out.view(W * 5, n), pack)
        slots, last, track, steps, rows = out.permute(1, 0, 2).reshape(5, W * n)
        scatter_mamba_states_after_mtp_verify(
            caches,
            slots,
            last,
            track,
            steps,
            src_indices_raw=rows,
            rows_loop=envs.SGLANG_MAMBA_SSM_COMMIT_ROW_LOOP.get(),
        )

    def admit(self, req_to_token_pool, req) -> bool:
        """Whether a transferred request can join the batch now (its state slot is bound)."""
        if not getattr(req, "kda_tp_lazy_mamba", False) or req.kv.holds_mamba:
            return True
        mid = req_to_token_pool.mamba_allocator.alloc(1)
        if mid is None:
            return False
        req.kv.mamba_pool_idx = mid[0]
        req.kv.mamba_needs_clear = True
        req_to_token_pool.req_index_to_mamba_index_mapping[req.kv.req_pool_idx] = mid[
            0
        ].to(torch.int32)
        # The one host read of the new slot; release keys the stage by it.
        slot = int(mid[0])
        req.kda_tp_state_slot = slot
        key = lazy_stage_key(req.kv.req_pool_idx)
        if getattr(req, "kda_tp_staged_slot", None) == key:
            self.rekey(key, slot)
            req.kda_tp_staged_slot = slot
        req.kda_tp_lazy_mamba = False
        return True

    def mark_lazy(self, req_to_token_pool, req) -> None:
        if not req.kv.holds_mamba and lazy_mamba_slot(
            req_to_token_pool.enable_mamba_extra_buffer
        ):
            req.kda_tp_lazy_mamba = True

    def has_prealloc_room(self, req_to_token_pool) -> bool:
        return self.has_room(req_to_token_pool.mamba_allocator, req_to_token_pool)

    def reserve_for(self, req) -> None:
        # Deferred publish assigns the stage later; hold it so has_room counts it.
        self.reserve()
        req.kda_tp_stage_reserved = True

    def drop_reservation(self, req) -> None:
        if getattr(req, "kda_tp_stage_reserved", False):
            self.unreserve()
            req.kda_tp_stage_reserved = False

    def state_destination(self, req, slots: np.ndarray) -> np.ndarray:
        """The stage the prefill writes the request's state into, in place of its slot ids."""
        slot = int(slots.reshape(-1)[0])
        if getattr(req, "kda_tp_lazy_mamba", False):
            slot = lazy_stage_key(req.kv.req_pool_idx)
        else:
            req.kda_tp_state_slot = slot
        req.kda_tp_staged_slot = slot
        stage = self.assign(slot, reserved=getattr(req, "kda_tp_stage_reserved", False))
        req.kda_tp_stage_reserved = False
        return np.full(slots.shape, stage, dtype=slots.dtype)

    def queue_imports(self, reqs) -> None:
        for req in reqs:
            if getattr(req, "kda_tp_staged_slot", None) is not None:
                self.queue_import(req.kda_tp_staged_slot)
                req.kda_tp_staged_slot = None


def mirror_deferred_state_ops(model_runner, forward_batch) -> None:
    # Runs on every rank at every target forward start; no-op unless a DP rank extends.
    modes = getattr(forward_batch, "dp_forward_modes", None)
    if modes is not None:
        from sglang.srt.model_executor.forward_mode import ForwardMode

        if not any(
            ForwardMode(m).is_extend() and not ForwardMode(m).is_target_verify()
            for m in modes
        ):
            return
    state = model_runner.token_to_kv_pool_allocator.state
    if state.ckpt_pool is not None:
        raise NotImplementedError("KDA TP: int8 mamba checkpoints are not supported")

    def ids(t):
        return [] if t is None else [int(x) for x in t.tolist()]

    local = (
        ids(forward_batch.mamba_clear_indices),
        ids(forward_batch.mamba_cow_src_indices),
        ids(forward_batch.mamba_cow_dst_indices),
    )
    forward_batch.mamba_clear_indices = None
    forward_batch.mamba_cow_src_indices = None
    forward_batch.mamba_cow_dst_indices = None
    out: List[Optional[tuple]] = [None] * width()
    torch.distributed.all_gather_object(out, local, group=group().cpu_group)
    clear = [i for o in out for i in o[0]]
    src = [i for o in out for i in o[1]]
    dst = [i for o in out for i in o[2]]
    dev = state.pool.device
    # Global slot ids index physical storage directly (no compaction under KDA TP).
    if clear:
        state.pool.clear_slots(torch.tensor(clear, dtype=torch.int64, device=dev))
    if src:
        from sglang.srt.mem_cache.kv.decl import STATE_PART

        # The copy reads every state layer of the source slot.
        model_runner.token_to_kv_pool.layer_router.wait_for_part(STATE_PART)
        state.copy(
            torch.tensor(src, dtype=torch.int64, device=dev),
            torch.tensor(dst, dtype=torch.int64, device=dev),
        )
