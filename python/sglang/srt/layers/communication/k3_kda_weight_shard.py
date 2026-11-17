"""DWDP for K3 KDA projection and shared-expert weights (``--kda-dwdp-size N``).

Each shard-group rank keeps 1/N of every KDA layer's projections; a layer's full weights
are gathered into a rotating slot by copy-engine peer pulls before it runs.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.utils.cuda_vmm_utils import (
    VmmReservation,
    align_up,
    check_drv,
    exchange_shareable_handles,
    get_device_granularity,
    import_and_map_alloc,
    make_device_allocation_prop,
    release_mappings,
    tensor_from_pointer,
)

logger = logging.getLogger(__name__)


# 256 B for bf16, so parameter views keep the alignment cuBLAS and TMA expect.
ALIGN = 128


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: Tuple[int, ...]
    offset: int
    # "rows": each of ``blocks`` row blocks is cut N ways; "cols": columns cut N ways.
    split: str = "rows"
    blocks: int = 1

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def layer_layout(shapes: Sequence[tuple]):
    specs, off = [], 0
    for name, shape, split, blocks in shapes:
        if split not in ("rows", "cols") or blocks < 1 or len(shape) != 2:
            raise ValueError(f"{name}: bad shard spec {shape} {split} x{blocks}")
        spec = ParamSpec(name, tuple(shape), off, split, blocks)
        specs.append(spec)
        off += spec.numel
        off += (-off) % ALIGN
    return specs, off


def kda_unit_shapes(proj: int, hidden: int) -> list:
    return [
        ("qkvg", (4 * proj, hidden), "rows", 4),
        ("o_proj", (hidden, proj), "cols", 1),
    ]


def shared_expert_unit_shapes(intermediate: int, hidden: int) -> list:
    return [
        ("gate_up", (2 * intermediate, hidden), "rows", 2),
        ("down", (hidden, intermediate), "cols", 1),
    ]


@dataclass(frozen=True)
class _Piece:
    """This rank's part of one row block or column slice of a parameter."""

    name: str
    block: int
    off: int
    rows: int
    cols: int


def unit_pieces(
    specs: Sequence[ParamSpec], group_size: int
) -> Tuple[List[_Piece], int]:
    pieces, off = [], 0
    for sp in specs:
        R, C = sp.shape
        if sp.split == "rows":
            if R % (sp.blocks * group_size):
                raise ValueError(
                    f"{sp.name}: {R} rows do not split into {sp.blocks} blocks x "
                    f"{group_size} ranks"
                )
            dims = [(b, R // (sp.blocks * group_size), C) for b in range(sp.blocks)]
        else:
            if C % group_size:
                raise ValueError(
                    f"{sp.name}: {C} columns do not split over {group_size} ranks"
                )
            dims = [(0, R, C // group_size)]
        for b, rows, cols in dims:
            pieces.append(_Piece(sp.name, b, off, rows, cols))
            off += rows * cols
            off += (-off) % ALIGN
    return pieces, off


class LayerWeightShard:
    """Per-rank shard store, gather slots and gather schedule for weight-sharded layers."""

    def __init__(
        self,
        *,
        units: Sequence[Tuple[int, Sequence[tuple]]],
        num_layers: int,
        group_rank: int,
        group_size: int,
        dtype: torch.dtype,
        device: torch.device,
        alloc_shard: Callable[[int], torch.Tensor],
        peer_shards: Callable[[torch.Tensor], List[torch.Tensor]],
        group_barrier: Optional[Callable[[], None]] = None,
    ):
        if group_size < 2:
            raise ValueError(f"weight shard group size must be >= 2, got {group_size}")
        if not 0 <= group_rank < group_size:
            raise ValueError(
                f"group rank {group_rank} out of range for size {group_size}"
            )
        self.unit_layers = [layer for layer, _ in units]
        if not self.unit_layers or self.unit_layers != sorted(set(self.unit_layers)):
            raise ValueError(
                "units must be non-empty with strictly ascending layer indices"
            )
        if self.unit_layers[-1] >= num_layers:
            raise ValueError(
                f"sharded layer {self.unit_layers[-1]} does not fit {num_layers} decoder layers"
            )
        self.num_units = len(self.unit_layers)
        self.unit_index: Dict[int, int] = {l: u for u, l in enumerate(self.unit_layers)}
        # First unit at or after each decoder layer (num_units past the last one).
        self.next_unit: List[int] = []
        u = 0
        for layer in range(num_layers):
            while u < self.num_units and self.unit_layers[u] < layer:
                u += 1
            self.next_unit.append(u)
        self.group_rank, self.group_size = group_rank, group_size
        self.specs: List[Dict[str, ParamSpec]] = []
        self.e_size: List[int] = []
        self.pieces: List[List[_Piece]] = []
        self.shard_size: List[int] = []
        self.shard_offset: List[int] = []
        off = 0
        for _, shapes in units:
            specs, e_size = layer_layout(shapes)
            pieces, size = unit_pieces(specs, group_size)
            self.specs.append({sp.name: sp for sp in specs})
            self.e_size.append(e_size)
            self.pieces.append(pieces)
            self.shard_size.append(size)
            self.shard_offset.append(off)
            off += size
        self.dtype, self.device = dtype, device
        self.shard = alloc_shard(off)
        if self.shard.numel() != off or self.shard.dtype != dtype:
            raise ValueError("alloc_shard returned a buffer of the wrong size or dtype")
        self.peers = peer_shards(self.shard)
        if len(self.peers) != group_size:
            raise ValueError(
                f"peer_shards returned {len(self.peers)} buffers, want {group_size}"
            )
        # The running layer's slot and the next sharded layer's, gathered while it runs.
        self.num_slots = 2
        self.slots = torch.empty(
            (self.num_slots, max(self.e_size)), dtype=dtype, device=device
        )
        self._group_barrier = group_barrier
        self.dummy_load = False
        # Per unit: piece index -> disjoint row ranges [x, y) loaded so far.
        self._loaded: List[Dict[int, List[Tuple[int, int]]]] = [
            {} for _ in range(self.num_units)
        ]

        cuda = device.type == "cuda"
        self._copy_stream = torch.cuda.Stream(device=device) if cuda else None
        # unit -> ready event (None on CPU) for gathers issued and not yet waited on.
        self._pending: Dict[int, Optional[torch.cuda.Event]] = {}
        # slot -> unit holding it until that unit's layer finishes; None when free.
        self._slot_owner: List[Optional[int]] = [None] * self.num_slots
        self._running: Optional[int] = None
        # Units whose gather this layer's start issued (the MoE stage waits for them).
        self._issued_this_layer: List[int] = []
        self._pull_ops: Optional[List[list]] = None
        self._ready_events: Optional[List[torch.cuda.Event]] = None
        self._join_event = torch.cuda.Event() if cuda else None
        self._copy_ctx = torch.cuda.stream(self._copy_stream) if cuda else None
        self._device_index = device.index if cuda else None

    def slot_of(self, u: int) -> int:
        return u % self.num_slots

    def unit_buffer(self, u: int) -> torch.Tensor:
        return self.slots[self.slot_of(u), : self.e_size[u]]

    def param_view(self, u: int, name: str) -> torch.Tensor:
        sp = self.specs[u][name]
        return self.unit_buffer(u)[sp.offset : sp.offset + sp.numel].view(sp.shape)

    def own_view(self, u: int, name: str) -> torch.Tensor:
        mine = [pc for pc in self.pieces[u] if pc.name == name]
        if not mine:
            raise KeyError(f"unit {u} has no parameter {name!r}")
        start, cols = mine[0].off, mine[0].cols
        rows = 0
        for pc in mine:
            if pc.off != start + rows * cols:
                raise RuntimeError(f"{name}: pieces are not contiguous in the shard")
            rows += pc.rows
        base = self.shard_offset[u] + start
        return self.shard[base : base + rows * cols].view(rows, cols)

    def load(
        self, u: int, name: str, loaded: torch.Tensor, row_offset: int = 0
    ) -> None:
        sp = self.specs[u][name]
        R, C = sp.shape
        if loaded.dim() != 2 or loaded.shape[1] != C:
            raise ValueError(
                f"{name}: loaded shape {tuple(loaded.shape)} does not match {sp.shape}"
            )
        n = loaded.shape[0]
        if row_offset < 0 or row_offset + n > R:
            raise ValueError(f"{name}: rows [{row_offset}, +{n}) outside {sp.shape}")
        r = self.group_rank
        for i, pc in enumerate(self.pieces[u]):
            if pc.name != name:
                continue
            piece = self.shard[
                self.shard_offset[u] + pc.off : self.shard_offset[u]
                + pc.off
                + pc.rows * pc.cols
            ].view(pc.rows, pc.cols)
            if sp.split == "rows":
                lo = (pc.block * self.group_size + r) * pc.rows
                x, y = max(lo, row_offset), min(lo + pc.rows, row_offset + n)
                if x < y:
                    self._mark_loaded(u, i, x - lo, y - lo)
                    piece[x - lo : y - lo].copy_(
                        loaded[x - row_offset : y - row_offset]
                    )
            else:
                self._mark_loaded(u, i, row_offset, row_offset + n)
                piece[row_offset : row_offset + n].copy_(
                    loaded[:, r * pc.cols : (r + 1) * pc.cols]
                )

    def _mark_loaded(self, u: int, i: int, x: int, y: int) -> None:
        ranges = self._loaded[u].setdefault(i, [])
        for a, b in ranges:
            if a < y and x < b:
                pc = self.pieces[u][i]
                raise RuntimeError(
                    f"weight shard: layer {self.unit_layers[u]} {pc.name} block {pc.block} "
                    f"rows [{max(a, x)}, {min(b, y)}) of this rank's slice loaded twice"
                )
        ranges.append((x, y))

    def check_loaded(self) -> None:
        bad = []
        for u, pieces in enumerate(self.pieces):
            for i, pc in enumerate(pieces):
                got = sum(b - a for a, b in self._loaded[u].get(i, ()))
                if got != pc.rows:
                    bad.append((self.unit_layers[u], pc.name, pc.block, got, pc.rows))
        if bad:
            raise RuntimeError(
                "weight shard: slices not fully loaded, as (layer, parameter, block, rows "
                f"loaded, rows expected): {bad[:8]}"
            )

    def init_dummy(self, low: float = -1e-3, high: float = 1e-3, seed: int = 1234):
        if any(self._loaded):
            raise RuntimeError(
                "weight shard: dummy init after checkpoint slices loaded"
            )
        g = torch.Generator(device=self.shard.device).manual_seed(
            seed + self.group_rank
        )
        self.shard.uniform_(low, high, generator=g)
        for u, pieces in enumerate(self.pieces):
            self._loaded[u] = {i: [(0, pc.rows)] for i, pc in enumerate(pieces)}

    def finish_load(self) -> None:
        if self.dummy_load and not any(self._loaded):
            self.init_dummy()
        self.check_loaded()
        # A peer must not pull this shard before its host-to-device copies land.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self._group_barrier is not None:
            self._group_barrier()

    def _copy_ops(self, u: int) -> list:
        slot = self.unit_buffer(u)
        off, n, r = self.shard_offset[u], self.group_size, self.group_rank
        es = self.shard.element_size()
        ops = []
        # Rotated peer order so no source serves two readers at once; own pieces last.
        for p in [(r + i) % n for i in range(1, n)] + [r]:
            src_shard = self.shard if p == r else self.peers[p]
            for pc in self.pieces[u]:
                sp = self.specs[u][pc.name]
                src = src_shard[off + pc.off : off + pc.off + pc.rows * pc.cols]
                if sp.split == "rows":
                    d0 = sp.offset + (pc.block * n + p) * pc.rows * pc.cols
                    ops.append((slot[d0 : d0 + pc.rows * pc.cols], src, None))
                    continue
                R, C = sp.shape
                dst = slot[sp.offset : sp.offset + R * C].view(R, C)[
                    :, p * pc.cols : (p + 1) * pc.cols
                ]
                spec2d = None
                if self.device.type == "cuda":
                    spec2d = _memcpy2d_args(
                        dst=dst.data_ptr(),
                        dst_pitch=C * es,
                        src=src.data_ptr(),
                        src_pitch=pc.cols * es,
                        width=pc.cols * es,
                        height=R,
                    )
                ops.append((dst, src.view(pc.rows, pc.cols), spec2d))
        return ops

    @staticmethod
    def _run_op(op, stream) -> None:
        dst, src, spec2d = op
        if spec2d is None:
            dst.copy_(src, non_blocking=True)
        else:
            _memcpy2d(spec2d, stream)

    def issue(self, k: int) -> None:
        if k in self._pending:
            return
        owner = self._slot_owner[self.slot_of(k)]
        if owner is not None:
            raise RuntimeError(
                f"weight shard: gather of unit {k} would overwrite slot {self.slot_of(k)}, "
                f"still held by unit {owner}"
            )
        if self._copy_stream is not None and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "weight-shard gathers must be issued outside CUDA graph capture: run prefill "
                "eager or with the breakable prefill graph (the hooks are eager breaks), "
                "and disable decode / full graphs"
            )
        self._slot_owner[self.slot_of(k)] = k
        if self._copy_stream is None:
            for op in self._copy_ops(k):
                self._run_op(op, None)
            self._pending[k] = None
            return
        self._pending[k] = self._issue_on_copy_stream(k)

    def _issue_on_copy_stream(self, k: int) -> torch.cuda.Event:
        if self._pull_ops is None:
            self._pull_ops = [self._copy_ops(u) for u in range(self.num_units)]
            # Safe to reuse: a unit is reissued only after its last gather was waited on.
            self._ready_events = [torch.cuda.Event() for _ in range(self.num_units)]
        # Everything enqueued so far, including the last reader of this slot, precedes the copies.
        self._join_event.record(torch.cuda.current_stream(self._device_index))
        self._copy_stream.wait_event(self._join_event)
        ev = self._ready_events[k]
        with self._copy_ctx:
            for op in self._pull_ops[k]:
                self._run_op(op, self._copy_stream)
            ev.record(self._copy_stream)
        return ev

    def _release_running(self) -> None:
        if self._running is not None:
            self._slot_owner[self.slot_of(self._running)] = None
            self._running = None

    def wait(self, k: int) -> None:
        if self._running != k:
            self._release_running()
        if k not in self._pending:
            self.issue(k)
        ev = self._pending.pop(k)
        if ev is not None:
            torch.cuda.current_stream(self.device).wait_event(ev)
        self._running = k

    def on_layer_start(self, layer_idx: int) -> None:
        # The previous sharded layer's kernels are all enqueued, so its slot is free.
        self._release_running()
        self._issued_this_layer = []
        k = self.unit_index.get(layer_idx)
        if k is not None:
            self.wait(k)
            self._issue_ahead(k + 1)
        else:
            self._issue_ahead(self.next_unit[layer_idx])

    def _issue_ahead(self, u: int) -> None:
        u %= self.num_units
        if u in self._pending or self._slot_owner[self.slot_of(u)] is not None:
            return
        self.issue(u)
        self._issued_this_layer.append(u)

    def needs_layer_hook(self, layer_idx: int) -> bool:
        return layer_idx in self.unit_index or layer_idx > self.unit_layers[-1]

    def on_moe_start(self) -> None:
        # Keeps gather NVLink traffic off the MoE all-to-all.
        issued, self._issued_this_layer = self._issued_this_layer, []
        if self._copy_stream is None:
            return
        cur = torch.cuda.current_stream(self.device)
        for u in issued:
            ev = self._pending.get(u)
            if ev is not None:
                cur.wait_event(ev)

    @property
    def pending(self) -> List[int]:
        return sorted(self._pending)

    def bytes_per_rank(self) -> Tuple[int, int]:
        shard_b = self.shard.numel() * self.shard.element_size()
        return shard_b, self.slots.numel() * self.slots.element_size()


def _memcpy2d_args(
    *, dst: int, dst_pitch: int, src: int, src_pitch: int, width: int, height: int
):
    from cuda.bindings import driver as drv

    args = drv.CUDA_MEMCPY2D()
    args.srcMemoryType = drv.CUmemorytype.CU_MEMORYTYPE_DEVICE
    args.srcDevice = src
    args.srcPitch = src_pitch
    args.dstMemoryType = drv.CUmemorytype.CU_MEMORYTYPE_DEVICE
    args.dstDevice = dst
    args.dstPitch = dst_pitch
    args.WidthInBytes = width
    args.Height = height
    return args


def _memcpy2d(args, stream: torch.cuda.Stream) -> None:
    from cuda.bindings import driver as drv

    # Copy-engine strided copy; a torch copy into the strided view would launch a kernel.
    check_drv(drv.cuMemcpy2DAsync(args, stream.cuda_stream), "cuMemcpy2DAsync")


class _PeerShards:
    """This rank's shard as one exportable cuMem allocation, and every group peer's
    shard mapped into this rank's address space."""

    def __init__(self, group, dtype: torch.dtype, device: torch.device):
        self.group, self.dtype = group, dtype
        self.device_id = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        # The primary context must exist before any driver call.
        torch.empty(0, device=device)
        self._local: Optional[VmmReservation] = None
        self._handle = None
        self._numel = self._nbytes = 0
        self._peer_maps: list = []

    def _view(self, va: int) -> torch.Tensor:
        es = torch.empty((), dtype=self.dtype).element_size()
        return tensor_from_pointer(
            va,
            self._numel * es,
            shape=(self._numel,),
            dtype=self.dtype,
            device_id=self.device_id,
        )

    def alloc(self, numel: int) -> torch.Tensor:
        gran = get_device_granularity(self.device_id)
        es = torch.empty((), dtype=self.dtype).element_size()
        self._numel, self._nbytes = numel, align_up(numel * es, gran)
        self._local = VmmReservation(
            self._nbytes,
            make_device_allocation_prop(self.device_id),
            self.device_id,
            alignment=gran,
        )
        self._handle = self._local.map(0, self._nbytes, retain_handle=True)
        return self._view(self._local.base)

    def peers(self, shard: torch.Tensor) -> List[torch.Tensor]:
        dist = torch.distributed
        rank, n = dist.get_rank(self.group), dist.get_world_size(self.group)
        t0 = time.perf_counter()
        logger.info("K3 KDA weight shard: mapping %d peer shards", n - 1)
        use_fabric, peer_handles = exchange_shareable_handles(
            [self._handle], self.group, rank, n
        )
        out = []
        try:
            for p in range(n):
                if p == rank:
                    out.append(shard)
                    continue
                handle = peer_handles[(p, 0)]
                va = import_and_map_alloc(
                    handle if use_fabric else None,
                    None if use_fabric else handle,
                    self._nbytes,
                    self.device_id,
                    use_fabric=use_fabric,
                    peer_rank=p,
                )
                self._peer_maps.append((va, self._nbytes, [(0, self._nbytes)]))
                out.append(self._view(va))
        finally:
            if not use_fabric:
                for fd in peer_handles.values():
                    os.close(fd)
        dist.barrier(group=self.group)  # every peer mapped before anyone pulls
        logger.info(
            "K3 KDA weight shard: peer shard mappings via %s done in %.2f s",
            "FABRIC" if use_fabric else "POSIX fd",
            time.perf_counter() - t0,
        )
        return out

    def release(self) -> None:
        try:
            torch.cuda.synchronize(self.device_id)
        except Exception:
            # Interpreter shutdown: the driver tears the context down anyway.
            return
        release_mappings(self._peer_maps)
        if self._local is not None:
            self._local.close()
            self._local = None


_ACTIVE: Optional[LayerWeightShard] = None
_ACTIVE_SHARED: Optional[LayerWeightShard] = None
_BUILDING = False


@contextlib.contextmanager
def building_target_model():
    """KDA layers built in this scope are sharded; the draft model keeps replicated weights."""
    global _BUILDING
    prev, _BUILDING = _BUILDING, True
    try:
        yield
    finally:
        _BUILDING = prev


def active() -> Optional[LayerWeightShard]:
    return _ACTIVE


def get_for_construction() -> Optional[LayerWeightShard]:
    return _ACTIVE if _BUILDING else None


def get_shared_for_construction() -> Optional[LayerWeightShard]:
    return _ACTIVE_SHARED if _BUILDING else None


def _stores() -> List[LayerWeightShard]:
    return [s for s in (_ACTIVE, _ACTIVE_SHARED) if s is not None]


def needs_layer_hook(layer_idx: int) -> bool:
    return any(s.needs_layer_hook(layer_idx) for s in _stores())


def on_layer_start(layer_idx: int) -> None:
    for s in _stores():
        if s.needs_layer_hook(layer_idx):
            s.on_layer_start(layer_idx)


def on_moe_start(layer_idx: int) -> None:
    for s in _stores():
        if s.needs_layer_hook(layer_idx):
            s.on_moe_start()


def check_all_loaded() -> None:
    for s in _stores():
        s.check_loaded()


def finish_load() -> None:
    for s in _stores():
        s.finish_load()


def shard_group_layout(
    ranks: Sequence[int], attn_tp: int, group_size: int, tp_rank: int
) -> tuple[list[list[int]], int, int]:
    """(every shard group's member ranks, this rank's group index, its rank in the group)."""
    dp_size = len(ranks) // attn_tp
    if len(ranks) % attn_tp != 0 or dp_size % group_size != 0:
        raise ValueError(
            f"--kda-dwdp-size {group_size} must divide the attention-DP size "
            f"{dp_size} (TP {len(ranks)}, attention TP {attn_tp})"
        )
    groups_per_slice = dp_size // group_size
    members = [
        [ranks[(g * group_size + i) * attn_tp + a] for i in range(group_size)]
        for a in range(attn_tp)
        for g in range(groups_per_slice)
    ]
    attn_tp_rank, dp_rank = tp_rank % attn_tp, tp_rank // attn_tp
    return (
        members,
        attn_tp_rank * groups_per_slice + dp_rank // group_size,
        dp_rank % group_size,
    )


def init_for_model(
    *,
    config,
    group_size: int,
    dtype: torch.dtype,
    shared_expert_layers: Sequence[int] = (),
    shared_intermediate_size: int = 0,
    dummy_load: bool = False,
) -> LayerWeightShard:
    global _ACTIVE, _ACTIVE_SHARED
    if _ACTIVE is not None or _ACTIVE_SHARED is not None:
        raise RuntimeError(
            "--kda-dwdp-size: the weight shard stores were already built for a "
            "Kimi-K3 model in this process; a second K3-architecture model (e.g. a draft) "
            "cannot be weight-sharded too. Use a draft of another architecture, or drop "
            "--kda-dwdp-size."
        )
    from sglang.srt.runtime_context import get_parallel

    par = get_parallel()
    attn_tp = par.attn_tp_size
    tp_group = par.tp_group
    ranks = list(tp_group.ranks)
    tp_rank = tp_group.rank_in_group
    group_members, my_group, group_rank = shard_group_layout(
        ranks, attn_tp, group_size, tp_rank
    )
    cpu_groups = []
    # Every rank creates every subgroup, in the same order (torch.distributed contract).
    for members in group_members:
        cpu_groups.append(torch.distributed.new_group(members, backend="gloo"))
    device = torch.device("cuda", torch.cuda.current_device())
    num_layers = config.num_hidden_layers
    kda_ids = [l for l in config.linear_layer_ids if l < num_layers]
    H = config.hidden_size
    P = (
        config.linear_attn_config["num_heads"]
        * config.linear_attn_config["head_dim"]
        // attn_tp
    )
    group = cpu_groups[my_group]

    def build(units) -> LayerWeightShard:
        vmm = _PeerShards(group, dtype, device)
        store = LayerWeightShard(
            units=units,
            num_layers=num_layers,
            group_rank=group_rank,
            group_size=group_size,
            dtype=dtype,
            device=device,
            alloc_shard=vmm.alloc,
            peer_shards=vmm.peers,
            group_barrier=functools.partial(torch.distributed.barrier, group=group),
        )
        store.dummy_load = dummy_load
        # The shard and peer views do not own their memory; it lives as long as the store.
        weakref.finalize(store, vmm.release)
        return store

    if shared_expert_layers:
        I = shared_intermediate_size
        _ACTIVE_SHARED = build(
            [(layer, shared_expert_unit_shapes(I, H)) for layer in shared_expert_layers]
        )
        s_shard_b, s_slot_b = _ACTIVE_SHARED.bytes_per_rank()
        logger.info(
            "K3 shared-expert weight sharding: %d MoE layers: shard %.2f GB + %d slots %.2f GB "
            "per rank instead of %.2f GB replicated",
            len(shared_expert_layers),
            s_shard_b / 1e9,
            _ACTIVE_SHARED.num_slots,
            s_slot_b / 1e9,
            len(shared_expert_layers) * 3 * I * H * 2 / 1e9,
        )
    _ACTIVE = build([(layer, kda_unit_shapes(P, H)) for layer in kda_ids])
    shard_b, slot_b = _ACTIVE.bytes_per_rank()
    replicated = len(kda_ids) * (4 * P * H + H * P) * 2
    logger.info(
        "K3 KDA weight sharding: shard group %d (%d ranks, this is rank %d), %d KDA "
        "layers: head-aligned shard %.2f GB + %d slots %.2f GB per rank instead of "
        "%.2f GB replicated",
        my_group,
        group_size,
        group_rank,
        len(kda_ids),
        shard_b / 1e9,
        _ACTIVE.num_slots,
        slot_b / 1e9,
        replicated / 1e9,
    )
    return _ACTIVE
