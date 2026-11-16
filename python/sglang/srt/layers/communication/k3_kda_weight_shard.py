"""K3 KDA weight sharding for attention-DP prefill (``--kda-weight-shard-size N``).

Under attention DP every rank runs every KDA layer on its own requests with all
96 heads, so each rank would hold all 69 KDA layers' projection weights (~61 GB
BF16). Here each rank of a shard group (N attention-DP ranks holding the same
attention-TP slice) keeps 1/N of every KDA layer's large projections ([q,k,v,g]
in-projection and o_proj), and the layer's full weights are gathered into one of
``D + 1`` rotating slots before the layer runs (``D`` = prefetch depth,
SGLANG_K3_KDA_WS_PREFETCH_DEPTH, default 1). The KDA math, its state pool and its
scheduling are untouched: the gathered slot is byte-identical to the replicated
weights, so outputs are bitwise identical to replicated ADP.

Layout (head-aligned): rank ``r`` keeps exactly what a tensor-parallel rank ``r`` of
``N`` holds of its slice: heads/N of every row block of the merged [q | k | v | g]
in-projection and the matching heads/N input columns of o_proj (``unit_pieces``), so
with N == TP and attention TP 1 a rank's shard is byte-for-byte KDA TP rank r's KDA
weights (``own_view``). The slot holds the replicated layout ``[qkvg (4P x H) | o_proj
(H x P)]``; sharded layer ``k`` reads slot ``k % (D + 1)`` through permanent parameter
views, so nothing is rebound per forward. Shared experts (when replicated under an EP
all-to-all backend) use the same scheme: [gate | up] rows and down columns.

Gather (``mover="ce"``, default): the shard is one cuMem allocation exported to the
group and mapped on every peer (``_vmm_alloc_and_peers``: POSIX fds within a host,
fabric handles across hosts), and it never changes after loading, so a gather is
peer pulls by the copy engines on a private copy stream, with zero SMs: per peer one
contiguous copy per in-projection row block and one ``cuMemcpy2DAsync`` for its o_proj
column slice (the slot's row pitch). This rank's own pieces are copied last from its
own shard. No barrier and no collective: a rank never waits on a peer, so DP ranks
idle in a forward need not take part. ``ce_ms`` uses one copy stream per peer.

Schedule (``on_layer_start`` at every decoder layer start, i.e. after the previous
MoE stage): a sharded layer waits for its gather, then the next ``D`` sharded
layers' gathers are issued (into the next forward once this one has none left).
With ``issue_at="core"`` (SGLANG_K3_KDA_WS_ISSUE_AT) a sharded layer issues them from
``on_core_start`` instead, after its in-projection GEMM, so the pulls overlap the KDA
core rather than the HBM-bound in-projection.
The slot a gather overwrites was last read by a layer whose kernels precede the
issue point on the compute stream; the copy stream waits on it, and a slot is
never reissued while its layer is pending or running. ``on_moe_start`` (before the
DEP dispatch) makes the compute stream wait for the gathers this layer issued, so
gather traffic never overlaps the MoE all-to-all; a gather longer than the
attention stage becomes a stall there instead (SGLANG_K3_KDA_WS_STATS times it).
SGLANG_K3_KDA_WS_MOE_WAIT selects when that MoE-stage wait applies: "always" (default),
"never", or a token count N (only in forwards whose largest per-DP-rank token count is
>= N). Without it the gather runs on through the MoE stage and the only wait is the
consuming layer's ``on_layer_start``; the slot rules above do not depend on it.

CUDA graphs: issuing a gather inside a capture is refused. The model calls both
hooks through ``eager_on_graph``, so under the breakable prefill graph they run
between captured segments in eager stream order; full graphs are not supported.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import logging
import os
import weakref
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


# Element alignment of every shard boundary and parameter offset (256 B for bf16), so
# the parameter views keep the alignment cuBLAS and TMA see on the replicated weights.
ALIGN = 128
MOVERS = ("ce", "ce_ms")


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: Tuple[int, ...]  # 2-D [rows, cols]: the parameter as the layer reads it
    offset: int  # elements, from the start of the layer's flat buffer
    # How the shard group splits it, as tensor parallelism does: "rows" cuts each of
    # ``blocks`` row blocks N ways (a merged column-parallel weight: rank r keeps its
    # rows of every block); "cols" cuts the columns N ways (a row-parallel weight).
    split: str = "rows"
    blocks: int = 1

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def parse_moe_wait(value: str) -> Optional[int]:
    """SGLANG_K3_KDA_WS_MOE_WAIT as the smallest forward token count whose MoE stage
    waits for the layer's gathers: "always" -> 0, "never" -> None, "N" -> N."""
    v = value.strip().lower()
    if v == "always":
        return 0
    if v == "never":
        return None
    if v.isdigit():
        return int(v)
    raise ValueError(
        f"SGLANG_K3_KDA_WS_MOE_WAIT must be 'always', 'never' or a token count, got {value!r}"
    )


def layer_layout(shapes: Sequence[tuple]):
    """Param specs at ALIGN-aligned offsets and the layer's flat size. ``shapes``:
    ``(name, (rows, cols), split, blocks)`` per parameter (``split`` / ``blocks`` as
    ParamSpec)."""
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
    """A KDA layer's sharded projections as tensor parallelism slices them: the merged
    [q | k | v | g] in-projection (4 row blocks) and o_proj (columns)."""
    return [
        ("qkvg", (4 * proj, hidden), "rows", 4),
        ("o_proj", (hidden, proj), "cols", 1),
    ]


def shared_expert_unit_shapes(intermediate: int, hidden: int) -> list:
    """Shared experts as tensor parallelism slices them: [gate | up] and down."""
    return [
        ("gate_up", (2 * intermediate, hidden), "rows", 2),
        ("down", (hidden, intermediate), "cols", 1),
    ]


@dataclass(frozen=True)
class _Piece:
    """This rank's part of one row block (``split="rows"``) or of the columns
    (``split="cols"``) of a parameter, stored row-major ``[rows, cols]`` at ``off``
    elements into the unit's shard."""

    name: str
    block: int
    off: int
    rows: int
    cols: int


def unit_pieces(
    specs: Sequence[ParamSpec], group_size: int
) -> Tuple[List[_Piece], int]:
    """Head-aligned shard of one unit (every group rank has the same piece sizes): per
    parameter, in block order, the rank's rows of each row block, or its column slice.
    Returns the pieces and the shard elements; every piece starts ALIGN-aligned."""
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
    """Per-rank shard store, gather slots and gather schedule for weight-sharded layers.

    ``units``: ``(decoder layer index, [(name, (rows, cols), split, blocks), ...])`` per
    sharded layer (``layer_layout``), in ascending layer order; units may have different
    shapes (the slots are sized to the largest). Unit ``u`` is gathered into slot
    ``u % (prefetch_depth + 1)``.
    ``num_layers``: decoder layers in the model (the schedule walks all of them).
    ``alloc_shard``: allocates this rank's shard buffer (elements).
    ``peer_shards``: called once with this rank's shard buffer, returns every group
    rank's shard buffer (index = rank in group) as tensors this rank can copy from.
    ``mover``: "ce" (one copy stream) or "ce_ms" (one copy stream per peer).
    ``moe_wait_min_tokens``: ``on_moe_start`` waits for the layer's gathers only in forwards
    of at least this many tokens (``parse_moe_wait``); 0 always, None never.
    ``group_barrier``: a barrier over the shard group (``finish_load``).
    """

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
        mover: str = "ce",
        prefetch_depth: int = 1,
        issue_at: str = "layer",
        moe_wait_min_tokens: Optional[int] = 0,
        group_barrier: Optional[Callable[[], None]] = None,
    ):
        if moe_wait_min_tokens is not None and moe_wait_min_tokens < 0:
            raise ValueError(
                f"moe_wait_min_tokens must be >= 0 or None, got {moe_wait_min_tokens}"
            )
        if issue_at not in ("layer", "core"):
            raise ValueError(f"issue_at must be 'layer' or 'core', got {issue_at!r}")
        if prefetch_depth < 1:
            raise ValueError(f"prefetch depth must be >= 1, got {prefetch_depth}")
        if group_size < 2:
            raise ValueError(f"weight shard group size must be >= 2, got {group_size}")
        if not 0 <= group_rank < group_size:
            raise ValueError(
                f"group rank {group_rank} out of range for size {group_size}"
            )
        if mover not in MOVERS:
            raise ValueError(f"unknown weight gather mover {mover!r} (one of {MOVERS})")
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
        self.mover = mover
        self.shard = alloc_shard(off)
        if self.shard.numel() != off or self.shard.dtype != dtype:
            raise ValueError("alloc_shard returned a buffer of the wrong size or dtype")
        self.peers = peer_shards(self.shard)
        if len(self.peers) != group_size:
            raise ValueError(
                f"peer_shards returned {len(self.peers)} buffers, want {group_size}"
            )
        self.prefetch_depth = prefetch_depth
        self.issue_at = issue_at
        self.moe_wait_min_tokens = moe_wait_min_tokens
        self.num_slots = prefetch_depth + 1
        self.slots = torch.empty(
            (self.num_slots, max(self.e_size)), dtype=dtype, device=device
        )
        self._group_barrier = group_barrier
        # --load-format dummy: finish_load fills the shard with random values.
        self.dummy_load = False
        # Per unit: piece index -> the disjoint row ranges [x, y) of the piece loaded so far.
        self._loaded: List[Dict[int, List[Tuple[int, int]]]] = [
            {} for _ in range(self.num_units)
        ]

        cuda = device.type == "cuda"
        self._copy_stream = torch.cuda.Stream(device=device) if cuda else None
        # ce_ms: one extra copy stream per peer, so the copy engines run the pulls concurrently.
        self._peer_streams = (
            [torch.cuda.Stream(device=device) for _ in range(group_size - 1)]
            if cuda and mover == "ce_ms"
            else []
        )
        # unit -> ready event (None on CPU) for gathers issued and not yet waited on.
        self._pending: Dict[int, Optional[torch.cuda.Event]] = {}
        # slot -> unit whose weights it holds or will hold and whose layer has not finished
        # (pending, or waited on and running); None when the slot may be overwritten.
        self._slot_owner: List[Optional[int]] = [None] * self.num_slots
        self._running: Optional[int] = None
        # Units whose gather this layer's start issued (the MoE stage waits for them).
        self._issued_this_layer: List[int] = []
        # Benchmarks: list of (unit, start ev, copies-done ev, ready ev) per gather.
        self.trace: Optional[list] = None
        # Stall accounting (stats=True): the MoE-stage wait and the layer-start wait.
        self.stats = False
        self._stall_evs: List[tuple] = []  # (kind, layer, start ev, end ev)
        self.moe_wait_ms_total = 0.0
        self.moe_wait_max_ms = 0.0
        self.moe_wait_count = 0
        self.layer_wait_ms_total = 0.0
        self.layer_wait_max_ms = 0.0
        self.layer_wait_count = 0
        self._last_hooked_layer = num_layers
        # Lean issue (mover "ce" on CUDA): per-unit (dst, src) views and ready events built
        # once, so a gather is one event wait, its peer copies and one event record.
        self.lean_issue = (
            cuda and mover == "ce" and envs.SGLANG_K3_KDA_WS_LEAN_ISSUE.get()
        )
        self._pull_ops: Optional[List[list]] = None
        self._ready_events: Optional[List[torch.cuda.Event]] = None
        self._join_event = torch.cuda.Event() if cuda else None
        self._copy_ctx = torch.cuda.stream(self._copy_stream) if cuda else None
        self._device_index = device.index if cuda else None

    # ---- layout ---------------------------------------------------------------------------

    def slot_of(self, u: int) -> int:
        return u % self.num_slots

    def unit_buffer(self, u: int) -> torch.Tensor:
        """Unit ``u``'s full flat weights as the layer reads them."""
        return self.slots[self.slot_of(u), : self.e_size[u]]

    def param_view(self, u: int, name: str) -> torch.Tensor:
        """Parameter ``name`` of unit ``u`` (full, replicated layout): a view of its slot."""
        sp = self.specs[u][name]
        return self.unit_buffer(u)[sp.offset : sp.offset + sp.numel].view(sp.shape)

    def own_view(self, u: int, name: str) -> torch.Tensor:
        """This rank's part of parameter ``name`` of unit ``u``, a view of its shard:
        exactly what a tensor-parallel rank ``group_rank`` of ``group_size`` holds
        (``[blocks * rows / N, cols]`` for a row split, ``[rows, cols / N]`` for a
        column split)."""
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

    # ---- loading ----------------------------------------------------------------------------

    def load(
        self, u: int, name: str, loaded: torch.Tensor, row_offset: int = 0
    ) -> None:
        """Store this rank's part of rows ``[row_offset, row_offset + loaded.shape[0])`` of
        parameter ``name`` of unit ``u`` (full, unsharded rows)."""
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
                lo = (
                    pc.block * self.group_size + r
                ) * pc.rows  # first row of the piece
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
        """Record rows [x, y) of piece ``i`` of unit ``u``; raises before the write if any
        of them was already loaded."""
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
        """Every piece of every unit was loaded exactly once (``_mark_loaded`` keeps the
        loaded ranges disjoint, so full row counts mean full coverage)."""
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
        """--load-format dummy: fill this rank's shard with random values, as
        ``initialize_dummy_weights`` does for parameters, and mark every slice loaded."""
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
        """Check the load, then make sure every group rank's shard has landed on its
        device before any rank's first gather pulls it."""
        if self.dummy_load and not any(self._loaded):
            self.init_dummy()
        self.check_loaded()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self._group_barrier is not None:
            self._group_barrier()

    # ---- gather + schedule ------------------------------------------------------------------

    def _copy_ops(self, u: int) -> list:
        """Unit ``u``'s gather as copy ops into the replicated layout of its slot, in
        rotated peer order (at step i every rank pulls from rank + i, so no source serves
        two readers at once), this rank's own pieces last. A row-split piece is one
        contiguous copy; a column-split piece lands with a row pitch, as one 2-D copy
        (``_memcpy2d``: copy engine, no SMs)."""
        slot = self.unit_buffer(u)
        off, n, r = self.shard_offset[u], self.group_size, self.group_rank
        es = self.shard.element_size()
        ops = []
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

    def _move(self, u: int) -> None:
        """Enqueue the gather of unit ``u`` (the current stream is the copy stream)."""
        ops = self._copy_ops(u)
        if self.mover == "ce" or self._copy_stream is None:
            cur = torch.cuda.current_stream(self.device) if self._copy_stream else None
            for op in ops:
                self._run_op(op, cur)
            return
        cur = torch.cuda.current_stream(self.device)
        streams = [cur] + self._peer_streams
        per_peer = len(self.pieces[u])
        for i, op in enumerate(ops):
            st = streams[(i // per_peer) % len(streams)]
            if st is not cur:
                st.wait_stream(cur)
            with torch.cuda.stream(st):
                self._run_op(op, st)
        for st in self._peer_streams:
            cur.wait_stream(st)

    def issue(self, k: int) -> None:
        """Start gathering unit ``k`` into its slot (no-op if already pending)."""
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
            self._move(k)
            self._pending[k] = None
            return
        if self.lean_issue and self.trace is None:
            self._pending[k] = self._issue_lean(k)
            return
        cur = torch.cuda.current_stream(self.device)
        # Everything enqueued so far, including the last reader of this slot, precedes the copies.
        self._copy_stream.wait_stream(cur)
        with torch.cuda.stream(self._copy_stream):
            tr = None
            if self.trace is not None:
                tr = [k] + [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                tr[1].record(self._copy_stream)
            self._move(k)
            ev = torch.cuda.Event()
            ev.record(self._copy_stream)
            if tr is not None:
                tr[2].record(self._copy_stream)
                tr[3].record(self._copy_stream)
                self.trace.append(tuple(tr))
        self._pending[k] = ev

    def _issue_lean(self, k: int) -> torch.cuda.Event:
        """``issue``'s copy-stream part with everything per unit precomputed: the copy
        stream joins the compute stream through one reused event, the peer copies run on
        cached views, and the unit's own reused ready event marks them done. Reusing the
        events is safe: a stream wait binds the event's latest record at enqueue time, and
        a unit is reissued only after its previous gather was waited on."""
        if self._pull_ops is None:
            self._pull_ops = [self._copy_ops(u) for u in range(self.num_units)]
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
        """Make the current stream wait until unit ``k``'s slot holds its weights; ``k``
        becomes the running unit (the previous one's kernels are enqueued, so its slot
        is released)."""
        if self._running != k:
            self._release_running()
        if k not in self._pending:
            self.issue(k)
        ev = self._pending.pop(k)
        if ev is not None:
            torch.cuda.current_stream(self.device).wait_event(ev)
        self._running = k

    def on_layer_start(self, layer_idx: int) -> None:
        """Call at the start of decoder layer ``layer_idx``, before its attention.

        The previous sharded layer's kernels are all enqueued before this point, so its
        slot is free for a gather issued from here on. The running unit waits for its
        gather; then the next ``prefetch_depth`` units are issued, in order, into the
        following forward once this one has none left, stopping at the first unit whose
        slot is still held."""
        self._release_running()
        if self.stats and layer_idx <= self._last_hooked_layer:
            self._log_forward_stalls()  # first hooked layer of a new forward
        self._last_hooked_layer = layer_idx
        self._issued_this_layer = []
        k = self.unit_index.get(layer_idx)
        if k is not None:
            if self.stats:
                e0 = self._stall_start()
                self.wait(k)
                self._stall_end("layer", layer_idx, e0)
            else:
                self.wait(k)
            if self.issue_at == "layer":
                self._issue_ahead(k + 1)
        else:
            self._issue_ahead(self.next_unit[layer_idx])

    def on_core_start(self, layer_idx: int) -> None:
        """issue_at="core": call where a sharded layer's attention core starts (after its
        in-projection GEMM, the HBM-heaviest kernel of the layer); the next units'
        gathers are issued here instead of at the layer start."""
        k = self.unit_index.get(layer_idx)
        if self.issue_at != "core" or k is None:
            return
        self._issue_ahead(k + 1)

    def _issue_ahead(self, first: int) -> None:
        """Issue, in order, the gathers of the ``prefetch_depth`` units from ``first``
        (into the next forward once this one has none left), stopping at the first unit
        whose slot is still held."""
        for j in range(first, first + self.prefetch_depth):
            if j >= self.num_units and (j - self.num_units) >= first:
                break  # wrapped past this forward's own window
            u = j % self.num_units
            if u in self._pending:
                continue
            if self._slot_owner[self.slot_of(u)] is not None:
                break
            self.issue(u)
            self._issued_this_layer.append(u)

    def needs_layer_hook(self, layer_idx: int) -> bool:
        """Whether the hooks can act at this layer: sharded layers wait and issue; other
        layers can only issue the wrap-around prefetch after the last sharded layer (in
        between, the next sharded layer's gather is already in flight). Under a breakable
        prefill CUDA graph each hooked call is an eager break, so the other layers stay
        inside the captured segments."""
        if self.issue_at == "core":
            return True  # non-sharded layers' starts issue the next unit's gather
        return layer_idx in self.unit_index or layer_idx > self.unit_layers[-1]

    def moe_wait_needs_tokens(self) -> bool:
        """Whether ``on_moe_start`` needs the forward's token count to decide."""
        return self.moe_wait_min_tokens not in (0, None)

    def moe_wait_units(self, num_tokens: Optional[int] = None) -> List[int]:
        """Units the MoE stage of the current layer waits for (and forgets them): the
        gathers this layer issued, unless SGLANG_K3_KDA_WS_MOE_WAIT exempts this forward
        (those stay pending and are waited on at their own layer's start)."""
        issued, self._issued_this_layer = self._issued_this_layer, []
        if not issued or self.moe_wait_min_tokens is None:
            return []
        if self.moe_wait_min_tokens:
            if num_tokens is None:
                raise ValueError(
                    "SGLANG_K3_KDA_WS_MOE_WAIT is a token threshold, but on_moe_start got no "
                    "token count"
                )
            if num_tokens < self.moe_wait_min_tokens:
                return []
        return [u for u in issued if u in self._pending]

    def on_moe_start(self, layer_idx: int, num_tokens: Optional[int] = None) -> None:
        """Call where layer ``layer_idx``'s MoE stage starts, before the DEP dispatch: the
        compute stream waits for the gathers this layer's start issued, so gather traffic
        never overlaps the MoE all-to-all (an overrun becomes a stall here).
        ``num_tokens``: the forward's largest per-DP-rank token count (needed only when
        ``moe_wait_needs_tokens``)."""
        units = self.moe_wait_units(num_tokens)
        if self._copy_stream is None or not units:
            return
        cur = torch.cuda.current_stream(self.device)
        e0 = self._stall_start()
        for u in units:
            ev = self._pending[u]
            if ev is not None:
                cur.wait_event(ev)
        self._stall_end("moe", layer_idx, e0)

    def _stall_start(self) -> Optional[torch.cuda.Event]:
        if not self.stats or self._copy_stream is None:
            return None
        self._fold_stalls()
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record(torch.cuda.current_stream(self.device))
        return e0

    def _stall_end(
        self, kind: str, layer_idx: int, e0: Optional[torch.cuda.Event]
    ) -> None:
        if e0 is None:
            return
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record(torch.cuda.current_stream(self.device))
        self._stall_evs.append((kind, layer_idx, e0, e1))

    def _log_forward_stalls(self) -> None:
        """Log the MoE-stage and layer-start wait stalls of the forwards finished since
        the last call (SGLANG_K3_KDA_WS_STATS)."""
        before = (
            self.moe_wait_ms_total,
            self.moe_wait_count,
            self.layer_wait_ms_total,
            self.layer_wait_count,
        )
        self._fold_stalls()
        n_moe = self.moe_wait_count - before[1]
        n_layer = self.layer_wait_count - before[3]
        if n_moe or n_layer:
            logger.info(
                "K3 weight shard (%d units): MoE-wait stall %.3f ms over %d layers, "
                "layer-start wait stall %.3f ms over %d layers; aggregate MoE %.1f ms / %d "
                "(max %.3f), layer-start %.1f ms / %d (max %.3f)",
                self.num_units,
                self.moe_wait_ms_total - before[0],
                n_moe,
                self.layer_wait_ms_total - before[2],
                n_layer,
                self.moe_wait_ms_total,
                self.moe_wait_count,
                self.moe_wait_max_ms,
                self.layer_wait_ms_total,
                self.layer_wait_count,
                self.layer_wait_max_ms,
            )

    def _fold_stalls(self) -> None:
        """Fold finished stall measurements into the aggregate counters."""
        keep = []
        for kind, layer, e0, e1 in self._stall_evs:
            if not e1.query():
                keep.append((kind, layer, e0, e1))
                continue
            ms = e0.elapsed_time(e1)
            if kind == "moe":
                self.moe_wait_ms_total += ms
                self.moe_wait_max_ms = max(self.moe_wait_max_ms, ms)
                self.moe_wait_count += 1
            elif kind == "layer":
                self.layer_wait_ms_total += ms
                self.layer_wait_max_ms = max(self.layer_wait_max_ms, ms)
                self.layer_wait_count += 1
            else:
                raise AssertionError(f"unknown stall kind {kind!r}")
        self._stall_evs = keep

    @property
    def pending(self) -> List[int]:
        return sorted(self._pending)

    def bytes_per_rank(self) -> Tuple[int, int]:
        """(shard bytes, slot bytes) held on this rank."""
        shard_b = self.shard.numel() * self.shard.element_size()
        return shard_b, self.slots.numel() * self.slots.element_size()


# ---- CUDA driver: VMM shard allocations shared with the group, 2-D copies ------------------


class _CuMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CuMemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CuMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CuMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CuMemAllocFlags),
    ]


class _CuMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CuMemLocation), ("flags", ctypes.c_int)]


_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_CU_MEM_HANDLE_TYPE_POSIX_FD = 1
_CU_MEM_HANDLE_TYPE_FABRIC = 8
# None, POSIX fd, fabric: the handle types a shard can be allocated with.
_CU_MEM_HANDLE_TYPES = (0, _CU_MEM_HANDLE_TYPE_POSIX_FD, _CU_MEM_HANDLE_TYPE_FABRIC)


class _CudaArray:
    """``__cuda_array_interface__`` over a raw device range, so ``torch.as_tensor`` wraps it
    without a copy; holds the owning ``_Vmm`` so the mappings outlive every view."""

    def __init__(self, ptr: int, nbytes: int, owner: _Vmm):
        self.owner = owner
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "strides": None,
            "version": 3,
        }


class _Vmm:
    """CUDA driver VMM (libcuda) on one device: physical allocations, VA reservations mapped
    piecewise onto them, torch views of the VAs. Everything is released once the store and
    every view are gone."""

    def __init__(self, device: torch.device):
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise RuntimeError(
                f"K3 KDA weight shard VMM: cannot load libcuda.so.1 ({e})"
            ) from e
        self.device = device
        self.dev_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        torch.empty(
            0, device=device
        )  # the primary context exists before any driver call
        self._handles: List[int] = []  # created or retained, released at teardown
        self._maps: List[Tuple[int, int]] = []  # (va, bytes)
        self._reserved: List[Tuple[int, int]] = []
        self._finalizer = weakref.finalize(
            self,
            _Vmm._release,
            self.lib,
            self.dev_index,
            self._maps,
            self._reserved,
            self._handles,
        )

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            msg = ctypes.c_char_p()
            self.lib.cuGetErrorString(rc, ctypes.byref(msg))
            raise RuntimeError(
                f"K3 KDA weight shard VMM: {what} failed: CUresult {rc} ({msg.value})"
            )

    def _prop(self, handle_type: int) -> _CuMemAllocationProp:
        return _CuMemAllocationProp(
            type=_CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes=handle_type,
            location=_CuMemLocation(_CU_MEM_LOCATION_TYPE_DEVICE, self.dev_index),
        )

    def _granularity_of(self, prop: _CuMemAllocationProp) -> int:
        g = ctypes.c_size_t()
        rc = self.lib.cuMemGetAllocationGranularity(
            ctypes.byref(g), ctypes.byref(prop), _CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        return g.value if rc == 0 else 0

    def granularity(self) -> int:
        """Mapping granularity that serves every handle type the shard could be allocated with."""
        gs = [self._granularity_of(self._prop(ht)) for ht in _CU_MEM_HANDLE_TYPES]
        gs = [g for g in gs if g]
        if not gs:
            raise RuntimeError(
                "K3 KDA weight shard VMM: no allocation granularity on this device"
            )
        return max(gs)

    def create(self, nbytes: int, handle_type: int = 0) -> int:
        h = ctypes.c_ulonglong()
        prop = self._prop(handle_type)
        self._check(
            self.lib.cuMemCreate(
                ctypes.byref(h),
                ctypes.c_size_t(nbytes),
                ctypes.byref(prop),
                ctypes.c_ulonglong(0),
            ),
            "cuMemCreate",
        )
        self._handles.append(h.value)
        return h.value

    def export(self, handle: int, handle_type: int):
        """Shareable handle: an fd (POSIX) or the 64-byte fabric handle."""
        if handle_type == _CU_MEM_HANDLE_TYPE_POSIX_FD:
            fd = ctypes.c_int()
            self._check(
                self.lib.cuMemExportToShareableHandle(
                    ctypes.byref(fd),
                    ctypes.c_ulonglong(handle),
                    handle_type,
                    ctypes.c_ulonglong(0),
                ),
                "cuMemExportToShareableHandle",
            )
            return fd.value
        fh = (ctypes.c_ubyte * 64)()
        self._check(
            self.lib.cuMemExportToShareableHandle(
                fh, ctypes.c_ulonglong(handle), handle_type, ctypes.c_ulonglong(0)
            ),
            "cuMemExportToShareableHandle",
        )
        return bytes(fh)

    def import_(self, shareable, handle_type: int) -> int:
        h = ctypes.c_ulonglong()
        if handle_type == _CU_MEM_HANDLE_TYPE_POSIX_FD:
            arg = ctypes.c_void_p(shareable)
        else:
            arg = (ctypes.c_ubyte * 64).from_buffer_copy(shareable)
        self._check(
            self.lib.cuMemImportFromShareableHandle(ctypes.byref(h), arg, handle_type),
            "cuMemImportFromShareableHandle",
        )
        self._handles.append(h.value)
        return h.value

    def map_flat(
        self, handles: Sequence[int], nbytes: Sequence[int], dtype: torch.dtype
    ) -> torch.Tensor:
        """Whole allocations mapped back to back into one new VA, as one flat tensor."""
        total = sum(nbytes)
        va = self.reserve(total, self.granularity())
        off = 0
        for h, nb in zip(handles, nbytes):
            self.map(va + off, nb, h)
            off += nb
        self.set_access(va, total)
        return self.tensor(va, total, dtype)

    def reserve(self, nbytes: int, align: int) -> int:
        va = ctypes.c_ulonglong()
        self._check(
            self.lib.cuMemAddressReserve(
                ctypes.byref(va),
                ctypes.c_size_t(nbytes),
                ctypes.c_size_t(align),
                ctypes.c_ulonglong(0),
                ctypes.c_ulonglong(0),
            ),
            "cuMemAddressReserve",
        )
        self._reserved.append((va.value, nbytes))
        return va.value

    def map(self, va: int, nbytes: int, handle: int) -> None:
        """Map the whole allocation ``handle`` (``nbytes``) at ``va``."""
        self._check(
            self.lib.cuMemMap(
                ctypes.c_ulonglong(va),
                ctypes.c_size_t(nbytes),
                ctypes.c_size_t(0),
                ctypes.c_ulonglong(handle),
                ctypes.c_ulonglong(0),
            ),
            "cuMemMap",
        )
        self._maps.append((va, nbytes))

    def set_access(self, va: int, nbytes: int) -> None:
        desc = _CuMemAccessDesc(
            _CuMemLocation(_CU_MEM_LOCATION_TYPE_DEVICE, self.dev_index),
            _CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        )
        self._check(
            self.lib.cuMemSetAccess(
                ctypes.c_ulonglong(va),
                ctypes.c_size_t(nbytes),
                ctypes.byref(desc),
                ctypes.c_size_t(1),
            ),
            "cuMemSetAccess",
        )

    def tensor(self, va: int, nbytes: int, dtype: torch.dtype) -> torch.Tensor:
        t = torch.as_tensor(_CudaArray(va, nbytes, self), device=self.device)
        return t.view(dtype)

    @staticmethod
    def _release(lib, dev_index, maps, reserved, handles) -> None:
        try:
            torch.cuda.synchronize(dev_index)
        except (
            Exception
        ):  # interpreter shutdown: the driver tears the context down anyway
            return
        for va, nbytes in maps:
            lib.cuMemUnmap(ctypes.c_ulonglong(va), ctypes.c_size_t(nbytes))
        for va, nbytes in reserved:
            lib.cuMemAddressFree(ctypes.c_ulonglong(va), ctypes.c_size_t(nbytes))
        for h in handles:
            lib.cuMemRelease(ctypes.c_ulonglong(h))
        maps.clear(), reserved.clear(), handles.clear()


class _CudaMemcpy2D(ctypes.Structure):
    _fields_ = [
        ("srcXInBytes", ctypes.c_size_t),
        ("srcY", ctypes.c_size_t),
        ("srcMemoryType", ctypes.c_int),
        ("srcHost", ctypes.c_void_p),
        ("srcDevice", ctypes.c_ulonglong),
        ("srcArray", ctypes.c_void_p),
        ("srcPitch", ctypes.c_size_t),
        ("dstXInBytes", ctypes.c_size_t),
        ("dstY", ctypes.c_size_t),
        ("dstMemoryType", ctypes.c_int),
        ("dstHost", ctypes.c_void_p),
        ("dstDevice", ctypes.c_ulonglong),
        ("dstArray", ctypes.c_void_p),
        ("dstPitch", ctypes.c_size_t),
        ("WidthInBytes", ctypes.c_size_t),
        ("Height", ctypes.c_size_t),
    ]


_CU_MEMORYTYPE_DEVICE = 2
_LIBCUDA = None


def _libcuda():
    global _LIBCUDA
    if _LIBCUDA is None:
        try:
            _LIBCUDA = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise RuntimeError(
                f"K3 KDA weight shard: cannot load libcuda.so.1 ({e})"
            ) from e
    return _LIBCUDA


def _memcpy2d_args(
    *, dst: int, dst_pitch: int, src: int, src_pitch: int, width: int, height: int
) -> _CudaMemcpy2D:
    """A device-to-device 2-D copy (``height`` rows of ``width`` bytes) for ``_memcpy2d``."""
    return _CudaMemcpy2D(
        srcMemoryType=_CU_MEMORYTYPE_DEVICE,
        srcDevice=src,
        srcPitch=src_pitch,
        dstMemoryType=_CU_MEMORYTYPE_DEVICE,
        dstDevice=dst,
        dstPitch=dst_pitch,
        WidthInBytes=width,
        Height=height,
    )


def _memcpy2d(args: _CudaMemcpy2D, stream: torch.cuda.Stream) -> None:
    """cuMemcpy2DAsync on ``stream``: a strided device-to-device copy that the copy
    engines execute (a column slice of o_proj lands with the full row pitch), where a
    torch copy into the strided view would launch a kernel."""
    rc = _libcuda().cuMemcpy2DAsync_v2(
        ctypes.byref(args), ctypes.c_void_p(stream.cuda_stream)
    )
    if rc != 0:
        raise RuntimeError(
            f"K3 KDA weight shard: cuMemcpy2DAsync failed: CUresult {rc}"
        )


# ---- model wiring ---------------------------------------------------------------------------

_ACTIVE: Optional[LayerWeightShard] = None
# Shared-expert store (replicated shared experts under EP a2a): one unit per MoE layer.
_ACTIVE_SHARED: Optional[LayerWeightShard] = None
_BUILDING = False


@contextlib.contextmanager
def building_target_model():
    """Scope in which constructed KDA layers belong to the sharded target model (the
    draft / NextN model is built outside it and keeps replicated weights)."""
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
    """Whether any store's layer/MoE hooks act at decoder layer ``layer_idx``."""
    return any(s.needs_layer_hook(layer_idx) for s in _stores())


def on_layer_start(layer_idx: int) -> None:
    for s in _stores():
        if s.needs_layer_hook(layer_idx):
            s.on_layer_start(layer_idx)


def moe_wait_needs_tokens() -> bool:
    """Whether ``on_moe_start`` needs the forward's token count (SGLANG_K3_KDA_WS_MOE_WAIT
    is a threshold), so callers skip computing it otherwise."""
    return any(s.moe_wait_needs_tokens() for s in _stores())


def on_moe_start(layer_idx: int, num_tokens: Optional[int] = None) -> None:
    for s in _stores():
        if s.needs_layer_hook(layer_idx):
            s.on_moe_start(layer_idx, num_tokens)


def check_all_loaded() -> None:
    """Every store received exactly its checkpoint slices."""
    for s in _stores():
        s.check_loaded()


def finish_load() -> None:
    """Call from the model's post_load_weights, which every loader runs (load_weights or
    not): checks each store's slices (random under --load-format dummy), then syncs the
    shard group so no peer pulls a shard before its host-to-device copies land."""
    for s in _stores():
        s.finish_load()


def _exchange_fds(fds: List[int], group, timeout_s: float = 120.0) -> List[List[int]]:
    """Every group rank's fds (own entry: ``fds``), passed over abstract unix sockets with
    SCM_RIGHTS: each rank serves its fds to every peer that connects."""
    import socket
    import threading
    import uuid

    dist = torch.distributed
    rank, n = dist.get_rank(group), dist.get_world_size(group)
    tokens = [None] * n
    dist.all_gather_object(tokens, uuid.uuid4().hex, group=group)
    name = lambda r: f"\0k3-kda-ws-{tokens[0]}-{r}"  # noqa: E731
    chunk = 200  # under SCM_MAX_FD (253) per message
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    srv.settimeout(timeout_s)
    srv.bind(name(rank))
    srv.listen(n)
    errors: List[BaseException] = []

    def serve():
        try:
            for _ in range(n - 1):
                conn, _ = srv.accept()
                with conn:
                    for i in range(0, len(fds), chunk):
                        socket.send_fds(conn, [b"f"], fds[i : i + chunk])
        except BaseException as e:  # re-raised on the caller's thread
            errors.append(e)

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    dist.barrier(group=group)  # every listener is bound
    out: List[List[int]] = []
    try:
        for p in range(n):
            if p == rank:
                out.append(list(fds))
                continue
            got: List[int] = []
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as c:
                c.settimeout(timeout_s)
                c.connect(name(p))
                while len(got) < len(fds):
                    _, rfds, _, _ = socket.recv_fds(c, 16, chunk)
                    if not rfds:
                        raise RuntimeError(
                            f"K3 KDA weight shard: rank {p} closed the fd channel early"
                        )
                    got += rfds
            out.append(got)
        th.join(timeout_s)
        if th.is_alive():
            raise RuntimeError(
                "K3 KDA weight shard: fd exchange timed out serving peers"
            )
        if errors:
            raise errors[0]
    finally:
        srv.close()
    dist.barrier(group=group)
    return out


def _vmm_alloc_and_peers(group, dtype: torch.dtype, device: torch.device):
    """The shard as one cuMem allocation, exported to the group (POSIX fds within a host,
    fabric handles across hosts) and mapped on every peer, so the gathers are peer pulls
    by the copy engines."""
    import socket
    import time

    dist = torch.distributed
    vmm = _Vmm(device)
    es = torch.empty((), dtype=dtype).element_size()
    rank, n = dist.get_rank(group), dist.get_world_size(group)
    hosts = [None] * n
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    handle_type = (
        _CU_MEM_HANDLE_TYPE_POSIX_FD
        if len(set(hosts)) == 1
        else _CU_MEM_HANDLE_TYPE_FABRIC
    )
    holder = {}

    def alloc(numel: int) -> torch.Tensor:
        g = vmm.granularity()
        nbytes = [-(-numel * es // g) * g]
        handles = [vmm.create(b, handle_type) for b in nbytes]
        holder.update(handles=handles, nbytes=nbytes)
        return vmm.map_flat(handles, nbytes, dtype)[:numel]

    def peers(t: torch.Tensor) -> List[torch.Tensor]:
        t0 = time.perf_counter()
        logger.info(
            "K3 KDA weight shard: exporting %d shard allocations to %d peers",
            len(holder["handles"]),
            n - 1,
        )
        shareable = [vmm.export(h, handle_type) for h in holder["handles"]]
        if handle_type == _CU_MEM_HANDLE_TYPE_POSIX_FD:
            per_rank = _exchange_fds(shareable, group)
        else:
            per_rank = [None] * n
            dist.all_gather_object(per_rank, shareable, group=group)
        out = []
        for p in range(n):
            if p == rank:
                out.append(t)
                continue
            imported = [vmm.import_(x, handle_type) for x in per_rank[p]]
            out.append(vmm.map_flat(imported, holder["nbytes"], dtype))
        if handle_type == _CU_MEM_HANDLE_TYPE_POSIX_FD:
            for p, fds in enumerate(per_rank):
                for fd in fds:
                    os.close(fd)
        dist.barrier(group=group)  # every peer mapped before anyone pulls
        logger.info(
            "K3 KDA weight shard: peer shard mappings done in %.2f s",
            time.perf_counter() - t0,
        )
        return out

    return alloc, peers


def shard_group_layout(
    ranks: Sequence[int], attn_tp: int, group_size: int, tp_rank: int
) -> tuple[list[list[int]], int, int]:
    """(every shard group's member ranks, this rank's group index, its rank in the group).

    A shard group holds ``group_size`` attention-DP ranks with the same attention-TP slice
    (tp rank = dp rank * attn_tp + attention-TP rank); each keeps 1/group_size of that
    slice. Under attention TP 1 these are runs of consecutive TP ranks."""
    dp_size = len(ranks) // attn_tp
    if len(ranks) % attn_tp != 0 or dp_size % group_size != 0:
        raise ValueError(
            f"--kda-weight-shard-size {group_size} must divide the attention-DP size "
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
    mover: str,
    dtype: torch.dtype,
    shared_expert_layers: Sequence[int] = (),
    shared_intermediate_size: int = 0,
    dummy_load: bool = False,
) -> LayerWeightShard:
    """Build the process-wide shard stores for the K3 target model (call before its layers):
    the KDA projections and, for ``shared_expert_layers`` (the MoE layers whose shared
    experts would be replicated on every rank), the shared experts' gate_up and down."""
    global _ACTIVE, _ACTIVE_SHARED
    if _ACTIVE is not None or _ACTIVE_SHARED is not None:
        raise RuntimeError(
            "--kda-weight-shard-size: the weight shard stores were already built for a "
            "Kimi-K3 model in this process; a second K3-architecture model (e.g. a draft) "
            "cannot be weight-sharded too. Use a draft of another architecture, or drop "
            "--kda-weight-shard-size."
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
    if mover not in MOVERS:
        raise ValueError(
            f"SGLANG_K3_KDA_WS_MOVER={mover!r} is not supported (one of {MOVERS}): the "
            "head-aligned shards are gathered by copy-engine peer pulls"
        )
    cpu_groups = []
    # Every rank creates every subgroup, in the same order (torch.distributed contract).
    for members in group_members:
        cpu_groups.append(torch.distributed.new_group(members, backend="gloo"))
    device = torch.device("cuda", torch.cuda.current_device())
    num_layers = config.num_hidden_layers
    kda_ids = [l for l in config.linear_layer_ids if l < num_layers]
    H = config.hidden_size
    # This rank's attention-TP slice of the KDA projections.
    P = (
        config.linear_attn_config["num_heads"]
        * config.linear_attn_config["head_dim"]
        // attn_tp
    )
    moe_wait_min_tokens = parse_moe_wait(envs.SGLANG_K3_KDA_WS_MOE_WAIT.get())
    group_barrier = functools.partial(
        torch.distributed.barrier, group=cpu_groups[my_group]
    )
    if shared_expert_layers:
        s_alloc, s_peers = _vmm_alloc_and_peers(cpu_groups[my_group], dtype, device)
        I = shared_intermediate_size
        _ACTIVE_SHARED = LayerWeightShard(
            units=[
                (layer, shared_expert_unit_shapes(I, H))
                for layer in shared_expert_layers
            ],
            num_layers=num_layers,
            group_rank=group_rank,
            group_size=group_size,
            dtype=dtype,
            device=device,
            alloc_shard=s_alloc,
            peer_shards=s_peers,
            mover=mover,
            prefetch_depth=envs.SGLANG_K3_KDA_WS_PREFETCH_DEPTH.get(),
            issue_at="layer",
            moe_wait_min_tokens=moe_wait_min_tokens,
            group_barrier=group_barrier,
        )
        _ACTIVE_SHARED.dummy_load = dummy_load
        _ACTIVE_SHARED.stats = envs.SGLANG_K3_KDA_WS_STATS.get()
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
    alloc, peers = _vmm_alloc_and_peers(cpu_groups[my_group], dtype, device)
    _ACTIVE = LayerWeightShard(
        units=[(layer, kda_unit_shapes(P, H)) for layer in kda_ids],
        num_layers=num_layers,
        group_rank=group_rank,
        group_size=group_size,
        dtype=dtype,
        device=device,
        alloc_shard=alloc,
        peer_shards=peers,
        mover=mover,
        prefetch_depth=envs.SGLANG_K3_KDA_WS_PREFETCH_DEPTH.get(),
        issue_at=envs.SGLANG_K3_KDA_WS_ISSUE_AT.get(),
        moe_wait_min_tokens=moe_wait_min_tokens,
        group_barrier=group_barrier,
    )
    _ACTIVE.dummy_load = dummy_load
    _ACTIVE.stats = envs.SGLANG_K3_KDA_WS_STATS.get()
    shard_b, slot_b = _ACTIVE.bytes_per_rank()
    replicated = len(kda_ids) * (4 * P * H + H * P) * 2
    logger.info(
        "K3 KDA weight sharding: shard group %d (%d ranks, this is rank %d), mover=%s, "
        "prefetch depth %d, MoE-stage wait %s, %d KDA layers: head-aligned shard %.2f GB + %d slots "
        "%.2f GB per rank instead of %.2f GB replicated",
        my_group,
        group_size,
        group_rank,
        mover,
        _ACTIVE.prefetch_depth,
        envs.SGLANG_K3_KDA_WS_MOE_WAIT.get(),
        len(kda_ids),
        shard_b / 1e9,
        _ACTIVE.num_slots,
        slot_b / 1e9,
        replicated / 1e9,
    )
    return _ACTIVE
