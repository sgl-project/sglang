# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942/gfx950 TP∈{2,4,8} INT4/INT6 all-reduce.

Public type ``QuickAllReduceInt4``, with two interchangeable schedules selected
by ``algorithm``. Both are two-shot -- reduce-scatter then all-gather -- so
they are named for the topology of each lap instead:

* ``"mesh"`` the default: fanout to all N-1 peers, twice.
* ``"ring"`` 2(N-1) single-destination hops.

Super-tile ST∈{1,8} on the mesh, ST∈{1,8,16,32} on the ring. INT4 nibble or
INT6 bit-plane pair, both with group-16 E5M2 scales. Payload HBM is bf16.

The kernel configuration comes from a tuning ladder of ``TwoShotRung``, keyed
on ``(arch, link, world_size)`` in ``allreduce_policy``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from flydsl.expr.typing import Float32, Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .allreduce_host_shared import (
    _cuda_index,
    _launch_artifacts,
    _resolve_inbox_flags,
    _StEngine,
    _validate_ipc_process_group,
    check_payload,
    check_tensors,
    check_world_size,
    kernel_symbol,
    launcher_loaded,
    payload_probes,
    preload_launcher,
    require_arch,
)
from .allreduce_policy import (
    MIN_BATCH_BLOCKS,
    RS_INT6_MIN_WORLD,
    mesh_norm_ladder,
    mesh_st_ladder,
    resolve_link,
    ring_st_ladder,
)
from .kernels.comm.allreduce_shared import (
    BLOCK,
    MAX_PAYLOAD_BYTES,
    QUANT_MIN_PAYLOAD_BYTES,
    SUPPORTED_BLOCKS,
    WORLD,
    TwoShotRung,
    has_release_fence,
    select_rung,
)
from .kernels.comm.quick_allreduce_int4 import (
    DEFAULT_ATOMS,
    DEFAULT_DIVISOR,
    MESH_CODECS,
    MODE_AG,
    MODE_RS,
    SUPER_TILES,
    clamp_grid_cap,
    latency_max_tiles,
    make_quick_allreduce_int4_kernel,
    max_payload_bytes,
    mesh_lds_bytes,
    mxfp4_scale_shape,
    tile_atoms_for,
)
from .kernels.comm.quick_allreduce_int4_ring import (
    AG_CODECS,
    RING_SUPER_TILES,
    RS_CODECS,
    make_quick_allreduce_ring_kernel,
)
from .kernels.tensor_shim import _run_compiled

logger = logging.getLogger("aiter")

# The sum is restored in fp16 before the bf16 store, by scaling each group's
# dequant step (group scale * 1/bias) by the divisor. With the E5M2 group
# scale every fp16 extremum fits at divisor 1, so the divisor is opt-in and the
# scale range no longer sets these bounds. The smallest step is 2**-15 (INT4
# and INT6 alike, see Codec.scale_min_e5); at DIVISOR_MIN it is restored as
# 2**-19, an fp16 subnormal that still holds the 2-bit mantissa exactly
# (subnormal spacing 2**-24). Inputs are divided by the divisor before the
# first pack, so restored values keep the input's magnitude at DIVISOR_MAX.
DIVISOR_MIN = 2.0**-4
DIVISOR_MAX = 128.0


def _check_divisor(divisor: float) -> float:
    """One shared positive divisor. ``1`` disables the scaling."""
    msg = (
        f"QuickAllReduceInt4 divisor must be a finite float in "
        f"[{DIVISOR_MIN:g}, {DIVISOR_MAX:g}], got {divisor!r}"
    )
    try:
        value = float(divisor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(msg) from exc
    if not math.isfinite(value) or not DIVISOR_MIN <= value <= DIVISOR_MAX:
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class _Algorithm:
    """One all-reduce schedule, plus the host-side policy that tunes it."""

    name: str
    build: Callable[..., dict]
    super_tiles: tuple[int, ...]
    rs_codecs: tuple[str, ...]
    ag_codecs: tuple[str, ...]
    # ``(*, world_size, link, arch) -> (TwoShotRung, ...)``: the tuning ladder.
    st_ladder: Callable[..., tuple[TwoShotRung, ...]]
    # ``(*, world_size, link, arch, mesh) -> (TwoShotRung, ...)``: the ladder
    # of the fused add+RMSNorm, *mesh* where it has no tuned row; ``None`` for
    # a schedule without one.
    norm_ladder: Callable[..., tuple[TwoShotRung, ...]] | None = None
    # Trailing ``Int64`` kernel arguments before the grid size: the fused
    # RMSNorm's residual, residual_out, weight, MXFP4 out and scale pointers,
    # then its ``Float32`` epsilon. Plain launches pass zeros. A schedule
    # without the fused RMSNorm has none of them.
    norm_ptr_args: int = 0
    # Whether the kernel takes the shared ``divisor`` as its last argument.
    has_divisor: bool = False


def _build_mesh(
    *,
    world_size,
    rank,
    super_tile,
    grid,
    inbox_memory,
    rs_codec,
    ag_codec,
    block,
    **norm,
):
    if rs_codec != ag_codec:
        raise ValueError(
            f"mesh algorithm has one wire format for both laps, got "
            f"rs_codec={rs_codec!r} != ag_codec={ag_codec!r}"
        )
    return make_quick_allreduce_int4_kernel(
        world_size=world_size,
        super_tile=super_tile,
        grid=grid,
        inbox_memory=inbox_memory,
        codec=rs_codec,
        block=block,
        rank=rank,
        **norm,
    )


NORM_PTR_ARGS = 5

ALGORITHMS = {
    "mesh": _Algorithm(
        name="mesh",
        build=_build_mesh,
        super_tiles=SUPER_TILES,
        rs_codecs=MESH_CODECS,
        ag_codecs=MESH_CODECS,
        st_ladder=mesh_st_ladder,
        norm_ladder=mesh_norm_ladder,
        norm_ptr_args=NORM_PTR_ARGS,
        has_divisor=True,
    ),
    "ring": _Algorithm(
        name="ring",
        build=make_quick_allreduce_ring_kernel,
        super_tiles=RING_SUPER_TILES,
        rs_codecs=RS_CODECS,
        ag_codecs=AG_CODECS,
        st_ladder=ring_st_ladder,
    ),
}
DEFAULT_ALGORITHM = "mesh"


def _resolve_codecs(algo, world_size, rs_codec, ag_codec):
    """Codecs for one engine: explicit argument > per-N default.

    ``None`` means "not specified", and the caller is expected to pass it
    whenever it wants the schedule's own default rather than a pinned wire
    format. That distinction matters from TP4, where the ring's reduce-scatter
    lap defaults to INT6: an explicit ``rs_codec="int4"`` there is a real
    downgrade, not a restatement of the default.

    A codec the selected schedule cannot build raises. The per-N *default*
    silently narrows to what the schedule supports -- the mesh has no INT6
    path, so the ring's INT6 reduce-scatter default does not leak into it.

    The mesh has one wire format for both laps, so there ``rs_codec`` and
    ``ag_codec`` follow each other: give either one and the other takes the
    same value. Giving two different values raises when the kernel is built.
    """
    if algo.name == "mesh" and (rs_codec is None) != (ag_codec is None):
        rs_codec = ag_codec = rs_codec if rs_codec is not None else ag_codec

    rs_default = "int6" if world_size >= RS_INT6_MIN_WORLD else "int4"

    # The all-gather lap forwards bytes verbatim and so contributes exactly one
    # quantization. It is the dominant error term if the RS lap is INT6.
    ag_default = "int4"

    def _pick(requested, default, supported, label):
        if requested is not None:
            if requested not in supported:
                raise ValueError(
                    f"{label} must be one of {supported} for "
                    f"algorithm={algo.name!r}, got {requested!r}"
                )
            return requested
        # The default is a property of the world size, not of the schedule.
        if default not in supported:
            default = supported[0]
        return default

    resolved_rs = _pick(rs_codec, rs_default, algo.rs_codecs, "rs_codec")
    resolved_ag = _pick(ag_codec, ag_default, algo.ag_codecs, "ag_codec")
    return resolved_rs, resolved_ag


def batches_publishes(inbox_memory: str, algorithm: str, link: str) -> bool:
    """Whether ``QuickAllReduceInt4`` batches publishes into super-tiles.

    Always with a release fence, where every publish is an L2 writeback. The
    PCIe ring batches without one too: each of its ``2(N-1)`` hops ends in a
    handshake that is a PCIe round trip whether or not a writeback precedes it.
    """
    return has_release_fence(inbox_memory) or (algorithm == "ring" and link == "pcie")


def _check_sp_residency(launch, grid: int, cu_count: int) -> None:
    """Raise unless a *grid*-block launch of *launch* can have every workgroup
    resident at once on *cu_count* CUs.

    The single-phase kernels are persistent and spin on their peers' flags, so
    a block that is still queued while the resident ones wait on it deadlocks
    the launch. The limit is the occupancy HIP reports for the compiled kernel
    (``max_blocks_per_cu`` of its artifact, read once it is built), not the
    all-reduce's measured table.
    """
    need = -(-grid // cu_count)
    for artifact in _launch_artifacts(launch):
        fits = artifact.max_blocks_per_cu()
        if fits < need:
            raise RuntimeError(
                f"{kernel_symbol(launch)} fits {fits} workgroups per CU, but a "
                f"grid of {grid} blocks on {cu_count} CUs needs {need}: its "
                "persistent blocks would wait on peers that are not resident"
            )


class QuickAllReduceInt4:
    """IPC inbox + flag buffer and launch wrapper for ``quick_allreduce_int4``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange.

    ``algorithm`` selects the schedule. Both are two-shot -- reduce-scatter
    then all-gather -- so they are named for the topology of each lap:

    * ``"mesh"`` (default): each rank pushes to every one of the ``N-1``
      peers, twice. Two hops. Optimal on a meshed xGMI node.
    * ``"ring"``: ``2(N-1)`` hops, each a single contiguous run into exactly
      one peer's inbox. Same wire volume (``2(N-1)/N`` of the payload), traded
      for per-destination locality. Optimal for large messages on PCIe.

    ``rs_codec`` and ``ag_codec`` are the wire formats of the ring's two laps.
    The reduce-scatter lap requantizes ``N-1`` times and it defaults to ``"int6"``
    at TP8, where INT4 would cost too much accuracy. The all-gather lap forwards
    bytes verbatim and contributes a single quantization, so it defaults to ``"int4"``
    everywhere.

    ``inbox_memory`` selects how the IPC inbox is allocated:

    * ``"auto"`` (default) -- ``uncached`` on hosts with xGMI peer links,
      ``finegrained`` on PCIe-attached hosts, decided from the KFD topology.
      TP2 is ``uncached`` on PCIe too: one remote peer cannot collapse.
    * ``"uncached"`` -- correct everywhere, but peer writes collapse on PCIe.
    * ``"finegrained"`` -- device-coherent, full PCIe rate. Cacheable, so each
      publish writes the payload back from the writer's L2 with a release fence
      before the flag goes out write-through (``sc0 sc1``).

    ``min_bytes`` is the payload below which ``allreduce`` refuses to run,
    defaulting to ``QUANT_MIN_PAYLOAD_BYTES``.

    ``link`` selects the tuning ladder and is detected from the KFD topology
    when not given. ``ladder`` replaces the tuning ladder of the policy, for
    tests that run other configurations, and ``norm_ladder`` does the same for
    the fused add+RMSNorm of the mesh algorithm. Both schedules specialise their
    binary to this rank, so the JIT symbol carries an ``_r<n>_`` field.

    ``rmsnorm_hidden`` lists the row widths ``allreduce_rmsnorm`` will be
    called with.

    Every launch, and every replay of a CUDA graph that captured one, of any
    mode (``allreduce``, ``allreduce_rmsnorm``, ``reduce_scatter``,
    ``all_gather``) must run in one serialized device order on each rank.
    Launches through the same engine share its inbox slots and colors: the
    single-phase modes use the ``super_tile=1`` engine at the default block,
    and the all-reduce and the fused RMSNorm use it for the sizes whose rung
    selects it (or falls back to it). Launches in flight at once can read each
    other's payloads; one order for all launches is the rule that holds for
    every ladder. The same host call order
    on every rank is not enough when the calls sit on independent streams; a
    switch of stream needs an event dependency (``wait_event`` /
    ``wait_stream``).
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int = WORLD,
        inbox_memory: str = "auto",
        min_bytes: int | None = None,
        algorithm: str = DEFAULT_ALGORITHM,
        rs_codec: str | None = None,
        ag_codec: str | None = None,
        link: str | None = None,
        ladder: tuple[TwoShotRung, ...] | None = None,
        norm_ladder: tuple[TwoShotRung, ...] | None = None,
        rmsnorm_hidden: tuple[int, ...] = (),
    ):
        check_world_size(world_size)
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {tuple(ALGORITHMS)}, got {algorithm!r}"
            )
        algo = ALGORITHMS[algorithm]
        link = resolve_link(link)
        rs_codec, ag_codec = _resolve_codecs(algo, int(world_size), rs_codec, ag_codec)
        _validate_ipc_process_group(
            group, rank=int(rank), world_size=world_size, owner="QuickAllReduceInt4"
        )
        arch = require_arch(owner="QuickAllReduceInt4")
        if ladder is None:
            ladder = algo.st_ladder(world_size=world_size, link=link, arch=arch)
        if not ladder:
            raise ValueError("ladder must have at least one rung")
        if norm_ladder is None and algo.norm_ladder is not None:
            # Untuned, the fused RMSNorm runs on this object's own ladder, so a
            # ``ladder`` override reaches it too.
            norm_ladder = algo.norm_ladder(
                world_size=world_size, link=link, arch=arch, mesh=ladder
            )
        if norm_ladder is not None and not norm_ladder:
            raise ValueError("norm_ladder must have at least one rung")
        inbox_flags, resolved_inbox = _resolve_inbox_flags(
            inbox_memory, world_size, link
        )
        self._device_index = _cuda_index(device, owner="QuickAllReduceInt4")
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.link = link
        self.inbox_memory = resolved_inbox
        self.algorithm = algorithm
        self.rs_codec = rs_codec
        self.ag_codec = ag_codec
        self._algo = algo
        self._has_launched = False
        cu_count = int(
            torch.cuda.get_device_properties(self._device_index).multi_processor_count
        )
        self._latency_max_tiles = latency_max_tiles(cu_count)
        # v_cvt_scalef32_pk_fp4_bf16 is gfx950-only.
        self.supports_mxfp4 = arch == "gfx950"
        lds_capacity = get_lds_capacity_bytes(arch)

        self._batch_publishes = batches_publishes(resolved_inbox, algorithm, link)

        self.min_bytes = (
            QUANT_MIN_PAYLOAD_BYTES if min_bytes is None else int(min_bytes)
        )
        if self.min_bytes < 0:
            raise ValueError(f"min_bytes must be non-negative, got {self.min_bytes}")

        self._ladder = ladder
        self._norm_ladder = norm_ladder
        self._arch = arch
        self._cu_count = cu_count
        self._lds_capacity = lds_capacity
        self._inbox_flags = inbox_flags
        self._caps = self._engine_caps(ladder)
        self._by_cfg = {}
        # Per engine, the arguments that rebuild its kernel.
        self._build_kw = {}
        # The fused RMSNorm's engines and their arguments, keyed by tile width
        # and then like ``_by_cfg``, built by ``compile_rmsnorm``, and the
        # fused builds ``norm_launch`` adds to them on first use.
        self._norm_by_atoms = {}
        self._norm_build_kw = {}
        self._norm_launch = {}
        # The single-phase builds ``_sp_launcher`` adds on first use, by mode.
        self._sp_launch = {}
        # Whether ``_sp_preflight`` last passed on every rank, which the
        # single-phase launches require.
        self._sp_ready = False
        # Row widths whose fused builds are all compiled and loaded.
        self._rmsnorm_hidden = set()
        try:
            self._build_engines(self._caps, self._by_cfg, self._build_kw)
            for hidden in sorted({int(h) for h in rmsnorm_hidden}):
                self.compile_rmsnorm(hidden)
        except Exception:
            self.close()
            raise

        # The first rung's shape, reported as this object's own.
        self.super_tile, self.block = ladder[0].super_tile, ladder[0].block
        primary = self._by_cfg[(self.super_tile, self.block)]
        self.buf_bytes = primary.buf_bytes
        self.lds_bytes = primary.lds_bytes
        self.tile_bytes = primary.tile_bytes
        self.tile_fp16 = primary.tile_fp16
        self.rank_tile_bytes = primary.rank_tile_bytes
        self.wire_tile_bytes = primary.wire_tile_bytes

    @staticmethod
    def _engine_caps(ladder) -> dict[tuple[int, int], int]:
        """The grid cap of each engine *ladder* needs, keyed
        ``(super_tile, block)``.

        One engine per distinct ``(super_tile, block)``. Each ``block`` also
        gets an ST=1 engine: _select_cfg falls back to it when a payload has
        fewer tiles than the chosen super-tile, and the fallback has to share
        the rung's block -- a different tile size would change the tile count
        it was picked for.

        The rungs go in first so an ST=1 that the ladder *sites* keeps its own
        cap. Only then is the fallback filled in, and at the smallest cap
        among its rungs rather than at the global default: the fallback fires
        only when a payload has fewer tiles than the super-tile it would
        otherwise take, so ``_grid_x`` there is bounded by that super-tile
        (<= 32) with a release fence, and by the chosen rung's own cap without
        one -- under 128 either way. Seeding it with the 1216 default instead
        built a 194 MiB inbox to launch at most 32 blocks into, and did it on
        every object ever constructed.
        """
        caps = {}
        for r in ladder:
            caps.setdefault((r.super_tile, r.block), r.grid_cap)
        for b in {b for _st, b in list(caps)}:
            caps.setdefault((1, b), min(c for (_st, cb), c in caps.items() if cb == b))
        return caps

    def _build_engines(
        self, caps, engines, build_kws, tile_atoms: int = DEFAULT_ATOMS
    ) -> None:
        """Build an engine for each entry of *caps* into *engines*, and the
        arguments that rebuild its kernel into *build_kws*. *tile_atoms*
        other than the default is the fused RMSNorm's wide tile (mesh only).

        Collective: each engine does its own IPC handle exchange, so the
        engines are built in a fixed order -- ranks disagreeing on the order
        would deadlock.
        """
        with torch.cuda.device(self._device_index):
            for key in sorted(caps):
                st, b = key
                # A persistent kernel deadlocks if it launches more workgroups
                # than fit, and the ranks have to agree on the number: take the
                # minimum across the group so a heterogeneous node converges.
                grid = clamp_grid_cap(
                    caps[key],
                    arch=self._arch,
                    world_size=self.world_size,
                    super_tile=st,
                    cu_count=self._cu_count,
                    block=b,
                )
                if tile_atoms != DEFAULT_ATOMS:
                    grid = min(grid, self._lds_resident_grid(b, tile_atoms))
                shared_grid = torch.tensor(grid, dtype=torch.int64)
                dist.all_reduce(shared_grid, op=dist.ReduceOp.MIN, group=self.group)
                build_kw = {
                    "world_size": self.world_size,
                    "rank": self.rank,
                    "super_tile": st,
                    "grid": int(shared_grid.item()),
                    "inbox_memory": self.inbox_memory,
                    "rs_codec": self.rs_codec,
                    "ag_codec": self.ag_codec,
                    "block": b,
                }
                if tile_atoms != DEFAULT_ATOMS:
                    build_kw["tile_atoms"] = tile_atoms
                spec = self._algo.build(**build_kw)
                if spec["lds_bytes"] > self._lds_capacity:
                    raise ValueError(
                        f"{self.algorithm} (rs_codec={self.rs_codec}, "
                        f"ag_codec={self.ag_codec}) at block={b} needs "
                        f"{spec['lds_bytes']} B of LDS, over the "
                        f"{self._lds_capacity} B {self._arch} has"
                    )
                build_kws[key] = build_kw
                engines[key] = _StEngine(
                    spec=spec,
                    group=self.group,
                    rank=self.rank,
                    world_size=self.world_size,
                    inbox_flags=self._inbox_flags,
                )

    def _lds_resident_grid(self, block: int, tile_atoms: int) -> int:
        """Largest persistent grid whose workgroups all fit in the CUs' LDS,
        counting the fused RMSNorm's partial sums."""
        wg_lds = mesh_lds_bytes(
            world_size=self.world_size,
            block=block,
            codec=self.rs_codec,
            tile_atoms=tile_atoms,
            fuse_norm=True,
        )
        per_cu = max(1, self._lds_capacity // wg_lds)
        # A multiple of 8 keeps the flag prefix 64 B aligned at every TP size.
        return max(8, (per_cu * self._cu_count) // 8 * 8)

    def _engines(self) -> list[_StEngine]:
        """Every engine this object holds, each once: the fused RMSNorm
        shares those it has in common with the plain all-reduce."""
        by_id = {}
        for engines in (self._by_cfg, *self._norm_by_atoms.values()):
            for eng in engines.values():
                by_id.setdefault(id(eng), eng)
        return list(by_id.values())

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on *this* rank, across every rung.

        ``buf_bytes`` is the primary engine's alone, which understates a
        ladder-driven object by however many rungs it built. The total is what
        actually has to fit: the wire buffer is
        ``2(N-1) * grid * (ST * rank_atoms * tile + 64)``, so a high rung is
        large on its own and a sweep holding several tuning variants live at
        once is the realistic way to exhaust a device.
        """
        return sum(eng.buf_bytes for eng in self._engines())

    @staticmethod
    def _num_tiles(live_bytes: int, block: int, tile_atoms: int = DEFAULT_ATOMS) -> int:
        tile_bytes = int(block) * int(tile_atoms) * 16
        return max(1, (live_bytes + tile_bytes - 1) // tile_bytes)

    def _select_cfg(
        self, ladder, engines, live_bytes: int, tile_atoms: int = DEFAULT_ATOMS
    ) -> tuple[tuple[int, int], int]:
        """Key in *engines*, built for *ladder*, of a *live_bytes* payload,
        and its tile count.

        The ladder chooses the rung, which fixes the block and so the tile
        count; the tile count then only has to confirm there is a whole
        super-tile to take, falling back to the same block's ST=1 engine when
        there is not.
        """
        rung = select_rung(ladder, live_bytes)
        want, b = rung.super_tile, rung.block
        num_tiles = self._num_tiles(live_bytes, b, tile_atoms)
        if want == 1:
            st = 1
        elif self._batch_publishes:
            st = want if num_tiles >= want else 1
        else:
            st = want if num_tiles > engines[(want, b)].grid else 1
        return (st, b), num_tiles

    def _pick_cfg(self, live_bytes: int) -> tuple[tuple[int, int], int]:
        """``_by_cfg`` key for a *live_bytes* payload, and its tile count."""
        return self._select_cfg(self._ladder, self._by_cfg, live_bytes)

    def _grid_x(self, num_tiles: int, super_tile: int, grid: int) -> int:
        """Blocks to launch for *num_tiles* tiles under *super_tile*.

        *grid* is the compile-time cap of the engine that will run, which is
        per-super-tile once a ladder is in play -- the wire buffer scales with
        ``ST * grid``, so a high rung pairs a large ST with a small cap.

        Batching publishes only pays if a block actually owns a super-tile's
        worth of work: ST=8 across 448 blocks holding one tile each still
        publishes per tile. Hand each block a full super-tile instead, which
        cuts publishes to ``num_tiles / ST`` per phase.
        """
        if self._batch_publishes and super_tile != 1:
            batched = max(-(-num_tiles // super_tile), MIN_BATCH_BLOCKS)
            num_tiles = min(num_tiles, batched)
        return max(1, min(num_tiles, grid))

    def _launch_args(
        self,
        eng: _StEngine,
        inp_ptr,
        out_ptr,
        stream,
        *,
        live_bytes,
        num_tiles,
        norm_ptrs=None,
        norm_eps: float = 0.0,
        divisor: float = DEFAULT_DIVISOR,
    ):
        if stream is None:
            stream = Stream(torch.cuda.current_stream(self._device_index))
        elif not isinstance(stream, Stream):
            stream = Stream(stream)
        return (
            Int32(self.rank),
            Int64(live_bytes),
            Int32(num_tiles),
            Int64(inp_ptr),
            Int64(out_ptr),
            Int64(int(eng._gpu_peer_ptrs)),
            Int64(int(eng._colors)),
            *(Int64(p) for p in self._norm_ptrs(norm_ptrs)),
            *((Float32(norm_eps),) if self._algo.norm_ptr_args else ()),
            Int32(self._grid_x(num_tiles, eng.super_tile, eng.grid)),
            *((Float32(divisor),) if self._algo.has_divisor else ()),
            stream,
        )

    def _norm_ptrs(self, norm_ptrs) -> tuple[int, ...]:
        """*norm_ptrs* padded with zeros to what this algorithm's kernel takes."""
        n = self._algo.norm_ptr_args
        norm_ptrs = tuple(norm_ptrs or ())
        return norm_ptrs + (0,) * (n - len(norm_ptrs))

    def _launch_eng(
        self,
        eng: _StEngine,
        inp,
        out,
        stream,
        *,
        live_bytes: int,
        launch=None,
        norm_ptrs=None,
        norm_eps: float = 0.0,
        divisor: float = DEFAULT_DIVISOR,
    ) -> None:
        num_tiles = max(1, (live_bytes + eng.tile_bytes - 1) // eng.tile_bytes)
        args = self._launch_args(
            eng,
            int(inp.data_ptr()),
            int(out.data_ptr()),
            stream,
            live_bytes=live_bytes,
            num_tiles=num_tiles,
            norm_ptrs=norm_ptrs,
            norm_eps=norm_eps,
            divisor=divisor,
        )
        # A launch may still be using the raw HIP allocations when Python drops
        # the communicator. Keep cleanup conservative even if launch raises.
        self._has_launched = True
        with torch.cuda.device(self._device_index):
            _run_compiled(eng.launch if launch is None else launch, *args)

    def cfgs_for(self, lo: int, hi: int) -> list[tuple[int, int]]:
        """``_by_cfg`` keys a payload of ``lo..hi`` bytes (inclusive) can
        select, in build order."""
        floors = [rung.min_bytes for rung in self._ladder]
        picked = {self._pick_cfg(n)[0] for n in payload_probes(floors, lo, hi)}
        return [key for key in self._by_cfg if key in picked]

    def preload(self, *, payload_range=None) -> None:
        """JIT-compile engine binaries and load them onto the device, without
        launching any of them.

        ``payload_range=(lo, hi)`` takes only the binaries ``allreduce`` would
        run for a payload of ``lo..hi`` bytes (inclusive); ``None`` takes every
        one. The ladder builds engines for the whole size range, while a
        dispatcher routes only its own window here, so the rest never run.

        Local to this rank, not a collective. ``QuickAllReduce`` calls it once
        at init, via ``preload_fly_engines``, to keep JIT compiles and module
        loads out of HIP graph capture.
        """
        keys = self._by_cfg if payload_range is None else self.cfgs_for(*payload_range)
        with torch.cuda.device(self._device_index):
            for key in keys:
                eng = self._by_cfg[key]
                args = self._launch_args(eng, 0, 0, None, live_bytes=0, num_tiles=0)
                preload_launcher(eng.launch, *args)

    def norm_launch(self, cfg, hidden, gemma_norm, latency=False, mxfp4=False):
        """Fused add+RMSNorm variant of the fused RMSNorm's engine *cfg* at
        *hidden*'s tile width; shares its inbox and colors."""
        key = (cfg, int(hidden), bool(gemma_norm), bool(latency), bool(mxfp4))
        launch = self._norm_launch.get(key)
        if launch is None:
            launch = self._algo.build(
                **self._norm_build_kw[self._tile_atoms(hidden)][cfg],
                norm_hidden=key[1],
                gemma_norm=key[2],
                norm_latency=key[3],
                norm_mxfp4=key[4],
            )["launch"]
            self._norm_launch[key] = launch
        return launch

    def _tile_atoms(self, hidden: int) -> int | None:
        """Tile width of the fused RMSNorm at *hidden*, ``None`` if no block
        holds the row.
        """
        for b in SUPPORTED_BLOCKS:
            tile_atoms = tile_atoms_for(hidden, self.world_size, b)
            if tile_atoms is not None:
                return tile_atoms
        return None

    def supports_rmsnorm(self, hidden: int, nbytes: int) -> bool:
        """Whether ``allreduce_rmsnorm`` can take this row width and payload."""
        # Atom byte offsets travel in the signed 32-bit soffset.
        limits = [
            max_payload_bytes(tile_atoms, b)
            for b in SUPPORTED_BLOCKS
            if (tile_atoms := tile_atoms_for(hidden, self.world_size, b)) is not None
        ]
        return bool(limits) and int(nbytes) <= min(limits)

    @staticmethod
    def mxfp4_scale_shape(rows: int, hidden: int) -> tuple[int, int]:
        """Shape of the e8m0 ``mxfp4_scale`` buffer for ``rows x hidden``."""
        return mxfp4_scale_shape(rows, hidden)

    def _check_norm_schedule(self) -> None:
        """Raise unless this object's schedule has the fused RMSNorm."""
        if self.algorithm != "mesh" or (self.rs_codec, self.ag_codec) != (
            "int4",
            "int4",
        ):
            raise ValueError(
                "fused RMSNorm needs algorithm='mesh' with INT4 codecs, got "
                f"{self.algorithm!r} with rs_codec={self.rs_codec!r}, "
                f"ag_codec={self.ag_codec!r}"
            )

    def _check_norm(self, live_bytes: int, hidden: int) -> None:
        """Raise unless the fused RMSNorm can take *hidden* and *live_bytes*."""
        self._check_norm_schedule()
        if not self.supports_rmsnorm(hidden, live_bytes):
            raise ValueError(
                "fused RMSNorm needs a row that fits whole atoms in a tile "
                "split across the ranks, and a payload whose atom offsets fit "
                f"the signed 32-bit soffset, got hidden={hidden}, "
                f"{live_bytes} bytes at world_size={self.world_size}"
            )

    def compile_rmsnorm(self, hidden: int) -> None:
        """Eager-JIT and preload every fused build ``allreduce_rmsnorm`` can pick
        for hidden, without launching any.

        Collective the first time, when it builds the engines: every rank calls
        it, outside CUDA graph capture. The constructor calls it for each width
        in ``rmsnorm_hidden``, and the first eager ``allreduce_rmsnorm`` at an
        unprepared width does too.

        Args:
            hidden: row width; 2048, 4096, 8192 or 16384, or 6144 at
                world_size 2 or 4.
        """
        hidden = int(hidden)
        if hidden in self._rmsnorm_hidden:
            return
        self._check_norm(hidden * 2, hidden)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"QuickAllReduceInt4: the fused RMSNorm at hidden={hidden} is "
                "not prepared, and preparing it builds IPC engines and compiles "
                "kernels, which cannot run under CUDA graph capture; pass "
                f"rmsnorm_hidden=({hidden},) to the constructor, or call "
                f"compile_rmsnorm({hidden}) on every rank before the capture"
            )
        self._build_norm_engines(hidden)
        # Raises if no engine has a block that holds the row.
        self._norm_cfg(hidden * 2, hidden)
        variants = [
            (gemma_norm, mxfp4)
            for gemma_norm in (False, True)
            for mxfp4 in ((False, True) if self.supports_mxfp4 else (False,))
        ]
        with torch.cuda.device(self._device_index):
            for cfg, eng in self._norm_by_atoms[self._tile_atoms(hidden)].items():
                if tile_atoms_for(hidden, self.world_size, cfg[1]) is None:
                    continue
                args = self._launch_args(
                    eng, 0, 0, None, live_bytes=0, num_tiles=0, norm_ptrs=()
                )
                for latency in (True, False) if cfg == (1, BLOCK) else (False,):
                    for gemma_norm, mxfp4 in variants:
                        launch = self.norm_launch(
                            cfg, hidden, gemma_norm, latency, mxfp4
                        )
                        preload_launcher(launch, *args)
        self._rmsnorm_hidden.add(hidden)

    def _build_norm_engines(self, hidden: int) -> None:
        """Build the engines of the fused add+RMSNorm at *hidden*'s tile
        width that are not built yet, those of its ladder in
        ``MESH_NORM_LADDER``. Collective, as the constructor is.

        At the default width an engine whose configuration the plain
        all-reduce already has is shared, not built again.
        """
        tile_atoms = self._tile_atoms(hidden)
        caps = self._engine_caps(self._norm_ladder)
        if tile_atoms == DEFAULT_ATOMS:
            # The fused RMSNorm cannot run at a block whose atom is wider
            # than a row.
            if all(b != BLOCK for _st, b in caps):
                caps[(1, BLOCK)] = min(caps.values())
            shared = [key for key, cap in caps.items() if self._caps.get(key) == cap]
        else:
            fits = [
                b
                for b in SUPPORTED_BLOCKS
                if tile_atoms_for(hidden, self.world_size, b) == tile_atoms
            ]
            wide = {key: cap for key, cap in caps.items() if key[1] in fits}
            if not wide:
                wide[(1, BLOCK if BLOCK in fits else fits[0])] = min(caps.values())
            caps, shared = wide, []
        engines = self._norm_by_atoms.setdefault(tile_atoms, {})
        build_kws = self._norm_build_kw.setdefault(tile_atoms, {})
        for key in shared:
            engines.setdefault(key, self._by_cfg[key])
            build_kws.setdefault(key, self._build_kw[key])
        built, built_kws = {}, {}
        try:
            self._build_engines(
                {key: cap for key, cap in caps.items() if key not in engines},
                built,
                built_kws,
                tile_atoms=tile_atoms,
            )
        except Exception:
            for eng in built.values():
                eng.close()
            raise
        engines.update(built)
        build_kws.update(built_kws)

    def _norm_cfg(self, live_bytes: int, hidden: int) -> tuple[tuple[int, int], int]:
        """Key in the fused RMSNorm's engines at *hidden*'s tile width, and
        tile count, for a fused-norm payload.

        The fused RMSNorm ladder's pick, unless its block cannot hold a
        hidden-wide row, in which case the ST=1 engine of the narrowest block
        that can. Needs ``compile_rmsnorm``.
        """
        self._check_norm(live_bytes, hidden)
        tile_atoms = self._tile_atoms(hidden)
        engines = self._norm_by_atoms.get(tile_atoms)
        if not engines:
            raise RuntimeError("the fused RMSNorm needs compile_rmsnorm first")

        def _holds(b):
            return tile_atoms_for(hidden, self.world_size, b) is not None

        # A wide tile has no engines at the blocks that cannot hold the row.
        if _holds(select_rung(self._norm_ladder, live_bytes).block):
            return self._select_cfg(self._norm_ladder, engines, live_bytes, tile_atoms)
        blocks = sorted(b for (_st, b) in engines if _holds(b))
        if not blocks:
            raise ValueError(
                f"no engine of this object has a block that supports hidden={hidden}"
            )
        return (1, blocks[0]), self._num_tiles(live_bytes, blocks[0], tile_atoms)

    def allreduce_rmsnorm(
        self,
        inp,
        residual,
        weight,
        eps: float,
        out,
        residual_out,
        *,
        gemma_norm: bool = False,
        stream=None,
    ):
        """Fused INT4 all-reduce + residual add + RMSNorm (mesh algorithm).

        ``residual_out = allreduce(inp) + residual`` and
        ``out = rmsnorm(residual_out, eps) * w`` with ``w = weight`` or
        ``1 + weight`` when ``gemma_norm``.

        Runs on the engines of ``MESH_NORM_LADDER``. A call may be inside a
        CUDA graph capture, the first one included, if *hidden* is prepared:
        given in the constructor's ``rmsnorm_hidden``, or by an earlier
        ``compile_rmsnorm`` or eager call. An eager call prepares it, which is
        a collective the first time.

        Args:
            inp: this rank's bf16 partial sum, ``rows x hidden`` elements.
            residual: bf16 residual, same number of elements as ``inp``.
            weight: bf16 RMSNorm weight of ``hidden`` elements; hidden must
                be 2048, 4096, 8192 or 16384, or 6144 at world_size 2 or 4.
            eps: RMSNorm epsilon.
            out: bf16 output for the normed rows, same size as ``inp``.
            residual_out: bf16 output for ``allreduce(inp) + residual``; may
                be ``residual`` (in place), must not overlap ``inp`` or
                ``out``.
            gemma_norm: scale by ``1 + weight`` instead of ``weight``.
            stream: ``None`` uses the current PyTorch stream on this device.
        """
        self._allreduce_rmsnorm(
            inp, residual, weight, eps, out, residual_out, gemma_norm, stream
        )

    def allreduce_rmsnorm_mxfp4(
        self,
        inp,
        residual,
        weight,
        eps: float,
        out,
        residual_out,
        mxfp4_out,
        mxfp4_scale,
        *,
        gemma_norm: bool = False,
        stream=None,
    ):
        """``allreduce_rmsnorm`` that also MXFP4-quantizes ``out`` (gfx950).

        Args:
            inp, residual, weight, eps, out, residual_out, gemma_norm,
                stream: as in ``allreduce_rmsnorm``.
            mxfp4_out: 1-byte tensor of ``rows x hidden // 2`` bytes that
                receives the packed FP4 values.
            mxfp4_scale: 1-byte tensor of ``mxfp4_scale_shape(rows, hidden)``
                that receives the shuffled e8m0 scales.

        ``mxfp4_out`` and ``mxfp4_scale`` hold what
        ``per_1x32_f4_quant_hip(out, shuffle=True)`` returns, bit for bit.
        """
        self._allreduce_rmsnorm(
            inp,
            residual,
            weight,
            eps,
            out,
            residual_out,
            gemma_norm,
            stream,
            mxfp4=(mxfp4_out, mxfp4_scale),
        )

    def _allreduce_rmsnorm(
        self,
        inp,
        residual,
        weight,
        eps,
        out,
        residual_out,
        gemma_norm,
        stream,
        mxfp4=None,
    ):
        live_bytes = check_payload(
            inp, out, device_index=self._device_index, owner="QuickAllReduceInt4"
        )
        hidden = int(weight.numel())
        self._check_norm(live_bytes, hidden)
        if int(inp.numel()) % hidden != 0:
            raise ValueError("numel must be a multiple of hidden")
        for name, t in (
            ("residual", residual),
            ("residual_out", residual_out),
            ("weight", weight),
        ):
            if not isinstance(t, torch.Tensor) or t.dtype != torch.bfloat16:
                raise ValueError(f"{name} must be a bf16 tensor")
            if not t.is_cuda or t.device.index != self._device_index:
                raise ValueError(f"{name} must be on cuda:{self._device_index}")
            if not t.is_contiguous() or int(t.data_ptr()) % 16 != 0:
                raise ValueError(f"{name} must be contiguous and 16-byte aligned")
        if residual.numel() != inp.numel() or residual_out.numel() != inp.numel():
            raise ValueError("residual/residual_out must match inp numel")

        def _overlap(a, b):
            pa, pb = int(a.data_ptr()), int(b.data_ptr())
            return max(pa, pb) < min(pa, pb) + live_bytes

        if _overlap(residual_out, inp) or _overlap(residual_out, out):
            raise ValueError("residual_out must not overlap inp or out")
        if _overlap(residual, out) or (
            _overlap(residual, residual_out)
            and residual.data_ptr() != residual_out.data_ptr()
        ):
            raise ValueError(
                "residual must be residual_out or not overlap it, and must "
                "not overlap out"
            )
        mxfp4_ptrs = ()
        if mxfp4 is not None:
            if not self.supports_mxfp4:
                raise RuntimeError("fused MXFP4 quant needs gfx950")
            rows = int(inp.numel()) // hidden
            for name, t, nbytes in (
                ("mxfp4_out", mxfp4[0], live_bytes // 4),
                ("mxfp4_scale", mxfp4[1], math.prod(mxfp4_scale_shape(rows, hidden))),
            ):
                if not isinstance(t, torch.Tensor) or t.element_size() != 1:
                    raise ValueError(f"{name} must be a 1-byte tensor")
                if not t.is_cuda or t.device.index != self._device_index:
                    raise ValueError(f"{name} must be on cuda:{self._device_index}")
                if not t.is_contiguous() or t.numel() < nbytes:
                    raise ValueError(f"{name} must be contiguous with {nbytes} bytes")
            mxfp4_ptrs = (int(mxfp4[0].data_ptr()), int(mxfp4[1].data_ptr()))
        self.compile_rmsnorm(hidden)
        cfg, num_tiles = self._norm_cfg(live_bytes, hidden)
        eng = self._norm_by_atoms[self._tile_atoms(hidden)][cfg]
        # One tile per workgroup at <= 2 workgroups per CU: occupancy is
        # free, so take the low-latency build.
        latency = cfg == (1, BLOCK) and num_tiles <= min(
            eng.grid, self._latency_max_tiles
        )
        launch = self.norm_launch(cfg, hidden, gemma_norm, latency, mxfp4 is not None)
        if torch.cuda.is_current_stream_capturing() and not launcher_loaded(launch):
            raise RuntimeError(
                f"QuickAllReduceInt4: fused build {kernel_symbol(launch)} is not "
                "loaded, and loading it cannot run under CUDA graph capture"
            )
        self._launch_eng(
            eng,
            inp,
            out,
            stream,
            live_bytes=live_bytes,
            launch=launch,
            norm_eps=float(eps),
            norm_ptrs=(
                int(residual.data_ptr()),
                int(residual_out.data_ptr()),
                int(weight.data_ptr()),
                *mxfp4_ptrs,
            ),
        )

    def close(self):
        norm_by_atoms = getattr(self, "_norm_by_atoms", None) or {}
        if not getattr(self, "_by_cfg", None) and not any(norm_by_atoms.values()):
            return
        with torch.cuda.device(self._device_index):
            if getattr(self, "_has_launched", False):
                torch.cuda.synchronize(self._device_index)
                self._has_launched = False
            for eng in self._engines():
                eng.close()
            self._by_cfg.clear()
            norm_by_atoms.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            # Destructors must not raise, especially during interpreter shutdown.
            return

    def variant(self, nbytes: int) -> str:
        """Identity of the binary an *nbytes* payload would actually run."""
        cfg, num_tiles = self._pick_cfg(int(nbytes))
        eng = self._by_cfg[cfg]
        grid_x = self._grid_x(num_tiles, eng.super_tile, eng.grid)
        return f"{kernel_symbol(eng.launch)}/grid_x{grid_x}"

    def is_beneficial(self, nbytes: int) -> bool:
        """Whether *nbytes* is large enough for this kernel to be worth using.

        Callers with a fallback should route anything smaller to it; see
        ``QUANT_MIN_PAYLOAD_BYTES``. ``allreduce`` refuses payloads below the
        threshold rather than silently running them slowly.
        """
        return int(nbytes) >= self.min_bytes

    def _check_divisor_supported(self, divisor: float) -> None:
        """A divisor other than 1 needs the mesh schedule's INT4 wire format."""
        if divisor != DEFAULT_DIVISOR and (
            not self._algo.has_divisor or "int4" not in (self.rs_codec, self.ag_codec)
        ):
            raise ValueError(
                "QuickAllReduceInt4 divisor is only supported by the mesh "
                f"algorithm with the int4 codec, got algorithm={self.algorithm!r}, "
                f"rs_codec={self.rs_codec!r}, ag_codec={self.ag_codec!r}"
            )

    def allreduce(self, inp, out, stream=None, divisor: float = DEFAULT_DIVISOR):
        """Two-shot INT4 all-reduce into ``out``.

        ``stream=None`` uses the current PyTorch stream on this device; calls
        on one communicator run in one serialized device order (see the class
        docstring). ``divisor`` defaults to 1, which keeps the previous INT4
        result.
        Loaded values are divided by it before the first pack, and the
        finished sum is multiplied by it once. Every rank must pass the same
        divisor; this is not checked, and a mismatch silently gives each rank
        a different result. Only the mesh schedule with the INT4 codec takes
        a divisor other than 1.
        """
        divisor = _check_divisor(divisor)
        self._check_divisor_supported(divisor)
        live_bytes = check_payload(
            inp, out, device_index=self._device_index, owner="QuickAllReduceInt4"
        )
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"QuickAllReduceInt4.allreduce got a {live_bytes} B payload, "
                f"below the {self.min_bytes} B floor: at decode sizes this "
                "kernel saves a few microseconds on a collective that is not "
                "the bottleneck, and charges ~36 dB of SQNR for them. Route "
                "small messages to an exact all-reduce, or pass min_bytes=0 "
                "to override."
            )
        cfg, _num_tiles = self._pick_cfg(live_bytes)
        self._launch_eng(
            self._by_cfg[cfg],
            inp,
            out,
            stream,
            live_bytes=live_bytes,
            divisor=divisor,
        )

    def _sp_engine(self) -> _StEngine:
        """The engine the single-phase modes run on: ``super_tile=1`` at the
        default block, whose inbox and colors they share with the all-reduce
        launches that select the same engine."""
        if self.algorithm != "mesh":
            raise ValueError(
                "reduce_scatter / all_gather need the mesh algorithm, got "
                f"algorithm={self.algorithm!r}"
            )
        if self.inbox_memory != "uncached":
            raise ValueError(
                "reduce_scatter / all_gather are validated on the uncached inbox "
                f"only, got inbox_memory={self.inbox_memory!r}"
            )
        eng = self._by_cfg.get((1, BLOCK))
        if eng is None:
            raise ValueError(
                f"reduce_scatter / all_gather need a super_tile=1, block={BLOCK} "
                f"engine, this object has {sorted(self._by_cfg)}"
            )
        return eng

    def _sp_launcher(self, mode):
        """Build of the ``super_tile=1`` engine for *mode*, made on first use."""
        launch = self._sp_launch.get(mode)
        if launch is None:
            launch = self._algo.build(**self._build_kw[(1, BLOCK)], mode=mode)["launch"]
            self._sp_launch[mode] = launch
        return launch

    def _check_sp_payload(self, mode, inp, out, tile_bytes: int) -> int:
        """Validate a reduce-scatter / all-gather pair; return the full tensor's bytes."""
        inp_ptr, out_ptr = check_tensors(
            inp, out, device_index=self._device_index, owner="QuickAllReduceInt4"
        )
        inp_bytes = int(inp.numel()) * int(inp.element_size())
        out_bytes = int(out.numel()) * int(out.element_size())
        full_bytes, shard_bytes = (
            (inp_bytes, out_bytes) if mode == MODE_RS else (out_bytes, inp_bytes)
        )
        if full_bytes != self.world_size * shard_bytes:
            raise ValueError(
                f"{mode}: the full tensor must be world_size={self.world_size} "
                f"times the shard, got {full_bytes} and {shard_bytes} bytes"
            )
        if full_bytes == 0 or full_bytes % tile_bytes != 0:
            raise ValueError(
                f"{mode}: the full tensor must be a non-empty whole number of "
                f"{tile_bytes}-byte tiles, got {full_bytes} bytes"
            )
        if full_bytes > MAX_PAYLOAD_BYTES:
            raise ValueError(
                "QuickAllReduceInt4 payload must not exceed the 4 GiB buffer window"
            )
        if max(inp_ptr, out_ptr) < min(inp_ptr + inp_bytes, out_ptr + out_bytes):
            raise ValueError("QuickAllReduceInt4 requires non-overlapping input/output")
        return full_bytes

    def _sp_preflight(self, kind, inp, out) -> None:
        """Collective: build and load both single-phase modes and agree on the
        call, before any rank launches one.

        A rank that launches while a peer does not, or while a peer launches
        another size or mode, polls that peer's flags forever. So every rank
        catches whatever raises locally here (the engine and codec rules, the
        tensors, a build, the residency check) and the ranks vote on it through
        ``self.group``, together with the call kind and the full size (the shard
        size is fixed by it), which are all that can differ between ranks that
        get this far (the world size is the group's, the grid the minimum over
        the ranks). If any rank failed or the kinds or sizes differ, all raise:
        a failing rank its own exception, the others a ``RuntimeError``.

        *kind* is ``"compile_sp"`` or the mode of the eager launch that runs
        this; *inp* and *out* are that call's tensors, checked as it checks
        them (``compile_sp`` passes ``full`` and ``shard``). It loads modules,
        which a CUDA graph capture cannot do, so it is refused there before any
        vote; every rank must be outside a capture.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"QuickAllReduceInt4: {kind} needs the single-phase preflight, "
                "which loads kernels and agrees the configuration across ranks "
                "and cannot run under CUDA graph capture; call "
                "compile_sp(full, shard) on every rank before capture"
            )
        self._sp_ready = False
        error = fingerprint = None
        try:
            eng = self._sp_engine()
            full_bytes = self._check_sp_payload(
                MODE_AG if kind == MODE_AG else MODE_RS, inp, out, eng.tile_bytes
            )
            args = self._launch_args(eng, 0, 0, None, live_bytes=0, num_tiles=0)
            with torch.cuda.device(self._device_index):
                for mode in (MODE_RS, MODE_AG):
                    launch = self._sp_launcher(mode)
                    preload_launcher(launch, *args)
                    _check_sp_residency(launch, eng.grid, self._cu_count)
            fingerprint = (("kind", kind), ("full_bytes", full_bytes))
        except Exception as exc:  # noqa: BLE001 -- shared with the peers below
            error = exc
        message = None if error is None else f"{type(error).__name__}: {error}"
        votes = [None] * self.world_size
        dist.all_gather_object(votes, (message, fingerprint), group=self.group)
        failed = [
            f"rank {r}: {msg}" for r, (msg, _fp) in enumerate(votes) if msg is not None
        ]
        if not failed and len({fp for _msg, fp in votes}) == 1:
            self._sp_ready = True
            return
        if error is not None:
            raise error
        if failed:
            raise RuntimeError(
                "QuickAllReduceInt4 single-phase preflight failed on another "
                "rank, so nothing launches: " + "; ".join(failed)
            )
        differing = [
            i
            for i in range(len(votes[0][1]))
            if len({fp[i][1] for _msg, fp in votes}) > 1
        ]
        raise RuntimeError(
            "QuickAllReduceInt4 single-phase calls differ across ranks, so "
            "nothing launches: "
            + "; ".join(
                f"rank {r} " + ", ".join(f"{fp[i][0]}={fp[i][1]}" for i in differing)
                for r, (_msg, fp) in enumerate(votes)
            )
        )

    def _run_sp(self, mode, inp, out, stream) -> None:
        # No divisor: the single-phase builds take the default of 1, and the
        # all-gather's dequant would scale by any other. No ``min_bytes`` floor
        # either: it guards the all-reduce's small-message accuracy cost, and a
        # sequence-parallel caller picks this kernel for its layout.
        if not self._sp_ready:
            self._sp_preflight(mode, inp, out)
        eng = self._sp_engine()
        full_bytes = self._check_sp_payload(mode, inp, out, eng.tile_bytes)
        self._launch_eng(
            eng,
            inp,
            out,
            stream,
            live_bytes=full_bytes,
            launch=self._sp_launcher(mode),
        )

    def reduce_scatter(self, inp, out, stream=None):
        """INT4 reduce-scatter: this rank's contiguous block of the sum.

        ``inp`` is this rank's bf16 ``[T, H]`` partial and ``out`` the
        ``[T / world_size, H]`` block ``[rank * T/W, (rank + 1) * T/W)`` of the
        element-wise sum over ranks. Only byte sizes are checked: ``inp`` must
        be ``world_size`` times ``out`` and a whole number of tiles
        (``tile_bytes``, 32 KiB), so pass ``T`` divisible by ``world_size``.
        Every partial, this rank's own included, goes through the INT4 codec,
        so a row's value does not depend on which rank owns it (the all-reduce
        adds its own share exactly). Shares the inbox slots and colors with the
        all-reduce launches that go through the same engine; every rank must issue the same sequence of launches and
        sizes, and every launch and graph replay on this communicator must run
        in one serialized device order (see the class docstring). Needs the
        mesh algorithm, the uncached inbox and a ``super_tile=1`` engine at the
        default block.

        Until ``compile_sp`` has passed on every rank, the first eager call
        runs it with these tensors, which makes that call a collective (every
        rank makes it, in the same mode and with the same sizes), and a call
        under CUDA graph capture is refused.
        """
        self._run_sp(MODE_RS, inp, out, stream)

    def all_gather(self, inp, out, stream=None):
        """INT4 all-gather of contiguous row blocks.

        ``inp`` is this rank's bf16 ``[T / world_size, H]`` block and ``out``
        the ``[T, H]`` concatenation of every rank's block, in rank order. This
        rank's own block is round-tripped through the codec too, so every rank
        holds bitwise-equal output. Same size, sequence, ordering and
        ``compile_sp`` rules as ``reduce_scatter``.
        """
        self._run_sp(MODE_AG, inp, out, stream)

    def compile_sp(self, full, shard) -> None:
        """Collective preflight of ``reduce_scatter`` and ``all_gather``: JIT-
        compile both and load them onto the device, launching neither, and
        agree across the ranks that they can run. Call it on every rank, with
        the same sizes, before CUDA graph capture; the first eager
        ``reduce_scatter`` or ``all_gather`` runs it if nobody has.

        ``full`` is a ``[T, H]`` tensor and ``shard`` a ``[T / world_size, H]``
        one; they are checked as ``reduce_scatter`` would check them and left
        untouched. Each rank catches what fails locally (the engine and codec
        rules, the tensors, the builds, and whether every block of a build's
        grid fits the device at once, which the persistent kernels need), and
        the ranks compare the call kind and these sizes through ``group``. If
        any rank failed or they differ, every rank raises, the failing rank its
        own exception and the others a ``RuntimeError``, and no launch is
        accepted until a preflight passes (a failed repeat call revokes an
        earlier pass). The exception types differ between ranks, so do not
        branch on the type. Later shapes are checked on their own rank only,
        the checks that depend on addresses (overlap, alignment) included, so
        they must agree across ranks as well. Refused under CUDA graph capture.
        """
        self._sp_preflight("compile_sp", full, shard)
