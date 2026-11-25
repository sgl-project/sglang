# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} INT4 **mesh** all-reduce, plus its reduce-scatter
and all-gather phases on their own (``mode``, below).

Topology of each lap: every rank pushes directly to all ``N-1`` peers, twice.

INT4 nibble: [-8,+7], −1/8, 4 B/thread, 1152 B rank-tile. Scale is
group-16 signed E5M2 (bias 12) in the 128 B region. Super-tile ST∈{1,8}; host
uses ST=1 when ``num_tiles ≤`` the occupancy-clamped persistent grid.
Payload HBM is bf16; in-kernel math is packed fp16. Each rank owns
``tile_atoms / world_size`` atoms of a tile (8 GPUs → 1, 4 → 2, 2 → 4 at
the default 8 atoms).

The geometry and the cache policy are in ``allreduce_shared``.
The wire codec is in ``quick_allreduce_codec``.

One tuning knob besides the super-tile: ``block``, threads per workgroup. It
sets the tile (``block * tile_atoms * 16 B``), hence how many blocks a payload
gets and how many flags it costs. ``tile_atoms`` is 8 except for the fused
RMSNorm at a row width that needs the 12-atom tile (``tile_atoms_for``).

A rank never round-trips through its own inbox: its reduce-scatter share is
added from registers, and its reduced chunk is decoded from the same packet it
sends. That needs the rank at trace time, which costs one binary per rank.

``mode`` ``"rs"`` / ``"ag"`` run one phase on its own, for sequence
parallelism (``super_tile=1``, the 8-atom tile, INT4, no fused norm, and
``divisor == 1``). Rank ``d`` then owns a contiguous block of ``N / world_size``
bytes (rows ``[d*T/W, (d+1)*T/W)`` of a ``[T, H]`` tensor), not a slice of every
tile: the tile machinery is unchanged and only the HBM address of atom
``a = d*rank_atoms + k`` of tile ``v`` moves from ``v*tile_atoms + a`` to
``d*A_s + v*rank_atoms + k`` with ``A_s = num_tiles * rank_atoms``. ``N`` must be
a whole number of tiles. Unlike the all-reduce, these modes quantize this
rank's own partial too, in registers, and accumulate it in rank order with the
same dequant as a peer's packet, so a row's value does not depend on which rank
owns it and every rank's all-gather output is bitwise equal.

The all-reduce's second phase is the only guard against inbox reuse (a sender
writes only after the receiver's previous release flag). A single phase has
none, so both modes send their payload through the all-gather phase's slots
and start every round with ``ready``: the reduce-scatter phase's release flag
with no payload, published once this rank has finished reading the previous
round (or launch), awaited by every sender before its first payload store. The
exchange overlaps the round's load and quantize, and nothing is awaited after
the reads, so a following all-reduce is safe too: it writes the reduce-scatter
slots at once and waits for this rank's flag before it writes the all-gather
ones. No other inbox word is used.

``divisor`` defaults to ``DEFAULT_DIVISOR``. Loaded fp16 is divided by it
before the first pack; the gathered sum is multiplied by it once, through
the all-gather dequant scale, before the bf16 store. The second pack is
left unchanged. Only a codec with a scale plane (INT4) uses it.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import (
    Float32,
    Int32,
    Int64,
    ReductionOp,
    Stream,
    T,
    as_ir_value,
)

from .. import buffer_ops
from ..quant_utils import emit_mx_e8m0_scale
from .allreduce_shared import (
    _INBOX_POLICY,
    ATOMS,
    BLOCK,
    DEFAULT_GRID_CAP,
    FLAG_I32_PER_LANE,
    FLAG_LANES,
    QUAD_LANES,
    QUADS_PER_WAVE,
    SUPPORTED_ARCHS,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WAVE,
    WORLD,
    _acquire_inbox,
    _buffer_load,
    _buffer_ptr,
    _color_io,
    _global_ptr,
    _i32_to_bytes,
    _load_flag,
    _load_peers,
    _payload_io,
    _release_inbox,
    _store_flag_peer,
    _store_v4i32_peer,
    _to_sgpr_i64,
    make_pack_norm_storage,
    make_pack_storage,
)
from .quick_allreduce_codec import (
    SCALE_CODEC_TAG,
    SUPER_TILES,
    SUPPORTED_BLOCKS,
    _atom_bf16_to_f16,
    _atom_f16_to_bf16,
    _clamp_fp16_overflow,
    _codec_dequant,
    _codec_load,
    _codec_quant,
    _f16x2,
    _i32,
    _scale_from_word,
    _splat_f16x2,
    codecs_for_block,
    scale_slot_of,
    thread_lane,
)

# 1 keeps the undivided INT4 result; larger divisors are opt-in.
DEFAULT_DIVISOR = 1.0

# Re-exported for the host, which imports its tile math from this module.
__all__ = [
    "DEFAULT_ATOMS",
    "DEFAULT_DIVISOR",
    "DEFAULT_GRID_CAP",
    "MESH_CODECS",
    "MODES",
    "MODE_AG",
    "MODE_AR",
    "MODE_RS",
    "SUPER_TILES",
    "SUPPORTED_BLOCKS",
    "SUPPORTED_WORLDS",
    "TILE_ATOMS",
    "TILE_BYTES",
    "WIDE_ATOMS",
    "WORLD",
    "clamp_grid_cap",
    "latency_max_tiles",
    "make_quick_allreduce_int4_kernel",
    "max_payload_bytes",
    "mesh_lds_bytes",
    "mxfp4_scale_shape",
    "tile_atoms_for",
]

PHASES = 2
PHASE_REDUCE_SCATTER = 0
PHASE_ALL_GATHER = 1

# The all-reduce, and its reduce-scatter / all-gather phase on its own.
MODE_AR = "ar"
MODE_RS = "rs"
MODE_AG = "ag"
MODES = (MODE_AR, MODE_RS, MODE_AG)

# (world_size, super_tile) → VGPR-limited workgroups per CU, measured on the
# mesh kernel. Super-tile widens the live atom list, so residency falls as it
# grows; world size narrows each rank's share of a tile, so it rises with N.
_RESIDENT_WGS_PER_CU = {
    (2, 1): 3,
    (2, 8): 4,
    (4, 1): 4,
    (4, 8): 5,
    (8, 1): 4,
    (8, 8): 6,
}


def _resident_wgs_per_cu(world_size: int, super_tile: int) -> int:
    """VGPR-limited workgroups per CU at ``BLOCK`` threads, or the world's
    smallest if the pair was not measured."""
    key = (int(world_size), int(super_tile))
    resident = _RESIDENT_WGS_PER_CU.get(key)
    if resident is None:
        for_world = [v for (w, _st), v in _RESIDENT_WGS_PER_CU.items() if w == key[0]]
        if not for_world:
            raise ValueError(
                "quick_allreduce_int4 has no residency measurement for "
                f"world_size={world_size}"
            )
        resident = min(for_world)
    return resident


# Workgroups per CU the low-latency fused-norm build keeps resident.
NORM_LATENCY_WGS_PER_CU = 2


def latency_max_tiles(cu_count: int) -> int:
    """Largest launch the ``norm_latency`` build keeps fully resident."""
    return NORM_LATENCY_WGS_PER_CU * int(cu_count)


# Tile widths in atoms. A 12-atom tile holds a whole number of the rows that
# never divide the default 8-atom one, e.g. four bf16 hidden-6144 rows at
# block=256.
DEFAULT_ATOMS = ATOMS
WIDE_ATOMS = 12
TILE_ATOMS = (DEFAULT_ATOMS, WIDE_ATOMS)
SIGNED_SOFFSET_MAX = (1 << 31) - 1


def tile_atoms_for(hidden: int, world_size: int, block: int = BLOCK) -> int | None:
    """Tile width in atoms that keeps every bf16 row of hidden in one tile.

    Fused RMSNorm reduces a row inside one workgroup, so a row must be a whole
    number of atoms, a tile a whole number of rows, and the tile has to split
    evenly across *world_size* ranks. ``None`` when no width fits.
    """
    row_bytes = int(hidden) * 2
    atom_bytes = int(block) * 16
    if row_bytes <= 0 or row_bytes % atom_bytes != 0:
        return None
    atoms_per_row = row_bytes // atom_bytes
    for atoms in TILE_ATOMS:
        if atoms % atoms_per_row == 0 and atoms % int(world_size) == 0:
            return atoms
    return None


def max_payload_bytes(tile_atoms: int, block: int = BLOCK) -> int:
    """Largest fused-RMSNorm payload whose last atom offset still fits the
    signed 32-bit ``soffset``."""
    atom_bytes = int(block) * 16
    tile_bytes = int(tile_atoms) * atom_bytes
    max_tiles = (SIGNED_SOFFSET_MAX - (tile_bytes - atom_bytes)) // tile_bytes + 1
    return max_tiles * tile_bytes


MXFP4_GROUP = 32


def mxfp4_scale_shape(rows: int, hidden: int) -> tuple[int, int]:
    """Shuffled e8m0 scale buffer of ``per_1x32_f4_quant_hip(shuffle=True)``."""
    return (int(rows) + 255) // 256 * 256, (int(hidden) // MXFP4_GROUP + 7) // 8 * 8


def clamp_grid_cap(
    requested: int,
    *,
    arch: str,
    world_size: int,
    super_tile: int,
    cu_count: int,
    block: int = BLOCK,
) -> int:
    """Clamp a requested persistent grid to what actually fits on the device.

    A persistent kernel deadlocks if it launches more workgroups than can be
    co-resident, so the cap has to respect VGPR-limited occupancy.

    Under-launching a persistent kernel is always safe (each block simply loops over more tiles)
    whereas over-launching is the failure mode.
    """
    if requested < 1 or cu_count < 1:
        raise ValueError("grid_cap and cu_count must be positive")
    if block < 1:
        raise ValueError("block must be positive")
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"quick_allreduce_int4 has no residency measurement for {arch!r}"
        )
    resident = _resident_wgs_per_cu(world_size, super_tile)
    if block > BLOCK:
        resident = max(1, (resident * BLOCK) // int(block))
    return min(int(requested), resident * int(cu_count))


# Wire formats the mesh can build.
MESH_CODECS = ("int4", "fp16")


def mesh_lds_bytes(
    *,
    world_size: int,
    block: int = BLOCK,
    codec: str = "int4",
    tile_atoms: int = DEFAULT_ATOMS,
    fuse_norm: bool = False,
) -> int:
    """LDS of one mesh workgroup: a packed rank-tile per (destination,
    rank-atom), plus the fused RMSNorm's per-(wave, row) partial sums."""
    c = codecs_for_block(block)[codec]
    pack_rows = (int(world_size) - 1) * (int(tile_atoms) // int(world_size))
    norm_red_bytes = (block // WAVE) * int(tile_atoms) * 4 if fuse_norm else 0
    return pack_rows * c.rank_tile_bytes + norm_red_bytes


def _sp_payload_io(
    inp_ptr, out_ptr, nbytes, num_tiles, mode, world_size, rank_atoms, block, tid
):
    """``(load, store)`` for this thread's 16 B of one atom of a ``mode="rs"`` /
    ``"ag"`` payload.

    ``"rs"`` reads the full ``[T, H]`` tensor and writes one rank's block,
    ``"ag"`` reads a block and writes the full tensor. The full tensor is
    ``(world_size, num_tiles, rank_atoms, block * 4)`` i32, rank ``d``'s block
    being its ``d``-th slab; a block is ``(num_tiles, rank_atoms, block * 4)``.
    ``load(*coord)`` and ``store(vec, *coord)`` take the coordinate of the input
    or output side without its last mode: ``(d, tile, k)`` for the full tensor,
    ``(tile, k)`` for a block. Thread *tid* owns i32 ``[4 * tid, 4 * tid + 4)``
    of every row. *nbytes* is the full tensor's size in both modes, and bounds
    both tensors, so a read past it returns 0 and a write past it is dropped.
    """
    row_i32 = block * 4
    full_layout = fx.make_layout(
        (world_size, num_tiles, rank_atoms, row_i32),
        (num_tiles * fx.Int32(rank_atoms * row_i32), rank_atoms * row_i32, row_i32, 1),
    )
    shard_layout = fx.make_layout(
        (num_tiles, rank_atoms, row_i32), (rank_atoms * row_i32, row_i32, 1)
    )
    row_layout = fx.make_layout((1, row_i32), (row_i32, 1))
    copy_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int32)
    thr_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_layout((1, block), (1, 1)),
        fx.make_layout((1, 4), (1, 1)),
    ).get_slice(tid)
    shard_bytes = nbytes // fx.Int64(world_size)

    def _tensor(addr, layout, live_bytes):
        view = fx.make_view(_global_ptr(addr, T.i32, 16), layout)
        return rocdl.make_buffer_tensor(
            view, max_size=False, num_records_bytes=live_bytes
        )

    full_in = mode == MODE_RS
    inp = _tensor(
        inp_ptr,
        full_layout if full_in else shard_layout,
        nbytes if full_in else shard_bytes,
    )
    out = _tensor(
        out_ptr,
        shard_layout if full_in else full_layout,
        shard_bytes if full_in else nbytes,
    )

    def _row(buf, coord):
        return fx.make_view(fx.get_iter(fx.slice(buf, (*coord, None))), row_layout)

    def load(*coord):
        src = thr_copy.partition_S(_row(inp, coord))
        frag = fx.make_fragment_like(src)
        fx.copy(copy_atom, src, frag)
        return _atom_bf16_to_f16(fx.Vector(frag.load()))

    def store(vec, *coord):
        dst = thr_copy.partition_D(_row(out, coord))
        frag = fx.make_fragment_like(dst)
        frag.store(_atom_f16_to_bf16(vec))
        fx.copy(copy_atom, frag, dst)

    return load, store


def make_quick_allreduce_int4_kernel(
    *,
    world_size: int = WORLD,
    super_tile: int = 1,
    grid: int,
    inbox_memory: str = "uncached",
    codec: str = "int4",
    block: int = BLOCK,
    rank: int,
    norm_hidden: int | None = None,
    gemma_norm: bool = False,
    norm_latency: bool = False,
    norm_mxfp4: bool = False,
    tile_atoms: int | None = None,
    mode: str = MODE_AR,
):
    """Build the mesh all-reduce kernel.

    ``tile_atoms`` (8 or 12) is the tile width in ``block * 16`` B atoms.
    ``None`` takes the width ``tile_atoms_for`` gives ``norm_hidden``, else 8.
    All ranks must build the same width.

    With ``norm_hidden`` set, the all-gather epilogue fuses
    ``residual_out = ar + residual`` and
    ``out = rmsnorm(residual_out) * weight`` (``1 + weight`` with
    ``gemma_norm``), the contract of vLLM's ``fused_add_rms_norm``. It needs
    the INT4 codec and a ``hidden`` for which ``tile_atoms_for`` finds a tile
    at ``world_size`` and ``block``. The RMSNorm epsilon is the runtime ``norm_eps``
    argument, so one build serves every epsilon.

    ``norm_mxfp4`` additionally quantizes the bf16 ``out`` to MXFP4 the way
    ``per_1x32_f4_quant_hip(out, shuffle=True)`` does: packed fp4x2
    ``(rows, hidden // 2)`` and shuffled e8m0 scales of
    ``mxfp4_scale_shape(rows, hidden)``, the layout ``gemm_a4w4`` reads.

    ``norm_latency`` (``super_tile=1``, ``block=BLOCK`` only) trades occupancy
    for latency: two workgroups per CU instead of the persistent-grid
    residency, so the host may only launch it with
    ``num_tiles <= latency_max_tiles(cu)``.

    ``mode`` is ``"ar"`` (default, the two-phase all-reduce), ``"rs"`` (the
    reduce-scatter phase alone: ``[T, H]`` partials in, this rank's contiguous
    ``[T/W, H]`` reduced block out) or ``"ag"`` (the all-gather phase alone:
    ``[T/W, H]`` in, ``[T, H]`` out). ``nbytes`` is the size of the full
    ``[T, H]`` tensor in both. They share the engine's inbox and colors with
    the all-reduce, take ``super_tile=1``, the 8-atom tile, the INT4 codec and
    no fused norm, and need ``nbytes`` to be a whole number of tiles.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    SP_MODE = mode != MODE_AR
    fuse_norm = norm_hidden is not None
    if SP_MODE and (fuse_norm or norm_latency or norm_mxfp4):
        raise ValueError(f"mode={mode!r} has no fused-norm epilogue")
    if norm_latency and (not fuse_norm or super_tile != 1 or block != BLOCK):
        raise ValueError("norm_latency needs norm_hidden, super_tile=1 and block=BLOCK")
    if norm_mxfp4 and not fuse_norm:
        raise ValueError("norm_mxfp4 needs norm_hidden")
    if fuse_norm and codec != "int4":
        raise ValueError(f"fused RMSNorm needs codec='int4', got {codec!r}")
    NORM_LATENCY = bool(norm_latency)
    NORM_MXFP4 = bool(norm_mxfp4)
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if inbox_memory not in _INBOX_POLICY:
        raise ValueError(
            f"inbox_memory must be one of {tuple(_INBOX_POLICY)}, got {inbox_memory!r}"
        )
    if codec not in MESH_CODECS:
        raise ValueError(f"codec must be one of {MESH_CODECS}, got {codec!r}")
    if block not in SUPPORTED_BLOCKS:
        raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
    if not 0 <= int(rank) < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    norm_atoms = tile_atoms_for(norm_hidden, world_size, block) if fuse_norm else None
    if fuse_norm and norm_atoms is None:
        raise ValueError(
            f"fused RMSNorm at block={block} and world_size={world_size} does "
            f"not support hidden={norm_hidden}"
        )
    if tile_atoms is None:
        tile_atoms = norm_atoms or DEFAULT_ATOMS
    if tile_atoms not in TILE_ATOMS:
        raise ValueError(f"tile_atoms must be one of {TILE_ATOMS}, got {tile_atoms}")
    if fuse_norm and tile_atoms != norm_atoms:
        raise ValueError(
            f"hidden={norm_hidden} needs a {norm_atoms}-atom tile, got {tile_atoms}"
        )
    tile_atoms = int(tile_atoms)
    c = codecs_for_block(block)[codec]
    tile_bytes = block * tile_atoms * 16
    quads_per_block = block // QUAD_LANES
    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    flag_policy = policy["flag"]
    release_scope = policy["release"]
    acquire_scope = policy["acquire"]
    recv_policy = policy["recv"]
    if tile_atoms % world_size != 0:
        raise ValueError(
            f"tile_atoms={tile_atoms} is not divisible by world_size={world_size}"
        )
    if super_tile not in SUPER_TILES:
        raise ValueError(f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}")
    if SP_MODE and (super_tile != 1 or tile_atoms != DEFAULT_ATOMS or codec != "int4"):
        raise ValueError(
            f"mode={mode!r} needs super_tile=1, the {DEFAULT_ATOMS}-atom tile and "
            f"codec='int4', got super_tile={super_tile!r}, tile_atoms={tile_atoms}, "
            f"codec={codec!r}"
        )
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    # Each rank owns this many atoms of a tile (8 GPUs → 1, 4 → 2, 2 → 4 at
    # the default 8 atoms).
    rank_atoms = tile_atoms // world_size
    # Last-sector pad is ST * rank_atoms * rank_tile_i32 after the ST tiles.
    rank_payload_i32 = rank_atoms * c.rank_tile_i32
    release_i32_off = super_tile * rank_payload_i32
    wire_tile_i32 = release_i32_off + 16
    wire_tile_bytes = wire_tile_i32 * 4

    # This rank's own index as a trace-time constant: its own slice of every
    # tile never leaves registers, and "all peers but me" is only expressible
    # over trace-time peer indices.
    self_rank = int(rank)
    # Destinations this rank pushes packets and flags to, in rank order. Every
    # per-destination structure below -- LDS pack rows, fanout quads, flag
    # lanes -- is indexed by position in this list, not by rank.
    push_peers = [p for p in range(world_size) if p != self_rank]
    n_push = len(push_peers)

    # A rank-tile's sectors, in stripes of up to 8, one quad per (destination,
    # sector) of a stripe, in a single pass.
    stripe_width = min(8, quads_per_block // n_push)
    stripes = [
        (b, min(stripe_width, c.n_sectors - b))
        for b in range(0, c.n_sectors, stripe_width)
    ]
    sector_fastest = policy["fanout"] != "peer"

    # One pack row per (destination, rank-atom).
    pack_rows = n_push * rank_atoms
    pack_i32 = pack_rows * c.rank_tile_i32
    lds_bytes = mesh_lds_bytes(
        world_size=world_size,
        block=block,
        codec=codec,
        tile_atoms=tile_atoms,
        fuse_norm=fuse_norm,
    )
    # One fp32 partial sum of squares per (wave, row) for the fused RMSNorm.
    norm_red_f32 = (block // WAVE) * tile_atoms
    if fuse_norm:
        PackStorage = make_pack_norm_storage(pack_i32, norm_red_f32)
        ATOMS_PER_ROW = (int(norm_hidden) * 2) // (block * 16)
        ROWS_PER_TILE = tile_atoms // ATOMS_PER_ROW
        SCALE_N = mxfp4_scale_shape(1, norm_hidden)[1]
        INV_HIDDEN = 1.0 / float(norm_hidden)
        # Atoms whose loads issue together. LLVM cannot hoist a load above
        # the previous atom's store (in place, residual is residual_out), so
        # this is the epilogue's memory-level parallelism; larger groups
        # spill under the pinned residency budget.
        NORM_GROUP = (
            tile_atoms
            if NORM_LATENCY
            else 2 if (world_size, super_tile) == (8, 1) else 1
        )
    else:
        PackStorage = make_pack_storage(pack_i32)

    # flags_i32 is also the i32 offset of the wire area, so the flag prefix has
    # to be a whole number of 64 B sectors (16 i32s). At a smaller multiple every
    # rank-tile and release sector straddles two hardware sectors, so the 64 B
    # fanout stores and the last-sector release stop being one sector wide.
    grid_multiple = 16 // (PHASES * world_size)
    if grid % grid_multiple != 0:
        raise ValueError(
            f"grid must be a multiple of {grid_multiple} at "
            f"world_size={world_size} to keep the wire area 64 B aligned, got "
            f"{grid}"
        )
    flags_i32 = PHASES * grid * world_size

    @flyc.kernel(known_block_size=[block, 1, 1])
    def quick_allreduce_int4(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        residual_ptr: Int64,
        residual_out_ptr: Int64,
        weight_ptr: Int64,
        mxfp4_ptr: Int64,
        mxfp4_scale_ptr: Int64,
        norm_eps: Float32,
        n_blocks: Int32,
        divisor: Float32 = Float32(DEFAULT_DIVISOR),  # noqa: B008
    ):
        _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        wave, lane = thread_lane(tid, block)
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad

        pack_layout = fx.make_layout((pack_rows, c.rank_tile_i32), (c.rank_tile_i32, 1))
        # 64 B NT sectors of one rank-tile: (sector, lane-in-quad) -> i32
        # start of the dwordx4. Isolated NT store stays explicit.
        nt_own_layout = fx.make_layout((c.n_sectors, QUAD_LANES), (16, 4))
        # Four group-16 scale bytes share the i32 slot eight threads already own.
        scale_slot, pair_in_slot = scale_slot_of(tid, block)
        wire_slot_layout = fx.make_layout(
            (PHASES, grid, world_size, super_tile),
            (
                grid * world_size * wire_tile_i32,
                world_size * wire_tile_i32,
                wire_tile_i32,
                rank_payload_i32,
            ),
        )

        lds = fx.SharedAllocator().allocate(PackStorage).peek()
        pack = lds.pack.view(pack_layout)
        smem_ptr = lds.pack.ptr

        peers = _load_peers(peer_ptrs, world_size)
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        inbox = _buffer_ptr(_to_sgpr_i64(peer_vec[rank]), T.i32, 4)

        def _push_base(j):
            """Inbox base of destination *j*, a lane-varying ``push_peers`` index."""

            base = peers[push_peers[0]]
            for i in range_constexpr(1, n_push):
                base = (j == fx.Int32(i)).select(peers[push_peers[i]], base)
            return base

        _load_atom_raw, _store_atom = _payload_io(
            inp_ptr,
            out_ptr,
            nbytes,
            num_tiles,
            tile_atoms,
            block,
            tid,
            decode=_atom_bf16_to_f16,
            encode=_atom_f16_to_bf16,
        )
        _load_color, _store_color = _color_io(colors_ptr, bid)
        if const_expr(SP_MODE):
            # ``nbytes`` is the full tensor in both modes; the all-reduce's
            # ``_payload_io`` above is unused here.
            _sp_load, _sp_store = _sp_payload_io(
                inp_ptr,
                out_ptr,
                nbytes,
                num_tiles,
                mode,
                world_size,
                rank_atoms,
                block,
                tid,
            )
        if const_expr(fuse_norm):
            res_rsrc = buffer_ops.create_buffer_resource_from_addr(
                residual_ptr, num_records_bytes=nbytes
            )
            res_out_rsrc = buffer_ops.create_buffer_resource_from_addr(
                residual_out_ptr, num_records_bytes=nbytes
            )
            out_rsrc = buffer_ops.create_buffer_resource_from_addr(
                out_ptr, num_records_bytes=nbytes
            )
            weight_rsrc = buffer_ops.create_buffer_resource_from_addr(
                weight_ptr, num_records_bytes=int(norm_hidden) * 2
            )
            red_ptr = lds.red.ptr
            lane_i32 = tid * 4
        if const_expr(NORM_MXFP4):
            mxfp4_rsrc = buffer_ops.create_buffer_resource_from_addr(
                mxfp4_ptr, num_records_bytes=nbytes // 4
            )
            n_rows = nbytes // fx.Int64(int(norm_hidden) * 2)
            mxfp4_scale_rsrc = buffer_ops.create_buffer_resource_from_addr(
                mxfp4_scale_ptr,
                num_records_bytes=(n_rows + 255) // 256 * fx.Int64(256 * SCALE_N),
            )
            # Lane part of the shuffled scale index for group ``tid // 4`` of
            # an atom (see mx_scale_shuffle_idx); the atom's group base is a
            # multiple of 8, so the row and atom parts fold into soffset.
            g = tid // fx.Int32(MXFP4_GROUP // 8)
            scale_lane = (g // 8) * 256 + (g % 4) * 64 + ((g % 8) // 4) * 2

        def _pack_off(peer, i32_idx):
            return fx.get_scalar(fx.crd2idx((peer, i32_idx), pack_layout))

        def _sub_tile_i32(phase, src, sub):
            slot = fx.get_scalar(
                fx.crd2idx((fx.Int32(phase), bid, src, sub), wire_slot_layout)
            )
            return fx.Int32(flags_i32) + slot

        def _load_tile_atoms(tile):
            return [_load_atom(tile, atom) for atom in range_constexpr(tile_atoms)]

        # One shared divisor. Every read of the input is scaled, so the group
        # max and scale byte of the first pack see the scaled values, and so
        # does our own share, which stays in registers or is reloaded. The
        # second pack's input is already a scaled sum, so _codec_quant must
        # not scale. A packed fp16 divide expands per element, so multiply by
        # a reciprocal taken once.
        inv_div_h = _splat_f16x2((fx.Float32(1.0) / divisor).to(fx.Float16))
        div_h = _splat_f16x2(divisor.to(fx.Float16))

        def _load_atom(tile, atom):
            raw = _load_atom_raw(tile, atom)
            if const_expr(not c.has_scale):
                return raw
            return fx.Vector.from_elements(
                [_i32(_f16x2(raw[j]) * inv_div_h) for j in range_constexpr(4)],
                fx.Int32,
            )

        def _ag_scale(scale):
            """The all-gather dequant scale, with the divisor multiply folded in:
            one packed multiply per packet instead of one per fp16 pair."""
            return scale * div_h if const_expr(c.has_scale) else scale

        def _store_tile_atoms(tile, atoms):
            """Store a gathered tile; ``None`` marks atoms already stored."""
            for atom in range_constexpr(tile_atoms):
                if const_expr(atoms[atom] is not None):
                    _store_atom(tile, atom, atoms[atom])

        def _own_atoms(atoms):
            """This rank's reduce-scatter share, out of a whole tile's atoms."""
            return atoms[self_rank * rank_atoms : (self_rank + 1) * rank_atoms]

        def _add_f16(a, b):
            """Packed fp16 ``a + b`` of two atoms -- the codec's own FP16 add."""
            return fx.Vector.from_elements(
                [_i32(_f16x2(a[i]) + _f16x2(b[i])) for i in range_constexpr(4)],
                fx.Int32,
            )

        def _atom_soff(tile, atom):
            return tile * fx.Int32(tile_bytes) + fx.Int32(atom * block * 16)

        def _load_atom_bf16(rsrc, soff):
            return fx.Vector(
                buffer_ops.buffer_load(
                    rsrc,
                    lane_i32,
                    vec_width=4,
                    dtype=T.i32,
                    mask=fx.Int64(soff) < nbytes,
                    soffset_bytes=soff,
                )
            )

        def _store_atom_bf16(rsrc, soff, packed):
            buffer_ops.buffer_store(
                packed,
                rsrc,
                lane_i32,
                mask=fx.Int64(soff) < nbytes,
                soffset_bytes=soff,
            )

        def _store_mxfp4_atom(tile, atom, yb):
            """MXFP4-quantize one atom of bf16 ``out``.

            The 32-element group spans four consecutive lanes. The round-up
            e8m0 scale and the hardware scaled convert match
            ``per_1x32_f4_quant_hip`` bit for bit.
            """
            amax = abs(yb.to(fx.Float32)).reduce(ReductionOp.MAX)
            for off in range_constexpr(2):
                amax = fx.maxnumf(amax, amax.shuffle_xor(1 << off, WAVE))
            e8m0 = fx.Int32(emit_mx_e8m0_scale(amax))
            scale = as_ir_value((e8m0 << 23).bitcast(fx.Float32))
            words = yb.bitcast(fx.Int32)
            pk = as_ir_value(fx.Int32(0))
            for j in range_constexpr(4):
                pk = rocdl.cvt_scalef32_pk_fp4_bf16(
                    T.i32,
                    pk,
                    as_ir_value(
                        fx.Vector.from_elements([words[j]], fx.Int32).bitcast(
                            fx.BFloat16
                        )
                    ),
                    scale,
                    j,
                )
            live = fx.Int64(_atom_soff(tile, atom)) < nbytes
            buffer_ops.buffer_store(
                fx.Int32(pk),
                mxfp4_rsrc,
                tid,
                mask=live,
                soffset_bytes=tile * fx.Int32(tile_bytes // 4)
                + fx.Int32(atom * block * 4),
            )
            # The group's four lanes store the same byte.
            x = tile * fx.Int32(ROWS_PER_TILE) + fx.Int32(atom // ATOMS_PER_ROW)
            scale_row = (
                (x // fx.Int32(32)) * fx.Int32(32 * SCALE_N)
                + (x % fx.Int32(16)) * fx.Int32(4)
                + (x % fx.Int32(32)) // fx.Int32(16)
            )
            groups_per_atom = block * 8 // MXFP4_GROUP
            buffer_ops.buffer_store(
                arith.trunci(T.i8, as_ir_value(e8m0)),
                mxfp4_scale_rsrc,
                scale_lane,
                mask=live,
                soffset_bytes=scale_row
                + fx.Int32((atom % ATOMS_PER_ROW) * groups_per_atom // 8 * 256),
            )

        def _prefetch_residual(tile):
            return [
                _load_atom_bf16(res_rsrc, _atom_soff(tile, atom))
                for atom in range_constexpr(tile_atoms)
            ]

        def _norm_store_tile_atoms(tile, sub, res_pref, own):
            """Fused add + RMSNorm epilogue over one reduced tile.

            Atom *a* holds row ``a // ATOMS_PER_ROW`` of the tile, so the row
            of every register is a compile-time constant. A row is a whole
            number of atoms, so in the partial last tile each atom is either
            live or past the payload; the uniform mask drops the dead ones.

            The atom's byte offset goes in ``soffset`` (SGPR) and every lane
            shares one ``tid * 16`` VGPR offset. Per-atom VGPR offsets would
            be loop-invariant, so LLVM hoists them and keeps them live through
            the reduce-scatter and all-gather phases.

            Like vLLM ``fused_add_rms_norm``, the sum of squares uses the fp32
            sum and the output normalizes the bf16 ``residual_out``.

            The second pass re-reads ``residual_out`` instead of keeping the
            tile in registers across the reduction barrier: each lane reads
            back only the 16 B it stored, so program order makes it visible.

            Each atom's all-gather packet is received inside the first pass.
            Receiving the whole tile first, as the plain store does, keeps
            all eight dequantized atoms live under the residual loads. Each
            row's partial is reduced and parked in LDS as soon as the row is
            complete, and its rstd is formed when the second pass reaches it,
            so one row's state is live at a time.

            Our own chunk is *own*, the registers it was decoded into, or
            ``None`` where the reduce-scatter loop already stored it to
            ``out``, which is read back here before the norm overwrites it.

            The ``norm_latency`` build instead takes the residual prefetched
            before the reduce-scatter (*res_pref*) and keeps the tile in
            registers between the passes.
            """
            sq = fx.Float32(0.0)
            xbs = []
            for g in range_constexpr(0, tile_atoms, NORM_GROUP):
                ars = []
                ress = []
                for atom in range_constexpr(g, g + NORM_GROUP):
                    src, k = atom // rank_atoms, atom % rank_atoms
                    if const_expr(src != self_rank):
                        ars.append(
                            _recv_quantized(PHASE_ALL_GATHER, fx.Int32(src), sub, k)
                        )
                    elif const_expr(own is not None):
                        ars.append(own[k])
                    else:
                        ars.append(_load_atom_bf16(out_rsrc, _atom_soff(tile, atom)))
                    if const_expr(NORM_LATENCY):
                        ress.append(res_pref[atom])
                    else:
                        ress.append(_load_atom_bf16(res_rsrc, _atom_soff(tile, atom)))
                for i in range_constexpr(NORM_GROUP):
                    atom = g + i
                    if const_expr(atom // rank_atoms != self_rank):
                        words, scale = ars[i]
                        ar = (
                            fx.Vector(_codec_dequant(c, words, _ag_scale(scale), tid))
                            .bitcast(fx.Float16)
                            .to(fx.Float32)
                        )
                    elif const_expr(own is not None):
                        ar = fx.Vector(ars[i]).bitcast(fx.Float16).to(fx.Float32)
                    else:
                        ar = ars[i].bitcast(fx.BFloat16).to(fx.Float32)
                    x = ar + ress[i].bitcast(fx.BFloat16).to(fx.Float32)
                    xb = x.to(fx.BFloat16).bitcast(fx.Int32)
                    _store_atom_bf16(res_out_rsrc, _atom_soff(tile, atom), xb)
                    if const_expr(NORM_LATENCY):
                        xbs.append(xb)
                    sq = sq + (x * x).reduce(ReductionOp.ADD)
                    if const_expr((atom + 1) % ATOMS_PER_ROW == 0):
                        row = atom // ATOMS_PER_ROW
                        for off in range_constexpr(6):
                            sq = sq + sq.shuffle_xor(1 << (5 - off), WAVE)
                        if lane == 0:
                            fx.ptr_store(
                                sq,
                                red_ptr + (wave * fx.Int32(tile_atoms) + fx.Int32(row)),
                            )
                        sq = fx.Float32(0.0)
            gpu.barrier()

            for g in range_constexpr(0, tile_atoms, NORM_GROUP):
                xs = []
                ws = []
                for atom in range_constexpr(g, g + NORM_GROUP):
                    if const_expr(NORM_LATENCY):
                        xs.append(xbs[atom])
                    else:
                        xs.append(_load_atom_bf16(res_out_rsrc, _atom_soff(tile, atom)))
                    # bf16 column (atom % ATOMS_PER_ROW) * block * 8 + tid * 8.
                    ws.append(
                        fx.Vector(
                            buffer_ops.buffer_load(
                                weight_rsrc,
                                lane_i32,
                                vec_width=4,
                                dtype=T.i32,
                                soffset_bytes=(atom % ATOMS_PER_ROW) * block * 16,
                            )
                        )
                    )
                for i in range_constexpr(NORM_GROUP):
                    atom = g + i
                    row = atom // ATOMS_PER_ROW
                    if const_expr(atom % ATOMS_PER_ROW == 0):
                        total = fx.Float32(fx.ptr_load(red_ptr + row))
                        for w in range_constexpr(1, block // WAVE):
                            total = total + fx.Float32(
                                fx.ptr_load(red_ptr + (w * tile_atoms + row))
                            )
                        rstd = fmath.rsqrt(total * fx.Float32(INV_HIDDEN) + norm_eps)
                    x = xs[i].bitcast(fx.BFloat16).to(fx.Float32)
                    w = ws[i].bitcast(fx.BFloat16).to(fx.Float32)
                    if const_expr(gemma_norm):
                        w = w + fx.full(8, 1.0, fx.Float32)
                    y = x * w * fx.full(8, rstd, fx.Float32)
                    yb = y.to(fx.BFloat16)
                    _store_atom_bf16(
                        out_rsrc, _atom_soff(tile, atom), yb.bitcast(fx.Int32)
                    )
                    if const_expr(NORM_MXFP4):
                        _store_mxfp4_atom(tile, atom, yb)
            # The next tile's partials reuse ``red``.
            gpu.barrier()

        def _epilogue(tile, sub, res_pref=None, own=None):
            if const_expr(fuse_norm):
                _norm_store_tile_atoms(tile, sub, res_pref, own)
            else:
                _store_tile_atoms(tile, _recv_all_gather(sub, own))

        def _lds_write_packet(slot, words, scale, is_leader):
            for (off, pred), word in zip(c.plane_slots(tid), words):
                if pred:
                    fx.memref_store(word, pack, (slot, off))
            if const_expr(c.has_scale):  # noqa: SIM102
                if is_leader:
                    fx.memref_store(
                        scale, pack, (slot, fx.Int32(c.scale_i32_off) + scale_slot)
                    )

        def _pack_reduce_scatter(atoms):
            """Quantize each destination's slice of this tile into LDS.

            *atoms* is this thread's share of the whole tile, as loaded by
            ``_load_tile_atoms``: always ``tile_atoms`` 16 B atoms. Destination *d*
            owns ``atoms[d * rank_atoms : (d+1) * rank_atoms]``. Those packets
            are later NT-stored into *d*'s reduce-scatter inbox. Our own
            slice is never packed: it stays in registers.
            """
            for j, dest in enumerate(push_peers):
                for k in range_constexpr(rank_atoms):
                    words, scale, is_leader = _codec_quant(
                        c, atoms[dest * rank_atoms + k], lane, tid
                    )
                    _lds_write_packet(
                        fx.Int32(j * rank_atoms + k), words, scale, is_leader
                    )

        def _pack_all_gather(accs):
            """Quantize the reduced slice and replicate it for every peer.

            After reduce-scatter this rank holds ``rank_atoms`` reduced
            atoms. Copy the same packets into every destination slot so the
            NT fanout can push them into every peer's all-gather inbox, and
            return our own chunk decoded from those same packets.
            """
            own = []
            for k in range_constexpr(rank_atoms):
                words, scale, is_leader = _codec_quant(c, accs[k], lane, tid)
                for j in range_constexpr(n_push):
                    _lds_write_packet(
                        fx.Int32(j * rank_atoms + k), words, scale, is_leader
                    )
                # The same scale arithmetic as a peer's decode of this packet,
                # so every rank writes identical bits.
                own.append(
                    _codec_dequant(
                        c,
                        words,
                        _ag_scale(_scale_from_word(c, scale, pair_in_slot)),
                        tid,
                    )
                )
            return own

        def _fanout_nt(phase, inbox_src, sub):
            """NT-store one rank-tile from LDS to every destination's inbox.

            Lockstep stripes of up to 8 sectors cover the rank-tile: at the
            default block INT4 is 8+8+2 (16 nibble sectors then the 2-sector
            scale tail), fp16 is eight full stripes. ``sector_base`` is the first
            sector of each stripe.

            One quad per (destination, sector) of a stripe; leftover quads sit
            idle. Which axis runs fastest across consecutive quads is a fabric
            question.
                - "sector": consecutive quads target consecutive peers of
                  one sector, so a single store instruction hits every GPU
                  -- ideal on xGMI, whose native packet is exactly the 64 B
                  a quad writes.
                - "peer": consecutive quads walk the sectors of one peer,
                  giving each destination a ``64 * width`` B contiguous run
                  -- ideal on PCIe.
            """
            for k in range_constexpr(rank_atoms):
                for sector_base, width in stripes:
                    n_quads = fx.Int32(n_push * width)
                    safe = (quad_id < n_quads).select(quad_id, fx.Int32(0))
                    if const_expr(sector_fastest):
                        # sector fastest
                        j = safe % fx.Int32(n_push)
                        sector_in_stripe = safe // fx.Int32(n_push)
                    else:
                        # peer fastest
                        j = safe // fx.Int32(width)
                        sector_in_stripe = safe % fx.Int32(width)
                    sector = fx.Int32(sector_base) + sector_in_stripe
                    if quad_id < n_quads:
                        vec_idx = fx.get_scalar(
                            fx.crd2idx((sector, lane_in_quad), nt_own_layout)
                        )
                        pack_row = j
                        wire_idx = vec_idx
                        if const_expr(rank_atoms != 1):
                            pack_row = j * fx.Int32(rank_atoms) + fx.Int32(k)
                            wire_idx = vec_idx + fx.Int32(k * c.rank_tile_i32)
                        # 4xi32 NT vector cannot go through the i32 pack view.
                        v4 = fx.ptr_load(
                            smem_ptr + _pack_off(pack_row, vec_idx),
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                        byte_off = _i32_to_bytes(
                            _sub_tile_i32(phase, inbox_src, sub) + wire_idx
                        )
                        _store_v4i32_peer(_push_base(j) + byte_off, v4, payload_policy)

        def _publish(phase, inbox_src, color):
            """Drain payload NT stores, then write *color* into every peer inbox.

            Last 64 B of this rank's slot (after the ST rank-tiles) is the
            handshake: 16 i32s all equal to *color*. Peers spin on that
            sector in their copy of our slot; seeing *color* means our
            payload is visible.

            ``vmcnt(0)``: this 64-lane wave's NT payload stores are done.
            The workgroup barrier: the other three 64-lane waves issued
            payload too; ``vmcnt`` is per-wave, so without the join a
            wave-0 handshake could race stores still in flight. Neither
            can move after the color store, and neither can be dropped.

            On a cacheable inbox retiring the stores is not enough -- they
            can be sitting in this XCD's L2. The release fence after the join
            writes them back (``buffer_wbl2``) and waits for that to land
            before the flag goes out. Every workgroup issues its own: L2 is
            per-XCD.

            ``FLAG_LANES`` lanes per destination, 8 B each, so at most 64
            lanes: the whole handshake is one store instruction from wave 0.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if const_expr(release_scope is not None):
                _release_inbox(release_scope)
            limit = fx.Int32(n_push)
            dest = tid // fx.Int32(FLAG_LANES)
            safe = (dest < limit).select(dest, fx.Int32(0))
            if dest < limit:
                elem = (
                    _sub_tile_i32(phase, inbox_src, fx.Int32(0))
                    + fx.Int32(release_i32_off)
                    + (tid % fx.Int32(FLAG_LANES)) * fx.Int32(FLAG_I32_PER_LANE)
                )
                _store_flag_peer(
                    _push_base(safe) + _i32_to_bytes(elem), color, flag_policy
                )

        def _wait_flag(flag, color):
            # No fence in the loop body: _load_flag carries `sc0 sc1`, so a
            # retry cannot be served from a stale line. The acquire that
            # orders the payload reads is in _wait_release, once, after the
            # join.
            current = _load_flag(flag)
            while current != color:
                current = _load_flag(flag)

        def _wait_release(phase, color):
            # Lane ``t`` watches one source. Our own flag is never published,
            # so the N-1 lanes step over our own index.
            spin_src = tid
            if tid >= fx.Int32(self_rank):
                spin_src = tid + fx.Int32(1)
            if tid < n_push:
                elem = _sub_tile_i32(phase, spin_src, fx.Int32(0)) + fx.Int32(
                    release_i32_off
                )
                _wait_flag(peer_vec[rank] + _i32_to_bytes(elem), color)
            gpu.barrier()
            # Unconditional and after the join. Only `tid < n_push` spun,
            # so scoping the acquire to the spin would leave the other waves
            # of this workgroup reading the payload unordered after it -- and
            # would also skip it entirely in the common case where the flag is
            # already set on the first read.
            #
            # The scope is the inbox policy's. On a fine-grained inbox it is
            # system scope, an L1+L2 invalidate, which the `nt` payload loads
            # rely on. On an uncached inbox it is workgroup scope, which
            # invalidates nothing and is only a compiler barrier; that is safe
            # because the policy pairs it with `sc0 sc1` payload loads
            # (`recv_policy`), which bypass both caches.
            _acquire_inbox(acquire_scope)

        def _recv_quantized(phase, src, sub, k=0):
            base = _sub_tile_i32(phase, src, sub)
            if const_expr(k):
                base = base + fx.Int32(k * c.rank_tile_i32)

            def _get(off):
                return _buffer_load(inbox, base + off, 1, fx.Int32, recv_policy)[0]

            words, word = _codec_load(c, _get, tid, scale_slot)
            return words, _scale_from_word(c, word, pair_in_slot)

        def _reduce_scattered(sub, own):
            """Dequant-accumulate every peer's reduce-scatter packet for *sub*,
            and *own*, our share from registers, in rank order."""
            accs = [None] * rank_atoms
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    if const_expr(src == self_rank):
                        if const_expr(accs[k] is None):
                            accs[k] = own[k]
                        else:
                            accs[k] = _add_f16(own[k], accs[k])
                    else:
                        words, scale = _recv_quantized(
                            PHASE_REDUCE_SCATTER, fx.Int32(src), sub, k
                        )
                        if const_expr(accs[k] is None):
                            accs[k] = _codec_dequant(c, words, scale, tid)
                        else:
                            accs[k] = _codec_dequant(c, words, scale, tid, accs[k])
            return accs

        def _recv_all_gather(sub, own=None):
            """Dequantize every peer's all-gather packet back into full-tile atoms.

            Our own chunk is *own*, or ``None`` where it has been stored already.
            Both are already multiplied back by the divisor, which is folded into
            each packet's group scale.
            """
            gathered = []
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    if const_expr(src == self_rank):
                        gathered.append(None if own is None else own[k])
                    else:
                        words, scale = _recv_quantized(
                            PHASE_ALL_GATHER, fx.Int32(src), sub, k
                        )
                        gathered.append(_codec_dequant(c, words, _ag_scale(scale), tid))
            return gathered

        def _own_packets(atoms):
            """Our own share of *atoms*, quantized as a peer's packet of it is.

            ``mode="rs"`` only. The all-reduce adds its own share exactly; here
            it goes through the codec like every other partial, so a row's sum
            does not depend on which rank owns it. The packet stays in
            registers: our own inbox slots are never written.
            """
            own = []
            for k in range_constexpr(rank_atoms):
                words, scale, _ = _codec_quant(
                    c, atoms[self_rank * rank_atoms + k], lane, tid
                )
                own.append((words, _scale_from_word(c, scale, pair_in_slot)))
            return own

        def _reduce_scattered_sp(own):
            """Dequant-accumulate every rank's packet of this tile, in rank order.

            A peer's packet arrives in the all-gather slots (``mode="rs"`` sends
            through them, module docstring); ours is *own*, from
            ``_own_packets``, accumulated by the same dequant so the arithmetic
            does not depend on where a packet came from.
            """
            accs = [None] * rank_atoms
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    if const_expr(src == self_rank):
                        words, scale = own[k]
                    else:
                        words, scale = _recv_quantized(
                            PHASE_ALL_GATHER, fx.Int32(src), fx.Int32(0), k
                        )
                    if const_expr(accs[k] is None):
                        accs[k] = _codec_dequant(c, words, scale, tid)
                    else:
                        accs[k] = _codec_dequant(c, words, scale, tid, accs[k])
            return accs

        # Stride by the *launched* grid, not the compile-time cap. The host
        # launches fewer blocks than `grid` whenever it wants each block to own
        # several tiles (see QuickAllReduceInt4._grid_x); striding by the cap instead would
        # silently leave every tile above n_blocks unprocessed. `grid` still
        # sizes the wire slots and colour array, so n_blocks <= grid always.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        color = _load_color()
        if const_expr(SP_MODE):
            # One tile per round, ``ready`` first (module docstring). It is
            # published after the tile's loads have landed, so the previous
            # round's HBM stores do not delay it, and awaited only before the
            # first payload store, so its flight overlaps the quantize.
            for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
                tile = bid + i * n_blocks
                if const_expr(mode == MODE_RS):
                    atoms = [
                        _sp_load(a // rank_atoms, tile, a % rank_atoms)
                        for a in range_constexpr(tile_atoms)
                    ]
                else:
                    atoms = [_sp_load(tile, k) for k in range_constexpr(rank_atoms)]
                _publish(PHASE_REDUCE_SCATTER, rank, color)
                if const_expr(mode == MODE_RS):
                    _pack_reduce_scatter(atoms)
                    own = _own_packets(atoms)
                else:
                    own = _pack_all_gather(atoms)
                gpu.barrier()
                _wait_release(PHASE_REDUCE_SCATTER, color)
                _fanout_nt(PHASE_ALL_GATHER, rank, fx.Int32(0))
                _publish(PHASE_ALL_GATHER, rank, color)

                _wait_release(PHASE_ALL_GATHER, color)
                if const_expr(mode == MODE_RS):
                    for k, acc in enumerate(_reduce_scattered_sp(own)):
                        _sp_store(acc, tile, k)
                else:
                    for a, atom in enumerate(_recv_all_gather(fx.Int32(0), own)):
                        _sp_store(atom, a // rank_atoms, tile, a % rank_atoms)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        elif const_expr(super_tile == 1):
            for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
                tile = bid + i * n_blocks
                res_pref = None
                if const_expr(NORM_LATENCY):
                    res_pref = _prefetch_residual(tile)
                atoms = _load_tile_atoms(tile)
                _pack_reduce_scatter(atoms)
                gpu.barrier()
                _fanout_nt(PHASE_REDUCE_SCATTER, rank, fx.Int32(0))
                _publish(PHASE_REDUCE_SCATTER, rank, color)

                _wait_release(PHASE_REDUCE_SCATTER, color)
                # ST=1 keeps the whole tile in registers across the wait, so
                # our share is simply read back out of it.
                acc = _reduce_scattered(fx.Int32(0), _own_atoms(atoms))

                own_ag = _pack_all_gather(acc)
                gpu.barrier()
                _fanout_nt(PHASE_ALL_GATHER, rank, fx.Int32(0))
                _publish(PHASE_ALL_GATHER, rank, color)

                _wait_release(PHASE_ALL_GATHER, color)
                _epilogue(tile, fx.Int32(0), res_pref, own_ag)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        else:
            st_i = fx.Int32(super_tile)
            for i in range(fx.Int32(0), n_block_tiles, st_i):
                remain = n_block_tiles - i
                n_this = (remain < st_i).select(remain, st_i)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    atoms = _load_tile_atoms(tile)
                    _pack_reduce_scatter(atoms)
                    gpu.barrier()
                    _fanout_nt(PHASE_REDUCE_SCATTER, rank, s)
                    if (s + fx.Int32(1)) < n_this:
                        # Drain this wave's LDS loads, then join the WG.
                        # world_size<8 leaves waves idle in fanout; without the
                        # barrier they pack the next sub-tile into LDS while
                        # a busy wave still ptr_loads it. lgkmcnt only: NT
                        # payload stays in flight until _publish.
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()

                _publish(PHASE_REDUCE_SCATTER, rank, color)
                _wait_release(PHASE_REDUCE_SCATTER, color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    # Nothing is carried from the first loop -- that would be ST
                    # tiles of registers across the wait -- so our share is
                    # reloaded from the input, a local cached read against the
                    # uncached inbox read it replaces. Likewise our reduced
                    # chunk is stored now rather than carried to the third loop;
                    # the all-gather publish below releases it.
                    tile = bid + (i + s) * n_blocks
                    own_rs = [
                        _load_atom(tile, self_rank * rank_atoms + k)
                        for k in range_constexpr(rank_atoms)
                    ]
                    acc = _reduce_scattered(s, own_rs)
                    own_ag = _pack_all_gather(acc)
                    for k in range_constexpr(rank_atoms):
                        _store_atom(tile, self_rank * rank_atoms + k, own_ag[k])
                    gpu.barrier()
                    _fanout_nt(PHASE_ALL_GATHER, rank, s)
                    if (s + fx.Int32(1)) < n_this:
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()

                _publish(PHASE_ALL_GATHER, rank, color)
                _wait_release(PHASE_ALL_GATHER, color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * n_blocks
                    _epilogue(tile, s)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{block},{block}"
    value_attrs = {"rocdl.flat_work_group_size": flat_wg}
    if NORM_LATENCY:
        value_attrs["rocdl.waves_per_eu"] = NORM_LATENCY_WGS_PER_CU
    elif fuse_norm or SP_MODE:
        # The host sizes the persistent grid from the plain kernel's
        # residency. One wave per SIMD per workgroup at BLOCK, so pin the
        # fused epilogue and the single-phase modes to the same VGPR budget.
        value_attrs["rocdl.waves_per_eu"] = _resident_wgs_per_cu(world_size, super_tile)

    @flyc.jit
    def launch_quick_allreduce_int4(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        residual_ptr: Int64,
        residual_out_ptr: Int64,
        weight_ptr: Int64,
        mxfp4_ptr: Int64,
        mxfp4_scale_ptr: Int64,
        norm_eps: Float32,
        grid_x: Int32,
        divisor: Float32 = DEFAULT_DIVISOR,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        quick_allreduce_int4(
            rank,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            peer_ptrs,
            colors_ptr,
            residual_ptr,
            residual_out_ptr,
            weight_ptr,
            mxfp4_ptr,
            mxfp4_scale_ptr,
            norm_eps,
            grid_x,
            divisor,
            value_attrs=value_attrs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    tag = f"{SCALE_CODEC_TAG}_ws{world_size}_r{self_rank}_st{super_tile}_g{grid}"
    tag += f"_{inbox_memory}_{codec}_b{block}"
    if SP_MODE:
        tag += f"_{mode}"
    if tile_atoms != DEFAULT_ATOMS:
        tag += f"_a{tile_atoms}"
    if fuse_norm:
        tag += f"_rms{norm_hidden}{'g' if gemma_norm else ''}"
        tag += f"{'_lat' if NORM_LATENCY else ''}{'_mxfp4' if NORM_MXFP4 else ''}"
    launch_quick_allreduce_int4.func.__name__ = f"launch_quick_allreduce_int4_{tag}"
    try:
        quick_allreduce_int4.func.__name__ = f"quick_allreduce_int4_{tag}"
    except AttributeError:
        pass
    return {
        "launch": launch_quick_allreduce_int4,
        "flags_bytes": flags_i32 * 4,
        "data_bytes": PHASES * grid * world_size * wire_tile_bytes,
        "lds_bytes": lds_bytes,
        "tile_atoms": tile_atoms,
        "tile_bytes": tile_bytes,
        "tile_fp16": tile_bytes // 2,
        "rank_tile_bytes": c.rank_tile_bytes,
        "wire_tile_bytes": wire_tile_bytes,
        "super_tile": super_tile,
        "world_size": world_size,
        "mode": mode,
        "inbox_memory": inbox_memory,
        "codec": codec,
        "payload_policy": payload_policy,
        "flag_policy": flag_policy,
        "release_scope": release_scope,
        "rank_atoms": rank_atoms,
        "grid": grid,
        "block": block,
        "rank": self_rank,
    }
