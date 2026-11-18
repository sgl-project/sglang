"""Where each DWDP group rank's slice of a layer's weights sits in its shard and in a gather slot."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import msgspec

# 256 B for bf16, so parameter views keep the alignment cuBLAS and TMA expect.
ALIGN = 128


def _align(numel: int) -> int:
    return numel + (-numel) % ALIGN


class ParamSpec(msgspec.Struct, frozen=True):
    shape: Tuple[int, int]
    slot_offset: int
    # > 0: each of this many row blocks is cut N ways; 0: the columns are cut N ways.
    row_blocks: int


class Piece(msgspec.Struct, frozen=True):
    """This rank's part of one row block, or its column slice, of a parameter."""

    param: int
    block: int
    shard_offset: int
    rows: int
    cols: int


class CopyOp(msgspec.Struct, frozen=True):
    src_rank: int
    src_offset: int
    dst_offset: int
    rows: int
    cols: int
    # Elements between destination rows; 0 for a contiguous copy.
    dst_pitch: int


class UnitLayout:
    """One layer's parameters, as ``(shape, row_blocks)``, split over ``group_size`` ranks."""

    def __init__(self, params: Sequence[Tuple[Tuple[int, int], int]], group_size: int):
        self.group_size = group_size
        self.specs: List[ParamSpec] = []
        self.slot_numel = 0
        for shape, row_blocks in params:
            self.specs.append(ParamSpec(tuple(shape), self.slot_numel, row_blocks))
            self.slot_numel = _align(self.slot_numel + shape[0] * shape[1])
        self.pieces: List[Piece] = []
        self.shard_numel = 0
        for i, spec in enumerate(self.specs):
            rows, cols = spec.shape
            if spec.row_blocks:
                if rows % (spec.row_blocks * group_size):
                    raise ValueError(
                        f"{rows} rows do not split into {spec.row_blocks} blocks x "
                        f"{group_size} ranks"
                    )
                dims = [
                    (b, rows // (spec.row_blocks * group_size), cols)
                    for b in range(spec.row_blocks)
                ]
            else:
                if cols % group_size:
                    raise ValueError(
                        f"{cols} columns do not split over {group_size} ranks"
                    )
                dims = [(0, rows, cols // group_size)]
            for block, piece_rows, piece_cols in dims:
                self.pieces.append(
                    Piece(i, block, self.shard_numel, piece_rows, piece_cols)
                )
                self.shard_numel = _align(self.shard_numel + piece_rows * piece_cols)

    def first_row(self, piece: Piece, rank: int) -> int:
        """The parameter row where ``rank``'s copy of a row-block piece starts."""
        return (piece.block * self.group_size + rank) * piece.rows

    def copy_ops(self, rank: int) -> List[CopyOp]:
        n = self.group_size
        ops = []
        # Rotated peer order so no source serves two readers at once; own pieces last.
        for src in [(rank + i) % n for i in range(1, n)] + [rank]:
            for pc in self.pieces:
                spec = self.specs[pc.param]
                if spec.row_blocks:
                    dst, pitch = spec.slot_offset + self.first_row(pc, src) * pc.cols, 0
                else:
                    dst, pitch = spec.slot_offset + src * pc.cols, spec.shape[1]
                ops.append(CopyOp(src, pc.shard_offset, dst, pc.rows, pc.cols, pitch))
        return ops


def shard_group_layout(
    ranks: Sequence[int], attn_tp: int, group_size: int, tp_rank: int
) -> Tuple[List[List[int]], int, int]:
    """(every DWDP group's member ranks, this rank's group index, its rank in the group)."""
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
