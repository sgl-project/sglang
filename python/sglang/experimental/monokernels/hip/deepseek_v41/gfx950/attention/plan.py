# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 geometry and scratch layout."""

from dataclasses import dataclass

HIDDEN = 5120
Q_RANK = 1280
KV_DIM = 512
QKV_ROWS = Q_RANK + KV_DIM  # 1792
HEAD_DIM = 512
NOPE = 448
ROPE = 64
HALF = ROPE // 2
N_HEADS = 64
O_GROUPS = 8
O_RANK = 1024
WINDOW = 128
TOPK = 512
EPS = 1e-20

ROWS = 16  # GEMV rows a task
TILE = 16  # tokens a GEMV pass (an MFMA's N)
HEAD_TILE = 16  # heads an attention MFMA takes
WOA_ROWS = 32  # a wo_a task's rows: one MX group of its output

# KV record (V4 fp8_ds_mla): 576 B rows (448 e4m3, 64 bf16), then 8 B of scales a token
DATA = 576

PAIR = 8  # a mailbox (value, tag) pair
KEYS = TOPK + WINDOW  # a token's attention keys at most


def cdiv(a, b):
    return -(-a // b)


def round_up(a, b):
    return cdiv(a, b) * b


@dataclass(frozen=True)
class Dims:
    tp: int

    def __post_init__(self):
        assert self.tp in (2, 4), self.tp

    @property
    def heads(self) -> int:
        return N_HEADS // self.tp

    @property
    def head_tiles(self) -> int:
        return self.heads // HEAD_TILE

    @property
    def groups(self) -> int:
        return O_GROUPS // self.tp

    @property
    def group_k(self) -> int:
        """wo_a's K a group: its eight heads' outputs."""
        return self.heads // self.groups * HEAD_DIM  # 4096

    @property
    def o_rows(self) -> int:
        return self.groups * O_RANK

    @property
    def q_rows(self) -> int:
        return self.heads * HEAD_DIM

    @property
    def wqb_tasks(self) -> int:
        return self.q_rows // ROWS

    @property
    def woa_tasks(self) -> int:
        return self.o_rows // WOA_ROWS

    @property
    def wob_tasks(self) -> int:
        return HIDDEN // ROWS  # 320


WQKV_TASKS = QKV_ROWS // ROWS  # 112
WQKV_KPARTS = 2  # X8's K halves (a flag each); an LDS window of the wqkv GEMV


def wqkv_order(s):
    """Return SGLang's wqkv_a K order at width s as (chains, halves, sum order)."""
    if s <= 32:
        w = 16 if s <= 4 else 8
        return w, False, tuple(range(w))
    assert s <= 64
    return 5, True, (0, 4, 1, 2, 3)


def token_tiles(s):
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


def tile_rows(s):
    return min(s, TILE)


def byte_layout(items, start=0, align=256):
    out, off = {}, start
    for name, nbytes in items:
        off = round_up(off, align)
        out[name] = (off, nbytes)
        off += nbytes
    return out


def front_scratch(s: int, d: Dims) -> dict:
    """Return K1's scratch regions -> (byte offset, bytes)."""
    return byte_layout(
        [
            ("x8", s * HIDDEN),  # e4m3 words (S > 16: quantized once, shared)
            ("x8s", s * HIDDEN // 32 * 4),  # an i32 code a group
            ("xrdy", WQKV_KPARTS * s * PAIR),  # a flag a (token, K half)
            ("qkvp", WQKV_KPARTS * s * QKV_ROWS * PAIR),
            ("qx8", s * Q_RANK),  # e4m3 words
            ("qx8s", s * Q_RANK // 32 * 4),  # an i32 code a group
            ("qrdy", s * PAIR),
        ]
    )


def back_scratch(s: int, d: Dims, start: int = 0) -> dict:
    """Return K2's scratch regions past ``start`` -> (byte offset, bytes)."""
    ck = (
        64 if d.head_tiles == 2 and s * cdiv(KEYS, 64) <= 256 else 128
    )  # back.chunk_keys
    nchunk = cdiv(KEYS, ck)
    return byte_layout(
        [
            ("part", s * nchunk * d.heads * HEAD_DIM * 2),  # bf16 partials
            ("pstat", s * nchunk * d.heads * 8),  # (max, sum) f32
            ("srdy", s * nchunk * PAIR),
            ("xo", s * d.q_rows * 2),  # bf16: wo_a reads it unquantized
            ("crdy", s * d.heads // 8 * PAIR),
            ("x8b", s * d.o_rows),
            ("x8bs", s * d.o_rows // 32 * 4),
            ("ardy", d.woa_tasks * PAIR),
        ],
        start=start,
    )
