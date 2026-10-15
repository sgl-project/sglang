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
EPS = 1e-20

ROWS = 16  # GEMV rows a task

# KV record (V4 fp8_ds_mla): 576 B rows (448 e4m3, 64 bf16), then 8 B of scales a token
DATA = 576

PAIR = 8  # a mailbox (value, tag) pair


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
    def q_rows(self) -> int:
        return self.heads * HEAD_DIM

    @property
    def wqb_tasks(self) -> int:
        return self.q_rows // ROWS


WQKV_TASKS = QKV_ROWS // ROWS  # 112
WQKV_KPARTS = 2  # X8's K halves (a flag each); an LDS window of the wqkv GEMV


def wqkv_order(s):
    """Return SGLang's wqkv_a K order at width s as (chains, halves, sum order)."""
    if s <= 32:
        w = 16 if s <= 4 else 8
        return w, False, tuple(range(w))
    assert s <= 64
    return 5, True, (0, 4, 1, 2, 3)


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
