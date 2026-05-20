"""Shared bench helpers for the kv_canary jit_kernel benchmarks.

Centralizes the sweep axes, ``BenchCase`` dataclass, fast / slow case lists, the ``do_bench`` wrapper,
and the two naive baselines (slot-copy for verify+write, cumsum for plan). Per-kernel input builders
and ``_run_one_case`` orchestration stay in each ``bench_*.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import triton.testing

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES

BS_AXIS: list[int] = [1, 4, 32, 128, 256, 1024]
PREFIX_AXIS: list[int] = [0, 128, 1024, 4096, 10240, 16384]
EXTEND_LEN_AXIS: list[int] = [128, 512, 4096]
POOL_AXIS: list[str] = ["full", "swa_window_128"]
SWA_WINDOW: int = 128
RING_CAPACITY: int = 256

QUANTILES: list[float] = [0.5, 0.2, 0.8]


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchCase:
    bs: int
    prefix_len: int
    mode: str
    extend_len: int
    pool_kind: str

    @property
    def case_id(self) -> str:
        return f"bs{self.bs}_prefix{self.prefix_len}_{self.mode}{self.extend_len}_{self.pool_kind}"


def build_fast_matrix_cases() -> list[BenchCase]:
    return [
        BenchCase(bs=1, prefix_len=0, mode="decode", extend_len=1, pool_kind="full"),
        BenchCase(
            bs=32, prefix_len=4096, mode="extend", extend_len=128, pool_kind="full"
        ),
        BenchCase(
            bs=256, prefix_len=4096, mode="decode", extend_len=1, pool_kind="full"
        ),
        BenchCase(
            bs=1024, prefix_len=1024, mode="decode", extend_len=1, pool_kind="full"
        ),
        BenchCase(
            bs=32, prefix_len=16384, mode="extend", extend_len=4096, pool_kind="full"
        ),
        BenchCase(
            bs=128,
            prefix_len=10240,
            mode="decode",
            extend_len=1,
            pool_kind="swa_window_128",
        ),
        BenchCase(
            bs=1, prefix_len=128, mode="extend", extend_len=128, pool_kind="full"
        ),
        BenchCase(
            bs=4, prefix_len=1024, mode="extend", extend_len=512, pool_kind="full"
        ),
        BenchCase(
            bs=128, prefix_len=4096, mode="decode", extend_len=1, pool_kind="full"
        ),
        BenchCase(
            bs=32, prefix_len=16384, mode="extend", extend_len=16384, pool_kind="full"
        ),
        BenchCase(
            bs=256,
            prefix_len=128,
            mode="decode",
            extend_len=1,
            pool_kind="swa_window_128",
        ),
        BenchCase(
            bs=4,
            prefix_len=10240,
            mode="decode",
            extend_len=1,
            pool_kind="swa_window_128",
        ),
    ]


def build_slow_matrix_cases() -> list[BenchCase]:
    fast_keys = {c.case_id for c in build_fast_matrix_cases()}
    slow: list[BenchCase] = []
    for bs in BS_AXIS:
        for prefix_len in PREFIX_AXIS:
            for pool_kind in POOL_AXIS:
                for mode_extend in (
                    ("decode", 1),
                    *((("extend", e) for e in EXTEND_LEN_AXIS)),
                ):
                    mode, extend_len = mode_extend
                    case = BenchCase(
                        bs=bs,
                        prefix_len=prefix_len,
                        mode=mode,
                        extend_len=extend_len,
                        pool_kind=pool_kind,
                    )
                    if case.case_id in fast_keys:
                        continue
                    slow.append(case)
    return slow


def select_matrix_cases() -> list:
    fast = [pytest.param(c, id=c.case_id) for c in build_fast_matrix_cases()]
    slow = [
        pytest.param(c, id=c.case_id, marks=pytest.mark.slow)
        for c in build_slow_matrix_cases()
    ]
    return fast + slow


def do_bench(fn) -> float:
    ms_median, _, _ = triton.testing.do_bench(fn, quantiles=QUANTILES)
    return float(ms_median) * 1000.0


def baseline_us_slot_copy(*, total: int, device: torch.device) -> float:
    """Naive ``kv_buf[slot] = payload`` baseline for verify / write benches."""
    total = max(total, 1)
    payload = torch.zeros(total, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    sink = torch.zeros_like(payload)
    indices = torch.arange(total, device=device, dtype=torch.int64) % sink.shape[0]

    def baseline() -> None:
        sink.index_copy_(0, indices, payload)

    return do_bench(baseline)


def baseline_us_cumsum(*, bs: int, device: torch.device) -> float:
    """``torch.cumsum`` baseline for the plan bench (matches plan kernel's per-req-prefix-sum step)."""
    counts = torch.zeros(max(bs, 1), dtype=torch.int32, device=device)

    def baseline() -> None:
        torch.cumsum(counts, dim=0)

    return do_bench(baseline)
