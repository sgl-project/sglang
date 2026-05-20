"""Shared bench helpers for the kv_canary jit_kernel benchmarks (perf_report style).

Exposes the sweep axes, ``BenchCase`` dataclass, fast / full case factories, and the two naive
baselines used by the 3 per-kernel bench files. Per-kernel input builders and the
``triton.testing.perf_report`` decorators stay in each ``bench_*.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES

BS_AXIS: list[int] = [1, 4, 32, 128, 256, 1024]
PREFIX_AXIS: list[int] = [0, 128, 1024, 4096, 10240, 16384]
EXTEND_LEN_AXIS: list[int] = [128, 512, 4096, 16384]
POOL_AXIS: list[str] = ["full", "swa_window_128"]
SWA_WINDOW: int = 128
RING_CAPACITY: int = 256


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
    """~12 representative points covering the corners of the sweep matrix.

    Includes the two scenarios named in user-instruction b (bs=32 isl=16384 extend, bs=256 isl=4096 decode)
    plus latency / throughput / SWA-window samples. Mirror set kept identical across plan / verify / write
    so cross-kernel comparisons line up.
    """
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


def build_full_matrix_cases() -> list[BenchCase]:
    """Full cartesian product (~360 cases); superset of build_fast_matrix_cases."""
    fast = build_fast_matrix_cases()
    fast_keys = {c.case_id for c in fast}
    full: list[BenchCase] = list(fast)
    for bs in BS_AXIS:
        for prefix_len in PREFIX_AXIS:
            for pool_kind in POOL_AXIS:
                for mode, extend_len in (
                    ("decode", 1),
                    *(("extend", e) for e in EXTEND_LEN_AXIS),
                ):
                    case = BenchCase(
                        bs=bs,
                        prefix_len=prefix_len,
                        mode=mode,
                        extend_len=extend_len,
                        pool_kind=pool_kind,
                    )
                    if case.case_id in fast_keys:
                        continue
                    full.append(case)
    return full


def cases_to_x_vals(cases: list[BenchCase]) -> list[tuple[int, int, str, int, str]]:
    """Flatten BenchCase list into the tuple form ``triton.testing.Benchmark`` expects for x_vals."""
    return [(c.bs, c.prefix_len, c.mode, c.extend_len, c.pool_kind) for c in cases]


def naive_slot_copy_fn(*, total: int, device: torch.device) -> Callable[[], None]:
    """Return a no-arg callable that does a naive ``kv_buf[slot] = payload`` of ``total`` slots."""
    n_slots = max(total, 1)
    payload = torch.zeros(n_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    sink = torch.zeros_like(payload)
    indices = torch.arange(n_slots, device=device, dtype=torch.int64) % sink.shape[0]

    def baseline() -> None:
        sink.index_copy_(0, indices, payload)

    return baseline


def naive_cumsum_fn(*, bs: int, device: torch.device) -> Callable[[], None]:
    """Return a no-arg callable that runs ``torch.cumsum`` on a ``bs``-length int32 vector."""
    counts = torch.zeros(max(bs, 1), dtype=torch.int32, device=device)

    def baseline() -> None:
        torch.cumsum(counts, dim=0)

    return baseline
