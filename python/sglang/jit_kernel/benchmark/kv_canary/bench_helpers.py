"""Shared bench helpers for the kv_canary jit_kernel benchmarks (perf_report style).

Exposes the sweep axes, ``BenchCase`` dataclass, fast / full case factories, and the two naive
baselines used by the 3 per-kernel bench files. Per-kernel input builders and the
``triton.testing.perf_report`` decorators stay in each ``bench_*.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES, RealKvSource

BS_AXIS: list[int] = [1, 4, 32, 128, 256, 1024]
PREFIX_AXIS: list[int] = [0, 128, 1024, 4096, 10240, 16384]
EXTEND_LEN_AXIS: list[int] = [128, 512, 4096, 16384]
POOL_AXIS: list[str] = ["full", "swa_window_128"]
REAL_KV_AXIS: list[str] = ["none", "small_1src", "med_2src", "max_4src"]
HASH_MODE_AXIS: list[str] = ["off", "partial", "all"]
SWA_WINDOW: int = 128
RING_CAPACITY: int = 256


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchCase:
    bs: int
    prefix_len: int
    mode: str
    extend_len: int
    pool_kind: str
    real_kv_kind: str
    hash_mode: str

    @property
    def case_id(self) -> str:
        return (
            f"bs{self.bs}_prefix{self.prefix_len}_{self.mode}{self.extend_len}"
            f"_{self.pool_kind}_rkv{self.real_kv_kind}_hash{self.hash_mode}"
        )


def build_fast_matrix_cases() -> list[BenchCase]:
    """~17 representative points covering the corners of the sweep matrix.

    Includes the two scenarios named in user-instruction b (bs=32 isl=16384 extend, bs=256 isl=4096 decode)
    plus latency / throughput / SWA-window samples. Mirror set kept identical across plan / verify / write
    so cross-kernel comparisons line up. The first 12 cases keep ``real_kv_kind="none"`` / ``hash_mode="off"``
    so the fast matrix retains its historical shape; the trailing 5 cases exercise the
    ``fold_real_kv_sources`` byte-fold path that the production hot path always hits.
    """
    return [
        BenchCase(
            bs=1,
            prefix_len=0,
            mode="decode",
            extend_len=1,
            pool_kind="full",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=32,
            prefix_len=4096,
            mode="extend",
            extend_len=128,
            pool_kind="full",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=256,
            prefix_len=4096,
            mode="decode",
            extend_len=1,
            pool_kind="full",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=1024,
            prefix_len=1024,
            mode="decode",
            extend_len=1,
            pool_kind="full",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=32,
            prefix_len=16384,
            mode="extend",
            extend_len=4096,
            pool_kind="full",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=128,
            prefix_len=10240,
            mode="decode",
            extend_len=1,
            pool_kind="swa_window_128",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=1,
            prefix_len=128,
            mode="extend",
            extend_len=128,
            pool_kind="full",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=4,
            prefix_len=1024,
            mode="extend",
            extend_len=512,
            pool_kind="full",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=128,
            prefix_len=4096,
            mode="decode",
            extend_len=1,
            pool_kind="full",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=32,
            prefix_len=16384,
            mode="extend",
            extend_len=16384,
            pool_kind="full",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=256,
            prefix_len=128,
            mode="decode",
            extend_len=1,
            pool_kind="swa_window_128",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=4,
            prefix_len=10240,
            mode="decode",
            extend_len=1,
            pool_kind="swa_window_128",
            real_kv_kind="none",
            hash_mode="off",
        ),
        BenchCase(
            bs=32,
            prefix_len=4096,
            mode="extend",
            extend_len=128,
            pool_kind="full",
            real_kv_kind="small_1src",
            hash_mode="partial",
        ),
        BenchCase(
            bs=32,
            prefix_len=16384,
            mode="extend",
            extend_len=4096,
            pool_kind="full",
            real_kv_kind="med_2src",
            hash_mode="all",
        ),
        BenchCase(
            bs=256,
            prefix_len=4096,
            mode="decode",
            extend_len=1,
            pool_kind="full",
            real_kv_kind="max_4src",
            hash_mode="all",
        ),
        BenchCase(
            bs=128,
            prefix_len=10240,
            mode="decode",
            extend_len=1,
            pool_kind="swa_window_128",
            real_kv_kind="med_2src",
            hash_mode="partial",
        ),
        BenchCase(
            bs=1,
            prefix_len=0,
            mode="decode",
            extend_len=1,
            pool_kind="full",
            real_kv_kind="small_1src",
            hash_mode="all",
        ),
    ]


def build_full_matrix_cases() -> list[BenchCase]:
    """Full cartesian product + (real_kv × hash_mode) cross only on the 12 fast-matrix base points.

    Pruning: ``hash_mode == "off"`` forces ``real_kv_kind == "none"`` (kernel short-circuits the fold path,
    so multi-source variants would be redundant). The fast 12 base points are then cross-multiplied with
    the 10 surviving ``(real_kv, hash)`` combinations to expose fold cost without blowing up to the full
    360 x 10 cartesian.
    """
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
                        real_kv_kind="none",
                        hash_mode="off",
                    )
                    if case.case_id in fast_keys:
                        continue
                    full.append(case)

    fast_base_points = [
        (c.bs, c.prefix_len, c.mode, c.extend_len, c.pool_kind)
        for c in fast
        if c.real_kv_kind == "none" and c.hash_mode == "off"
    ]
    for bs, prefix_len, mode, extend_len, pool_kind in fast_base_points:
        for hash_mode in HASH_MODE_AXIS:
            if hash_mode == "off":
                continue
            for real_kv_kind in REAL_KV_AXIS:
                if real_kv_kind == "none":
                    continue
                case = BenchCase(
                    bs=bs,
                    prefix_len=prefix_len,
                    mode=mode,
                    extend_len=extend_len,
                    pool_kind=pool_kind,
                    real_kv_kind=real_kv_kind,
                    hash_mode=hash_mode,
                )
                if case.case_id in fast_keys:
                    continue
                full.append(case)
                fast_keys.add(case.case_id)

    return full


def cases_to_x_vals(
    cases: list[BenchCase],
) -> list[tuple[int, int, str, int, str, str, str]]:
    """Flatten BenchCase list into the tuple form ``triton.testing.Benchmark`` expects for x_vals."""
    return [
        (
            c.bs,
            c.prefix_len,
            c.mode,
            c.extend_len,
            c.pool_kind,
            c.real_kv_kind,
            c.hash_mode,
        )
        for c in cases
    ]


def _one_real_kv_source(
    *, num_slots: int, num_bytes: int, read_bytes: int, device: torch.device
) -> RealKvSource:
    tensor = torch.zeros(max(1, num_slots), num_bytes, dtype=torch.uint8, device=device)
    return RealKvSource(
        tensor=tensor,
        page_size=1,
        num_bytes_per_token=num_bytes,
        read_bytes=read_bytes,
    )


def make_real_kv_sources(
    *, kind: str, num_slots: int, device: torch.device
) -> tuple[RealKvSource, ...]:
    """Map a ``real_kv_kind`` axis label to a tuple of ``RealKvSource`` configs.

    Byte-volume ladder (none -> small_1src -> med_2src -> max_4src) so the bench exposes the
    ``fold_real_kv_sources`` PARTIAL/ALL cost gradient. ``max_4src`` hits the
    ``consts.MAX_REAL_KV_SOURCES = 4`` ABI ceiling.
    """
    if kind == "none":
        return ()
    if kind == "small_1src":
        return (
            _one_real_kv_source(
                num_slots=num_slots, num_bytes=8, read_bytes=4, device=device
            ),
        )
    if kind == "med_2src":
        return tuple(
            _one_real_kv_source(
                num_slots=num_slots, num_bytes=16, read_bytes=8, device=device
            )
            for _ in range(2)
        )
    if kind == "max_4src":
        return tuple(
            _one_real_kv_source(
                num_slots=num_slots, num_bytes=32, read_bytes=16, device=device
            )
            for _ in range(4)
        )
    raise ValueError(f"kv-canary bench: unknown real_kv_kind {kind!r}")


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
