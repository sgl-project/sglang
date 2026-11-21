from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from sglang.kernels.ops.kv_canary.verify import CANARY_SLOT_BYTES, RealKvSource

BS_AXIS: list[int] = [1, 4, 32, 128, 256, 1024]
PREFIX_AXIS: list[int] = [0, 128, 1024, 4096, 10240, 16384]
EXTEND_LEN_AXIS: list[int] = [128, 512, 4096, 16384]
POOL_AXIS: list[str] = ["full", "swa_window_128"]
REAL_KV_AXIS: list[str] = ["none", "small_1src", "med_2src", "max_4src"]
HASH_MODE_AXIS: list[str] = ["none", "partial", "all"]
SWA_WINDOW: int = 128
RING_CAPACITY: int = 256
MAX_EXTEND_TOKENS_PER_FORWARD: int = 4096


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchCase:
    scenario: str
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
            f"{self.scenario}_bs{self.bs}_prefix{self.prefix_len}_{self.mode}{self.extend_len}"
            f"_{self.pool_kind}_rkv{self.real_kv_kind}_hash{self.hash_mode}"
        )


def _case(
    *,
    scenario: str,
    bs: int,
    prefix_len: int,
    mode: str,
    extend_len: int,
    pool_kind: str,
    real_kv_kind: str = "none",
    hash_mode: str = "none",
) -> BenchCase:
    return BenchCase(
        scenario=scenario,
        bs=bs,
        prefix_len=prefix_len,
        mode=mode,
        extend_len=extend_len,
        pool_kind=pool_kind,
        real_kv_kind=real_kv_kind,
        hash_mode=hash_mode,
    )


def _is_realistic_extend_case(case: BenchCase) -> bool:
    if case.mode != "extend":
        return True
    return case.bs * case.extend_len <= MAX_EXTEND_TOKENS_PER_FORWARD


def _dedupe_cases(cases: list[BenchCase]) -> list[BenchCase]:
    seen: set[str] = set()
    result: list[BenchCase] = []

    for case in cases:
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        result.append(case)

    return result


def build_fast_matrix_cases() -> list[BenchCase]:
    return _dedupe_cases(
        [
            _case(
                scenario="smoke_decode_empty",
                bs=1,
                prefix_len=0,
                mode="decode",
                extend_len=1,
                pool_kind="full",
            ),
            _case(
                scenario="small_extend_batch",
                bs=32,
                prefix_len=4096,
                mode="extend",
                extend_len=128,
                pool_kind="full",
            ),
            _case(
                scenario="e2e_decode_steady",
                bs=256,
                prefix_len=4096,
                mode="decode",
                extend_len=1,
                pool_kind="full",
            ),
            _case(
                scenario="decode_large_batch_short_prefix",
                bs=1024,
                prefix_len=1024,
                mode="decode",
                extend_len=1,
                pool_kind="full",
            ),
            _case(
                scenario="e2e_prefill_chunk_first",
                bs=1,
                prefix_len=0,
                mode="extend",
                extend_len=4096,
                pool_kind="full",
            ),
            _case(
                scenario="e2e_prefill_chunk_mid",
                bs=1,
                prefix_len=8192,
                mode="extend",
                extend_len=4096,
                pool_kind="full",
            ),
            _case(
                scenario="e2e_prefill_chunk_last",
                bs=1,
                prefix_len=12288,
                mode="extend",
                extend_len=4096,
                pool_kind="full",
            ),
            _case(
                scenario="e2e_decode_tail",
                bs=1,
                prefix_len=5120,
                mode="decode",
                extend_len=1,
                pool_kind="full",
            ),
            _case(
                scenario="swa_decode_long_prefix",
                bs=128,
                prefix_len=10240,
                mode="decode",
                extend_len=1,
                pool_kind="swa_window_128",
            ),
            _case(
                scenario="small_extend_single_req",
                bs=1,
                prefix_len=128,
                mode="extend",
                extend_len=128,
                pool_kind="full",
            ),
            _case(
                scenario="medium_extend_chunk",
                bs=4,
                prefix_len=1024,
                mode="extend",
                extend_len=512,
                pool_kind="full",
            ),
            _case(
                scenario="decode_mid_batch",
                bs=128,
                prefix_len=4096,
                mode="decode",
                extend_len=1,
                pool_kind="full",
            ),
            _case(
                scenario="e2e_prefill_chunk_second",
                bs=1,
                prefix_len=4096,
                mode="extend",
                extend_len=4096,
                pool_kind="full",
            ),
            _case(
                scenario="swa_decode_short_prefix",
                bs=256,
                prefix_len=128,
                mode="decode",
                extend_len=1,
                pool_kind="swa_window_128",
            ),
            _case(
                scenario="swa_decode_tail",
                bs=4,
                prefix_len=10240,
                mode="decode",
                extend_len=1,
                pool_kind="swa_window_128",
            ),
            _case(
                scenario="small_extend_batch_hash",
                bs=32,
                prefix_len=4096,
                mode="extend",
                extend_len=128,
                pool_kind="full",
                real_kv_kind="small_1src",
                hash_mode="partial",
            ),
            _case(
                scenario="e2e_prefill_chunk_hash",
                bs=1,
                prefix_len=12288,
                mode="extend",
                extend_len=4096,
                pool_kind="full",
                real_kv_kind="med_2src",
                hash_mode="all",
            ),
            _case(
                scenario="e2e_decode_steady_hash",
                bs=256,
                prefix_len=4096,
                mode="decode",
                extend_len=1,
                pool_kind="full",
                real_kv_kind="max_4src",
                hash_mode="all",
            ),
            _case(
                scenario="swa_decode_long_prefix_hash",
                bs=128,
                prefix_len=10240,
                mode="decode",
                extend_len=1,
                pool_kind="swa_window_128",
                real_kv_kind="med_2src",
                hash_mode="partial",
            ),
            _case(
                scenario="smoke_decode_empty_hash",
                bs=1,
                prefix_len=0,
                mode="decode",
                extend_len=1,
                pool_kind="full",
                real_kv_kind="small_1src",
                hash_mode="all",
            ),
        ]
    )


def build_full_matrix_cases() -> list[BenchCase]:
    """Full matrix plus targeted e2e points.

    Extend cases are pruned to a maximum token chunk per forward because the scheduler chunks long
    prefills; for example, a 4096-token extend is represented as ``bs=1``, not ``bs=32``.
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
                    case = _case(
                        scenario="matrix",
                        bs=bs,
                        prefix_len=prefix_len,
                        mode=mode,
                        extend_len=extend_len,
                        pool_kind=pool_kind,
                    )
                    if not _is_realistic_extend_case(case):
                        continue
                    if case.case_id in fast_keys:
                        continue
                    full.append(case)

    fast_base_points = [
        (c.bs, c.prefix_len, c.mode, c.extend_len, c.pool_kind)
        for c in fast
        if c.real_kv_kind == "none" and c.hash_mode == "none"
    ]
    for bs, prefix_len, mode, extend_len, pool_kind in fast_base_points:
        for hash_mode in HASH_MODE_AXIS:
            if hash_mode == "none":
                continue
            for real_kv_kind in REAL_KV_AXIS:
                if real_kv_kind == "none":
                    continue
                case = _case(
                    scenario="fold_matrix",
                    bs=bs,
                    prefix_len=prefix_len,
                    mode=mode,
                    extend_len=extend_len,
                    pool_kind=pool_kind,
                    real_kv_kind=real_kv_kind,
                    hash_mode=hash_mode,
                )
                if not _is_realistic_extend_case(case):
                    continue
                if case.case_id in fast_keys:
                    continue
                full.append(case)
                fast_keys.add(case.case_id)

    return full


def cases_to_x_vals(
    cases: list[BenchCase],
) -> list[tuple[str, int, int, str, int, str, str, str]]:
    return [
        (
            c.scenario,
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
    ``real_kv_fold_sources`` PARTIAL/ALL cost gradient. ``max_4src`` hits the
    ``consts.MAX_REAL_KV_SOURCES = 4`` ABI ceiling.
    """
    if kind == "none":
        return ()
    if kind == "small_1src":
        return (
            _one_real_kv_source(
                num_slots=num_slots, num_bytes=16, read_bytes=16, device=device
            ),
        )
    if kind == "med_2src":
        return tuple(
            _one_real_kv_source(
                num_slots=num_slots, num_bytes=32, read_bytes=16, device=device
            )
            for _ in range(2)
        )
    if kind == "max_4src":
        return tuple(
            _one_real_kv_source(
                num_slots=num_slots, num_bytes=64, read_bytes=32, device=device
            )
            for _ in range(4)
        )
    raise ValueError(f"kv-canary bench: unknown real_kv_kind {kind!r}")


def naive_slot_copy_fn(*, total: int, device: torch.device) -> Callable[[], None]:
    n_slots = max(total, 1)
    payload = torch.zeros(n_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    sink = torch.zeros_like(payload)
    indices = torch.arange(n_slots, device=device, dtype=torch.int64) % sink.shape[0]

    def baseline() -> None:
        sink.index_copy_(0, indices, payload)

    return baseline


def naive_cumsum_fn(*, bs: int, device: torch.device) -> Callable[[], None]:
    counts = torch.zeros(max(bs, 1), dtype=torch.int32, device=device)

    def baseline() -> None:
        torch.cumsum(counts, dim=0)

    return baseline
