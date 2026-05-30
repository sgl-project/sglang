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
MAX_EXTEND_TOKENS_PER_FORWARD: int = 4096


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchCase:
    scenario: str
    bs: int
    prefix_len: int
    mode: str
    extend_len: int
    pool_kind: str

    @property
    def case_id(self) -> str:
        return (
            f"{self.scenario}_bs{self.bs}_prefix{self.prefix_len}_{self.mode}{self.extend_len}"
            f"_{self.pool_kind}"
        )


def _case(
    *,
    scenario: str,
    bs: int,
    prefix_len: int,
    mode: str,
    extend_len: int,
    pool_kind: str,
) -> BenchCase:
    return BenchCase(
        scenario=scenario,
        bs=bs,
        prefix_len=prefix_len,
        mode=mode,
        extend_len=extend_len,
        pool_kind=pool_kind,
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

    return full


def cases_to_x_vals(
    cases: list[BenchCase],
) -> list[tuple[str, int, int, str, int, str]]:
    return [
        (
            c.scenario,
            c.bs,
            c.prefix_len,
            c.mode,
            c.extend_len,
            c.pool_kind,
        )
        for c in cases
    ]


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
