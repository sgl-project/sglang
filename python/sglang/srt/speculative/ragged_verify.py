from __future__ import annotations

import bisect
import logging
from enum import Enum
from typing import Optional, Sequence

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class RaggedVerifyMode(str, Enum):
    STATIC = "static"
    CAP_ACCEPT = "cap-accept"
    COMPACT = "compact"


def read_ragged_verify_mode() -> RaggedVerifyMode:
    value = envs.SGLANG_RAGGED_VERIFY_MODE.get()
    for mode in RaggedVerifyMode:
        if value == mode.value:
            return mode
    raise ValueError(
        f"invalid SGLANG_RAGGED_VERIFY_MODE={value!r}; expected one of "
        f"{', '.join(repr(m.value) for m in RaggedVerifyMode)}"
    )


def ragged_verify_compact_enabled() -> bool:
    return read_ragged_verify_mode() == RaggedVerifyMode.COMPACT


def round_up_grid(total: int, grid: Sequence[int]) -> int:
    if not grid:
        raise ValueError("round_up_grid requires a non-empty grid")
    if total > grid[-1]:
        raise ValueError(
            f"total {total} exceeds max grid tier {grid[-1]}; "
            "the caller must reject this batch before selecting a graph tier"
        )
    index = bisect.bisect_left(grid, total)
    return grid[index]


def build_graph_num_tokens_grid(
    *,
    capture_bs: Sequence[int],
    num_tokens_per_bs: int,
    token_grid: Sequence[int],
) -> list[int]:
    if num_tokens_per_bs < 1:
        raise ValueError(f"num_tokens_per_bs must be >= 1, got {num_tokens_per_bs}")
    uniform_tiers = {bs * num_tokens_per_bs for bs in capture_bs}
    merged = uniform_tiers.union(token_grid)
    if any(tier < 1 for tier in merged):
        raise ValueError(
            f"graph num-tokens grid tiers must be >= 1, got {sorted(merged)}"
        )
    return sorted(merged)


class RaggedVerifyLayout(msgspec.Struct, frozen=True):
    verify_lens: torch.Tensor
    graph_num_tokens: int
    extend_start_loc: torch.Tensor
    qo_indptr_device: torch.Tensor
    verify_lens_cpu: Optional[list[int]] = None
    total_verify_tokens: Optional[int] = None
    qo_indptr_host: Optional[torch.Tensor] = None
    kv_indptr_host: Optional[torch.Tensor] = None
    kv_lens_host: Optional[torch.Tensor] = None
    max_q_len: Optional[int] = None
    max_kv_len: Optional[int] = None

    def __post_init__(self) -> None:
        if self.verify_lens_cpu is None:
            return
        if not self.verify_lens_cpu:
            raise ValueError("RaggedVerifyLayout requires at least one request")
        if min(self.verify_lens_cpu) < 1:
            raise ValueError(
                f"every request must verify the anchor (verify_len >= 1), got "
                f"{self.verify_lens_cpu}"
            )
        if self.total_verify_tokens != sum(self.verify_lens_cpu):
            raise ValueError(
                f"total_verify_tokens {self.total_verify_tokens} != "
                f"sum(verify_lens_cpu) {sum(self.verify_lens_cpu)}"
            )
        if not (self.total_verify_tokens <= self.graph_num_tokens):
            raise ValueError(
                f"total_verify_tokens {self.total_verify_tokens} exceeds "
                f"graph_num_tokens {self.graph_num_tokens}"
            )

    @property
    def bs(self) -> int:
        return int(self.verify_lens.shape[0])

    @classmethod
    def _assemble_device(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        verify_lens_cpu: Optional[list[int]] = None,
        total_verify_tokens: Optional[int] = None,
    ) -> RaggedVerifyLayout:
        from sglang.srt.speculative.dspark_components.kernels.dspark_ragged import (
            BuildQoIndptr,
        )

        verify_lens = verify_lens.to(torch.int32)
        indptr = BuildQoIndptr.execute(verify_lens=verify_lens)
        return cls(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            extend_start_loc=indptr.extend_start_loc,
            qo_indptr_device=indptr.qo_indptr,
            verify_lens_cpu=verify_lens_cpu,
            total_verify_tokens=total_verify_tokens,
        )

    @classmethod
    def _assemble(
        cls,
        *,
        verify_lens_cpu: list[int],
        total_verify_tokens: int,
        graph_num_tokens: int,
        device: torch.device,
    ) -> RaggedVerifyLayout:
        verify_lens = torch.tensor(verify_lens_cpu, dtype=torch.int32, device=device)
        return cls._assemble_device(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            verify_lens_cpu=verify_lens_cpu,
            total_verify_tokens=total_verify_tokens,
        )

    @classmethod
    def from_verify_lens_device(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
    ) -> RaggedVerifyLayout:
        return cls._assemble_device(
            verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
        )

    @classmethod
    def from_verify_lens(
        cls,
        *,
        verify_lens_cpu: Sequence[int],
        device: torch.device,
        grid: Sequence[int],
        graph_num_tokens_floor: int = 0,
    ) -> RaggedVerifyLayout:
        verify_lens_list = [int(v) for v in verify_lens_cpu]
        total_verify_tokens = sum(verify_lens_list)
        bucket_input = max(total_verify_tokens, graph_num_tokens_floor)
        graph_num_tokens = round_up_grid(total=bucket_input, grid=grid)

        return cls._assemble(
            verify_lens_cpu=verify_lens_list,
            total_verify_tokens=total_verify_tokens,
            graph_num_tokens=graph_num_tokens,
            device=device,
        )

    @classmethod
    def uniform(
        cls,
        *,
        bs: int,
        num_draft_tokens: int,
        device: torch.device,
        grid: Sequence[int],
    ) -> RaggedVerifyLayout:
        return cls.from_verify_lens(
            verify_lens_cpu=[num_draft_tokens] * bs,
            device=device,
            grid=grid,
        )

    def padded_to_bucket(self, *, padded_bs: int) -> RaggedVerifyLayout:
        from sglang.srt.speculative.dspark_components.kernels.dspark_ragged import (
            PaddedToBucket,
        )

        padded = PaddedToBucket.execute(
            verify_lens=self.verify_lens,
            graph_num_tokens=self.graph_num_tokens,
            bs=self.bs,
            padded_bs=padded_bs,
        )

        return RaggedVerifyLayout._assemble_device(
            verify_lens=padded,
            graph_num_tokens=self.graph_num_tokens,
            total_verify_tokens=self.graph_num_tokens,
        )


def build_capture_verify_lens(
    *,
    num_tokens: int,
    num_slots: int,
    num_draft_tokens: int,
) -> list[int]:
    if num_slots < 1 or num_tokens < num_slots:
        raise ValueError(
            f"capture layout needs 1 <= num_slots <= num_tokens, got "
            f"num_slots={num_slots}, num_tokens={num_tokens}"
        )
    if num_tokens > num_slots * num_draft_tokens:
        raise ValueError(
            f"capture layout cannot pack num_tokens={num_tokens} into "
            f"{num_slots} rows of at most {num_draft_tokens} tokens"
        )
    base = num_tokens // num_slots
    rem = num_tokens - base * num_slots
    return [base + 1] * rem + [base] * (num_slots - rem)


class RaggedTargetVerifyGeometry(msgspec.Struct):
    cache_seqlens_int32: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seq_len_q: Optional[int]


def build_ragged_target_verify_geometry(
    *,
    seq_lens: torch.Tensor,
    layout: RaggedVerifyLayout,
) -> RaggedTargetVerifyGeometry:
    cache_seqlens_int32 = (seq_lens + layout.verify_lens).to(torch.int32)
    cu_seqlens_q = layout.qo_indptr_device.to(torch.int32)
    cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
    )
    max_seq_len_q = (
        max(layout.verify_lens_cpu) if layout.verify_lens_cpu is not None else None
    )
    return RaggedTargetVerifyGeometry(
        cache_seqlens_int32=cache_seqlens_int32,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seq_len_q=max_seq_len_q,
    )
