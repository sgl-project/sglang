from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch

from sglang.jit_kernel.dsv4 import CompressorDecodePlan, CompressorPrefillPlan


@dataclass
class LegacyContext:
    """Per-request ring buffer (no req_to_token / full_to_swa).

    `req_pool_indices[i]` directly maps to the request's ring base slot.
    """

    bs: int
    head_dim: int
    compress_ratio: int
    req_pool_indices: torch.Tensor  # int64 [bs] on cuda
    pages_per_req: int

    @property
    def num_pages(self) -> int:
        # Reserve enough pages to hold all batched requests' rings.
        return int(self.req_pool_indices.max().item() + 1) * self.pages_per_req

    def state_loc(self, b: int, position: int) -> int:
        rid = int(self.req_pool_indices[b].item())
        if self.compress_ratio == 4:
            page = rid * 2 + (position // 4) % 2
        else:
            page = rid
        return page * self.compress_ratio + position % self.compress_ratio

    def make_prefill_plan(
        self,
        seq_lens_cpu: torch.Tensor,
        extend_lens_cpu: torch.Tensor,
        num_q_tokens: int,
    ) -> CompressorPrefillPlan:
        return CompressorPrefillPlan.generate_legacy(
            compress_ratio=self.compress_ratio,  # type: ignore
            req_pool_indices=self.req_pool_indices,
            seq_lens=seq_lens_cpu,
            extend_lens=extend_lens_cpu,
            num_q_tokens=num_q_tokens,
            device=torch.device("cuda"),
        )

    def make_decode_plan(self, seq_lens_gpu: torch.Tensor) -> CompressorDecodePlan:
        return CompressorDecodePlan.generate_legacy(
            compress_ratio=self.compress_ratio,  # type: ignore
            req_pool_indices=self.req_pool_indices,
            seq_lens=seq_lens_gpu,
        )


@dataclass
class PagedContext:
    """SWA paged layout with identity req_to_token + identity full_to_swa.

    Each request occupies `num_swa_pages_per_req` contiguous swa_pages, so
    `req_to_token[r, p] = r * (num_swa_pages_per_req * swa_page_size) + p`.
    """

    bs: int
    head_dim: int
    compress_ratio: int
    swa_page_size: int
    ring_size: int
    num_swa_pages_per_req: int
    req_pool_indices: torch.Tensor  # int64 [bs] on cuda
    req_to_token: torch.Tensor  # int64 [num_reqs_capacity, max_tokens_per_req] on cuda
    full_to_swa: torch.Tensor  # int64 [num_swa_slots] on cuda

    @property
    def num_pages(self) -> int:
        # Upper bound: every (request, position) state slot fits.
        max_state_loc = (
            self.bs * self.num_swa_pages_per_req * self.ring_size
            + self.swa_page_size  # slack for the largest tail
        )
        return max_state_loc // self.compress_ratio + 1

    def state_loc(self, b: int, position: int) -> int:
        rid = int(self.req_pool_indices[b].item())
        loc = int(self.req_to_token[rid, position].item())
        swa_loc = int(self.full_to_swa[loc].item())
        swa_page = swa_loc // self.swa_page_size
        return swa_page * self.ring_size + swa_loc % self.ring_size

    def make_prefill_plan(
        self,
        seq_lens_cpu: torch.Tensor,
        extend_lens_cpu: torch.Tensor,
        num_q_tokens: int,
    ) -> CompressorPrefillPlan:
        return CompressorPrefillPlan.generate(
            compress_ratio=self.compress_ratio,  # type: ignore
            req_pool_indices=self.req_pool_indices,
            seq_lens=seq_lens_cpu,
            extend_lens=extend_lens_cpu,
            req_to_token=self.req_to_token,
            full_to_swa=self.full_to_swa,
            swa_page_size=self.swa_page_size,
            ring_size=self.ring_size,
            num_q_tokens=num_q_tokens,
            device=torch.device("cuda"),
        )

    def make_decode_plan(self, seq_lens_gpu: torch.Tensor) -> CompressorDecodePlan:
        return CompressorDecodePlan.generate(
            compress_ratio=self.compress_ratio,  # type: ignore
            req_pool_indices=self.req_pool_indices,
            req_to_token=self.req_to_token,
            full_to_swa=self.full_to_swa,
            seq_lens=seq_lens_gpu,
            swa_page_size=self.swa_page_size,
            ring_size=self.ring_size,
        )


def make_legacy_context(
    bs: int,
    compress_ratio: Literal[4, 128],
    head_dim: int = 512,
) -> LegacyContext:
    pages_per_req = 2 if compress_ratio == 4 else 1
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device="cuda")
    return LegacyContext(
        bs=bs,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        req_pool_indices=req_pool_indices,
        pages_per_req=pages_per_req,
    )


def make_paged_context(
    bs: int,
    compress_ratio: Literal[4, 128],
    head_dim: int = 512,
    swa_page_size: int = 256,
    ring_size: Optional[int] = None,
    num_swa_pages_per_req: int = 8,
    max_tokens_per_req: int = 8192,
    num_reqs_capacity: int = 16,
) -> PagedContext:
    if ring_size is None:
        ring_size = 8 if compress_ratio == 4 else 128
    assert swa_page_size % ring_size == 0
    assert ring_size % compress_ratio == 0
    assert num_swa_pages_per_req * swa_page_size <= max_tokens_per_req

    stride = num_swa_pages_per_req * swa_page_size
    req_to_token = torch.zeros(
        (num_reqs_capacity, max_tokens_per_req), dtype=torch.int32
    )
    for r in range(bs):
        req_to_token[r, :stride] = torch.arange(
            r * stride, (r + 1) * stride, dtype=torch.int32
        )
    total_swa_slots = num_reqs_capacity * stride
    full_to_swa = torch.arange(total_swa_slots, dtype=torch.int64)
    req_pool_indices = torch.arange(bs, dtype=torch.int64)
    return PagedContext(
        bs=bs,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        swa_page_size=swa_page_size,
        ring_size=ring_size,
        num_swa_pages_per_req=num_swa_pages_per_req,
        req_pool_indices=req_pool_indices.cuda(),
        req_to_token=req_to_token.cuda(),
        full_to_swa=full_to_swa.cuda(),
    )


def make_state_pool(num_pages: int, compress_ratio: int, head_dim: int) -> torch.Tensor:
    last_dim = head_dim * (4 if compress_ratio == 4 else 2)
    return torch.zeros(
        (num_pages, compress_ratio, last_dim),
        dtype=torch.float32,
        device="cuda",
    )


def to_seq_extend(
    seq_extend_pairs: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seq_lens = torch.tensor([s for s, _ in seq_extend_pairs], dtype=torch.int64)
    extend_lens = torch.tensor([e for _, e in seq_extend_pairs], dtype=torch.int64)
    num_q = int(extend_lens.sum().item())
    return seq_lens, extend_lens, num_q
