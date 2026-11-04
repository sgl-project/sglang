from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
)
from sglang.srt.utils import get_device


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
            device=torch.device(get_device()),
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
            full_to_state=self.full_to_swa,
            swa_page_size=self.swa_page_size,
            ring_size=self.ring_size,
            num_q_tokens=num_q_tokens,
        )

    def make_decode_plan(self, seq_lens_gpu: torch.Tensor) -> CompressorDecodePlan:
        return CompressorDecodePlan.generate(
            compress_ratio=self.compress_ratio,  # type: ignore
            req_pool_indices=self.req_pool_indices,
            req_to_token=self.req_to_token,
            full_to_state=self.full_to_swa,
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
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=get_device())
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
        req_pool_indices=req_pool_indices.to(get_device()),
        req_to_token=req_to_token.to(get_device()),
        full_to_swa=full_to_swa.to(get_device()),
    )


def make_state_pool(num_pages: int, compress_ratio: int, head_dim: int) -> torch.Tensor:
    last_dim = head_dim * (4 if compress_ratio == 4 else 2)
    return torch.zeros(
        (num_pages, compress_ratio, last_dim),
        dtype=torch.float32,
        device=get_device(),
    )


def to_seq_extend(
    seq_extend_pairs: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seq_lens = torch.tensor([s for s, _ in seq_extend_pairs], dtype=torch.int64)
    extend_lens = torch.tensor([e for _, e in seq_extend_pairs], dtype=torch.int64)
    num_q = int(extend_lens.sum().item())
    return seq_lens, extend_lens, num_q


def run_compress_prefill(
    ctx: LegacyContext | PagedContext,
    pool: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
) -> torch.Tensor:
    num_q = int(extend_lens_cpu.sum().item())
    plan = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, num_q)
    return compress_forward(
        pool,
        kv_score_input,
        ape,
        plan,
        head_dim=ctx.head_dim,
        compress_ratio=ctx.compress_ratio,
    )


def run_compress_decode(
    ctx: LegacyContext | PagedContext,
    pool: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    seq_lens_gpu: torch.Tensor,
) -> torch.Tensor:
    plan = ctx.make_decode_plan(seq_lens_gpu)
    return compress_forward(
        pool,
        kv_score_input,
        ape,
        plan,
        head_dim=ctx.head_dim,
        compress_ratio=ctx.compress_ratio,
    )


def ceil_ue8m0_exp_ref(x: torch.Tensor) -> torch.Tensor:
    """UE8M0 exponent of ``x`` rounded up, as the FP4 indexer quantizers store it."""
    bits = x.to(torch.float32).contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp = exp + (mantissa != 0).to(torch.int32)
    return exp.clamp(1, 254)


def fp4_e2m1_code_ref(x: torch.Tensor) -> torch.Tensor:
    """E2M1 nibble of ``x`` (sign in bit 3), saturating at 6.0."""
    ax = torch.minimum(x.abs(), torch.tensor(6.0, device=x.device))
    idx = torch.zeros_like(ax, dtype=torch.uint8)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        idx += (ax > threshold).to(torch.uint8)
    sign = ((x < 0) & (idx != 0)).to(torch.uint8) * 8
    return idx | sign


# DSV4 FP4 index-K row: 128 values as 64 packed e2m1 bytes plus four ue8m0
# exponents (one per 32 values), stored page-major as 64 payload bytes per slot
# followed by 4 scale bytes per slot.
_INDEX_HEAD_DIM = 128
_INDEX_FP4_DIM = _INDEX_HEAD_DIM // 2
_INDEX_GROUP_SIZE = 32
_INDEX_SCALE_GROUPS = _INDEX_HEAD_DIM // _INDEX_GROUP_SIZE
_INDEX_SCALE_BYTES = 4
_INDEX_PAGE_SIZE = 64


def quantize_fp4_indexer_ref(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch ``quantize_fp4_indexer_tensor`` (default rounding): packed e2m1 codes
    uint8 ``[n, 64]`` and the four ue8m0 exponents packed into int32 ``[n]``."""
    x = x.contiguous().view(-1, _INDEX_HEAD_DIM).float()
    groups = x.view(-1, _INDEX_SCALE_GROUPS, _INDEX_GROUP_SIZE)
    scale_raw = (groups.abs().amax(dim=-1) / 6.0).clamp_min(1.0e-4)
    scale_exp = ceil_ue8m0_exp_ref(scale_raw)
    scale = (scale_exp << 23).contiguous().view(torch.float32)

    scaled = (groups / scale.unsqueeze(-1)).view(-1, _INDEX_HEAD_DIM)
    code = fp4_e2m1_code_ref(scaled)
    packed = (code[:, 0::2].to(torch.int16) | (code[:, 1::2].to(torch.int16) << 4)).to(
        torch.uint8
    )

    packed_sf = scale_exp[:, 0].clone()
    for group_id in range(1, _INDEX_SCALE_GROUPS):
        packed_sf |= scale_exp[:, group_id] << (8 * group_id)
    return packed, packed_sf


def store_fp4_index_cache_ref(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    loc: torch.Tensor,
    num_pages: int,
) -> torch.Tensor:
    """The 64-slot paged index-K cache after writing row ``i`` to slot ``loc[i]``."""
    expected = torch.zeros(
        num_pages,
        _INDEX_PAGE_SIZE * (_INDEX_FP4_DIM + _INDEX_SCALE_BYTES),
        device=x_fp4.device,
        dtype=torch.uint8,
    )
    sf_shifts = torch.arange(0, 32, 8, device=x_fp4.device, dtype=torch.int32)
    for token_id in range(x_fp4.shape[0]):
        cache_loc = int(loc[token_id].item())
        page = cache_loc // _INDEX_PAGE_SIZE
        offset = cache_loc % _INDEX_PAGE_SIZE
        expected[page, offset * _INDEX_FP4_DIM : (offset + 1) * _INDEX_FP4_DIM] = x_fp4[
            token_id
        ]
        sf_start = _INDEX_PAGE_SIZE * _INDEX_FP4_DIM + offset * _INDEX_SCALE_BYTES
        expected[page, sf_start : sf_start + _INDEX_SCALE_BYTES] = (
            (x_sf[token_id] >> sf_shifts) & 0xFF
        ).to(torch.uint8)
    return expected
