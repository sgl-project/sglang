"""Fixtures for the DSA replay metadata kernels: a req_to_token pool, sentinel
filled destination buffers and launchers with the replay's arguments, shared by
the bounded-metadata tests and benchmarks. No model runner or weights."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch

PAGE = 64
TOPK = 2048
SENTINEL = -7
BLOCK_N = 128
# DSA models' default --context-length (1M tokens) plus speculative headroom.
WIDTH_1M = 1048576 + 128


class MetadataWorld:
    """A req_to_token pool of random slot ids; row 0 is the padding request."""

    def __init__(self, width: int, pool: int = 40, seed: int = 0, device="cuda"):
        self.device = device
        self.width = width
        self.real_cols = (width + PAGE - 1) // PAGE
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.req_to_token = torch.randint(
            0, 1 << 26, (pool, width), generator=self.gen, dtype=torch.int32
        ).to(device)
        self.pool = pool

    def batch(self, kv_lens: Sequence[int], pad_to=None, seq_dtype=torch.int64):
        """seq_lens / req_pool_indices for ``kv_lens``, padded like a CUDA graph
        batch (padding rows: request 0, length 1)."""
        bs = len(kv_lens)
        pad_to = pad_to or bs
        req = torch.zeros(pad_to, dtype=torch.int64)
        req[:bs] = torch.randperm(self.pool - 1, generator=self.gen)[:bs] + 1
        seq = torch.ones(pad_to, dtype=seq_dtype)
        seq[:bs] = torch.tensor(kv_lens, dtype=seq_dtype)
        return seq.to(self.device), req.to(self.device)


def int32_buf(n: int, device, fill: int = SENTINEL) -> torch.Tensor:
    return torch.full((n,), fill, dtype=torch.int32, device=device)


def table(rows: int, cols: int, device) -> torch.Tensor:
    return torch.full((rows, cols), SENTINEL, dtype=torch.int32, device=device)


def clone_bufs(bufs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in bufs.items()}


def decode_bufs(world: MetadataWorld, bs: int) -> Dict[str, torch.Tensor]:
    d = world.device
    cu, dsa_cu = int32_buf(bs + 1, d), int32_buf(bs + 1, d)
    # Capture-time value: the precompute + copy replay never rewrites element 0.
    cu[0], dsa_cu[0] = 0, 0
    return dict(
        cache_seqlens=int32_buf(bs, d),
        cu_seqlens_k=cu,
        page_table_1=table(bs, world.width, d),
        dsa_cache_seqlens=int32_buf(bs, d),
        dsa_cu_seqlens_k=dsa_cu,
        real_page_table=table(bs, world.real_cols, d),
    )


def verify_bufs(
    world: MetadataWorld, bs: int, next_n: int, with_ctx: bool = False
) -> Dict[str, torch.Tensor]:
    d = world.device
    rows = bs * next_n
    bufs = dict(
        cache_seqlens=int32_buf(bs, d),
        cu_seqlens_k=int32_buf(bs + 1, d),
        page_table_1=table(rows, world.width, d),
        seqlens_expanded=int32_buf(rows, d),
        dsa_cache_seqlens=int32_buf(rows, d),
        dsa_cu_seqlens_k=int32_buf(rows + 1, d),
        real_page_table=table(rows, world.real_cols, d),
    )
    if with_ctx:
        bufs["paged_mqa_ctx_lens_2d"] = torch.full(
            (bs, next_n), SENTINEL, dtype=torch.int32, device=d
        )
    return bufs


def run_decode(fn: Callable, world, seq, req, bs, bufs, **kw) -> None:
    fn(
        seq_lens=seq,
        req_pool_indices=req,
        req_to_token=world.req_to_token,
        cache_seqlens=bufs["cache_seqlens"],
        cu_seqlens_k=bufs["cu_seqlens_k"],
        page_table_1=bufs["page_table_1"],
        dsa_cache_seqlens=bufs["dsa_cache_seqlens"],
        dsa_cu_seqlens_k=bufs["dsa_cu_seqlens_k"],
        real_page_table=bufs["real_page_table"],
        bs=bs,
        max_len=world.width,
        dsa_index_topk=TOPK,
        physical_page_size=PAGE,
        **kw,
    )


def run_verify(fn: Callable, world, seq, req, bs, next_n, bufs, **kw) -> None:
    fn(
        seq_lens=seq,
        req_pool_indices=req,
        req_to_token=world.req_to_token,
        cache_seqlens=bufs["cache_seqlens"],
        cu_seqlens_k=bufs["cu_seqlens_k"],
        page_table_1=bufs["page_table_1"],
        seqlens_expanded=bufs["seqlens_expanded"],
        dsa_cache_seqlens=bufs["dsa_cache_seqlens"],
        dsa_cu_seqlens_k=bufs["dsa_cu_seqlens_k"],
        real_page_table=bufs["real_page_table"],
        bs=bs,
        max_seqlen_k=world.width,
        dsa_index_topk=TOPK,
        physical_page_size=PAGE,
        next_n=next_n,
        paged_mqa_ctx_lens_2d=bufs.get("paged_mqa_ctx_lens_2d"),
        **kw,
    )


def run_draft_extend(
    fn: Callable,
    world,
    seq,
    req,
    bs,
    next_n,
    bufs,
    extend_seq_lens: Optional[torch.Tensor],
    **kw,
) -> None:
    """The static-length draft extend of a speculative decode step."""
    fn(
        seq_lens=seq,
        extend_seq_lens=extend_seq_lens,
        req_pool_indices=req,
        req_to_token=world.req_to_token,
        cache_seqlens=bufs["cache_seqlens"],
        cu_seqlens_k=bufs["cu_seqlens_k"],
        page_table_1=bufs["page_table_1"],
        seqlens_expanded=bufs["seqlens_expanded"],
        dsa_cache_seqlens=bufs["dsa_cache_seqlens"],
        dsa_cu_seqlens_k=bufs["dsa_cu_seqlens_k"],
        real_page_table=bufs["real_page_table"],
        bs=bs,
        total_len=bs * next_n,
        max_seqlen_k=world.width,
        dsa_index_topk=TOPK,
        physical_page_size=PAGE,
        max_extend_len=next_n,
        max_total_len=bs * next_n,
        static_extend_len=True,
        **kw,
    )


def live_columns(kv_len: int, width: int):
    """(page_table_1 columns, real_page_table columns) a kernel writes for a row
    of ``kv_len`` tokens: whole 128-column blocks below kv_len, within width."""
    live = min(-(-kv_len // BLOCK_N) * BLOCK_N, width)
    return live, -(-live // PAGE)
