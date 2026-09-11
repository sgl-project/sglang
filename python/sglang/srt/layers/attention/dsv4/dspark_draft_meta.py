"""Fused attention-metadata build for the DSpark draft block.

The DSpark draft head runs its ``(bs, gamma)`` block through the DSV4 HIP
backend as a ``TARGET_VERIFY`` forward, so it used to reuse the target's
metadata builder: ``expand_prefill_casually`` (a python loop of two launches
per request), ``get_swa_page_indices`` (a 128-wide gather plus two masked
fills), ``init_compression_metadata``, the C4/C128 compressor planners, the
FP4 indexer workspace, and ``build_decode_streams`` for all three streams.
Eighty-five tiny kernels, every one launched eagerly from python, and ~1.27 ms
of pure GPU bubble per decode step at bs=8.

Almost none of it is read. Every ``DSparkV4Stage`` pins ``compress_ratio=0``,
so with the unified-KV triton path the draft forward touches exactly four
tensors: ``unified.swa_indices``, ``unified.swa_indptr``, ``unified.swa_loc``
and ``unified.verify_store_state_slot``. All four are pure functions of
``seq_lens`` and ``req_pool_indices``, which means the whole build collapses
into one Triton kernel over device tensors -- no host values, no allocations,
so it can be recorded inside the draft CUDA graph instead of launched eagerly.

Shapes are static for a given graph bucket: the block is always ``gamma`` wide
and every request extends by exactly ``gamma``, so token ``t = b * gamma + j``
has

    seq_len_casual = seq_lens[b] + 1 + j      (the target builder's
                                              seq_lens + gamma, expanded
                                              causally over the block)
    position       = seq_len_casual - 1
    swa_len        = min(position + 1, win)
    swa_loc        = req_pool_indices[b] * ring + position % ring

and the ragged SWA index stream is the ``swa_len``-long run
``slot * ring + (position - swa_len + 1 + i) % ring``, packed at the exclusive
prefix sum of ``swa_len``. The kernel computes that prefix sum in-register
(one block covers every token: ``bs * gamma`` is at most a few hundred) so no
separate cumsum launch is needed.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _dspark_draft_meta_kernel(
    seq_lens_ptr,  # [bs] prefix lengths (int32/int64)
    req_pool_indices_ptr,  # [bs] req-pool slot per request
    seq_lens_casual_ptr,  # [N] int32 out
    positions_casual_ptr,  # [N] int32 out
    swa_len_ptr,  # [N] int32 out
    state_slot_ptr,  # [N] int32 out
    swa_loc_ptr,  # [N] int32 out
    swa_indptr_ptr,  # [N + 1] int32 out
    swa_indices_ptr,  # [N * win] int32 out
    ring_stride,
    N,
    gamma: tl.constexpr,
    win: tl.constexpr,
    BLOCK_N: tl.constexpr,  # next_pow2(N)
    BLOCK_W: tl.constexpr,  # next_pow2(win)
):
    """One program per block token; grid = (N,).

    Every program re-derives the full ``[N]`` length vector so it can take the
    exclusive prefix sum locally. ``N`` is ``bs * gamma`` (a few hundred at
    most), so the redundant work is far cheaper than a second launch.
    """
    t = tl.program_id(0)

    i = tl.arange(0, BLOCK_N)
    live = i < N
    b_all = i // gamma
    j_all = i % gamma
    # Clamp the gather so padded lanes stay in bounds; `live` masks them out.
    sl_all = tl.load(seq_lens_ptr + b_all, mask=live, other=0).to(tl.int32)
    # Padded graph slots carry seq_lens == 0; the eager builder padded
    # seq_lens_casual with 1, so clamp to keep a single valid slot.
    casual_all = tl.maximum(sl_all + 1 + j_all, 1)
    len_all = tl.minimum(casual_all, win)
    len_all = tl.where(live, len_all, 0)

    # Exclusive prefix sum at t, and the total for indptr[N].
    base = tl.sum(tl.where(i < t, len_all, 0))
    total = tl.sum(len_all)

    tl.store(swa_indptr_ptr + t, base)
    if t == 0:
        tl.store(swa_indptr_ptr + N, total)

    b = t // gamma
    j = t % gamma
    seq_len = tl.load(seq_lens_ptr + b).to(tl.int32)
    casual = tl.maximum(seq_len + 1 + j, 1)
    pos = casual - 1
    n = tl.minimum(casual, win)
    slot = tl.load(req_pool_indices_ptr + b).to(tl.int32)

    tl.store(seq_lens_casual_ptr + t, casual)
    tl.store(positions_casual_ptr + t, pos)
    tl.store(swa_len_ptr + t, n)
    tl.store(state_slot_ptr + t, slot)
    tl.store(swa_loc_ptr + t, slot * ring_stride + pos % ring_stride)

    # Ragged SWA prefix: abs_pos in [pos - n + 1, pos].
    w = tl.arange(0, BLOCK_W)
    wmask = w < n
    abs_pos = pos - n + 1 + w
    paged = slot * ring_stride + abs_pos % ring_stride
    tl.store(swa_indices_ptr + base + w, paged, mask=wmask)


class DSparkDraftMetaBuffers:
    """Persistent output buffers for one ``(bs, gamma)`` graph bucket.

    Allocated once, outside any CUDA-graph capture, and refilled in place by
    the kernel. Keeping the addresses pinned is what lets the fill be recorded
    inside the draft decode graph: the captured node reads the same static
    ``seq_lens`` / ``req_pool_indices`` buffers the runner refreshes per step,
    and writes the same metadata tensors the captured attention already binds.
    """

    def __init__(self, *, bs: int, gamma: int, win: int, device: torch.device):
        n = bs * gamma
        self.bs = bs
        self.gamma = gamma
        self.win = win
        self.n = n
        kw = {"dtype": torch.int32, "device": device}
        self.seq_lens_casual = torch.zeros(n, **kw)
        self.positions_casual = torch.zeros(n, **kw)
        self.swa_len = torch.zeros(n, **kw)
        self.state_slot = torch.zeros(n, **kw)
        self.swa_loc = torch.zeros(n, **kw)
        self.swa_indptr = torch.zeros(n + 1, **kw)
        self.swa_indices = torch.zeros(n * win, **kw)
        # Placeholders for DSV4AttnMetadata fields the compress_ratio==0
        # unified-KV path never reads. Zero-element so metadata copy_ is a
        # no-op and nothing can silently consume a stale value.
        self.empty = torch.zeros(0, **kw)
        self.empty_2d = torch.zeros(0, 0, **kw)
        self._block_n = max(16, triton.next_power_of_2(n))
        self._block_w = max(16, triton.next_power_of_2(win))

    def fill(
        self,
        *,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        ring_stride: int,
    ) -> None:
        _dspark_draft_meta_kernel[(self.n,)](
            seq_lens,
            req_pool_indices,
            self.seq_lens_casual,
            self.positions_casual,
            self.swa_len,
            self.state_slot,
            self.swa_loc,
            self.swa_indptr,
            self.swa_indices,
            ring_stride,
            self.n,
            gamma=self.gamma,
            win=self.win,
            BLOCK_N=self._block_n,
            BLOCK_W=self._block_w,
            num_warps=4,
        )


def make_dspark_draft_metadata(
    *,
    buffers: DSparkDraftMetaBuffers,
    page_size: int,
    c4_sparse_topk: int,
    cuda_int32_kwargs: dict,
    metadata_cls,
    attn_metadata_cls,
    unified_cls,
):
    """Wrap pre-allocated buffers in the DSV4 metadata objects.

    Called once per bucket; the returned object is reused for every step so
    the captured graph's bound addresses stay valid.
    """
    core = attn_metadata_cls(
        page_size=page_size,
        raw_out_loc=buffers.empty,
        seq_lens_casual=buffers.seq_lens_casual,
        cuda_int32_kwargs=cuda_int32_kwargs,
        positions_casual=buffers.positions_casual,
        page_table=buffers.empty_2d,
        swa_page_indices=buffers.empty_2d,
        swa_topk_lengths=buffers.swa_len,
        c4_sparse_topk=c4_sparse_topk,
    )
    # init=False fields: the compress_ratio==0 draft owns no C4 stream and HIP
    # has no flash_mla scheduler metadata, so pin them all to None rather than
    # leaving them unset.
    core.c4_sparse_topk_lengths = None
    core.c4_sparse_topk_lengths_raw = None
    core.c4_sparse_page_indices = None
    core.c4_sparse_raw_indices = None
    core.c1_flashmla_metadata = None
    core.c4_flashmla_metadata = None
    core.c128_flashmla_metadata = None

    unified = unified_cls()
    unified.swa_indices = buffers.swa_indices
    unified.swa_indptr = buffers.swa_indptr
    unified.swa_loc = buffers.swa_loc
    unified.verify_store_state_slot = buffers.state_slot
    core.unified = unified

    return metadata_cls(core, None)
