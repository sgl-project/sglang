# MSA (fmha_sm100) drop-in for the MiniMax-M3 main sparse-attention step.
#
# Replaces only step 3 of MiniMax sparse prefill/decode. The lightning indexer
# (steps 1-2) is unchanged and still produces `topk_idx`.
# NVIDIA Blackwell (SM100/sm_103) only; callers gate on `msa_available()`.

from __future__ import annotations

import functools
from typing import Optional

import torch


@functools.lru_cache(maxsize=1)
def msa_available() -> bool:
    """True iff the fmha_sm100 sparse kernels are importable on this device."""
    try:
        cap = torch.cuda.get_device_capability()
    except Exception:
        return False
    # SM100 family: B200 (10,0) and B300 (10,3). fmha_sm100/jit.py emits both
    # sm_100a and sm_103a; the kernels run on either.
    if cap[0] != 10 or cap[1] not in (0, 3):
        return False
    try:
        import fmha_sm100  # noqa: F401

        return True
    except Exception:
        return False


def _build_page_table(
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len], physical slot per logical pos
    slot_ids: torch.Tensor,  # [batch]
    seq_lens: torch.Tensor,  # [batch] total K length (prefix + chunk)
    page_size: int,
) -> torch.Tensor:
    """Flattened physical page ids per request (MSA `kv_indices`).

    sglang's paged allocator stores page_size contiguous physical slots per page, so the
    physical page of logical position p is ``req_to_token[req, p] // page_size`` and is the
    same for every p within a page. We read one slot per logical page to recover the table.

    Vectorized (no per-request Python loop): pages are packed contiguously by request in the
    same order MSA's planner expects (``kv_page_indptr = cumsum(ceil(seq_lens/page_size))``).
    ``searchsorted`` maps each packed page slot back to its request; one ``.item()`` recovers
    the total page count (this runs eagerly, outside CUDA-graph capture).
    """
    P = page_size
    n_pages = (seq_lens.to(torch.int64) + (P - 1)) // P  # [batch]
    offsets = (
        torch.cumsum(n_pages, 0) - n_pages
    )  # [batch] exclusive page offset per request
    total = int(n_pages.sum().item())
    idx = torch.arange(
        total, device=req_to_token.device
    )  # packed page slot -> (req, page)
    req = torch.searchsorted(
        offsets + n_pages, idx, right=True
    )  # request id per packed slot
    logical_first = (idx - offsets[req]) * P  # first logical position of that page
    rows = slot_ids[req].to(torch.int64)
    return (req_to_token[rows, logical_first] // P).to(torch.int32)


def msa_sparse_prefill_main(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (slot-major NHD)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk] (0-based, -1 pad) -- step1/2 output
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch]
    cu_seqlens: torch.Tensor,  # [batch+1] cumulative Q lengths
    seq_lens: torch.Tensor,  # [batch] total K length (prefix + chunk)
    prefix_lens: torch.Tensor,  # [batch]
    block_size_k: int,  # == page_size == 128 for M3
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Drop-in for flash_prefill_with_gqa_share_sparse using MSA fmha_sm100.

    Returns o [total_q, num_q_heads, head_dim].
    """
    from fmha_sm100 import fmha_sm100, fmha_sm100_plan

    max_slots, num_kv_heads, head_dim = k_cache.shape
    num_q_heads = q.shape[1]
    P = block_size_k
    topk = topk_idx.shape[-1]
    if max_slots % P != 0:
        raise ValueError(f"max_slots={max_slots} not divisible by page_size={P}")
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # Whole pool as MSA paged KV: [num_phys_pages, num_kv_heads, P, head_dim].
    n_phys_pages = max_slots // P
    k_paged = k_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v_paged = v_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)

    # Per-request Q lengths (extend) and physical page table.
    qo_segment_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
    kv_indices = _build_page_table(req_to_token, slot_ids, seq_lens, P)

    # topk_idx [Hkv, total_q, topk] -> kv_block_indexes [total_q, Hkv, topk].
    kv_block_indexes = topk_idx.permute(1, 0, 2).contiguous().to(torch.int32)

    plan = fmha_sm100_plan(
        qo_segment_lens,
        seq_lens.to(torch.int32),
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=P,
        kv_block_num=topk,
        causal=True,
        qo_offset=prefix_lens.to(torch.int32),
    )
    o, _ = fmha_sm100(
        q,
        k_paged,
        v_paged,
        plan,
        sm_scale=sm_scale,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
    )
    return o


def build_msa_decode_meta(
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,  # [batch]
    seq_lens: torch.Tensor,  # [batch] cached KV length per request
    num_q_heads: int,
    block_size_k: int,
    topk: int,
):
    """Per-forward MSA decode metadata (page table + fmha plan), shared across layers.

    Within one decode forward every sparse layer has the same batch, seq_lens, page size
    and topk, so the physical page table and the fmha_sm100 plan are identical for all of
    them — only ``kv_block_indexes`` (the per-layer top-k selection) changes. Building these
    once per forward instead of once per layer removes the dominant host-side overhead
    (page-table build + the host-side ``fmha_sm100_plan``) from the 57-layer decode loop.

    Used only by the standalone parity harnesses; the serving backend builds eager-decode
    metadata via ``build_msa_decode_cg_plan`` + ``update_msa_decode_cg_meta`` instead.
    """
    from fmha_sm100 import fmha_sm100_plan

    max_slots, num_kv_heads, _ = k_cache.shape
    P = block_size_k
    if max_slots % P != 0:
        raise ValueError(f"max_slots={max_slots} not divisible by page_size={P}")
    B = slot_ids.shape[0]
    kv_indices = _build_page_table(req_to_token, slot_ids, seq_lens, P)
    seq_lens_i32 = seq_lens.to(torch.int32)
    plan = fmha_sm100_plan(
        torch.ones(B, dtype=torch.int32),
        seq_lens_i32,
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=P,
        kv_block_num=topk,
        causal=False,
        qo_offset=seq_lens_i32 - 1,  # decode query sits at the last cached position
    )
    return kv_indices, plan


# ---------------------------------------------------------------------------
# Eager-only MSA decode plan (NOT used under CUDA graph)
#
# WARNING: the fmha_sm100 sparse decode kernel is NOT cuda-graph-safe — captured
# and replayed it returns silently wrong results that compound across replays
# (~14% GSM8K loss on B200). The backend routes decode to the cuda-graph-safe
# Triton sparse path whenever decode runs under a CUDA graph (see
# MiniMaxSparseAttnBackend._use_msa_decode); this plan is reachable ONLY in eager
# decode (no decode CUDA graph), where there is no capture/replay. Do NOT wire it
# back into a captured graph — that reintroduces the ~14% regression.
#
# The build-once / replay-update structure below (refreshing the four length
# tensors ``{kv_segment_lens, kv_segment_offsets, kv_page_indptr, qo_offset}`` and
# the page table in place) is a leftover from the abandoned capture-once attempt;
# it is kept only because eager decode reuses one per-forward plan across layers.
# ---------------------------------------------------------------------------

_MSA_CG_LEN_KEYS = (
    "kv_segment_lens",
    "kv_segment_offsets",
    "kv_page_indptr",
    "qo_offset",
)


def _check_cg_plan_layout(plan) -> None:
    """Fail fast if fmha_sm100's plan layout drifted from what replay-update
    assumes (dict at tuple index 3 holding the four length tensors) — these are
    undocumented fmha_sm100 internals."""
    if not (isinstance(plan, tuple) and len(plan) > 3 and isinstance(plan[3], dict)):
        raise RuntimeError(
            "fmha_sm100_plan no longer returns a tuple with a metadata dict at index 3; "
            "the MSA CUDA-graph decode path must be revalidated against this fmha_sm100 "
            "version. Set SGLANG_DISABLE_MSA=1 to use the Triton path meanwhile."
        )
    missing = [k for k in _MSA_CG_LEN_KEYS if not torch.is_tensor(plan[3].get(k))]
    if missing:
        raise RuntimeError(
            f"fmha_sm100 plan is missing length tensors {missing}; the MSA CUDA-graph "
            "decode path must be revalidated against this fmha_sm100 version. "
            "Set SGLANG_DISABLE_MSA=1 to use the Triton path meanwhile."
        )


def build_msa_decode_cg_plan(
    num_q_heads: int,
    num_kv_heads: int,
    block_size_k: int,
    topk: int,
    batch_size: int,
    device: Optional[torch.device] = None,
):
    """Persistent fmha_sm100 decode plan for one batch size (CUDA-graph stable).

    Built once per captured batch size. The worklist is length-independent (it uses
    ``topk * page_size`` internally), so the reference KV length here only has to make
    every topk block valid; the length-dependent tensors are overwritten each step by
    ``update_msa_decode_cg_meta``. Returns the plan tuple to pass to ``fmha_sm100``.
    """
    from fmha_sm100 import fmha_sm100_plan

    P = block_size_k
    ref_len = topk * P  # length at which all topk blocks exist -> full worklist
    qo = torch.ones(batch_size, dtype=torch.int32)
    kv = torch.full((batch_size,), ref_len, dtype=torch.int32)
    plan = fmha_sm100_plan(
        qo,
        kv,
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=P,
        kv_block_num=topk,
        causal=False,
        qo_offset=kv - 1,
        device=device,
    )
    _check_cg_plan_layout(plan)
    return plan


def update_msa_decode_cg_meta(
    plan,
    kv_indices_buf: torch.Tensor,  # persistent page-table buffer [batch * max_pages]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,  # [batch]
    seq_lens: torch.Tensor,  # [batch] cached KV length per request
    block_size_k: int,
    topk: int,
    num_q_heads: int,
    num_kv_heads: int,
):
    """Refresh the persistent decode plan's length-dependent tensors + page table IN PLACE.

    Host-side (calls fmha_sm100_plan and one ``.item()``); MUST run outside CUDA-graph
    capture — i.e. only from ``init_forward_metadata_out_graph``. The captured graph then
    reads the same plan-tensor and ``kv_indices_buf`` addresses on replay.
    """
    from fmha_sm100 import fmha_sm100_plan

    P = block_size_k
    B = seq_lens.shape[0]
    seq_lens_i32 = seq_lens.to(torch.int32)
    # Fresh plan for the real lengths; copy only its four length-dependent tensors into the
    # persistent plan (same shapes — they depend on batch size, not length). The fresh
    # worklist is identical to the persistent one (topk*P based) and is discarded.
    # qo_offset is clamped: graph replay pads the batch with seq_len==0 slots
    # (masked via kv_segment_lens==0, but seq_len-1 would be -1).
    fresh = fmha_sm100_plan(
        torch.ones(B, dtype=torch.int32),
        seq_lens_i32,
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=P,
        kv_block_num=topk,
        causal=False,
        qo_offset=(seq_lens_i32 - 1).clamp_min(0),
        device=seq_lens.device,
    )
    _check_cg_plan_layout(fresh)
    pd, fd = plan[3], fresh[3]
    for k in _MSA_CG_LEN_KEYS:
        pd[k].copy_(fd[k])
    table = _build_page_table(req_to_token, slot_ids, seq_lens, P)
    kv_indices_buf[: table.numel()].copy_(table)


def msa_sparse_decode_main(
    q: torch.Tensor,  # [batch, num_q_heads, head_dim] (1 query token per request)
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (slot-major NHD)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, batch, topk] (0-based, -1 pad)
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch]
    seq_lens: torch.Tensor,  # [batch] cached KV length per request
    block_size_k: int,  # == page_size == 128
    sm_scale: Optional[float] = None,
    kv_indices: Optional[
        torch.Tensor
    ] = None,  # precomputed page table (per-forward cache)
    plan=None,  # precomputed fmha_sm100 plan (per-forward cache)
) -> torch.Tensor:
    """Drop-in for flash_decode_with_gqa_share_sparse using MSA fmha_sm100.

    Each request is one decode query at absolute position seq_len-1 attending to its
    cached KV through the topk selected 128-blocks. Returns o [batch, num_q_heads, head_dim].

    ``kv_indices`` / ``plan`` are shared across all sparse layers of a forward; the serving
    backend builds them once via ``build_msa_decode_cg_plan`` + ``update_msa_decode_cg_meta``
    (eager decode only) and passes them in. When omitted (only the standalone parity
    harnesses) they are built here via ``build_msa_decode_meta``.
    """
    from fmha_sm100 import fmha_sm100

    max_slots, num_kv_heads, head_dim = k_cache.shape
    H = q.shape[1]
    P = block_size_k
    topk = topk_idx.shape[-1]
    if max_slots % P != 0:
        raise ValueError(f"max_slots={max_slots} not divisible by page_size={P}")
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    n_phys_pages = max_slots // P
    k_paged = k_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v_paged = v_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)

    if kv_indices is None or plan is None:
        kv_indices, plan = build_msa_decode_meta(
            k_cache, req_to_token, slot_ids, seq_lens, H, P, topk
        )
    kv_block_indexes = topk_idx.permute(1, 0, 2).contiguous().to(torch.int32)

    o, _ = fmha_sm100(
        q,
        k_paged,
        v_paged,
        plan,
        sm_scale=sm_scale,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
    )
    return o
