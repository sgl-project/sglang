"""Paged-experts forward.

Per step the active experts are paged into the K-slot pool and the real fused-MoE GEMM runs over it, in
two regimes:

* ``distinct active experts <= K``: keep-warm. Page only the misses (resident experts are reused across
  steps), remap, one GEMM.
* ``distinct active experts > K`` (e.g. prefill, or batched decode): the pool can't hold them at once, so
  serve them in ``ceil(distinct / K)`` **waves** — each wave pages <=K experts, masks the routing to that
  wave, runs the GEMM, and the per-wave partials are **summed**. Each active expert is in exactly one wave
  and out-of-wave experts are masked to weight 0, so the sum equals the full MoE output (lossless).

*Where* the decision + page-in run — host (eager) or on the GPU inside the captured decode graph — is a
``Placement`` strategy (``placement.py``); ``paged_apply`` just dispatches to it. The shared building
blocks (``mask_and_remap_expert_ids``, ``_gemm_hidden``, and the two wave helpers) live here.

Routing stays E-wide; only the table is K.
"""

from __future__ import annotations

import torch


def mask_and_remap_expert_ids(
    topk_ids: torch.Tensor, logical_to_gpu_index: torch.Tensor
) -> torch.Tensor:
    """Logical expert ids -> GPU slot ids; non-resident experts map to -1 (masked below).
    ``logical_to_gpu_index[e]`` is the slot of expert e (-1 if absent)."""
    return logical_to_gpu_index[topk_ids]


def _gemm_hidden(
    method, layer, dispatch_output, remap: torch.Tensor, *, clone_hidden: bool
):
    """Run the base fused-MoE over the K-slot pool for one (wave's) remap, returning the hidden output.

    Zero the routing weight where the expert is masked out (remap == -1) so its contribution is provably
    0, and clamp masked ids -1 -> 0 (slot-0 output x 0 = exact 0; required for marlin's moe_align binning,
    bit-identical for triton). ``clone_hidden`` is set on the wave path, where the same input is reused
    across waves and the base method may consume it in place.
    """
    topk_output = dispatch_output.topk_output
    tw = topk_output.topk_weights
    masked_tw = torch.where(remap >= 0, tw, torch.zeros_like(tw))
    safe_ids = torch.where(remap >= 0, remap, torch.zeros_like(remap))
    hidden = dispatch_output.hidden_states
    md = dispatch_output._replace(
        hidden_states=hidden.clone() if clone_hidden else hidden,
        topk_output=topk_output._replace(topk_ids=safe_ids, topk_weights=masked_tw),
    )
    out = method.base_method.apply(layer, md)
    return out.hidden_states if hasattr(out, "hidden_states") else out


def _gemm_hidden_fused(
    method, layer, dispatch_output, safe_ids, masked_tw, *, clone_hidden: bool
):
    """Like :func:`_gemm_hidden`, but the masking/remap chain was already computed by the fused
    ``remap_mask`` kernel (``pager.remap_mask_ondevice``) — just swap the buffers in and run the GEMM.
    """
    topk_output = dispatch_output.topk_output
    hidden = dispatch_output.hidden_states
    md = dispatch_output._replace(
        hidden_states=hidden.clone() if clone_hidden else hidden,
        topk_output=topk_output._replace(topk_ids=safe_ids, topk_weights=masked_tw),
    )
    out = method.base_method.apply(layer, md)
    return out.hidden_states if hasattr(out, "hidden_states") else out


def _wave_apply(method, layer, dispatch_output, topk_ids: torch.Tensor, distinct):
    """Serve > K distinct experts in ceil(len(distinct)/K) waves; sum the per-wave partials (lossless)."""
    pager = method._pager
    K, E, dev = pager.K, pager.E, pager.device
    store = pager.store
    if hasattr(store, "prefetch_cold"):
        # Queue the WHOLE step's cold reads up front (all waves, one deep madvise batch), and read the
        # NEXT layer's cold tier ahead while this layer transfers/computes — in the wave regime the
        # distinct set saturates toward E, so the next layer's cold set is a predictable sequential
        # read. Both are no-ops off the disk tier.
        store.prefetch_cold(distinct)
        store._step_prefetched = (
            True  # page_in skips its per-wave re-prefetch of the same ranges
        )
        from sglang.srt.layers.moe.paged_experts.pager import next_layer_pager

        nxt = next_layer_pager(pager)
        if nxt is not None and hasattr(nxt.store, "prefetch_cold_all"):
            nxt.store.prefetch_cold_all()
    l2g = torch.full((E,), -1, dtype=torch.int32, device=dev)
    out = None
    group = []
    for w in range(0, len(distinct), K):
        group = distinct[w : w + K]
        src = torch.tensor(group, dtype=torch.int64, device=dev)
        dst = torch.arange(len(group), dtype=torch.int64, device=dev)
        pager.page_in(src, dst)
        l2g.fill_(-1)
        l2g[src] = dst.to(torch.int32)
        partial = _gemm_hidden(
            method, layer, dispatch_output, l2g[topk_ids], clone_hidden=True
        )
        out = partial if out is None else out + partial
    pager.set_residency(group)  # leave the maps consistent for the next keep-warm step
    if hasattr(store, "_step_prefetched"):
        store._step_prefetched = False
    return out


def _ondevice_wave_apply(method, layer, dispatch_output, topk_ids):
    """On-device static-wave path (distinct > K, e.g. prefill): ceil(E/K) waves, each planned+gathered
    on-device, GEMM'd and summed. No host sync. Resyncs the keep-warm state to the last wave so a
    following decode step is consistent. Lossless (each active expert is served in exactly one wave).
    """
    pager = method._pager
    nwaves = (pager.E + pager.K - 1) // pager.K
    out = None
    for w in range(nwaves):
        pager.decide_and_page_wave_ondevice(topk_ids, w)
        remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
        partial = _gemm_hidden(method, layer, dispatch_output, remap, clone_hidden=True)
        out = partial if out is None else out + partial
    pager.resync_residency_ondevice(nwaves - 1)
    return out


def paged_apply(method, layer, dispatch_output):
    """Dispatch the step to the method's decode placement (eager host vs captured on-device).

    The placement (``method._placement``) owns the decide + page-in flow; both end in ``_gemm_hidden``
    over the K-slot pool. See ``placement.py``.
    """
    return method._placement.apply(method, layer, dispatch_output)
