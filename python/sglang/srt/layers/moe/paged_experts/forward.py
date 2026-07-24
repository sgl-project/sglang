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


def _refresh_nvfp4_scalars(method, layer, logical_to_slot=None):
    """nvfp4: scatter the resident full-E per-expert scalars (g*_alphas, w*_input_scale_quant) into the
    K slots by the live residency map, so slot s carries the scalar of whatever logical expert is paged
    there. The big weights + swizzled block scales page normally; only these sub-8-byte scalars need the
    per-step refresh (they can't ride the pinned gather). No-op for every other quant method.

    ``logical_to_slot`` overrides the map used by the eager path. The wave path (``_wave_apply``) positions
    each wave's weights via a LOCAL logical->slot map and leaves ``pager.logical_to_gpu_index_cuda`` stale
    until ``set_residency`` at the end — so the scalars MUST be scattered by that same local map, else slot
    s gets the right weight but a DIFFERENT expert's alpha (wrong output magnitude, confident-wrong logits;
    it is the multi-token verify batch, not single-token decode, that trips this).
    """
    fe = getattr(method, "_nvfp4_full_e", None)
    if fe is None:
        return
    pager = method._pager
    s2l = getattr(pager, "_slot_expert_d", None)
    if logical_to_slot is None and s2l is not None:
        # Captured / on-device path: a FIXED-[K] gather by the slot->logical map (updated in-graph by
        # the decide kernel) — capture-safe, re-read each replay. Only valid when the ON-DEVICE decide
        # positioned the weights (captured keep-warm / on-device waves), i.e. no explicit wave map was
        # passed. Empty slots (-1) clamp to 0; their GEMM output is masked out downstream, so the
        # borrowed scalar is inert.
        idx = s2l.clamp(min=0).long()
        for nm, full in fe.items():
            getattr(layer, nm).data.copy_(full[idx])
    else:
        # Eager scatter by the logical->slot map. The HOST wave path (_wave_apply) positions weights by
        # its LOCAL l2g (host page_in, not the on-device decide) and passes it as logical_to_slot — that
        # map, NOT the on-device _slot_expert_d (which the host wave leaves stale, and which can even
        # hold an out-of-range logical id from a prior spec-verify step -> a scalar-gather OOB), is what
        # matches where the weights landed. Falls back to the live pager map for the eager keep-warm path
        # (logical_to_slot=None, s2l=None). Boolean-mask indexing is data-dependent-shape (NOT
        # capturable), but _wave_apply is only ever reached OFF the capture path, so this is safe.
        l2g = (
            logical_to_slot
            if logical_to_slot is not None
            else pager.logical_to_gpu_index_cuda
        )  # [E] int32: slot of each logical expert, -1 if not
        resident = l2g >= 0
        slots = l2g[resident].long()
        for nm, full in fe.items():
            getattr(layer, nm).data[slots] = full[resident]


def _gemm_hidden(
    method,
    layer,
    dispatch_output,
    remap: torch.Tensor,
    *,
    clone_hidden: bool,
    logical_to_slot: torch.Tensor = None,
):
    """Run the base fused-MoE over the K-slot pool for one (wave's) remap, returning the hidden output.

    Zero the routing weight where the expert is masked out (remap == -1) so its contribution is provably
    0, and clamp masked ids -1 -> 0 (slot-0 output x 0 = exact 0; required for marlin's moe_align binning,
    bit-identical for triton). ``clone_hidden`` is set on the wave path, where the same input is reused
    across waves and the base method may consume it in place. ``logical_to_slot`` is the wave's local
    logical->slot map, threaded to the nvfp4 scalar refresh (see ``_refresh_nvfp4_scalars``).
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
    _refresh_nvfp4_scalars(method, layer, logical_to_slot=logical_to_slot)
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
    _refresh_nvfp4_scalars(method, layer)
    out = method.base_method.apply(layer, md)
    return out.hidden_states if hasattr(out, "hidden_states") else out


def _scratch_fill(pager, bufs, bank, ts, ev_ready, ev_gemm) -> None:
    """Enqueue a full-store read into scratch bank ``bank`` on the transfer stream. The bank is free
    once the GEMM that last read it (two layers ago) completed; the staging pin buffers (windowed cold
    rows) are free once this bank's previous fill drained (CPU wait — the gather writes them host-side).
    """
    ev_ready.synchronize()
    with torch.cuda.stream(ts):
        ts.wait_event(ev_gemm)
        # resident-aware: D2D the K residents out of the layer's own pool, stream only the complement
        if not pager.scratch_fill_resident_aware(bufs):
            pager.store.read_full(bufs, stage_key=bank)
        ev_ready.record(ts)


def _scratch_prefill_apply(method, layer, dispatch_output, topk_ids, distinct=None):
    """Streaming prefill: run this layer's MoE as ONE vanilla E-wide fused-MoE pass out of a full-E
    scratch pool instead of ``ceil(E/K)`` masked waves through the K-slot pool.

    Two global scratch banks ping-pong across layers: while this layer's GEMM computes out of bank
    ``layer_ord & 1``, the NEXT layer's whole expert set streams into the other bank on the transfer
    stream — cross-layer overlap with none of the residency hazards of pre-paging the K-slot pool
    (scratch is not serving state; the K slots and every residency map stay untouched, so decode
    resumes warm after prefill). The layer's paged params are swapped to the scratch views only for
    the duration of the base-method call (every supported backend derives its expert count from the
    weight shapes), and ``topk_ids`` pass through UNREMAPPED — bit-identical math to a fully-resident
    serve. Returns the hidden output, or ``None`` when the scratch pool is unavailable (caller falls
    back to the wave path).
    """
    pager = method._pager
    if torch.cuda.is_current_stream_capturing():
        return None
    if getattr(method, "_nvfp4_full_e", None) is not None:
        # nvfp4 scalar params (g*_alphas, w*_input_scale_quant) are K-sized and refreshed by the
        # K-slot residency map; the E-wide unremapped scratch pass would index them out of range.
        # Route nvfp4 through the wave path, where the per-wave residency refresh is correct.
        return None
    if distinct is not None and len(distinct) < (6 * pager.E) // 10:
        return None  # sparse big batch: waves move fewer bytes
    store = pager.store
    if getattr(store, "host", None) is None or not store.pinned:
        # full-pin stores only: any cold tier (windowed RAM or disk) needs a per-layer CPU-side
        # staging gather in read_full, which stalls the pipeline on the CPU — measured net-negative,
        # so windowed stores keep the wave path.
        return None
    ctx = pager.scratch_ctx()
    if ctx is None:
        return None
    bufs, ev_ready, ev_gemm = ctx
    bank = getattr(pager, "_layer_ord", 0) & 1
    ts = pager.wave_ctx()[0]
    cs = torch.cuda.current_stream()
    if not getattr(pager, "_scratch_prefilled", False):
        ts.wait_stream(cs)  # first fill of the pass: order behind enqueued compute
        _scratch_fill(pager, bufs[bank], bank, ts, ev_ready[bank], ev_gemm[bank])
    pager._scratch_prefilled = False
    cs.wait_event(ev_ready[bank])
    gpu = pager.store.gpu
    saved = {name: p.data for name, p in gpu.items()}
    try:
        for name, p in gpu.items():
            p.data = bufs[bank][name]
        hidden = _gemm_hidden(
            method, layer, dispatch_output, topk_ids, clone_hidden=False
        )
    finally:
        for name, p in gpu.items():
            p.data = saved[name]
    ev_gemm[bank].record(cs)
    # stream the NEXT layer's experts into the other bank while its attention runs
    from sglang.srt.layers.moe.paged_experts.pager import next_layer_pager

    nxt = next_layer_pager(pager)
    if nxt is not None and nxt.scratch_ctx() is not None:
        nbank = getattr(nxt, "_layer_ord", 0) & 1
        _scratch_fill(nxt, bufs[nbank], nbank, ts, ev_ready[nbank], ev_gemm[nbank])
        nxt._scratch_prefilled = True
    return hidden


def _wave_apply(method, layer, dispatch_output, topk_ids: torch.Tensor, distinct):
    """Serve > K distinct experts in waves; sum the per-wave partials (lossless).

    DOUBLE-BUFFERED: the K slots are split into two banks of K//2 and the waves ping-pong between them —
    wave w+1's page-in (hot ``transfer_kv`` + cold staged H2D, on a dedicated transfer stream) overlaps
    wave w's GEMM (compute stream), and the CPU-side cold gather (which faults the disk tier in) runs
    while the GPU is busy. Events sequence the two hazards: a bank's slots are rewritten only after the
    GEMM that read them (gemm-done), and a bank's staging buffer is refilled only after its H2D drained
    (h2d-done). Each active expert is still served in exactly one wave, so the partial-sum stays
    lossless.
    """
    hidden = _scratch_prefill_apply(
        method, layer, dispatch_output, topk_ids, distinct=distinct
    )
    if hidden is not None:
        return hidden
    pager = method._pager
    K, E, dev = pager.K, pager.E, pager.device
    store = pager.store
    half = K // 2
    # Bank (double-buffer) the host wave path only where the overlap pays: the DISK cold tier, whose
    # CPU-side gather (page faults) hides under the previous wave's GEMM. On RAM-windowed stores the
    # waves are transfer-bound with little to hide, and halving the wave size nearly doubles the
    # per-wave fixed costs (masked-GEMM pass + transfer launches) — measured net-negative there, so
    # those keep serial full-K waves.
    banked = (
        half > 0
        and bool(getattr(store, "_cold_mm", {}))
        and not torch.cuda.is_current_stream_capturing()
    )
    wave_k = half if banked else K
    groups = [distinct[w : w + wave_k] for w in range(0, len(distinct), wave_k)]
    rolling = False
    if hasattr(store, "prefetch_cold"):
        # Disk cold tier read-ahead. When the cold tier fits comfortably in the page cache, queue the
        # WHOLE step's reads up front (one deep madvise batch) and the NEXT layer's cold tier behind it.
        # When it does NOT (a true >RAM store), a whole-step WILLNEED evicts its own tail before the
        # later waves arrive — roll the read-ahead one wave ahead of the gather instead.
        from sglang.srt.layers.moe.paged_experts.store import _host_available_bytes

        mm_total = sum(len(m) for m in getattr(store, "_cold_mm", {}).values())
        avail = _host_available_bytes()
        rolling = bool(mm_total) and bool(avail) and mm_total > avail // 2
        if rolling:
            store.prefetch_cold(groups[0])
        else:
            store.prefetch_cold(distinct)
            from sglang.srt.layers.moe.paged_experts.pager import next_layer_pager

            nxt = next_layer_pager(pager)
            if nxt is not None and hasattr(nxt.store, "prefetch_cold_all"):
                nxt.store.prefetch_cold_all()
        store._step_prefetched = (
            True  # page_in skips its per-wave re-prefetch of the same ranges
        )
    l2g = torch.full((E,), -1, dtype=torch.int32, device=dev)
    cs = torch.cuda.current_stream()
    if banked:
        ts, ev_h2d, ev_gemm, _ = pager.wave_ctx()
        ts.wait_stream(cs)
    out = None
    group, base = [], 0
    for i, group in enumerate(groups):
        b = i & 1
        base = b * half if banked else 0
        if rolling and i + 1 < len(groups):
            store.prefetch_cold(groups[i + 1])  # keep the disk one wave ahead
        if banked:
            # staging buffers for bank b are free once wave i-2's H2D drained (CPU wait: the gather
            # below writes them from the CPU side)
            ev_h2d[b].synchronize()
            with torch.cuda.stream(ts):
                # the plan tensors MUST be created on ts: an arange enqueued on the compute stream is
                # not synchronized with ts, and the transfer kernels reading a not-yet-materialized
                # dst was a real (intermittent, load-dependent) illegal-memory-access
                src = torch.tensor(group, dtype=torch.int64, device=dev)
                dst = torch.arange(
                    base, base + len(group), dtype=torch.int64, device=dev
                )
                # bank b's slots are free once wave i-2's GEMM finished reading them
                ts.wait_event(ev_gemm[b])
                pager.page_in(src, dst, stage_bank=b, async_h2d=True, src_host=group)
                ev_h2d[b].record(ts)
            cs.wait_event(ev_h2d[b])
        else:
            src = torch.tensor(group, dtype=torch.int64, device=dev)
            dst = torch.arange(base, base + len(group), dtype=torch.int64, device=dev)
            pager.page_in(src, dst, src_host=group)
        l2g.fill_(-1)
        l2g[src] = dst.to(torch.int32)
        partial = _gemm_hidden(
            method,
            layer,
            dispatch_output,
            l2g[topk_ids],
            clone_hidden=True,
            logical_to_slot=l2g,  # nvfp4: scatter scalars by THIS wave's map, not the stale pager map
        )
        if banked:
            ev_gemm[b].record(cs)
        out = partial if out is None else out + partial
    if banked:
        # the NEXT step's page_in gathers into the shared staging buffers from the CPU side, which no
        # stream ordering protects — drain this step's H2D before returning (GEMMs stay async)
        ev_h2d[0].synchronize()
        ev_h2d[1].synchronize()
    pager.set_residency(
        group, base=base
    )  # leave the maps consistent for the next keep-warm step
    if hasattr(store, "_step_prefetched"):
        store._step_prefetched = False
    return out


def _ondevice_wave_apply(method, layer, dispatch_output, topk_ids):
    """On-device static-wave path (distinct > K, e.g. prefill): waves planned+gathered on-device,
    GEMM'd and summed. No host sync. Resyncs the keep-warm state to the last wave so a following decode
    step is consistent. Lossless (each active expert is served in exactly one wave).

    Outside graph capture the waves are DOUBLE-BUFFERED (banked K//2 slots, decide+gather on a transfer
    stream overlapping the previous wave's GEMM, per-bank idx buffers so the next decide can't race the
    current remap). Under capture the serial full-K wave path runs unchanged — the cross-stream event
    choreography is not worth capturing.
    """
    hidden = _scratch_prefill_apply(method, layer, dispatch_output, topk_ids)
    if hidden is not None:
        return hidden
    pager = method._pager
    E, K = pager.E, pager.K
    half = K // 2
    banked = half > 0 and not torch.cuda.is_current_stream_capturing()
    if not banked:
        nwaves = (E + K - 1) // K
        out = None
        for w in range(nwaves):
            pager.decide_and_page_wave_ondevice(topk_ids, w)
            remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
            partial = _gemm_hidden(
                method, layer, dispatch_output, remap, clone_hidden=True
            )
            out = partial if out is None else out + partial
        lo = (nwaves - 1) * K
        pager.resync_residency_ondevice(lo, min(K, E - lo))
        return out

    ts, ev_h2d, ev_gemm, idx_banks = pager.wave_ctx()
    cs = torch.cuda.current_stream()
    ts.wait_stream(
        cs
    )  # topk_ids (and wave 0's _prep copy) depend on compute-stream work
    nwaves = (E + half - 1) // half
    out = None
    for w in range(nwaves):
        b = w & 1
        with torch.cuda.stream(ts):
            ts.wait_event(ev_gemm[b])  # bank b free once wave w-2's GEMM read it
            pager.decide_and_page_wave_ondevice(
                topk_ids, w, wave_k=half, slot_base=b * half, idx_out=idx_banks[b]
            )
            ev_h2d[b].record(ts)
        cs.wait_event(ev_h2d[b])
        remap = mask_and_remap_expert_ids(topk_ids, idx_banks[b])
        partial = _gemm_hidden(method, layer, dispatch_output, remap, clone_hidden=True)
        ev_gemm[b].record(cs)
        out = partial if out is None else out + partial
    lo = (nwaves - 1) * half
    pager.resync_residency_ondevice(lo, E - lo, slot_base=((nwaves - 1) & 1) * half)
    return out


def paged_apply(method, layer, dispatch_output):
    """Dispatch the step to the method's decode placement (eager host vs captured on-device).

    The placement (``method._placement``) owns the decide + page-in flow; both end in ``_gemm_hidden``
    over the K-slot pool. See ``placement.py``.
    """
    return method._placement.apply(method, layer, dispatch_output)
