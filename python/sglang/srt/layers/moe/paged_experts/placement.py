"""Decode-placement strategy: *where* the per-step residency decision + page-in run.

Both placements end in the same K-slot fused-MoE GEMM (``forward._gemm_hidden``); they differ only in
where the per-step decide + page-in happen — and therefore whether sglang's decode CUDA graph can capture
the step:

* ``EagerPlacement`` — a host-side keep-warm/LRU decision + ``transfer_kv`` page-in. Data-dependent, so it
  runs outside any graph (requires ``--disable-cuda-graph``). Kernel-free.
* ``CapturedPlacement`` — the decide + UVA gather run on the GPU with no host sync, so the decode step is
  captured. The keep-warm vs static-wave regime is chosen from shapes alone (``num_tokens*top_k <= K``),
  which is static under capture; it needs the pager's on-device state (``setup_ondevice``), flagged by
  ``needs_ondevice_store``.

Selected once per layer (from ``--disable-cuda-graph``; see ``method.make_for_layer``). A third placement
is a new subclass — no ``use_ondevice`` bool threaded through method / pager / forward.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Placement(ABC):
    """Strategy for where a paged-experts decode step decides residency + pages experts in."""

    #: whether the pager must allocate on-device residency state (``setup_ondevice``) for this placement
    needs_ondevice_store: bool = False

    @abstractmethod
    def apply(self, method, layer, dispatch_output):
        """Decide + page-in + run the K-slot GEMM for one step; return a ``StandardCombineInput``."""


class EagerPlacement(Placement):
    """Host decide (keep-warm + LRU/LFU) + ``transfer_kv`` page-in. Kernel-free; requires
    ``--disable-cuda-graph`` (the host decision is data-dependent, so the step is not capturable).
    """

    needs_ondevice_store = False

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.paged_experts.forward import (
            _gemm_hidden,
            _wave_apply,
            mask_and_remap_expert_ids,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        distinct = pager.distinct_active(topk_ids)
        if len(distinct) <= pager.K:  # keep-warm: page only the misses
            src, dst = pager.decide_keep_warm(topk_ids, distinct=distinct)
            pager.page_in(src, dst)
            remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
            hidden = _gemm_hidden(
                method, layer, dispatch_output, remap, clone_hidden=False
            )
        else:  # distinct > K: serve in waves, sum the partials (lossless)
            hidden = _wave_apply(method, layer, dispatch_output, topk_ids, distinct)
        return StandardCombineInput(hidden_states=hidden)


def _keep_warm_gemm(method, layer, dispatch_output, pager):
    """The keep-warm GEMM tail shared by the captured placements: fused remap+mask (ONE launch replacing
    the gather + 2x where + 2x zeros_like chain) with the python chain as fallback for weight layouts the
    kernel doesn't handle."""
    from sglang.srt.layers.moe.paged_experts.forward import (
        _gemm_hidden,
        _gemm_hidden_fused,
        mask_and_remap_expert_ids,
    )

    topk_output = dispatch_output.topk_output
    fused = pager.remap_mask_ondevice(topk_output.topk_ids, topk_output.topk_weights)
    if fused is not None:
        return _gemm_hidden_fused(
            method, layer, dispatch_output, fused[0], fused[1], clone_hidden=False
        )
    remap = mask_and_remap_expert_ids(
        topk_output.topk_ids, pager.logical_to_gpu_index_cuda
    )
    return _gemm_hidden(method, layer, dispatch_output, remap, clone_hidden=False)


_WAVE_CAPTURE_WARNED = False


def _warn_wave_capture_once(pager, topk_ids):
    """Full-pin path: a capture batch with ``bs*top_k > K`` is servable (on-device waves) but pays
    ``ceil(E/K)`` GEMMs per layer per step — a silent multi-x cliff. Say so once, with the fix.
    """
    global _WAVE_CAPTURE_WARNED
    import torch

    if _WAVE_CAPTURE_WARNED or not torch.cuda.is_current_stream_capturing():
        return
    _WAVE_CAPTURE_WARNED = True
    bs, top_k = topk_ids.shape[0], topk_ids.shape[-1]
    nwaves = (pager.E + pager.K - 1) // pager.K
    logger.warning(
        "[paged-experts] capture batch bs=%d exceeds the keep-warm bound (bs*top_k=%d > K=%d): decode at "
        "this batch size serves every MoE layer in %d waves (~%dx the expert GEMM cost). Cap "
        "--cuda-graph-max-bs at %d (K//top_k) to keep captured decode in the keep-warm regime.",
        bs,
        bs * top_k,
        pager.K,
        nwaves,
        nwaves,
        max(1, pager.K // top_k),
    )


class CapturedPlacement(Placement):
    """On-device decide + UVA gather, run inside sglang's captured decode graph (no host sync). The
    keep-warm vs static-wave regime is chosen from shapes alone (``num_tokens*top_k <= K``).
    """

    needs_ondevice_store = True

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.paged_experts.forward import _ondevice_wave_apply
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        keep_warm = topk_ids.shape[0] * topk_ids.shape[-1] <= pager.K
        if keep_warm:
            pager.decide_and_page_ondevice(topk_ids)
            hidden = _keep_warm_gemm(method, layer, dispatch_output, pager)
        else:  # distinct can exceed K (prefill / big batch): static waves, summed
            _warn_wave_capture_once(pager, topk_ids)
            hidden = _ondevice_wave_apply(method, layer, dispatch_output, topk_ids)
        return StandardCombineInput(hidden_states=hidden)


def _reject_wave_under_capture(pager, topk_ids):
    """The windowed placements' distinct>K fallback is a HOST wave (syncs) — fine for prefill, fatal
    inside a decode graph capture. Fires when a capture batch needs more distinct experts than the K-slot
    pool holds; fail with the fix instead of letting the raise abort capture mid-region (which resurfaces
    as a cryptic cudaErrorStreamCaptureUnjoined).

    ``topk_ids.shape[0]`` is the captured TOKEN count: for a plain decode graph that equals the batch
    size, but for a speculative TARGET_VERIFY graph it is ``batch_size * num_draft_tokens`` — so capping
    ``--cuda-graph-max-bs`` alone does NOT shrink it under spec; the tree width must shrink too.
    """
    import torch

    if torch.cuda.is_current_stream_capturing():
        num_tokens, top_k = topk_ids.shape[0], topk_ids.shape[-1]
        raise RuntimeError(
            f"Paged Experts (windowed): the captured batch needs up to num_tokens*top_k="
            f"{num_tokens * top_k} distinct-expert slots but the pool has only K={pager.K}, and the "
            f"windowed wave fallback cannot be captured. Reduce the captured token count so "
            f"num_tokens*top_k <= K: cap --cuda-graph-max-bs (and, for speculative decoding, also "
            f"--speculative-num-draft-tokens — the verify graph captures batch_size*num_draft_tokens "
            f"tokens, so with top_k={top_k} and K={pager.K} keep num_draft_tokens <= {max(1, pager.K // top_k)} "
            f"at batch size 1), or run with --disable-cuda-graph."
        )


class CapturedWindowedPlacement(Placement):
    """Captured decode for the pinned-WINDOW store (the >pin-ceiling fallback). Keep-warm decode runs the
    on-device ``decide_bounded`` + windowed gather: window hits gather in-graph from ``host_hot``, while cold
    (window-missing) experts are deferred and staged out-of-graph by the replay-twice post-replay hook
    (registered when the pager set up its window state). The rare ``distinct > K`` step (prefill / big batch)
    falls back to the eager host wave path — the window store pages hot via ``transfer_kv`` and cold via an
    indexed copy — since prefill is one-shot and not on the captured decode path."""

    needs_ondevice_store = True

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.paged_experts.forward import _wave_apply
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        keep_warm = topk_ids.shape[0] * topk_ids.shape[-1] <= pager.K
        if keep_warm:
            pager.decide_and_page_bounded_ondevice(topk_ids)
            hidden = _keep_warm_gemm(method, layer, dispatch_output, pager)
        else:  # prefill / big batch: eager host wave (window store pages hot+cold); not captured
            _reject_wave_under_capture(pager, topk_ids)
            distinct = pager.distinct_active(topk_ids)
            hidden = _wave_apply(method, layer, dispatch_output, topk_ids, distinct)
        return StandardCombineInput(hidden_states=hidden)


_bcg_break = None


def _bcg_cold_break():
    """The eager break that stages a windowed layer's cold experts (BCG break-and-page-in). Wrapped with
    ``eager_on_graph`` so, under breakable-decode capture, calling it ends the decide+gather segment, runs
    the staging eager (host_cold -> slots), and starts the GEMM segment — eliminating the replay-twice
    second full-graph replay. Built lazily (eager_on_graph hard-raises off CUDA)."""
    global _bcg_break
    if _bcg_break is None:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            eager_on_graph,
        )

        def _stage(pager, hidden_states):
            pager.stage_cold_at_break()  # side effect: refill cold into slots + update residency maps
            # Return None on purpose: a pass-through tensor made the backend's output-copy launch a
            # redundant D2D self-copy every break (48/token); None falls through with no copy.

        _bcg_break = eager_on_graph(True)(_stage)
    return _bcg_break


class CapturedWindowedBCGPlacement(Placement):
    """Captured windowed decode under the *breakable* backend (BCG break-and-page-in). Same on-device
    decide_bounded + windowed gather as the replay-twice variant, but the deferred cold experts are staged
    at an in-layer eager break (between decide and the expert GEMM) — so a cold miss is paged inline in the
    same forward pass, with NO second full-graph replay. Requires --cuda-graph-backend-decode breakable.
    """

    needs_ondevice_store = True

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.paged_experts.forward import _wave_apply
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        keep_warm = topk_ids.shape[0] * topk_ids.shape[-1] <= pager.K
        if keep_warm:
            pager.decide_and_page_bounded_ondevice(
                topk_ids
            )  # segment 1: decide + window-hit gather
            # eager break: stage this step's cold experts into their slots, then the GEMM segment runs with
            # them resident (no replay-twice). Called for its side effect + the segment boundary; it passes
            # hidden_states through unchanged (the GEMM below reads the same fixed-address buffer), so the
            # return is ignored (dispatch_output.hidden_states is a read-only property).
            _bcg_cold_break()(pager, dispatch_output.hidden_states)
            # Remap AFTER the break: the fused remap_mask reads the LIVE map, so it sees the experts the
            # break just staged (segment 2 of the broken graph).
            hidden = _keep_warm_gemm(method, layer, dispatch_output, pager)
        else:  # prefill / big batch: eager host wave (not on the captured decode path)
            _reject_wave_under_capture(pager, topk_ids)
            distinct = pager.distinct_active(topk_ids)
            hidden = _wave_apply(method, layer, dispatch_output, topk_ids, distinct)
        return StandardCombineInput(hidden_states=hidden)


def make_placement(
    use_ondevice: bool, windowed: bool = False, breakable_decode: bool = False
) -> Placement:
    """Captured when CUDA graphs are on (and a pinned store is available), else eager host. A windowed
    (>pin-ceiling) store uses the captured replay-twice variant when on-device — or the BCG break-and-page-in
    variant when decode runs under the breakable backend (no second full-graph replay).
    """
    if not use_ondevice:
        return EagerPlacement()
    if windowed:
        return (
            CapturedWindowedBCGPlacement()
            if breakable_decode
            else CapturedWindowedPlacement()
        )
    return CapturedPlacement()
