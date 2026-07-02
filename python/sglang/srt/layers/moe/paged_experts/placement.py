"""Decode-placement strategy: *where* the per-step residency decision + page-in run.

``EagerPlacement`` makes a host-side keep-warm/LRU decision and pages the misses in with ``transfer_kv``,
then runs the K-slot fused-MoE GEMM (``forward._gemm_hidden``). The decision is data-dependent, so it
runs outside any CUDA graph — Paged Experts requires ``--disable-cuda-graph`` for now. A captured,
on-device placement (no host sync, so the decode step can be graph-captured) is a follow-up that builds
on the on-device decide kernel; it is intentionally not part of this change.

Selected once per layer in ``method.make_for_layer``. A future captured placement is a new subclass — no
``use_ondevice`` bool threaded through method / pager / forward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Placement(ABC):
    """Strategy for where a paged-experts decode step decides residency + pages experts in."""

    @abstractmethod
    def apply(self, method, layer, dispatch_output):
        """Decide + page-in + run the K-slot GEMM for one step; return a ``StandardCombineInput``."""


class EagerPlacement(Placement):
    """Host decide (keep-warm + LRU/LFU) + ``transfer_kv`` page-in. Kernel-free; requires
    ``--disable-cuda-graph`` (the host decision is data-dependent, so the step is not capturable).
    """

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


def make_placement() -> Placement:
    """Paged Experts' decode placement: eager host paging (requires ``--disable-cuda-graph``). The
    captured on-device placement is a follow-up."""
    return EagerPlacement()
