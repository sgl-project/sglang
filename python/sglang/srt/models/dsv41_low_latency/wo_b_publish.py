# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""wo_b's GEMM epilogue stores its rows straight into the mailbox of the Lamport
all-reduce + mHC kernel, so the attention boundary runs only the consumer."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.models.dsv41_low_latency.switch import MXFP8_MAX_ROWS


class WoBPublisher:
    """One mailbox and publishing GEMM per process, built collectively on the
    first eager forward: symmetric memory cannot be allocated during capture."""

    def __init__(self, *, hidden: int, hc_mult: int, top_k: int):
        self._hidden = hidden
        self._hc_mult = hc_mult
        self._top_k = top_k
        self._all_reduce = None
        self._gemm = None
        self._built = False
        # The partial wo_b rows the epilogue published, until the post consumes them.
        self._published: Optional[torch.Tensor] = None

    def _build(self, wo_b: torch.nn.Module) -> None:
        from sglang.srt.models.dsv41_low_latency.lamport.all_reduce_mhc import (
            AllReduceMHC,
        )
        from sglang.srt.models.dsv41_low_latency.lamport.publishing_gemm import (
            PublishingMxfp8Linear,
        )
        from sglang.srt.models.dsv41_low_latency.mxfp8_linear import pinned_tactics
        from sglang.srt.models.dsv41_low_latency.switch import WO_B_MAILBOX_MAX_TOKENS

        self._built = True
        self._all_reduce = AllReduceMHC(
            hidden_size=self._hidden,
            hc_mult=self._hc_mult,
            max_num_tokens=WO_B_MAILBOX_MAX_TOKENS,
            top_k=self._top_k,
            device=wo_b.weight.device,
        )
        self._gemm = PublishingMxfp8Linear(self._all_reduce)
        # Compile before capture; a publish without its consumer would desync the
        # mailbox, so nothing launches. Under an autotune pass this tunes wo_b, so
        # it offers only the pinned tactics the plain projections use.
        with pinned_tactics():
            for num_tokens in range(1, MXFP8_MAX_ROWS + 1):
                self._gemm.precompile(wo_b, num_tokens)

    def publish(self, wo_b: torch.nn.Module, x) -> Optional[torch.Tensor]:
        """wo_b's partial rows, also stored into the mailbox; None to decline. The
        caller publishes only when the attention post runs the fused all-reduce."""
        if not self._built:
            if torch.cuda.is_current_stream_capturing():
                return None
            self._build(wo_b)
        if self._published is not None:
            # The previous rows reached an unfused completion; the mailbox would desync.
            raise RuntimeError("published wo_b rows were not reduced by the fused post")
        rows = (x[0] if isinstance(x, tuple) else x).shape[0]
        if not self._gemm.supports(wo_b, rows):
            return None
        prepared = self._gemm.prepare(wo_b, x)
        if prepared is None:
            return None
        self._published = self._gemm(wo_b, prepared)
        return self._published

    def take(self, partial: torch.Tensor) -> bool:
        """Whether ``partial`` is the published output; its consumer must run next."""
        if self._published is None or partial is not self._published:
            return False
        self._published = None
        return True

    def reduce(self, partial, residual, post_mix, comb, pre, norm_weight, norm_eps):
        """The consumer of `take`n rows: (updated streams, normalized next input)."""
        return self._all_reduce(
            partial, residual, post_mix, comb, pre, norm_weight.detach(), norm_eps
        )


_publisher: Optional[WoBPublisher] = None


class _LayerPublish:
    """The publisher bound to one decoder layer: it publishes only when that layer's
    attention boundary leaves the TP sum to the fused post (the mailbox consumer)."""

    def __init__(self, layer: torch.nn.Module):
        self._layer = layer

    def __call__(self, wo_b: torch.nn.Module, x) -> Optional[torch.Tensor]:
        rows = (x[0] if isinstance(x, tuple) else x).shape[0]
        if not self._layer.defers_attn_sum(rows):
            return None
        return _publisher.publish(wo_b, x)


def install(layer: torch.nn.Module) -> None:
    """Let ``layer``'s attention publish wo_b into the shared mailbox."""
    global _publisher
    if _publisher is None:
        _publisher = WoBPublisher(
            hidden=layer.hc_cfg.hidden,
            hc_mult=layer.hc_cfg.mult,
            top_k=layer.config.num_experts_per_tok,
        )
    # A plain object, so the layer does not become a submodule of its attention.
    layer.self_attn.wo_b_publish = _LayerPublish(layer)


def take(partial: torch.Tensor) -> bool:
    return _publisher is not None and _publisher.take(partial)


def reduce(partial, residual, post_mix, comb, pre, norm_weight, norm_eps):
    return _publisher.reduce(
        partial, residual, post_mix, comb, pre, norm_weight, norm_eps
    )
