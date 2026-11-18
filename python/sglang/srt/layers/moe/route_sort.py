"""Fused route+sort: the radix-4 router also writes the experts' MoE sort.

At decode and verify batch sizes a MoE runner that sorts the routed tokens by
expert spends one or two launches on that sort, which also zeroes its output.
route_radix4_with_sort emits both from the routing launch, and the runner that
consumes the sort is handed it for its next experts call.

A MoE runner declares that it consumes a router-written sort by returning a
RouteSortConsumer from MoeRunnerCore.route_sort_consumer(); that consumer says
per batch size at which block_m, if any, the runner's next experts call takes
the sort; the router builds it only at a block_m the kernel is built for
(moe_route_radix4.SORT_BLOCK_MS). A model layer asks its experts' runner for
one and stages it through the consume-once stash in route_quant_handoff, so the
shared routing functions keep their signatures:

    KimiK3MoE.forward           stage(consumer) before routing, clear() after
                                the layer
    biased_grouped_topk_gpu     try_route_sort_fused() replaces route_radix4
                                when the staged consumer takes the batch
    the consuming runner core   reads staged_route_sort() to see its own
                                consumer staged
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, Tuple

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.moe import moe_route_radix4
from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe import route_quant_handoff
from sglang.srt.runtime_context import get_buffer
from sglang.srt.utils import is_device_stream_capturing

logger = logging.getLogger(__name__)


class RouteSortConsumer(Protocol):
    """The hook a MoE runner declares to consume the sort the router writes."""

    def sort_block_m(self, m: int) -> Optional[int]:
        """The block_m the runner's next experts call takes a sort of M tokens
        at, or None when it sorts on its own."""

    def accept(self, sort: dict) -> None:
        """Hand the router-written sort to the runner's next experts call."""

    def drop(self) -> None:
        """Drop a sort that no experts call consumed."""


def stage(consumer: RouteSortConsumer) -> None:
    """Ask the next radix-4 routing call for the fused sort, on the batch sizes
    the layer's consumer takes. The caller pairs this with clear() after the
    layer (try/finally)."""
    route_quant_handoff.stage_route_sort(consumer)


def clear(consumer: RouteSortConsumer) -> None:
    """Drop the request and a sort that no experts call consumed."""
    route_quant_handoff.clear()
    consumer.drop()


class _SortWorkspaces:
    """The sort's arrival counters per (device, stream): launches in flight on
    two streams must not share them. A captured graph keeps its capture stream's
    counters, so it must not replay alongside eager launches on that stream."""

    def __init__(self) -> None:
        self.by_stream: dict[tuple[torch.device, int], torch.Tensor] = {}

    def get(self, device: torch.device) -> Optional[torch.Tensor]:
        """The current stream's counters, or None when they would first be
        allocated inside a graph capture, where they are only zeroed when that
        one graph replays."""
        key = (device, torch.cuda.current_stream(device).cuda_stream)
        ws = self.by_stream.get(key)
        if ws is None:
            if is_device_stream_capturing(device):
                _warn_captured_without_workspace()
                return None
            ws = moe_route_radix4.new_sort_workspace(device)
            self.by_stream[key] = ws
        return ws


@cache_once
def _warn_captured_without_workspace() -> None:
    # A graph captured here keeps the unfused router (and the runner's own sort)
    # on every replay, which otherwise only shows up as a slower step.
    logger.warning(
        "radix-4 fused MoE sort skipped: first use on this stream is a CUDA graph "
        "capture; run the router once eagerly on the capture stream first"
    )


def _published_moe_output(n_tokens: int) -> Optional[torch.Tensor]:
    """The MoE output buffer the model published, when the kernel can zero it.

    The runner only trusts the zeroing when this is also the output its experts
    call writes.
    """
    out = zero_copy_context.ctx.moe_output
    if (
        out is not None
        and out.dim() == 2
        and out.shape[0] == n_tokens
        and out.shape[1] % 8 == 0
        and out.dtype == torch.bfloat16
        and out.is_contiguous()
        and out.data_ptr() % 16 == 0
    ):
        return out
    return None


def try_route_sort_fused(
    scores: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Route with the fused sort and hand it to the staged layer's consumer when
    that consumer takes this batch size. Returns (weights, ids) on a hit, None
    otherwise (the caller runs route_radix4).

    Caller must have checked moe_route_radix4.covered().
    """
    consumer, num_token_non_padded = route_quant_handoff.take_route_sort()
    block_m = None if consumer is None else consumer.sort_block_m(scores.shape[0])
    if block_m not in moe_route_radix4.SORT_BLOCK_MS:
        return None
    sort_ws = get_buffer("route_sort_ws", _SortWorkspaces).get(scores.device)
    if sort_ws is None:
        return None
    weights, ids, aux = moe_route_radix4.route_radix4_with_sort(
        scores=scores,
        bias=bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        sort_ws=sort_ws,
        block_m=block_m,
        moe_buf=_published_moe_output(scores.shape[0]),
        num_token_non_padded=num_token_non_padded,
    )
    consumer.accept(aux)
    return weights, ids
