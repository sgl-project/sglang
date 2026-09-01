"""Layerwise PD KV transfer hooks for the model forward loop.

When ``SGLANG_DISAGG_LAYERWISE`` is enabled, the model forward loop calls
these helpers at each layer boundary so the active PD KV senders can
dispatch the layer's KV cache the moment its attention op completes,
overlapping the RDMA write with the next layer's compute.  On the decode
side the receiver arms per-layer readiness events before the forward and
the model loop waits on each layer before its attention op reads the KV.

The helpers read the senders / receiver from the active ForwardContext,
so they are no-ops on non-disaggregation workers and on workers that have
not opted into layerwise transfer.  Models call them unconditionally; the
guard lives here.

Multi-request batches: ``layerwise_save_kv_layer`` iterates over ALL
layerwise-enabled senders in the batch (one per request) so every
request's KV is dispatched per layer, not just the first request's.
"""

from __future__ import annotations

import logging

from sglang.srt.model_executor.forward_context import (
    # get_disagg_kv_receiver,
    get_disagg_kv_senders,
)

logger = logging.getLogger(__name__)


def layerwise_start_send(num_layers: int) -> None:
    """Notify all active senders that a layerwise forward of *num_layers*
    layers is about to begin."""
    for sender in get_disagg_kv_senders():
        if sender.is_layerwise_enabled:
            sender.start_layerwise_send(num_layers)


def layerwise_save_kv_layer(layer_id: int) -> None:
    """Notify all active senders that layer *layer_id*'s attention has
    completed and its KV cache is ready to transfer."""
    for sender in get_disagg_kv_senders():
        if sender.is_layerwise_enabled:
            sender.save_kv_layer(layer_id)


def layerwise_wait_transfer_done() -> None:
    """Make the compute stream wait for all queued transfer-stream RDMA
    work to complete.  Called before EP combine communication to avoid
    RDMA / HCCL network resource contention."""
    for sender in get_disagg_kv_senders():
        if sender.is_layerwise_enabled:
            sender.wait_compute_on_transfer()


def layerwise_finalize_send() -> None:
    """Notify all active senders that the forward has finished and no more
    per-layer save calls will arrive."""
    for sender in get_disagg_kv_senders():
        if sender.is_layerwise_enabled:
            sender.finalize_layerwise_send()
