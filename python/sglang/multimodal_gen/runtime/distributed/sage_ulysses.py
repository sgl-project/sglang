# SPDX-License-Identifier: Apache-2.0
"""Prepared Sage2 QKV exchange and output gather over NCCL.

Attention is a caller-supplied consumer of FlashInfer's UlyssesQKV record.
This module owns communication workspaces and buffers, without attention
routing, sparse schedules, masks, layout conversion or kernel selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from flashinfer.comm.ulysses import UlyssesQKV


class _PreparedSageTransport:
    def __init__(self, group, query, used):
        import torch.distributed as dist
        from flashinfer.comm import UlyssesCommunicator

        if not hasattr(UlyssesCommunicator, "prepare_qkv"):
            raise ImportError(
                "Prepared Sage Ulysses requires FlashInfer with the prepared Sage2 "
                "QKV API (flashinfer-ai/flashinfer#4924); the pinned 0.7.3 "
                "release does not provide it"
            )
        self._local_used = max(
            0, min(query.shape[0], used - dist.get_rank(group) * query.shape[0])
        )
        # 3 * Q bytes conservatively covers the complete combined lowp operand,
        # including the received payload and scales, not just one source shard.
        # Keep both QKV exchange and output gather on the validated NCCL path.
        self.comm = UlyssesCommunicator(
            group,
            max_bytes=3 * query.numel() * query.element_size(),
            dtype=query.dtype,
            device=query.device,
            backend="nccl",
        )
        self.workspace = None
        self.encoded = None
        self.gathered = None
        try:
            self.workspace = self.comm.prepare_qkv(
                (1, *query.shape), quantization="sage2", used_sequence=used
            )
            self.gathered = torch.empty(
                (1, *query.shape), dtype=query.dtype, device=query.device
            )
        except BaseException:
            self.close()
            raise

    def exchange(self, q, k, v):
        # prepare_qkv requires zero input rows beyond the global live prefix.
        # These are disposable projection rows; clear only padding in place.
        if self._local_used < q.shape[0]:
            for operand in (q, k, v):
                operand[self._local_used :].zero_()
        self.encoded = self.comm.scatter_qkv(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            workspace=self.workspace,
            out=self.encoded,
        )
        return self.encoded

    def gather(self, output):
        return self.comm.gather_heads(output, out=self.gathered)

    def close(self):
        comm, self.comm = self.comm, None
        try:
            if comm is not None:
                comm.close()
        finally:
            # Pure NCCL close is local and owns no prepared buffers. Drop our
            # references before allocating a replacement geometry. Returned
            # output views retain their own storage through PyTorch.
            self.workspace = None
            self.encoded = None
            self.gathered = None


_TRANSPORTS: dict[tuple, tuple[tuple, _PreparedSageTransport]] = {}
_HARDWARE_SUPPORT: dict[tuple, bool] = {}


def _hardware_supported_by_all_ranks(group, device):
    # Negotiate once: a rank-local architecture rejection could leave peers
    # entering prepared collectives on a mixed-GPU group.
    import torch.distributed as dist

    key = (group, device)
    if key not in _HARDWARE_SUPPORT:
        supported = torch.version.hip is None and torch.cuda.get_device_capability(
            device
        ) == (12, 0)
        vote = torch.tensor([int(supported)], dtype=torch.int32, device=device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
        _HARDWARE_SUPPORT[key] = bool(vote.item())
    return _HARDWARE_SUPPORT[key]


def _get_transport(group, query, used):
    # Serialized inference reuses a geometry across layers and steps. Keep
    # only the current geometry for each group/device, releasing its buffers
    # before preparing a different shape, dtype or live prefix. Request order
    # and geometry must agree on all ranks, as for ordinary Ulysses.
    key = (group, query.device)
    geometry = (tuple(query.shape), query.dtype, used)
    cached = _TRANSPORTS.get(key)
    if cached is not None:
        cached_geometry, transport = cached
        if cached_geometry == geometry:
            return transport
        del _TRANSPORTS[key]
        transport.close()
    # Publish only a fully initialized transport. A preparation failure leaves
    # no stale entry, so a later call can retry the collective consistently.
    transport = _PreparedSageTransport(group, query, used)
    _TRANSPORTS[key] = (geometry, transport)
    return transport


def shutdown_prepared_sage_ulysses():
    transports = list(_TRANSPORTS.values())
    _TRANSPORTS.clear()
    _HARDWARE_SUPPORT.clear()
    for _, transport in transports:
        transport.close()


def prepared_sage_ulysses_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    group,
    used_sequence: int,
    attention: Callable[[UlyssesQKV], torch.Tensor],
    transport=None,
) -> torch.Tensor:
    """Exchange QKV, run the supplied attention consumer, and gather output.

    The caller admits compatible, rank-consistent geometry before invoking
    this collective path. ``attention`` receives FlashInfer's UlyssesQKV,
    including its layout, scales and live/logical sequence lengths, without
    dequantization or repacking here. It returns floating output in global
    BSHD/head-sharded layout matching ``operands.q.shape`` and
    ``operands.input_dtype``, with disposable padding rows cleared.

    A compatible dense Sage or sparse CAKE consumer can be supplied. Errors
    after preparation/exchange propagate; no backend fallback occurs here.
    """
    if transport is None:
        transport = _get_transport(group, query, used_sequence)
    operands = transport.exchange(query, key, value)
    output = attention(operands)
    return transport.gather(output)[0]
