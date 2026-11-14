# SPDX-License-Identifier: Apache-2.0
"""Optional BF16/FP16 Ulysses layout transforms using FlashInfer's public API.

Communicators and registered outputs are collective resources. Cache keys must
be identical on every rank; release them before destroying the process groups.
The cache admits at most 16 distinct (group, device, dtype, payload size)
contexts and fails explicitly beyond that limit. Set
SGLANG_DIFFUSION_FLASHINFER_ULYSSES=1 to opt in; the BACKEND suffix selects
FlashInfer's auto/nvlink/nccl/pcie policy (default auto). This path takes
precedence over the built-in IPC attention pipeline. It requires a FlashInfer
build exposing UlyssesCommunicator; newer builds also fuse the QKV pack.
CPU/ROCm, autograd, compile, capture and uneven sequence shards keep the
existing transport. Both variables must agree across all participating ranks.
"""

import logging

import torch

from sglang.multimodal_gen import envs

logger = logging.getLogger(__name__)
_CONTEXTS: dict[tuple, "_Context"] = {}
_MAX_CONTEXTS = 16


def enabled() -> bool:
    return envs.SGLANG_DIFFUSION_FLASHINFER_ULYSSES


class _Context:
    def __init__(self, group, x, numel):
        try:
            from flashinfer.comm import UlyssesCommunicator
        except ImportError as exc:
            raise ImportError(
                "SGLANG_DIFFUSION_FLASHINFER_ULYSSES requires a FlashInfer "
                "build exposing flashinfer.comm.UlyssesCommunicator"
            ) from exc
        self.comm = UlyssesCommunicator(
            group,
            max_bytes=numel * x.element_size(),
            dtype=x.dtype,
            device=x.device,
            backend=envs.SGLANG_DIFFUSION_FLASHINFER_ULYSSES_BACKEND,
        )
        self.outputs = {}
        self.workspace = None
        logger.info("FlashInfer Ulysses selected backend=%s", self.comm.backend)

    def exchange(self, x, op):
        if self.comm.backend != "pcie":
            return getattr(self.comm, op)(x)
        # allocate_output is collective on PCIe. Never enter it during capture.
        key = (op, tuple(x.shape))
        out = self.outputs.get(key)
        if out is None:
            if len(self.outputs) >= _MAX_CONTEXTS:
                raise RuntimeError("FlashInfer Ulysses output cache is full")
            out = self.comm.allocate_output(x, op)
            self.outputs[key] = out
        return getattr(self.comm, op)(x, out=out).clone()


def _eligible(x):
    return (
        enabled()
        and x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.float16, torch.bfloat16)
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
    )


def _context(group, x, numel):
    key = (group, x.device, x.dtype, numel)
    context = _CONTEXTS.get(key)
    if context is None:
        if len(_CONTEXTS) >= _MAX_CONTEXTS:
            raise RuntimeError(
                "FlashInfer Ulysses geometry cache is full; restart the worker "
                "before serving more sequence geometries"
            )
        context = _Context(group, x, numel)
        _CONTEXTS[key] = context
    return context


def exchange(x, group, *, scatter: bool, head_dim: int = 2):
    """Return an independently owned layout transform, or None to fall back.

    Q, K and V exchanged separately must remain live together, even when they
    have identical shapes. PCIe's persistent registered result is copied out;
    NVLink/NCCL return a fresh result directly.
    """
    if not _eligible(x) or x.ndim != 4 or head_dim not in (1, 2):
        return None
    world = torch.distributed.get_world_size(group)
    if world <= 1:
        return x
    x = x if head_dim == 2 else x.transpose(1, 2)
    split_dim = 2 if scatter else 1
    if x.shape[split_dim] % world:
        return None
    x = x.contiguous()
    context = _context(group, x, x.numel())
    op = "scatter_heads" if scatter else "gather_heads"
    out = context.exchange(x, op)
    return out if head_dim == 2 else out.transpose(1, 2).contiguous()


def exchange_qkv(q, k, v, group):
    """One ordinary (unquantized) QKV exchange for Wan's 4D / H3's 3D QKV."""
    if not _eligible(q) or q.shape != k.shape or q.shape != v.shape:
        return None
    if (
        q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.device != k.device
        or q.device != v.device
    ):
        return None
    if q.ndim not in (3, 4):
        return None
    world = torch.distributed.get_world_size(group)
    if q.shape[-2] % world:
        return None
    is_3d = q.ndim == 3
    if is_3d:
        q, k, v = (t.unsqueeze(0) for t in (q, k, v))
    context = _context(group, q, 3 * q.numel())
    fused = getattr(context.comm, "scatter_qkv_head_chunk", None)
    if fused is not None and context.comm.backend != "pcie":
        if context.workspace is None:
            context.workspace = context.comm.create_workspace()
        out = fused(
            q,
            k,
            v,
            head_offset=0,
            head_count=q.shape[2] // world,
            workspace=context.workspace,
            out=torch.empty(
                (q.shape[0], q.shape[1] * world, q.shape[2] // world, 3 * q.shape[3]),
                dtype=q.dtype,
                device=q.device,
            ),
        )
    else:
        # Older FlashInfer and PCIe: keep each head's Q,K,V together while
        # scattering H, using only the public scatter_heads API.
        packed = torch.stack((q, k, v), dim=-2).flatten(-2)
        out = context.exchange(packed, "scatter_heads")
    if is_3d:
        # H3 already consumes strided Q/K/V slices of one fused payload.
        return out.squeeze(0).split(q.shape[-1], dim=-1)
    return tuple(t.contiguous() for t in out.split(q.shape[-1], dim=-1))


def close_all() -> None:
    """Collectively release resources while their process groups are alive."""
    for context in _CONTEXTS.values():
        context.comm.close()
    _CONTEXTS.clear()
