"""MoE-aware weight offloading for expert layers.

Why this exists
---------------
SGLang's ``--cpu-offload-gb`` copies a module's entire ``state_dict`` to the
device on every forward pass. That is correct for a dense model; for an MoE
with 512 experts and top-10 routing it is roughly 40x more traffic than needed.

Measured on Qwen3.8-Flash-Next, RTX PRO 4000, PCIe Gen5 x16:
    rxpci 55-61 GB/s sustained, ~26 GB per token, 2.3 tok/s.
    What is actually needed: 10 x 48 x 1.31 MB = 0.63 GB per token.

How it works
------------
Nothing is moved or copied. After loading, both device and host memory are
full, so any additional allocation would trip the OOM killer. Instead the
expert tensors are excluded from the bulk transfer in ``offloader.py`` and stay
where the offloader already placed them (pinned host memory, or the GPU). This
streamer gathers exactly the selected experts into one shared staging buffer
per forward pass and renumbers the top-k ids onto it. The Triton kernel is
unchanged - it still sees one contiguous tensor.

Enabled via ``SGLANG_MOE_EXPERT_STREAM=1``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils import logger


@triton.jit
def _gather_rows_kernel(src_ptr, idx_ptr, out_ptr, row_bytes, BLOCK: tl.constexpr):
    """Copy selected rows bytewise; the source may live in host memory.

    The point is that expert ids stay on the GPU. The naive route
    (``uniq.to("cpu")`` followed by one copy per row) forces a device
    synchronization per layer - with 48 layers per token that is the single
    largest cost, and the GPU only waits. Pinned host memory is directly
    addressable from a kernel under unified addressing; SGLang does the same
    for the PLE table.
    """
    row = tl.load(idx_ptr + tl.program_id(0)).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_bytes
    vals = tl.load(src_ptr + row * row_bytes + offs, mask=mask, other=0)
    tl.store(
        out_ptr + tl.program_id(0).to(tl.int64) * row_bytes + offs, vals, mask=mask
    )


_STAGING: Dict[Tuple, torch.Tensor] = {}


@triton.jit
def _gather_rows_tab_kernel(tab_ptr, idx_ptr, out_ptr, row_bytes, BLOCK: tl.constexpr):
    """Row gather through an address table: expert id -> int64 row address (GPU or pinned host)."""
    e = tl.load(idx_ptr + tl.program_id(0)).to(tl.int64)
    base = tl.load(tab_ptr + e)
    src = base.to(tl.pointer_type(tl.uint8))
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_bytes
    vals = tl.load(src + offs, mask=mask, other=0)
    tl.store(
        out_ptr + tl.program_id(0).to(tl.int64) * row_bytes + offs, vals, mask=mask
    )


_TENSOR_NAMES = (
    "w13_qweight",
    "w2_qweight",
    "w13_scales",
    "w2_scales",
    "w13_qzeros",
    "w2_qzeros",
)
_LOGGED = False
# Work without deduplication up to this many ids (avoids the device sync).
_NO_DEDUP_LIMIT = 64
_ARANGE_CACHE: Dict[Tuple, torch.Tensor] = {}


def _cached_arange(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, str(device), dtype)
    t = _ARANGE_CACHE.get(key)
    if t is None:
        t = torch.arange(n, device=device, dtype=dtype)
        _ARANGE_CACHE[key] = t
    return t


def enabled() -> bool:
    return os.environ.get("SGLANG_MOE_EXPERT_STREAM") == "1"


def _staging(
    name: str,
    k: int,
    max_k: int,
    rest: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """One buffer per tensor kind, at full size, sliced to the first k rows.

    Important: do NOT allocate one buffer per distinct k. The number of
    selected experts varies from call to call (10 during decode, up to 512
    during prefill); a buffer per size fills the VRAM within a few requests.
    """
    key = (name, dtype, str(device), tuple(rest))
    buf = _STAGING.get(key)
    if buf is None:
        buf = torch.empty((max_k,) + rest, dtype=dtype, device=device)
        _STAGING[key] = buf
        logger.info(
            "MoE staging buffer %s: %s, %.0f MB",
            name,
            tuple(buf.shape),
            buf.numel() * buf.element_size() / 1e6,
        )
    return buf[:k]


class ExpertStreamer:
    """Gathers only the selected experts on each forward pass."""

    def __init__(self, layer: torch.nn.Module):
        self.layer = layer
        self.names: List[str] = []
        self.device = torch.device("cuda")
        for name in _TENSOR_NAMES:
            t = getattr(layer, name, None)
            if t is None:
                continue
            t = t.data if isinstance(t, torch.nn.Parameter) else t
            if t.numel() == 0 or t.shape[0] == 0:
                continue  # qzeros are empty tensors when sym=True
            self.names.append(name)

        global _LOGGED
        if not _LOGGED and self.names:
            _LOGGED = True
            t = getattr(layer, self.names[0])
            t = t.data if isinstance(t, torch.nn.Parameter) else t
            logger.info(
                "MoE expert streaming active: %d experts, source on %s, " "tensors %s",
                t.shape[0],
                t.device,
                ",".join(self.names),
            )

    def gather(
        self, topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flat = topk_ids.reshape(-1)
        n = flat.numel()
        if n <= _NO_DEDUP_LIMIT:
            # During decode this is 10 ids. torch.unique would have to
            # synchronize here because its output size is data-dependent, and
            # that per-layer sync is the most expensive item. Copying a
            # duplicated expert is cheaper than the sync: the renumbering is
            # then a plain arange.
            uniq = flat
            k = n
            inverse = _cached_arange(n, flat.device, topk_ids.dtype)
        else:
            uniq, inverse = torch.unique(flat, return_inverse=True)
            k = int(uniq.numel())

        out: Dict[str, torch.Tensor] = {}
        placed = getattr(self.layer, "_placed", None)
        for name in self.names:
            if placed is not None:
                proto = placed["proto"][name]  # a hot GPU tensor: shape[1:], dtype
                buf = _staging(
                    name,
                    k,
                    placed["E"],
                    tuple(proto.shape[1:]),
                    proto.dtype,
                    self.device,
                )
                row_bytes = proto[0].numel() * proto.element_size()
                BLOCK = 1024
                _gather_rows_tab_kernel[(k, triton.cdiv(row_bytes, BLOCK))](
                    placed["addr"][name],
                    uniq,
                    buf.view(torch.uint8),
                    row_bytes,
                    BLOCK=BLOCK,
                )
                out[name] = buf
                continue
            src = getattr(self.layer, name)
            src = src.data if isinstance(src, torch.nn.Parameter) else src
            buf = _staging(
                name, k, src.shape[0], tuple(src.shape[1:]), src.dtype, self.device
            )
            if src.is_cuda:
                torch.index_select(src, 0, uniq, out=buf)
            else:
                src_u8 = src.view(torch.uint8)
                buf_u8 = buf.view(torch.uint8)
                row_bytes = src_u8.numel() // src_u8.shape[0]
                BLOCK = 1024
                _gather_rows_kernel[(k, triton.cdiv(row_bytes, BLOCK))](
                    src_u8,
                    uniq,
                    buf_u8,
                    row_bytes,
                    BLOCK=BLOCK,
                )
            out[name] = buf
        return inverse.reshape(topk_ids.shape).to(topk_ids.dtype), out
