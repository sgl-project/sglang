"""Paged-experts residency kernels: the per-step decision and the expert page-in on the GPU.

Used by ``sglang.srt.layers.moe.paged_experts`` so a decode step pages experts without a host
sync and can be captured in a CUDA graph. All index tensors are int32 on the GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "paged_experts",
        cuda_files=["moe/paged_experts.cuh"],
        cuda_wrappers=[
            ("paged_experts_decide", "paged_experts_decide"),
            ("paged_experts_gather", "paged_experts_gather"),
            (
                "paged_experts_host_device_pointer",
                "paged_experts_host_device_pointer",
            ),
        ],
    )


def paged_experts_decide(
    topk: torch.Tensor,
    step: torch.Tensor,
    slot_expert: torch.Tensor,
    expert_slot: torch.Tensor,
    slot_lastuse: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    count: torch.Tensor,
) -> None:
    """Make every expert in ``topk`` (``[T]`` routed ids, negative = padding, ``T <= K``) resident.

    Keeps the resident ones, gives each missing one the least recently used slot the step does
    not need, and updates the residency state in place: ``step`` ``[1]``, ``slot_expert`` and
    ``slot_lastuse`` ``[K]``, ``expert_slot`` ``[E]``. Writes the page-in plan to ``src``/``dst``
    ``[K]`` (expert -> slot) and its length to ``count`` ``[1]``.
    """
    _jit_module().paged_experts_decide(
        topk, step, slot_expert, expert_slot, slot_lastuse, src, dst, count
    )


def paged_experts_gather(
    hosts: torch.Tensor,
    gpus: torch.Tensor,
    words: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    count: torch.Tensor,
) -> None:
    """Copy experts ``src[i]`` into slots ``dst[i]`` for ``i < count[0]``, for every paged tensor.

    ``hosts``/``gpus``/``words`` are int64 GPU tensors with one entry per paged tensor: the UVA
    device pointer of its pinned host store (``paged_experts_host_device_pointer``), the address of
    its GPU table, and its size per expert in 4-byte words. Rows of a multiple of 16 bytes are
    copied in 16-byte loads.
    """
    _jit_module().paged_experts_gather(hosts, gpus, words, src, dst, count)


def paged_experts_host_device_pointer(pinned: torch.Tensor) -> int:
    """The address under which the GPU reads the pinned host tensor ``pinned``."""
    return int(_jit_module().paged_experts_host_device_pointer(pinned))
