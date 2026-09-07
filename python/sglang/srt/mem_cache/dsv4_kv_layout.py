"""DeepSeek-V4 KV cache layout, selected by ``--dsv4-kv-layout``."""

from __future__ import annotations

from enum import Enum

from sglang.srt.runtime_context import get_memory


class DSV4KVLayout(str, Enum):
    # Separate packed-fp8 SWA / C4 / C128 pools; SWA is a paged token pool.
    PAGED = "paged"
    # One bf16 buffer per layer: a per-request SWA ring followed by the
    # compressed rows, served by the ROCm ring attention kernels.
    RING = "ring"


def dsv4_kv_layout() -> DSV4KVLayout:
    return DSV4KVLayout(get_memory().dsv4_kv_layout)


def is_dsv4_ring_kv() -> bool:
    # Plain string compare on a bag leaf: this is read inside compiled model code.
    return get_memory().dsv4_kv_layout == DSV4KVLayout.RING.value
