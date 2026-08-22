"""Platform-facing description of a token-to-KV pool to build.

``KVPoolRequest`` is the input to :meth:`SRTPlatform.build_kv_pool` — the
single seam through which a platform (in-tree NPU, out-of-tree plugins)
provides its own KV pool implementations. The configurator computes every
field once and the platform constructs the pool itself, so pool
``__init__`` signatures stay private to each platform: adding a field here
never breaks a platform that ignores it (unlike the deprecated
``get_*_kv_pool_cls`` class hooks, whose classes had to accept whatever
kwargs the call site passed).

The struct intentionally carries plain scalars rather than ``ModelConfig``
or ``ServerArgs`` so platforms stay decoupled from their internals.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch


class KVPoolRequest(msgspec.Struct, frozen=True, kw_only=True):
    """What the requested KV pool must hold, plus selection context.

    ``kind`` names the pool family the in-tree default would use:
    ``"mha"``, ``"mla"``, or ``"dsa"``. A platform returns ``None`` from
    ``build_kv_pool`` for any request it has no opinion on.
    """

    kind: str
    # Pool shape.
    size: int  # token capacity
    page_size: int
    dtype: torch.dtype
    device: str
    layer_num: int
    start_layer: int
    end_layer: int
    enable_memory_saver: bool
    enable_kv_cache_copy: bool = False
    # Whether the pool must defer its backing allocation until after graph
    # capture (torch-memory-saver post-capture backing).
    post_capture_active: bool = False
    # MHA fields (zero when kind != "mha").
    head_num: int = 0
    head_dim: int = 0
    v_head_dim: int = 0
    # MLA / DSA fields (zero when kind == "mha").
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    # Set only for DSA-indexed models; plain-MLA requests carry None so
    # platforms without an indexer path never see a phantom dimension.
    index_head_dim: Optional[int] = None
    kv_cache_dim: Optional[int] = None  # DSA only
    # Selection context: what the in-tree default selection would honor.
    # A platform that cannot express ``layout`` must either honor it or
    # raise; silently building another layout is how the old class-getter
    # priority chain went wrong.
    layout: str = "contiguous"  # "contiguous" | "page_major"
    kv_cache_dtype_str: str = ""
    attention_backend: str = ""
    is_hybrid_swa: bool = False
    swa_size: Optional[int] = None
    # True when this pool backs the full-attention layers inside a
    # hybrid-linear composite (HybridLinearKVPool) rather than standing alone.
    is_full_attention_leaf: bool = False
