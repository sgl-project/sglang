"""Compressed QSA data for a unified FULL / recurrent-state arena."""

import torch

from sglang.srt.layers.attention.qsa.config import QSAProfile
from sglang.srt.mem_cache.kv.decl import (
    DerivedPartDecl,
    FullLifetime,
    RequestPartDecl,
    TensorPartDecl,
)
from sglang.srt.mem_cache.kv.routed_cache import RoutedKVCache
from sglang.srt.mem_cache.qsa_kv_pool import QSAIndexAccess


def qsa_storage_parts(
    layer_decls, profile: QSAProfile, page_size: int, compressed_dtype: torch.dtype
) -> dict:
    """Price and allocate the full-page payload and the uncompressed request tail."""
    ratio = profile.compress_ratio
    if page_size < ratio or page_size % ratio:
        raise ValueError(
            f"QSA page size {page_size} must be a multiple of ratio {ratio}"
        )
    layer_ids = tuple(
        layer.layer_id
        for layer in layer_decls
        if any(isinstance(part.lifetime, FullLifetime) for part in layer.parts)
    )
    index_shape = (profile.kv_heads, profile.head_dim)
    compressed = TensorPartDecl(
        name="qsa_compressed_k",
        shape=(page_size // ratio, *index_shape),
        dtype=compressed_dtype,
        layer_ids=layer_ids,
    )
    pending = TensorPartDecl(
        name="qsa_pending_k",
        shape=(ratio, *index_shape),
        dtype=QSAIndexAccess.index_state_dtype,
        layer_ids=layer_ids,
    )
    rope = TensorPartDecl(name="qsa_pending_rope", shape=(ratio, 3), dtype=torch.int64)
    return {
        "derived_parts": (DerivedPartDecl(sub_pool="full", part=compressed),),
        "request_parts": (
            RequestPartDecl(part=pending),
            RequestPartDecl(part=rope),
        ),
    }


class QSAUnifiedKVCache(QSAIndexAccess, RoutedKVCache):
    """Physical FULL-page compressed keys and independently indexed request rings."""

    def __init__(self, *, profile: QSAProfile, request_memory, **kwargs):
        super().__init__(**kwargs)
        self.qsa_compress_ratio = profile.compress_ratio
        self.qsa_block_topk = profile.block_topk
        self.qsa_token_topk = profile.budget
        self.qsa_index_kv_heads = profile.kv_heads
        self.qsa_index_head_dim = profile.head_dim
        self.qsa_compressed_page_size = self.page_size // profile.compress_ratio
        # Flatten [page, groups-per-page] or [request, members-per-group].
        # Both are views over plan-owned allocations; moving a FULL page moves
        # its derived compressed keys through KVArena.move_units.
        page_memory = self._arena.page_parts[self._owner]
        compressed = page_memory.tensors["qsa_compressed_k"].flatten(1, 2)
        self.qsa_compressed_k_buffer_pool = compressed.unbind(0)
        self.qsa_compressed_dtype = compressed.dtype
        pending = request_memory.tensors["qsa_pending_k"].flatten(1, 2)
        self.qsa_key_state_buffer_pool = pending.unbind(0)
        self.qsa_rope_position_buffer = request_memory.tensors[
            "qsa_pending_rope"
        ].flatten(0, 1)

    def _transfer_full_attention_id(self, layer_id: int) -> int:
        return self.layer_router.route(layer_id).index

    def _wait_for_layer(self, layer_id: int) -> None:
        self.layer_router.read(layer_id)
