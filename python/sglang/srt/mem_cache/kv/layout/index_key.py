"""A sparse-attention indexer's keys as a part of a composite unit: one blob per
page per layer (the page's keys, then their scales), addressed by physical page,
read and written by the paged index kernels."""

from typing import Tuple

import torch

from sglang.srt.mem_cache.kv.decl import INDEX_K_PART, IndexKeyGeometry
from sglang.srt.mem_cache.kv.layout.composite import CompositeSubPoolSpec


class IndexKeyView:
    """Per layer, the index-key blobs of a composite sub-pool, ``(pages,
    page_size * bytes_per_token)`` bytes whose pages lie a unit apart. Its
    per-layer methods take the layer's index among the part's layers."""

    index_kpool = 1
    index_kpool_compress = False
    tail_extra_slots = 0
    quant_block_size = IndexKeyGeometry.quant_block_size

    def __init__(
        self,
        *,
        arena,
        spec: CompositeSubPoolSpec,
        geometry: IndexKeyGeometry,
        page_size: int,
        model_layer_ids: Tuple[int, ...],
    ):
        self._sub_pool_name = spec.name
        self.buffers = spec.part_views(arena, INDEX_K_PART)
        assert len(self.buffers) == len(model_layer_ids)
        self.model_layer_ids = tuple(model_layer_ids)
        self.page_size = page_size
        self.index_page_size = page_size
        self.device = arena.device
        self.index_head_dim = geometry.index_head_dim
        self.use_fp4_indexer = geometry.use_fp4
        self.index_bytes_per_token = geometry.bytes_per_token
        self.index_key_bytes_per_token = geometry.key_bytes_per_token

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.buffers[layer_id]

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        from sglang.kernels.ops.kvcache.dsa_index_k_cache import GetKAndS

        return GetKAndS.execute(
            self,
            self.buffers[layer_id],
            seq_len_tensor=seq_len_tensor,
            page_indices=page_indices,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        from sglang.kernels.ops.kvcache.dsa_index_k_cache import SetKAndS

        SetKAndS.execute(
            pool=self,
            buf=self.buffers[layer_id],
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
        )
